mod blosc;
mod ome;
//#[cfg(test)]
pub mod test;

pub use test::ConnectedFullMapVolume2;

pub use ome::OmeZarrContext;

use crate::volume::{ColorScheme, FourColors, PaintVolume, VoxelVolume};
use blosc::{BloscChunk, BloscContext};
use derive_more::Debug;
use egui::Color32;
use ehttp::Request;
use serde::{Deserialize, Serialize};
use sha2::Digest;
use sha2::Sha256;
use std::cell::RefCell;
use std::sync::atomic::Ordering;
use std::{
    collections::{HashMap, HashSet},
    ops::{Deref, DerefMut},
    sync::{Arc, Mutex},
};

#[derive(Serialize, Deserialize, Debug, Clone)]
enum ZarrDataType {
    #[serde(rename = "|u1")]
    U1,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
enum ZarrVersion {
    #[serde(rename = "2")]
    V2 = 2,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
enum ZarrOrder {
    #[serde(rename = "C")]
    ColumnMajor,
    #[serde(rename = "F")]
    RowMajor,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
enum ZarrCompressionName {
    #[serde(rename = "lz4")]
    Lz4,
    #[serde(rename = "zstd")]
    Zstd,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
enum ZarrCompressorId {
    #[serde(rename = "blosc")]
    Blosc,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ZarrCompressor {
    blocksize: u8,
    clevel: u8,
    #[serde(rename = "cname")]
    compression_name: ZarrCompressionName,
    id: String,
    shuffle: u8,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ZarrFilters {}
#[derive(Serialize, Deserialize, Debug, Clone)]
struct ZarrArrayDef {
    chunks: Vec<usize>,
    compressor: Option<ZarrCompressor>,
    dtype: String,
    fill_value: u8,
    filters: Option<ZarrFilters>,
    order: ZarrOrder,
    shape: Vec<usize>,
    zarr_format: u8,
    dimension_separator: Option<String>,
}

#[derive(Clone)]
pub struct ZarrArray<const N: usize, T> {
    access: Arc<dyn ZarrFileAccess>,
    def: ZarrArrayDef,
    phantom_t: std::marker::PhantomData<T>,
}

trait ZarrFileAccess: Send + Sync + Debug {
    fn load_array_def(&self) -> ZarrArrayDef;
    //fn load_chunk(&self, array_def: &ZarrArrayDef, chunk_no: &[usize]) -> Option<BloscChunk<u8>>;
    fn chunk_file_for(&self, array_def: &ZarrArrayDef, chunk_no: &[usize]) -> Option<String>;
    fn cache_missing(&self) -> bool;
}

#[derive(Debug, Clone)]
struct ZarrDirectory {
    path: String,
}
impl ZarrFileAccess for ZarrDirectory {
    fn load_array_def(&self) -> ZarrArrayDef {
        let zarray = std::fs::read_to_string(format!("{}/.zarray", self.path)).unwrap();
        serde_json::from_str::<ZarrArrayDef>(&zarray).unwrap()
    }

    fn chunk_file_for(&self, array_def: &ZarrArrayDef, chunk_no: &[usize]) -> Option<String> {
        let chunk_path = format!(
            "{}/{}",
            self.path,
            chunk_no
                .iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(array_def.dimension_separator.as_deref().unwrap_or("."))
        );
        if !std::path::Path::new(&chunk_path).exists() {
            None
        } else {
            Some(chunk_path)
        }
    }
    fn cache_missing(&self) -> bool {
        true
    }
}

trait Downloader: Sync + Send + Debug {
    fn download(&self, from_url: &str, to_path: &str);
}

#[derive(Debug)]
struct SimpleDownloader {
    channel: std::sync::mpsc::Sender<(String, String)>,
}
impl SimpleDownloader {
    fn new() -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<(String, String)>();
        std::thread::spawn(move || {
            let mut ongoing: HashSet<String> = HashSet::new();
            let downloading = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            for (from, to) in rx {
                if ongoing.contains(&from.to_string()) {
                    continue;
                }
                if downloading.load(Ordering::Relaxed) > 10 {
                    continue;
                }

                ongoing.insert(from.clone());
                downloading.fetch_add(1, Ordering::Acquire);
                println!("Starting download from {} to {}", from, to);
                let inner_counter = downloading.clone();
                ehttp::fetch(Request::get(&from), move |result| {
                    println!("Downloaded from {} to {}", from, to);
                    let response = result.unwrap();
                    inner_counter.fetch_sub(1, Ordering::Acquire);
                    if response.status == 200 {
                        let data = response.bytes.to_vec();
                        std::fs::create_dir_all(std::path::Path::new(&to).parent().unwrap()).unwrap();
                        let tmp_file = format!("{}.tmp", to);
                        std::fs::write(&tmp_file, &data).unwrap();
                        std::fs::rename(tmp_file, to).unwrap();
                    } else {
                        println!("Failed to download from {}, status {}", from, response.status);
                    }
                });
            }
        });
        Self { channel: tx }
    }
}
impl Downloader for SimpleDownloader {
    fn download(&self, from_url: &str, to_path: &str) {
        self.channel.send((from_url.to_string(), to_path.to_string())).unwrap();
    }
}

#[derive(Debug, Clone)]
struct RemoteZarrDirectory {
    url: String,
    local_cache_dir: String,
    downloader: Arc<dyn Downloader>,
}
impl ZarrFileAccess for RemoteZarrDirectory {
    fn load_array_def(&self) -> ZarrArrayDef {
        let target_file = format!("{}/.zarray", self.local_cache_dir);
        if !std::path::Path::new(&target_file).exists() {
            let data = ehttp::fetch_blocking(&Request::get(&format!("{}/.zarray", self.url)))
                .unwrap()
                .bytes
                .to_vec();
            std::fs::create_dir_all(std::path::Path::new(&target_file).parent().unwrap()).unwrap();
            std::fs::write(&target_file, &data).unwrap();
        }

        let zarray = std::fs::read_to_string(&target_file).unwrap();
        serde_json::from_str::<ZarrArrayDef>(&zarray).unwrap()
    }

    fn chunk_file_for(&self, array_def: &ZarrArrayDef, chunk_no: &[usize]) -> Option<String> {
        let target_file = format!(
            "{}/{}",
            self.local_cache_dir,
            chunk_no.iter().map(|i| i.to_string()).collect::<Vec<_>>().join("/")
        );

        if std::path::Path::new(&target_file).exists() {
            Some(target_file)
        } else {
            let target_url = format!(
                "{}/{}",
                self.url,
                chunk_no
                    .iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(array_def.dimension_separator.as_deref().unwrap_or("."))
            );
            self.downloader.download(&target_url, &target_file);

            None
        }
    }
    fn cache_missing(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone)]
struct BlockingRemoteZarrDirectory {
    url: String,
    local_cache_dir: String,
}
impl ZarrFileAccess for BlockingRemoteZarrDirectory {
    fn load_array_def(&self) -> ZarrArrayDef {
        let target_file = format!("{}/.zarray", self.local_cache_dir);
        if !std::path::Path::new(&target_file).exists() {
            let data = ehttp::fetch_blocking(&Request::get(&format!("{}/.zarray", self.url)))
                .unwrap()
                .bytes
                .to_vec();
            std::fs::create_dir_all(std::path::Path::new(&target_file).parent().unwrap()).unwrap();
            std::fs::write(&target_file, &data).unwrap();
        }

        let zarray = std::fs::read_to_string(&target_file).unwrap();
        serde_json::from_str::<ZarrArrayDef>(&zarray).unwrap()
    }

    fn chunk_file_for(&self, array_def: &ZarrArrayDef, chunk_no: &[usize]) -> Option<String> {
        let target_file = format!(
            "{}/{}",
            self.local_cache_dir,
            chunk_no.iter().map(|i| i.to_string()).collect::<Vec<_>>().join("/")
        );

        if std::path::Path::new(&target_file).exists() {
            Some(target_file)
        } else {
            let target_url = format!(
                "{}/{}",
                self.url,
                chunk_no
                    .iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(array_def.dimension_separator.as_deref().unwrap_or("."))
            );
            println!("Downloading chunk from {}", target_url);
            let response = ehttp::fetch_blocking(&Request::get(&target_url)).unwrap();
            if response.status != 200 {
                println!(
                    "Failed to download chunk from {}, status {}",
                    target_url, response.status
                );
                return None;
            }
            let data = response.bytes.to_vec();
            std::fs::create_dir_all(std::path::Path::new(&target_file).parent().unwrap()).unwrap();
            let tmp = format!("{}.tmp", target_file);
            std::fs::write(&tmp, &data).unwrap();
            std::fs::rename(&tmp, &target_file).unwrap();
            Some(target_file)
        }
    }
    fn cache_missing(&self) -> bool {
        true
    }
}

impl<const N: usize> ZarrArray<N, u8> {
    fn load_chunk_context(&self, chunk_no: [usize; N]) -> Option<ChunkContext> {
        self.access
            .chunk_file_for(&self.def, &chunk_no)
            .map(|chunk_file| match &self.def.compressor {
                Some(compressor) => match compressor.id.as_str() {
                    "blosc" => ChunkContext::Blosc(BloscChunk::load(&chunk_file).into_ctx()),
                    _ => panic!("Unsupported compressor: {}", compressor.id),
                },
                _ => ChunkContext::Raw(RawContext::load(&chunk_file)),
            })
    }
    pub fn from_path(path: &str) -> Self {
        println!("Loading ZarrArray from path: {}", path);
        Self::from_access(Arc::new(ZarrDirectory { path: path.to_string() }))
    }
    pub fn from_url_blocking(url: &str, local_cache_dir: &str) -> Self {
        println!("Loading ZarrArray from url: {}", url);
        Self::from_access(Arc::new(BlockingRemoteZarrDirectory {
            url: url.to_string(),
            local_cache_dir: local_cache_dir.to_string(),
        }))
    }
    pub fn from_url(url: &str, local_cache_dir: &str) -> Self {
        println!("Loading ZarrArray from url: {} to: {} ", url, local_cache_dir);
        Self::from_access(Arc::new(RemoteZarrDirectory {
            url: url.to_string(),
            local_cache_dir: local_cache_dir.to_string(),
            downloader: Arc::new(SimpleDownloader::new()),
        }))
    }
    pub fn from_url_to_default_cache_dir(url: &str) -> Self {
        let canonical_url = if url.ends_with("/") { &url[..url.len() - 1] } else { url };
        let sha256 = format!("{:x}", Sha256::digest(canonical_url.as_bytes()));
        let local_cache_dir = std::env::temp_dir().join("vesuvius-gui").join(sha256);
        Self::from_url(url, local_cache_dir.to_str().unwrap())
    }
    fn from_access(access: Arc<dyn ZarrFileAccess>) -> Self {
        let def = access.load_array_def();
        ZarrArray {
            access,
            def,
            phantom_t: std::marker::PhantomData,
        }
    }

    pub fn into_ctx(self) -> ZarrContextBase<N> {
        let cache = Arc::new(Mutex::new(ZarrContextCache::new(&self.def)));
        let cache_missing = self.access.cache_missing();
        ZarrContextBase {
            array: self,
            cache,
            cache_missing,
        }
    }
}

pub struct ZarrContextBase<const N: usize> {
    array: ZarrArray<N, u8>,
    cache: Arc<Mutex<ZarrContextCache<N>>>,
    cache_missing: bool,
}
impl<const N: usize> ZarrContextBase<N> {
    pub fn into_ctx(&self) -> ZarrContext<N> {
        ZarrContext {
            array: self.array.clone(),
            cache_missing: self.cache_missing,
            cache: self.cache.clone(),
            state: ZarrContextState {
                last_chunk_no: [usize::MAX; N],
                last_context: None,
            }
            .into(),
            color_scheme: Box::new(FourColors {}),
        }
    }
}

pub struct RawContext {
    data: memmap::Mmap,
}
impl RawContext {
    fn load(chunk_file: &str) -> RawContext {
        let file = std::fs::File::open(chunk_file).unwrap();
        let data = unsafe { memmap::Mmap::map(&file).unwrap() };
        RawContext { data }
    }
    fn get(&self, idx: usize) -> u8 {
        self.data[idx]
    }
}

enum ChunkContext {
    Blosc(BloscContext),
    Raw(RawContext),
}
impl ChunkContext {
    fn get(&mut self, idx: usize) -> u8 {
        match self {
            ChunkContext::Blosc(ctx) => ctx.get(idx),
            ChunkContext::Raw(raw) => raw.get(idx),
        }
    }
}

struct ZarrContextCacheEntry {
    ctx: ChunkContext,
    last_access: u64,
}
impl Deref for ZarrContextCacheEntry {
    type Target = ChunkContext;
    fn deref(&self) -> &Self::Target {
        &self.ctx
    }
}
impl DerefMut for ZarrContextCacheEntry {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.ctx
    }
}

struct ZarrContextCache<const N: usize> {
    cache: HashMap<[usize; N], Option<ZarrContextCacheEntry>>,
    access_counter: u64,
    non_empty_entries: usize,
    max_entries: usize,
}
impl<const N: usize> ZarrContextCache<N> {
    fn new(def: &ZarrArrayDef) -> Self {
        ZarrContextCache {
            cache: HashMap::new(),
            access_counter: 0,
            non_empty_entries: 0,
            max_entries: 6_000_000_000 / def.chunks.iter().product::<usize>(), // FIXME: make configurable
        }
    }
    fn entry(&self, ctx: Option<ChunkContext>) -> Option<ZarrContextCacheEntry> {
        ctx.map(|ctx| ZarrContextCacheEntry {
            ctx,
            last_access: self.access_counter,
        })
    }
    fn cleanup(&mut self) {
        if self.non_empty_entries > self.max_entries {
            // FIXME: make configurable
            // purge oldest n% of entries
            let mut entries = self
                .cache
                .iter()
                .filter_map(|(k, e)| e.as_ref().map(|e| (*k, e.last_access)))
                .collect::<Vec<_>>();
            entries.sort_by_key(|(_, e)| *e);
            let n = (self.non_empty_entries as f64 * 0.2) as usize; // FIXME: make configurable
            let before = self.non_empty_entries;
            let sorted_entries_len = entries.len();
            for (k, _) in entries.into_iter().take(n) {
                if self.cache.remove(&k).is_some() {
                    self.non_empty_entries -= 1;
                }
            }
            println!(
                "Purged {} entries {} from {} (sorted: {})",
                n, self.non_empty_entries, before, sorted_entries_len
            );
        }
    }
    fn purge_missing(&mut self) {
        self.cache.retain(|_, e| if e.is_none() { false } else { true });
    }
}

struct ZarrContextState<const N: usize> {
    last_chunk_no: [usize; N],
    last_context: Option<Option<ChunkContext>>,
}

pub struct ZarrContext<const N: usize> {
    array: ZarrArray<N, u8>,
    cache_missing: bool,
    cache: Arc<Mutex<ZarrContextCache<N>>>,
    state: RefCell<ZarrContextState<N>>,
    color_scheme: Box<dyn ColorScheme>,
}

impl ZarrContext<3> {
    fn get(&self, index: [usize; 3]) -> Option<u8> {
        if index[0] > self.array.def.shape[0]
            || index[1] > self.array.def.shape[1]
            || index[2] > self.array.def.shape[2]
        {
            return None; // FIXME: or just return 0?
        }
        let chunk_no = [
            index[0] / self.array.def.chunks[0],
            index[1] / self.array.def.chunks[1],
            index[2] / self.array.def.chunks[2],
        ];
        let chunk_offset = [
            index[0] % self.array.def.chunks[0],
            index[1] % self.array.def.chunks[1],
            index[2] % self.array.def.chunks[2],
        ];

        let idx = ((chunk_offset[0] * self.array.def.chunks[1]) + chunk_offset[1]) * self.array.def.chunks[2]
            + chunk_offset[2];

        // fast path
        let last_chunk_no = self.state.borrow().last_chunk_no;
        if chunk_no == last_chunk_no {
            let mut state = self.state.borrow_mut();
            if let Some(last) = state.last_context.as_mut().unwrap() {
                Some(last.get(idx))
            } else {
                None
            }
        } else {
            // slow path goes through mutex
            self.get_from_cache(chunk_no, idx)
        }
    }
    fn get_from_cache(&self, chunk_no: [usize; 3], idx: usize) -> Option<u8> {
        let mut access = self.cache.lock().unwrap();
        let state = &mut *self.state.borrow_mut();
        access.access_counter += 1;

        if let Some(last) = state.last_context.take() {
            let entry = access.entry(last);
            if entry.is_some() {
                access.non_empty_entries += 1;
            }
            access.cache.insert(state.last_chunk_no, entry);
        }

        access.cleanup();
        let cache = &mut access.cache;

        if cache.contains_key(&chunk_no) {
            let mut entry = cache.remove(&chunk_no).unwrap();
            if entry.is_some() {
                access.non_empty_entries -= 1;
            }
            let res = if let Some(entry) = entry.as_mut() {
                let res = entry.get(idx);
                Some(res)
            } else {
                None
            };
            state.last_chunk_no = chunk_no;
            state.last_context = Some(entry.map(|e| e.ctx));
            res
        } else {
            if let Some(mut ctx) = self.array.load_chunk_context(chunk_no) {
                //let mut ctx: BloscContext = chunk.into_ctx();
                let res = ctx.get(idx);
                state.last_chunk_no = chunk_no;
                state.last_context = Some(Some(ctx));
                Some(res)
            } else {
                state.last_chunk_no = chunk_no;
                state.last_context = Some(None);
                None
            }
        }
    }
    pub fn with_color_scheme(self, color_scheme: Box<dyn ColorScheme>) -> ZarrContext<3> {
        ZarrContext {
            array: self.array,
            cache_missing: self.cache_missing,
            cache: self.cache,
            state: self.state,
            color_scheme: color_scheme,
        }
    }
}

impl PaintVolume for ZarrContext<3> {
    fn paint(
        &self,
        xyz: [i32; 3],
        u_coord: usize,
        v_coord: usize,
        plane_coord: usize,
        width: usize,
        height: usize,
        _sfactor: u8,
        paint_zoom: u8,
        _config: &crate::volume::DrawingConfig,
        buffer: &mut crate::volume::Image,
    ) {
        //assert!(_sfactor == 1);
        let _sfactor = 1;
        if !self.cache_missing {
            // clean missing entries from cache
            let mut access = self.cache.lock().unwrap();
            access.purge_missing();
        }

        let fi32 = _sfactor as f64;

        for im_u in 0..width {
            for im_v in 0..height {
                let im_rel_u = (im_u as i32 - width as i32 / 2) * paint_zoom as i32;
                let im_rel_v = (im_v as i32 - height as i32 / 2) * paint_zoom as i32;

                let mut uvw: [f64; 3] = [0.; 3];
                uvw[u_coord] = (xyz[u_coord] + im_rel_u) as f64 / fi32;
                uvw[v_coord] = (xyz[v_coord] + im_rel_v) as f64 / fi32;
                uvw[plane_coord] = (xyz[plane_coord]) as f64 / fi32;

                let [x, y, z] = uvw;

                if x < 0.0 || y < 0.0 || z < 0.0 {
                    continue;
                }

                let v = self.get([z as usize, y as usize, x as usize]).unwrap_or(0);
                if v != 0 {
                    //println!("painting at {} {} {} {}", x, y, z, v);
                    let color = self.color_scheme.get_color(v);
                    buffer.blend(im_u, im_v, color, 0.4);
                }
            }
        }
    }
}
impl VoxelVolume for ZarrContext<3> {
    fn get(&self, xyz: [f64; 3], downsampling: i32) -> u8 {
        let f = downsampling as f64;
        let v = self
            .get([(xyz[2] / f) as usize, (xyz[1] / f) as usize, (xyz[0] / f) as usize])
            .unwrap_or(0);
        if v != 0 {
            255
        } else {
            0
        }
    }
    fn get_color(&self, xyz: [f64; 3], downsampling: i32) -> Color32 {
        let f = downsampling as f64;
        let v = self
            .get([(xyz[2] * f) as usize, (xyz[1] * f) as usize, (xyz[0] * f) as usize])
            .unwrap_or(0);
        let color = self.color_scheme.get_color(v);
        color
    }
    fn get_color_interpolated(&self, xyz: [f64; 3], downsampling: i32) -> Color32 {
        self.get_color(xyz, downsampling)
    }
}

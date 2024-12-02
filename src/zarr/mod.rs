mod blosc;
#[cfg(test)]
mod test;

use crate::volume::PaintVolume;
use blosc::{BloscChunk, BloscContext};
use derive_more::Debug;
use egui::Color32;
use ehttp::Request;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
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
    compressor: ZarrCompressor,
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
    fn load_chunk(&self, array_def: &ZarrArrayDef, chunk_no: &[usize]) -> Option<BloscChunk<u8>>;
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
    fn load_chunk(&self, array_def: &ZarrArrayDef, chunk_no: &[usize]) -> Option<BloscChunk<u8>> {
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
            Some(BloscChunk::load(&chunk_path))
        }
    }
}

impl<const N: usize> ZarrArray<N, u8> {
   fn load_chunk(&self, chunk_no: [usize; N]) -> Option<BloscChunk<u8>> {
       self.access.load_chunk(&self.def, &chunk_no)
    }
    pub fn from_path(path: &str) -> Self {
       println!("Loading ZarrArray from path: {}", path);
       Self::from_access(Arc::new(ZarrDirectory { path: path.to_string() }))
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
        ZarrContextBase { array: self, cache }
    }
}

pub struct ZarrContextBase<const N: usize> {
    array: ZarrArray<N, u8>,
    cache: Arc<Mutex<ZarrContextCache<N>>>,
}
impl<const N: usize> ZarrContextBase<N> {
    pub fn into_ctx(&self) -> ZarrContext<N> {
        ZarrContext {
            array: self.array.clone(),
            cache: self.cache.clone(),
            last_chunk_no: [usize::MAX; N],
            last_context: None,
        }
    }
}

struct ZarrContextCacheEntry {
    ctx: BloscContext,
    last_access: u64,
}
impl Deref for ZarrContextCacheEntry {
    type Target = BloscContext;
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
            max_entries: 2000000000 / def.chunks.iter().product::<usize>(),
        }
    }
    fn entry(&self, ctx: Option<BloscContext>) -> Option<ZarrContextCacheEntry> {
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
}

pub struct ZarrContext<const N: usize> {
    array: ZarrArray<N, u8>,
    cache: Arc<Mutex<ZarrContextCache<N>>>,
    last_chunk_no: [usize; N],
    last_context: Option<Option<BloscContext>>,
}

impl ZarrContext<3> {
    fn get(&mut self, index: [usize; 3]) -> u8 {
        if index[0] > self.array.def.shape[0]
            || index[1] > self.array.def.shape[1]
            || index[2] > self.array.def.shape[2]
        {
            return 0;
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
        if chunk_no == self.last_chunk_no {
            if let Some(last) = self.last_context.as_mut().unwrap() {
                last.get(idx)
            } else {
                0
            }
        } else {
            // slow path goes through mutex
            self.get_from_cache(chunk_no, idx)
        }
    }
    fn get_from_cache(&mut self, chunk_no: [usize; 3], idx: usize) -> u8 {
        let mut access = self.cache.lock().unwrap();
        access.access_counter += 1;

        if let Some(last) = self.last_context.take() {
            let entry = access.entry(last);
            if entry.is_some() {
                access.non_empty_entries += 1;
            }
            access.cache.insert(self.last_chunk_no, entry);
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
                res
            } else {
                // chunk wasn't found on disk
                0
            };
            self.last_chunk_no = chunk_no;
            self.last_context = Some(entry.map(|e| e.ctx));
            res
        } else {
            if let Some(chunk) = self.array.load_chunk(chunk_no) {
                let mut ctx = chunk.into_ctx();
                let res = ctx.get(idx);
                self.last_chunk_no = chunk_no;
                self.last_context = Some(Some(ctx));
                res
            } else {
                self.last_chunk_no = chunk_no;
                self.last_context = Some(None);
                0
            }
        }
    }
}

impl PaintVolume for ZarrContext<3> {
    fn paint(
        &mut self,
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
        assert!(_sfactor == 1);
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

                let v = self.get([z as usize, y as usize, x as usize]);
                if v != 0 {
                    let color = match v {
                        1 => Color32::RED,
                        2 => Color32::GREEN,
                        3 => Color32::YELLOW,
                        _ => Color32::BLUE,
                    };
                    buffer.set(im_u, im_v, color);
                }
            }
        }
    }
}

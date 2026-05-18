pub mod blosc;
pub mod ome;
pub mod sharding;
pub mod v3;

use blosc::BloscChunk;
use dashmap::DashMap;
use derive_more::with_trait::Debug;
use directories::BaseDirs;
use ehttp::Request;
use fxhash::{FxBuildHasher, FxHashMap, FxHashSet};
use libm::modf;
pub use ome::OmeZarrContext;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use sha2::Digest;
use sha2::Sha256;
use std::cell::RefCell;
use std::fs::File;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::{
    ops::Deref,
    sync::{Arc, Mutex},
};

type HashMap<K, V> = FxHashMap<K, V>;
type HashSet<K> = FxHashSet<K>;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ZarrDataType {
    #[serde(rename = "|u1")]
    U1,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(try_from = "u8", into = "u8")]
pub enum ZarrVersion {
    V2,
}

impl TryFrom<u8> for ZarrVersion {
    type Error = String;
    fn try_from(v: u8) -> Result<Self, Self::Error> {
        match v {
            2 => Ok(ZarrVersion::V2),
            other => Err(format!("unsupported zarr_format `{}`, expected 2", other)),
        }
    }
}

impl From<ZarrVersion> for u8 {
    fn from(v: ZarrVersion) -> u8 {
        match v {
            ZarrVersion::V2 => 2,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ZarrOrder {
    #[serde(rename = "C")]
    ColumnMajor,
    #[serde(rename = "F")]
    RowMajor,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ZarrCompressionName {
    #[serde(rename = "lz4")]
    Lz4,
    #[serde(rename = "zstd")]
    Zstd,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ZarrCompressorId {
    #[serde(rename = "blosc")]
    Blosc,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ZarrCompressor {
    blocksize: u8,
    clevel: u8,
    #[serde(rename = "cname")]
    compression_name: ZarrCompressionName,
    id: ZarrCompressorId,
    shuffle: u8,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ZarrFilters {}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ZarrArrayDef {
    pub chunks: Vec<usize>,
    pub compressor: Option<ZarrCompressor>,
    pub dtype: ZarrDataType,
    pub fill_value: u8,
    pub filters: Option<ZarrFilters>,
    pub order: ZarrOrder,
    pub shape: Vec<usize>,
    pub zarr_format: ZarrVersion,
    pub dimension_separator: Option<String>,
}

#[derive(Clone)]
pub struct ZarrArray<const N: usize, T> {
    access: Arc<dyn ZarrFileAccess>,
    def: ZarrArrayDef,
    phantom_t: std::marker::PhantomData<T>,
}

pub trait ZarrFileAccess: Send + Sync + Debug {
    fn load_array_def(&self) -> ZarrArrayDef;
    fn load_chunk(&self, array_def: &ZarrArrayDef, chunk_no: &[usize]) -> Option<ChunkContext>;
    fn cache_missing(&self) -> bool;
}

/// Apply a v2 compressor to a chunk file. Hoisted out of `ZarrArray::load_chunk_context`
/// so that v3 access impls can keep their own decompression path without sharing this.
fn decompress_v2_chunk(def: &ZarrArrayDef, file: Arc<File>) -> ChunkContext {
    match &def.compressor {
        Some(compressor) => match compressor.id {
            ZarrCompressorId::Blosc => ChunkContext::Heap(BloscChunk::load_data_from_file(&file)),
        },
        None => ChunkContext::Raw(RawContext::load_from_file(&file)),
    }
}

#[derive(Debug, Clone)]
struct ZarrDirectory {
    path: String,
}
impl ZarrDirectory {
    fn chunk_file_for(&self, array_def: &ZarrArrayDef, chunk_no: &[usize]) -> Option<Arc<File>> {
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
            Some(File::open(chunk_path).unwrap().into())
        }
    }
}
impl ZarrFileAccess for ZarrDirectory {
    fn load_array_def(&self) -> ZarrArrayDef {
        let path = format!("{}/.zarray", self.path);
        let zarray = std::fs::read_to_string(&path).unwrap();
        parse_json(&zarray, &path)
    }
    fn load_chunk(&self, def: &ZarrArrayDef, chunk_no: &[usize]) -> Option<ChunkContext> {
        self.chunk_file_for(def, chunk_no).map(|f| decompress_v2_chunk(def, f))
    }
    fn cache_missing(&self) -> bool {
        true
    }
}

trait Downloader: Sync + Send + Debug {
    fn download(&self, from_url: &str, to_path: &str);
}

const SIMPLE_DOWNLOADER_WORKERS: usize = 32;

/// Per-download log messages are at `info` level but gated by `VESUVIUS_LOG_DOWNLOADS` so
/// they don't spam at the workspace's default `RUST_LOG=info`. Set the env var to any
/// value to opt in; the check is memoized at first use.
fn download_logging_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("VESUVIUS_LOG_DOWNLOADS").is_ok())
}

#[derive(Debug)]
struct SimpleDownloader {
    channel: std::sync::mpsc::SyncSender<(String, String)>,
    ongoing: Arc<Mutex<HashSet<String>>>,
}
impl SimpleDownloader {
    fn new() -> Self {
        // Rendezvous channel: a send only succeeds when a worker is parked in recv().
        // Effective in-flight cap is exactly SIMPLE_DOWNLOADER_WORKERS — no buffer = no
        // buffer bloat of stale chunks from a viewport the user has already left behind.
        let (tx, rx) = std::sync::mpsc::sync_channel::<(String, String)>(0);
        let rx = Arc::new(Mutex::new(rx));

        let client = Client::builder()
            .pool_max_idle_per_host(SIMPLE_DOWNLOADER_WORKERS)
            .pool_idle_timeout(Some(std::time::Duration::from_secs(60)))
            .http2_adaptive_window(true)
            .tcp_keepalive(Some(std::time::Duration::from_secs(30)))
            .timeout(Some(std::time::Duration::from_secs(60)))
            .build()
            .expect("failed to build reqwest client for SimpleDownloader");

        let ongoing: Arc<Mutex<HashSet<String>>> = Arc::new(Mutex::new(HashSet::default()));

        for _ in 0..SIMPLE_DOWNLOADER_WORKERS {
            let rx = rx.clone();
            let client = client.clone();
            let ongoing = ongoing.clone();
            std::thread::spawn(move || loop {
                let (from, to) = {
                    let lock = rx.lock().unwrap();
                    match lock.recv() {
                        Ok(v) => v,
                        Err(_) => return,
                    }
                };
                if download_logging_enabled() {
                    log::info!("Starting download from {} to {}", from, to);
                }
                match client.get(&from).send() {
                    Ok(response) => {
                        let status = response.status();
                        if status.as_u16() == 200 {
                            match response.bytes() {
                                Ok(bytes) => {
                                    if download_logging_enabled() {
                                        log::info!("Downloaded from {} to {}", from, to);
                                    }
                                    if let Some(parent) = std::path::Path::new(&to).parent() {
                                        if let Err(e) = std::fs::create_dir_all(parent) {
                                            log::warn!("Failed to create parent dir for {}: {}", to, e);
                                        }
                                    }
                                    let tmp_file = format!("{}.tmp", to);
                                    if let Err(e) = std::fs::write(&tmp_file, bytes.as_ref()) {
                                        log::warn!("Failed to write {}: {}", tmp_file, e);
                                    } else if let Err(e) = std::fs::rename(&tmp_file, &to) {
                                        log::warn!("Failed to rename {} -> {}: {}", tmp_file, to, e);
                                    }
                                }
                                Err(e) => log::warn!("Failed to read body from {}: {}", from, e),
                            }
                        } else if status.as_u16() == 404 {
                            // expected for sparse zarrs — missing chunks are normal
                            if download_logging_enabled() {
                                log::info!("Missing chunk (404) at {}", from);
                            }
                        } else {
                            log::warn!("Failed to download from {}, status {}", from, status.as_u16());
                        }
                    }
                    Err(e) => log::warn!("Request error for {}: {}", from, e),
                }
                ongoing.lock().unwrap().remove(&from);
            });
        }

        Self { channel: tx, ongoing }
    }
}
impl Downloader for SimpleDownloader {
    fn download(&self, from_url: &str, to_path: &str) {
        let from = from_url.to_string();
        let to = to_path.to_string();
        if !self.ongoing.lock().unwrap().insert(from.clone()) {
            return;
        }
        // On Full (all workers busy), clear the ongoing entry so paint can re-emit later.
        if self.channel.try_send((from.clone(), to)).is_err() {
            self.ongoing.lock().unwrap().remove(&from);
        }
    }
}

#[derive(Debug, Clone)]
struct RemoteZarrDirectory {
    url: String,
    local_cache_dir: String,
    downloader: Arc<dyn Downloader>,
}
impl RemoteZarrDirectory {
    fn chunk_file_for(&self, array_def: &ZarrArrayDef, chunk_no: &[usize]) -> Option<Arc<File>> {
        let target_file = format!(
            "{}/{}",
            self.local_cache_dir,
            chunk_no.iter().map(|i| i.to_string()).collect::<Vec<_>>().join("/")
        );

        if std::path::Path::new(&target_file).exists() {
            Some(File::open(target_file).unwrap().into())
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
        parse_json(&zarray, &target_file)
    }
    fn load_chunk(&self, def: &ZarrArrayDef, chunk_no: &[usize]) -> Option<ChunkContext> {
        self.chunk_file_for(def, chunk_no).map(|f| decompress_v2_chunk(def, f))
    }
    fn cache_missing(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone)]
pub(crate) struct BlockingRemoteZarrDirectory {
    url: String,
    local_cache_dir: String,
    downloading: Arc<Mutex<HashMap<String, Arc<Mutex<Option<Arc<File>>>>>>>,
    client: Client,
}
impl BlockingRemoteZarrDirectory {
    pub(crate) fn new(url: &str, local_cache_dir: &str, client: Client) -> Self {
        Self {
            url: url.to_string(),
            local_cache_dir: local_cache_dir.to_string(),
            downloading: Arc::new(Mutex::new(HashMap::default())),
            client,
        }
    }
    fn chunk_file_for(&self, array_def: &ZarrArrayDef, chunk_no: &[usize]) -> Option<Arc<File>> {
        let target_file = format!(
            "{}/{}",
            self.local_cache_dir,
            chunk_no.iter().map(|i| i.to_string()).collect::<Vec<_>>().join("/")
        );

        if std::path::Path::new(&target_file).exists() {
            Some(File::open(target_file).unwrap().into())
        } else {
            let missing_marker_file = format!("{}.missing", target_file);
            if std::path::Path::new(&missing_marker_file).exists() {
                //TODO: and not older than negative TTL
                //println!("Chunk {} is missing, skipping download", target_file);
                return None;
            }

            let chunk_str = chunk_no
                .iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(array_def.dimension_separator.as_deref().unwrap_or("."));

            let entry = {
                let mut downloading = self.downloading.lock().unwrap();
                if downloading.contains_key(&chunk_str) {
                    let entry = downloading.get(&chunk_str).unwrap().clone();
                    return entry.lock().unwrap().clone();
                } else {
                    let entry = Arc::new(Mutex::new(None));
                    downloading.insert(chunk_str.clone(), entry.clone());
                    entry
                }
            };
            let mut entry = entry.lock().unwrap();

            let target_url = format!("{}/{}", self.url, chunk_str);
            //println!("Downloading chunk from {}", target_url);
            // run request with reqwest blocking
            let response = self.client.get(&target_url).send().unwrap();
            if response.status() != 200 {
                /* println!(
                    "Failed to download chunk from {}, status {}",
                    target_url, response.status
                ); */
                let missing_tmp = format!("{}.missing.tmp", target_file);
                std::fs::create_dir_all(std::path::Path::new(&target_file).parent().unwrap()).unwrap();
                std::fs::write(&missing_tmp, "").unwrap(); // create missing marker file
                std::fs::rename(&missing_tmp, &missing_marker_file).unwrap();
                return None;
            }
            let data = response.bytes().unwrap().to_vec();
            let parent_dir = std::path::Path::new(&target_file).parent().unwrap();
            std::fs::create_dir_all(parent_dir).unwrap();

            let mut file = File::create(&target_file).unwrap();
            file.write_all(&data).unwrap();
            let file = Arc::new(file);

            *entry = Some(file.clone());

            {
                self.downloading.lock().unwrap().remove(&chunk_str);
            }

            Some(file)
        }
    }
}
impl ZarrFileAccess for BlockingRemoteZarrDirectory {
    fn load_array_def(&self) -> ZarrArrayDef {
        let target_file = format!("{}/.zarray", self.local_cache_dir);
        if !std::path::Path::new(&target_file).exists() {
            let zarray_url = format!("{}/.zarray", self.url);
            let res = ehttp::fetch_blocking(&Request::get(&zarray_url)).unwrap();

            if res.status != 200 {
                panic!("Failed to download .zarray from {}, status: {}", zarray_url, res.status);
            }
            let data = res.bytes.to_vec();

            std::fs::create_dir_all(std::path::Path::new(&target_file).parent().unwrap()).unwrap();
            std::fs::write(&target_file, &data).unwrap();
        }

        let zarray = std::fs::read_to_string(&target_file).unwrap();
        parse_json(&zarray, &target_file)
    }
    fn load_chunk(&self, def: &ZarrArrayDef, chunk_no: &[usize]) -> Option<ChunkContext> {
        self.chunk_file_for(def, chunk_no).map(|f| decompress_v2_chunk(def, f))
    }
    fn cache_missing(&self) -> bool {
        true
    }
}

pub(crate) fn parse_json<T: serde::de::DeserializeOwned>(json: &str, source: &str) -> T {
    let de = &mut serde_json::Deserializer::from_str(json);
    serde_path_to_error::deserialize(de).unwrap_or_else(|err| {
        let path = err.path().to_string();
        panic!(
            "Failed to parse {} (source: {}) at .{}: {}\n--- content ---\n{}\n--- end content ---",
            std::any::type_name::<T>(),
            source,
            path,
            err.into_inner(),
            json,
        )
    })
}

pub fn default_cache_dir_for_url(url: &str) -> String {
    let canonical_url = if url.ends_with("/") { &url[..url.len() - 1] } else { url };
    let sha256 = format!("{:x}", Sha256::digest(canonical_url.as_bytes()));
    BaseDirs::new()
        .unwrap()
        .cache_dir()
        .join("vesuvius-gui")
        .join("zarr")
        .join(sha256)
        .to_str()
        .unwrap()
        .to_string()
}

impl<const N: usize> ZarrArray<N, u8> {
    fn load_chunk_context(&self, chunk_no: [usize; N]) -> Option<ChunkContext> {
        self.access.load_chunk(&self.def, &chunk_no)
    }
    pub fn from_path(path: &str) -> Self {
        Self::from_access(Arc::new(ZarrDirectory { path: path.to_string() }))
    }
    pub fn from_url_blocking(url: &str, local_cache_dir: &str, client: Client) -> Self {
        //println!("Loading ZarrArray from url: {}", url);
        Self::from_access(Arc::new(BlockingRemoteZarrDirectory::new(url, local_cache_dir, client)))
    }
    pub fn from_url_to_default_cache_dir_blocking(url: &str, client: Client) -> Self {
        Self::from_url_blocking(url, default_cache_dir_for_url(&url).as_str(), client)
    }
    pub fn from_url(url: &str, local_cache_dir: &str) -> Self {
        //println!("Loading ZarrArray from url: {} to: {} ", url, local_cache_dir);
        Self::from_access(Arc::new(RemoteZarrDirectory {
            url: url.to_string(),
            local_cache_dir: local_cache_dir.to_string(),
            downloader: Arc::new(SimpleDownloader::new()),
        }))
    }
    pub fn from_url_to_default_cache_dir(url: &str) -> Self {
        Self::from_url(url, &default_cache_dir_for_url(url))
    }
    pub fn from_access(access: Arc<dyn ZarrFileAccess>) -> Self {
        let def = access.load_array_def();
        ZarrArray {
            access,
            def,
            phantom_t: std::marker::PhantomData,
        }
    }
    pub fn from_def_and_access(def: ZarrArrayDef, access: Arc<dyn ZarrFileAccess>) -> Self {
        ZarrArray {
            access,
            def,
            phantom_t: std::marker::PhantomData,
        }
    }

    pub fn into_ctx(self) -> ZarrContextBase<N> {
        let cache = Arc::new(ZarrContextCache::new());
        let cache_missing = self.access.cache_missing();
        ZarrContextBase {
            array: self,
            cache,
            cache_missing,
        }
    }
}

impl ZarrArray<3, u8> {
    /// Open a 3D zarr at `path`. Tries zarr v3 (`zarr.json`) first and falls
    /// back to v2 (`.zarray`). Specialised to 3D because the only v3 codec
    /// chain we currently support (`sharding_indexed { c3d }`) is fixed at
    /// 3D 256³ sub-chunks.
    pub fn from_path_auto(path: &str) -> Self {
        if std::path::Path::new(&format!("{}/zarr.json", path)).exists() {
            v3::open_v3_array_local(path)
        } else {
            Self::from_path(path)
        }
    }

    /// Remote analog of [`from_path_auto`]. Probes the URL for `zarr.json`
    /// (v3) by issuing a HEAD-equivalent (a single GET — we cache the body
    /// either way) and falls back to v2 (`.zarray`) on 404.
    pub fn from_url_auto(url: &str, cache_dir: &str, client: Client) -> Self {
        // Quick local-cache shortcut: if we've already cached zarr.json from
        // a prior open, no need to probe again.
        let cached_v3 = std::path::Path::new(&format!("{}/zarr.json", cache_dir)).exists();
        let v3_present = cached_v3
            || matches!(
                client.get(&format!("{}/zarr.json", url.trim_end_matches('/'))).send().map(|r| r.status()),
                Ok(s) if s.is_success()
            );
        if v3_present {
            v3::open_v3_array_remote(url, cache_dir, client)
        } else {
            Self::from_url_blocking(url, cache_dir, client)
        }
    }
}

pub struct ZarrContextBase<const N: usize> {
    array: ZarrArray<N, u8>,
    cache: Arc<ZarrContextCache<N>>,
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
        }
    }
}

pub struct RawContext {
    data: memmap::Mmap,
}
impl RawContext {
    fn load_from_file(chunk_file: &File) -> RawContext {
        let data = unsafe { memmap::Mmap::map(chunk_file).unwrap() };
        RawContext { data }
    }
    /// Open and mmap an existing file. Returns `None` if the file cannot be
    /// opened or mmap'd — callers that use this as a cache lookup should
    /// treat that as a cache miss and fall back to whatever produces the
    /// bytes (e.g. re-decoding from a compressed source).
    pub fn open(path: &std::path::Path) -> Option<RawContext> {
        let file = std::fs::File::open(path).ok()?;
        let data = unsafe { memmap::Mmap::map(&file).ok()? };
        Some(RawContext { data })
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
    fn get(&self, idx: usize) -> u8 {
        self.data[idx]
    }
}

pub enum ChunkContext {
    Heap(Vec<u8>),
    Raw(RawContext),
}
impl ChunkContext {
    pub fn get(&self, idx: usize) -> u8 {
        match self {
            ChunkContext::Heap(data) => data[idx],
            ChunkContext::Raw(raw) => raw.get(idx),
        }
    }
}

struct ZarrContextCacheEntry {
    ctx: Arc<ChunkContext>,
    last_access: AtomicU64,
}
impl Deref for ZarrContextCacheEntry {
    type Target = ChunkContext;
    fn deref(&self) -> &Self::Target {
        &self.ctx
    }
}

struct ZarrContextCache<const N: usize> {
    cache: DashMap<[usize; N], Option<ZarrContextCacheEntry>, FxBuildHasher>,
    access_counter: AtomicU64,
    non_empty_entries: AtomicU64,
}
impl<const N: usize> ZarrContextCache<N> {
    fn new() -> Self {
        ZarrContextCache {
            cache: DashMap::with_hasher_and_shard_amount(FxBuildHasher::default(), 1024),
            access_counter: AtomicU64::new(0),
            non_empty_entries: AtomicU64::new(0),
        }
    }
    // TODO: for now we expect chunks to be uncompressed and memmapped so we piggy back on the OS page cache
    // for zarrs that are compressed, we currently have a memory leak that might need fixing in the future
    // To do that, cache entries should report on their memory usage and we should limit the total amount of memory used
    // in the cache.
    /*
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
            let _before = self.non_empty_entries;
            let _sorted_entries_len = entries.len();
            for (k, _) in entries.into_iter().take(n) {
                if self.cache.remove(&k).is_some() {
                    self.non_empty_entries -= 1;
                }
            }
            /* println!(
                "Purged {} entries {}/{} from {} (sorted: {})",
                n, self.non_empty_entries, self.max_entries, before, sorted_entries_len
            ); */
        }
    }*/
    fn get(&self, array: &ZarrArray<N, u8>, chunk_no: [usize; N]) -> Option<Arc<ChunkContext>> {
        let mut entry = self.cache.entry(chunk_no).or_insert_with(|| {
            let ctx = array.load_chunk_context(chunk_no);
            if ctx.is_none() {
                None
            } else {
                self.non_empty_entries.fetch_add(1, Ordering::Relaxed);
                Some(ZarrContextCacheEntry {
                    ctx: Arc::new(ctx.unwrap()),
                    last_access: AtomicU64::new(0),
                })
            }
        });
        let counter = self.access_counter.fetch_add(1, Ordering::Relaxed);
        if let Some(e) = entry.value_mut() {
            e.last_access.store(counter, Ordering::Relaxed);
            Some(e.ctx.clone())
        } else {
            None
        }
    }
    fn purge_missing(&self) {
        self.cache.retain(|_, e| if e.is_none() { false } else { true });
    }
}

struct ZarrContextState<const N: usize> {
    last_chunk_no: [usize; N],
    last_context: Option<Option<Arc<ChunkContext>>>,
}

pub struct ZarrContext<const N: usize> {
    array: ZarrArray<N, u8>,
    cache_missing: bool,
    cache: Arc<ZarrContextCache<N>>,
    state: RefCell<ZarrContextState<N>>,
}

impl<const N: usize, T> ZarrArray<N, T> {
    pub fn shape(&self) -> &[usize] {
        &self.def.shape
    }
}

impl<const N: usize> ZarrContextBase<N> {
    pub fn shape(&self) -> &[usize] {
        self.array.shape()
    }
}

impl<const N: usize> ZarrContext<N> {
    pub fn shape(&self) -> &[usize] {
        self.array.shape()
    }
}

impl ZarrContext<3> {
    pub fn cache_missing(&self) -> bool {
        self.cache_missing
    }
    pub fn purge_missing_cache(&self) {
        self.cache.purge_missing();
    }
    pub fn get(&self, index: [usize; 3]) -> Option<u8> {
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
        let state = self.state.borrow();
        let last_chunk_no = state.last_chunk_no;
        if chunk_no == last_chunk_no {
            if let Some(last) = state.last_context.as_ref().unwrap() {
                Some(last.get(idx))
            } else {
                None
            }
        } else {
            drop(state); // release borrow before acquiring mutex
            self.get_from_cache(chunk_no, idx)
        }
    }
    pub fn get_interpolated(&self, xyz: [f64; 3]) -> Option<u8> {
        let (dx, x0) = modf(xyz[0]);
        let x0 = x0 as usize;
        let x1 = x0 + 1;
        let (dy, y0) = modf(xyz[1]);
        let y0 = y0 as usize;
        let y1 = y0 + 1;
        let (dz, z0) = modf(xyz[2]);
        let z0 = z0 as usize;
        let z1 = z0 + 1;

        let cx = self.array.def.chunks[0];
        let cy = self.array.def.chunks[1];
        let cz = self.array.def.chunks[2];

        // fast-path: if all coordinates are in the same chunk, i.e. they are not on the upper chunk boundary
        let chunk_offset = [x0 % cx, y0 % cy, z0 % cz];

        let fast_path = chunk_offset[0] != cx - 1 && chunk_offset[1] != cy - 1 && chunk_offset[2] != cz - 1;

        let (p000, p100, p010, p110, p001, p101, p011, p111) = if fast_path {
            let chunk_no = [x0 / cx, y0 / cy, z0 / cz];

            let idx = ((chunk_offset[0] * cy) + chunk_offset[1]) * cz + chunk_offset[2];

            let idx_dx = cy * cz;
            let idx_dy = cz;
            let idx_dz = 1;

            // fast path
            let last_chunk_no = self.state.borrow().last_chunk_no;
            if chunk_no != last_chunk_no {
                // slow path goes through mutex
                self.get_from_cache(chunk_no, idx); // prime last cache
            }
            let state = self.state.borrow_mut();
            match state.last_context {
                Some(None) | None => return None,
                _ => {}
            }

            let last = state.last_context.as_ref().unwrap().as_ref().unwrap().as_ref();

            if let ChunkContext::Raw(raw) = last {
                let p000 = raw.get(idx);
                let p100 = raw.get(idx + idx_dx);
                let p010 = raw.get(idx + idx_dy);
                let p110 = raw.get(idx + idx_dx + idx_dy);
                let p001 = raw.get(idx + idx_dz);
                let p101 = raw.get(idx + idx_dx + idx_dz);
                let p011 = raw.get(idx + idx_dy + idx_dz);
                let p111 = raw.get(idx + idx_dx + idx_dy + idx_dz);
                (p000, p100, p010, p110, p001, p101, p011, p111)
            } else {
                let p000 = last.get(idx);
                let p100 = last.get(idx + idx_dx);
                let p010 = last.get(idx + idx_dy);
                let p110 = last.get(idx + idx_dx + idx_dy);
                let p001 = last.get(idx + idx_dz);
                let p101 = last.get(idx + idx_dx + idx_dz);
                let p011 = last.get(idx + idx_dy + idx_dz);
                let p111 = last.get(idx + idx_dx + idx_dy + idx_dz);
                (p000, p100, p010, p110, p001, p101, p011, p111)
            }
        } else {
            let p000 = self.get([x0, y0, z0]);
            let p100 = self.get([x1, y0, z0]);
            let p010 = self.get([x0, y1, z0]);
            let p110 = self.get([x1, y1, z0]);
            let p001 = self.get([x0, y0, z1]);
            let p101 = self.get([x1, y0, z1]);
            let p011 = self.get([x0, y1, z1]);
            let p111 = self.get([x1, y1, z1]);

            if let (Some(p000), Some(p100), Some(p010), Some(p110), Some(p001), Some(p101), Some(p011), Some(p111)) =
                (p000, p100, p010, p110, p001, p101, p011, p111)
            {
                (p000, p100, p010, p110, p001, p101, p011, p111)
            } else {
                return None;
            }
        };

        let c00 = p000 as f64 * (1.0 - dx) + p100 as f64 * dx;
        let c10 = p010 as f64 * (1.0 - dx) + p110 as f64 * dx;
        let c01 = p001 as f64 * (1.0 - dx) + p101 as f64 * dx;
        let c11 = p011 as f64 * (1.0 - dx) + p111 as f64 * dx;

        let c0 = c00 * (1.0 - dy) + c10 * dy;
        let c1 = c01 * (1.0 - dy) + c11 * dy;

        let c = c0 * (1.0 - dz) + c1 * dz;

        Some(c as u8)
    }

    fn get_from_cache(&self, chunk_no: [usize; 3], idx: usize) -> Option<u8> {
        let chunk = self.cache.get(&self.array, chunk_no);

        let mut state = self.state.borrow_mut();
        state.last_chunk_no = chunk_no;
        state.last_context = Some(chunk.clone());
        chunk.map(|c| c.get(idx))
    }
    pub fn shareable(&self) -> Box<dyn (FnOnce() -> ZarrContext<3>) + Send + Sync> {
        let array = self.array.clone();
        let cache = self.cache.clone();
        let cache_missing = self.cache_missing;
        Box::new(move || ZarrContext {
            array,
            cache_missing,
            cache,
            state: ZarrContextState {
                last_chunk_no: [usize::MAX; 3],
                last_context: None,
            }
            .into(),
        })
    }
}

impl Clone for ZarrContext<3> {
    fn clone(&self) -> Self {
        ZarrContext {
            array: self.array.clone(),
            cache_missing: self.cache_missing,
            cache: self.cache.clone(),
            state: ZarrContextState {
                last_chunk_no: [usize::MAX; 3],
                last_context: None,
            }
            .into(),
        }
    }
}

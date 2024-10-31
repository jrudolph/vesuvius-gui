use crate::{volume::PaintVolume, zstd_decompress};
use derive_more::Debug;
use egui::Color32;
use memmap::MmapOptions;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    fs::{File, OpenOptions},
    io::Write,
    ops::{Deref, DerefMut},
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc, Mutex,
    },
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

#[derive(Debug, Clone)]
pub struct ZarrArray<const N: usize, T> {
    path: String,
    def: ZarrArrayDef,
    phantom_t: std::marker::PhantomData<T>,
}

#[derive(Debug, Clone)]
pub enum BloscShuffle {
    None,
    Bit,
    Byte,
}

#[derive(Debug, Clone)]
pub enum BloscCompressor {
    Blosclz,
    Lz4,
    Snappy,
    Zlib,
    Zstd,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct BloscHeader {
    version: u8,
    version_lz: u8,
    flags: u8,
    typesize: usize,
    nbytes: usize,
    blocksize: usize,
    cbytes: usize,
    shuffle: BloscShuffle,
    compressor: BloscCompressor,
}
impl BloscHeader {
    fn from_bytes(bytes: &[u8]) -> Self {
        let flags = bytes[2];
        let shuffle = match flags & 0x7 {
            0 | 1 => BloscShuffle::None,
            2 => BloscShuffle::Byte,
            4 => BloscShuffle::Bit,
            x => panic!("Invalid shuffle value {x}"),
        };
        let compressor = match flags >> 5 {
            0 => BloscCompressor::Blosclz,
            1 => BloscCompressor::Lz4,
            2 => BloscCompressor::Snappy,
            3 => BloscCompressor::Zlib,
            4 => BloscCompressor::Zstd,
            x => panic!("Invalid compressor value {x}"),
        };

        BloscHeader {
            version: bytes[0],
            version_lz: bytes[1],
            flags,
            typesize: bytes[3] as usize,
            nbytes: u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize,
            blocksize: u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            cbytes: u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize,
            shuffle,
            compressor,
        }
    }
}

#[derive(Debug)]
pub struct BloscChunk<T> {
    header: BloscHeader,
    offsets: Vec<u32>,
    #[debug(skip)]
    data: memmap::Mmap,
    phantom_t: std::marker::PhantomData<T>,
}

#[allow(dead_code)]
struct BloscBlock {
    id: u16, // FIXME
    data: Vec<u8>,
}

struct BloscContext {
    chunk: BloscChunk<u8>,
    cache: HashMap<usize, BloscBlock>,
    last_block_idx: usize,
    last_entry: Option<BloscBlock>,
}
impl BloscContext {
    fn get(&mut self, index: usize) -> u8 {
        let block_idx = index * self.chunk.header.typesize as usize / self.chunk.header.blocksize as usize;
        let idx = (index * self.chunk.header.typesize as usize) % self.chunk.header.blocksize as usize;

        if block_idx == self.last_block_idx {
            self.last_entry.as_ref().unwrap().data[idx]
        } else if self.cache.contains_key(&block_idx) {
            let block = self.cache.remove(&block_idx).unwrap();
            if let Some(last_block) = self.last_entry.take() {
                self.cache.insert(self.last_block_idx, last_block);
            }
            let res = block.data[idx];
            self.last_block_idx = block_idx;
            self.last_entry = Some(block);
            res
        } else {
            if self.cache.len() > 1000 {
                self.cache.clear();
            }

            let uncompressed = self.chunk.decompress(block_idx);
            let res = uncompressed[idx];
            let block = BloscBlock {
                id: block_idx as u16,
                data: uncompressed,
            };
            self.cache.insert(block_idx, block);

            res
        }
    }
}

impl BloscChunk<u8> {
    fn load(file: &str) -> Self {
        let file = File::open(file).unwrap();
        let chunk = unsafe { MmapOptions::new().map(&file) }.unwrap();

        // parse 16 byte blosc header
        let header = BloscHeader::from_bytes(&chunk[0..16]);
        let mut offsets = vec![];
        for i in 0..((header.nbytes + header.blocksize - 1) / header.blocksize) as usize {
            offsets.push(u32::from_le_bytes([
                chunk[16 + i * 4],
                chunk[16 + i * 4 + 1],
                chunk[16 + i * 4 + 2],
                chunk[16 + i * 4 + 3],
            ]));
        }

        BloscChunk {
            header,
            offsets,
            data: chunk,
            phantom_t: std::marker::PhantomData,
        }
    }
    fn into_ctx(self) -> BloscContext {
        BloscContext {
            chunk: self,
            cache: HashMap::new(),
            last_block_idx: usize::MAX,
            last_entry: None,
        }
    }
    fn get(&self, index: usize) -> u8 {
        let block_idx = index * self.header.typesize as usize / self.header.blocksize as usize;
        let idx = (index * self.header.typesize as usize) % self.header.blocksize as usize;

        self.decompress(block_idx)[idx]
    }
    fn decompress(&self, block_idx: usize) -> Vec<u8> {
        let block_offset = self.offsets[block_idx] as usize;
        let block_compressed_length =
            u32::from_le_bytes(self.data[block_offset..block_offset + 4].try_into().unwrap()) as usize;
        let block_compressed_data = &self.data[block_offset + 4..block_offset + block_compressed_length + 4];

        match self.header.compressor {
            BloscCompressor::Lz4 => lz4_compression::decompress::decompress(&block_compressed_data).unwrap(),
            BloscCompressor::Zstd => zstd_decompress(block_compressed_data),
            _ => todo!(),
        }
    }
}

impl<const N: usize, T> ZarrArray<N, T> {
    fn chunk_path(&self, chunk_no: [usize; N]) -> String {
        format!(
            "{}/{}",
            self.path,
            chunk_no
                .iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(self.def.dimension_separator.as_deref().unwrap_or("."))
        )
    }
}

impl<const N: usize> ZarrArray<N, u8> {
    fn load_chunk(&self, chunk_no: [usize; N]) -> Option<BloscChunk<u8>> {
        let chunk_path = self.chunk_path(chunk_no);
        if !std::path::Path::new(&chunk_path).exists() {
            None
        } else {
            Some(BloscChunk::load(&chunk_path))
        }
    }
    pub fn from_path(path: &str) -> Self {
        // read and parse path/.zarray into ZarrArrayDef

        let zarray = std::fs::read_to_string(format!("{}/.zarray", path)).unwrap();
        println!("Read ZarrArrayDef: {}", zarray);
        let zarray_def = serde_json::from_str::<ZarrArrayDef>(&zarray).unwrap();

        println!("Loaded ZarrArrayDef: {:?}", zarray_def);

        assert!(zarray_def.shape.len() == N);

        ZarrArray {
            path: path.to_string(),
            def: zarray_def,
            phantom_t: std::marker::PhantomData,
        }
    }

    pub fn into_ctx(self) -> ZarrContextBase<N> {
        let cache = Arc::new(Mutex::new(ZarrContextCache::new(&self.def)));
        ZarrContextBase { array: self, cache }
    }
    #[allow(dead_code)]
    fn get(&self, index: [usize; N]) -> u8 {
        let chunk_no = index
            .iter()
            .zip(self.def.chunks.iter())
            .map(|(i, c)| i / c)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let chunk_offset = index
            .iter()
            .zip(self.def.chunks.iter())
            .map(|(i, c)| i % c)
            .collect::<Vec<_>>();

        if let Some(chunk) = self.load_chunk(chunk_no) {
            println!("Chunk: {:?}", chunk);
            let idx = chunk_offset
                .iter()
                .zip(self.def.chunks.iter())
                //.rev() // FIXME: only if row-major
                .fold(0, |acc, (i, c)| acc * c + i);
            println!("Index for {:?}: {:?}", chunk_offset, idx);
            chunk.get(idx)
        } else {
            0
        }
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
    fn deref(&self) -> &Self::Target { &self.ctx }
}
impl DerefMut for ZarrContextCacheEntry {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.ctx }
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

#[allow(dead_code)]
struct SparsePointCloud {
    points: HashMap<[u16; 3], u8>,
}
#[allow(dead_code)]
fn write_points(array: &mut ZarrContext<3>) -> SparsePointCloud {
    // write all points out into a simple binary file

    let file1 = File::create("fiber-points-1.raw").unwrap();
    let file2 = File::create("fiber-points-2.raw").unwrap();
    let mut writer1 = std::io::BufWriter::new(file1);
    let mut writer2 = std::io::BufWriter::new(file2);

    fn write(writer: &mut std::io::BufWriter<std::fs::File>, x: usize, y: usize, z: usize) {
        writer.write((x as u16).to_le_bytes().as_ref()).unwrap();
        writer.write((y as u16).to_le_bytes().as_ref()).unwrap();
        writer.write((z as u16).to_le_bytes().as_ref()).unwrap();
    }

    let shape = array.array.def.shape.clone();
    let mut count: u64 = 0;
    for z in 0..shape[0] {
        if z % 1 == 0 {
            println!("z: {} count: {}", z, count);
        }
        for y in 0..shape[1] {
            for x in 0..shape[2] {
                let idx = [z, y, x];
                let v = array.get(idx);
                if v != 0 {
                    count += 1;
                    if v == 1 {
                        write(&mut writer1, x, y, z);
                    } else {
                        write(&mut writer2, x, y, z);
                    }
                }
            }
        }
    }
    println!("Count: {}", count);

    todo!() //SparsePointCloud {  }
}

fn index_of(x: u16, y: u16, z: u16) -> usize {
    let x = x as usize;
    let y = y as usize;
    let z = z as usize;

    // instead of zzzzzzzzzzzzzzzzyyyyyyyyyyyyyyyyxxxxxxxxxxxxxxxx use
    //            zzzzzzzzyyyyyyyyxxxxxxxxzzzzyyyyxxxxzzzzyyyyxxxx

    // 4x4x4 => 4x4x4 =>

    let page = ((z & 0x1f00) << 2) | ((y & 0x1f00) >> 3) | ((x & 0x1f00) >> 8);
    let line = ((z & 0xf0) << 4) | (y & 0xf0) | ((x & 0xf0) >> 4);
    let addr = ((z & 0xf) << 8) | ((y & 0xf) << 4) | (x & 0xf);

    (page << 24) | (line << 12) | addr
}

pub struct FullMapVolume {
    mmap: memmap::Mmap,
}
impl FullMapVolume {
    pub fn new() -> FullMapVolume {
        let file = File::open("data/fiber-points.map").unwrap();
        let map = unsafe { MmapOptions::new().map(&file) }.unwrap();

        FullMapVolume { mmap: map }
    }
}
impl PaintVolume for FullMapVolume {
    fn paint(
        &mut self,
        xyz: [i32; 3],
        u_coord: usize,
        v_coord: usize,
        plane_coord: usize,
        width: usize,
        height: usize,
        sfactor: u8,
        paint_zoom: u8,
        _config: &crate::volume::DrawingConfig,
        buffer: &mut crate::volume::Image,
    ) {
        assert!(sfactor == 1);
        let fi32 = sfactor as f64;

        for im_u in 0..width {
            for im_v in 0..height {
                let im_rel_u = (im_u as i32 - width as i32 / 2) * paint_zoom as i32;
                let im_rel_v = (im_v as i32 - height as i32 / 2) * paint_zoom as i32;

                let mut uvw: [f64; 3] = [0.; 3];
                uvw[u_coord] = (xyz[u_coord] + im_rel_u) as f64 / fi32;
                uvw[v_coord] = (xyz[v_coord] + im_rel_v) as f64 / fi32;
                uvw[plane_coord] = (xyz[plane_coord]) as f64 / fi32;

                // x1961:5393 , y2135:5280, z7000:11249
                let x = -1961.0 + uvw[0];
                let y = -2135.0 + uvw[1];
                let z = -7000.0 + uvw[2];

                if x < 0.0 || y < 0.0 || z < 0.0 {
                    continue;
                }

                let v = self.mmap[index_of(x as u16, y as u16, z as u16)];
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

pub struct ConnectedFullMapVolume {
    mmap: memmap::Mmap,
}
impl ConnectedFullMapVolume {
    pub fn new() -> ConnectedFullMapVolume {
        let file = File::open("data/fiber-points-connected.map").unwrap();
        let map = unsafe { MmapOptions::new().map(&file) }.unwrap();

        ConnectedFullMapVolume { mmap: map }
    }
}
impl PaintVolume for ConnectedFullMapVolume {
    fn paint(
        &mut self,
        xyz: [i32; 3],
        u_coord: usize,
        v_coord: usize,
        plane_coord: usize,
        width: usize,
        height: usize,
        sfactor: u8,
        paint_zoom: u8,
        _config: &crate::volume::DrawingConfig,
        buffer: &mut crate::volume::Image,
    ) {
        assert!(sfactor == 1);
        //let mut color_idx = 0;
        //let mut colors = HashMap::new();
        let fi32 = sfactor as f64;

        for im_u in 0..width {
            for im_v in 0..height {
                let im_rel_u = (im_u as i32 - width as i32 / 2) * paint_zoom as i32;
                let im_rel_v = (im_v as i32 - height as i32 / 2) * paint_zoom as i32;

                let mut uvw: [f64; 3] = [0.; 3];
                uvw[u_coord] = (xyz[u_coord] + im_rel_u) as f64 / fi32;
                uvw[v_coord] = (xyz[v_coord] + im_rel_v) as f64 / fi32;
                uvw[plane_coord] = (xyz[plane_coord]) as f64 / fi32;

                // x1961:5393 , y2135:5280, z7000:11249
                let x = /* -1961.0 + */ uvw[0];
                let y = /* -2135.0 + */ uvw[1];
                let z = /* -7000.0 + */ uvw[2];

                if x < 0.0 || y < 0.0 || z < 0.0 {
                    continue;
                }

                let idx = index_of(x as u16, y as u16, z as u16);
                if self.mmap[idx] > 1 {
                    //let v = self.mmap[idx * 2] as u16 | ((self.mmap[idx * 2 + 1] as u16) << 8);
                    let v = self.mmap[idx];
                    /* let color = match v & 7 {
                        1 => Color32::RED,
                        2 => Color32::GREEN,
                        3 => Color32::YELLOW,
                        4 => Color32::BLUE,
                        5 => Color32::KHAKI,
                        6 => Color32::DARK_RED,
                        7 => Color32::GOLD,
                        _ => panic!(),
                    }; */
                    fn from_hsb(h: f64, s: f64, b: f64) -> Color32 {
                        let h = h % 360.0;
                        let s = s.clamp(0.0, 1.0);
                        let b = b.clamp(0.0, 1.0);

                        let c = b * s;
                        let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
                        let m = b - c;

                        let (r1, g1, b1) = match (h / 60.0) as u32 {
                            0 => (c, x, 0.0),
                            1 => (x, c, 0.0),
                            2 => (0.0, c, x),
                            3 => (0.0, x, c),
                            4 => (x, 0.0, c),
                            _ => (c, 0.0, x),
                        };

                        let to_byte = |v: f64| ((v + m) * 255.0).round() as u8;
                        Color32::from_rgb(to_byte(r1), to_byte(g1), to_byte(b1))
                    }
                    //Color32::
                    let color_id = v % 60;
                    let b = (color_id % 2) as f64 / 2.0 + 0.5;
                    let h = (color_id / 2) as f64 / 30.0 * 360.0;
                    let color = from_hsb(h, 1.0, b);
                    /* let color = if let Some(color) = colors.get(&v) {
                        *color
                    } else {
                        if colors.len() > 50 {
                            Color32::WHITE
                        } else {
                            let color = from_hsb(((color_idx % 50) as f64) / 50.0 * 360.0, 1.0, 1.0);
                            colors.insert(v, color);
                            println!("Color {} is {:?}", v, color);
                            color_idx += 1;
                            color
                        }
                    }; */
                    buffer.set(im_u, im_v, color);
                }
            }
        }
    }
}

#[allow(dead_code)]
fn connected_components(array: &ZarrContextBase<3> /* , full: &FullMapVolume */) {
    let shape = array.array.def.shape.clone();

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open("data/fiber-points-connected.map")
        .unwrap();

    file.set_len(8192 * 8192 * 8192 * 2).unwrap();
    let map = unsafe { MmapOptions::new().map_mut(&file).unwrap() };
    let read_map = unsafe { MmapOptions::new().map_mut(&file).unwrap() };

    //let mut id1 = 1u16;
    //let mut id2 = 3u16;
    let id = 2u8;
    let global_id = 1u32;

    let locked = Mutex::new((map, id, global_id));

    use rayon::prelude::*;

    let z_completed = AtomicU32::new(0);

    (20..shape[0] as u16).into_par_iter().for_each(|z| {
        let mut array = array.into_ctx();
        let mut work_list = vec![];
        let mut visited = HashSet::new();
        let mut selected = vec![];
        let mut selected_coords = vec![];

        for y in 0..shape[1] as u16 {
            for x in 0..shape[2] as u16 {
                let idx = index_of(x, y, z);
                //let idx = [z, y, x];
                //let v = full.mmap[idx];
                let v = array.get([z as usize, y as usize, x as usize]);
                let mut count = 0;
                if v != 0 && read_map[idx] == 0 {
                    /* if v == 1 {
                        // keep even odd for horizontal / vertical
                        let res = id1;
                        id1 += 4;
                        res
                    } else {
                        let res = id2;
                        id2 += 4;
                        res
                    }; */
                    //assert!(new_id > 2);

                    //println!("Found a value of {} at {:?}", v, [x, y, z]);

                    // flood current zone with fill_id
                    work_list.clear();
                    work_list.push([x, y, z]);
                    visited.clear();
                    selected.clear();
                    selected_coords.clear();

                    while let Some([x, y, z]) = work_list.pop() {
                        let next_idx = index_of(x, y, z);
                        //let v2 = full.mmap[next_idx];
                        let v2 = array.get([z as usize, y as usize, x as usize]);
                        //println!("Checking at {:?}, found {}", [x, y, z], v2);
                        if v2 == v && !visited.contains(&next_idx) {
                            // do erosion check
                            /* const MAX_DIST: i32 = 5;

                            fn is_inner(full: &FullMapVolume, x: u16, y: u16, z: u16, v: u8) -> bool {
                                for dx in -MAX_DIST..=MAX_DIST {
                                    for dy in -MAX_DIST..=MAX_DIST {
                                        for dz in -MAX_DIST..=MAX_DIST {
                                            let x = x as i32 + dx;
                                            let y = y as i32 + dy;
                                            let z = z as i32 + dz;

                                            if x >= 0
                                                && y >= 0
                                                && z >= 0
                                                && !full.mmap[index_of(x as u16, y as u16, z as u16)] == v
                                            {
                                                return false;
                                            }
                                        }
                                    }
                                }
                                true
                            }
                            if is_inner(full, x, y, z, v) { */
                            selected.push(next_idx);
                            selected_coords.push([x, y, z]);
                            count += 1;

                            /* if count % 1000000 == 0 {
                                println!(
                                    "new_id {} with value {} had {} elements (at x: {:4} y: {:4} z: {:4}",
                                    new_id, v, count, x, y, z
                                );
                            } */

                            // add neighbors to work_list
                            // for now just consider neighbors that share a full face of the voxel cube
                            work_list.push([x as u16 - 1, y as u16, z as u16]);
                            work_list.push([x as u16 + 1, y as u16, z as u16]);

                            work_list.push([x as u16, y as u16 - 1, z as u16]);
                            work_list.push([x as u16, y as u16 + 1, z as u16]);

                            work_list.push([x as u16, y as u16, z as u16 - 1]);
                            work_list.push([x as u16, y as u16, z as u16 + 1]);

                            /* for dx in -1..=MAX_DIST {
                                for dy in -1..=MAX_DIST {
                                    for dz in -1..=MAX_DIST {
                                        let x = x as i32 + dx;
                                        let y = y as i32 + dy;
                                        let z = z as i32 + dz;
                                        if x >= 0 && y >= 0 && z >= 20 {
                                            work_list.push([x as u16, y as u16, z as u16]);
                                        }
                                    }
                                }
                            } */
                            //println!("Worklist now has {} elements", work_list.len());
                            //}
                        }

                        visited.insert(next_idx);
                    }

                    let this_global = {
                        let (map, id, global_id) = &mut *locked.lock().unwrap();
                        if map[selected[0]] == 0 {
                            let new_id = if selected.len() > 200000 {
                                *id += 1;
                                if *id & 0xff < 2 {
                                    *id += 1;
                                }
                                *id
                            } else {
                                1
                            };

                            for idx in selected.iter() {
                                //map[idx * 2] = (new_id & 0xff) as u8;
                                //map[idx * 2 + 1] = ((new_id & 0xff00) >> 8) as u8;
                                map[*idx] = new_id;
                            }

                            let this_global = if selected.len() > 200000 {
                                *global_id += 1;
                                *global_id
                            } else {
                                1
                            };

                            if new_id > 1 {
                                println!(
                                    "new_id {} / {} starting at {:4}/{:4}/{:4} had {} elements",
                                    new_id, this_global, x, y, z, count
                                );
                            }

                            this_global
                        } else {
                            println!(
                                "Skipping at {}/{}/{} because already set to {}",
                                x, y, z, map[selected[0]]
                            );
                            1
                        }
                    };

                    if this_global != 1 {
                        let file_name = format!("data/connected/{}.raw", this_global);
                        let file = OpenOptions::new().write(true).create(true).open(file_name).unwrap();

                        let mut writer = std::io::BufWriter::new(file);

                        for [x, y, z] in selected_coords.iter() {
                            writer.write((*x).to_le_bytes().as_ref()).unwrap();
                            writer.write((*y).to_le_bytes().as_ref()).unwrap();
                            writer.write((*z).to_le_bytes().as_ref()).unwrap();
                        }
                    }
                }
            }
        }
        let completed = z_completed.fetch_add(1, Ordering::Relaxed);
        if z % 1 == 0 {
            //println!("z: {} id1: {} id2: {}", z, id1, id2);
            println!("z: {} id: {} completed: {}", z, id, completed);
        }
    });
}

#[allow(dead_code)]
fn write_points2(array: &mut ZarrContext<3>) -> SparsePointCloud {
    let shape = array.array.def.shape.clone();

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open("/tmp/fiber-points.map")
        .unwrap();

    file.set_len(8192 * 8192 * 16384).unwrap();
    let mut map = unsafe { MmapOptions::new().map_mut(&file).unwrap() };

    let mut count: u64 = 0;
    for z in 0..shape[0] {
        if z % 1 == 0 {
            println!("z: {} count: {}", z, count);
        }
        for y in 0..shape[1] {
            for x in 0..shape[2] {
                let idx = [z, y, x];
                let v = array.get(idx);
                if v != 0 {
                    let idx = index_of(x as u16, y as u16, z as u16); //(z * shape[1] + y) * shape[2] + x;
                    map[idx] = v;
                    count += 1;
                }
            }
        }
    }
    println!("Count: {}", count);

    todo!() //SparsePointCloud {  }
}

#[test]
pub fn test_zarr() {
    let zarr: ZarrArray<3, u8> = ZarrArray::from_path("/home/johannes/tmp/pap/bruniss/mask-fiber-only_rescaled.zarr");
    let mut zarr = zarr.into_ctx();
    //write_points2(&mut zarr);

    //write_points2(&mut zarr);
    //let full = FullMapVolume::new();
    connected_components(&mut zarr /* , &full */);

    let mut zarr = zarr.into_ctx();
    let at0 = [1, 21, 115];
    let at1 = [1, 21, 116];
    let at2 = [1, 21, 117];
    let at3 = [1, 21, 118];

    let start = std::time::Instant::now();
    let at = [1, 21, 115];
    let val = zarr.get(at);
    println!("Value at {:?}: {:?} elapsed: {:?}", at, val, start.elapsed());

    let start = std::time::Instant::now();
    let at = [1, 21, 116];
    let val = zarr.get(at);
    println!("Value at {:?}: {:?} elapsed: {:?}", at, val, start.elapsed());

    let start = std::time::Instant::now();
    let at = [1, 21, 117];
    let val = zarr.get(at);
    println!("Value at {:?}: {:?} elapsed: {:?}", at, val, start.elapsed());

    let start = std::time::Instant::now();
    let at = [1, 21, 118];
    let mut sum = 0;
    for i in 0..100000000 {
        sum += zarr.get(at0);
        sum += zarr.get(at1);
        sum += zarr.get(at2);
        sum += zarr.get(at3);
    }
    let elapsed = start.elapsed();
    println!(
        "Value at {:?} sum: {} elapsed: {:?} per element: {:?}",
        at,
        sum,
        &elapsed,
        elapsed / 100000000 / 4,
    );

    todo!()
}

/*
00000000  02 01 21 01 40 59 73 07  00 00 02 00 b4 02 69 00  |..!.@Ys.......i.|
00000010  93 12 00 00 f8 0e 00 00  a3 14 00 00 2a 38 00 00  |............*8..|
00000020  fd 24 00 00 ed 55 00 00  8a 71 00 00 82 b5 00 00  |.$...U...q......|
00000030  4c 87 00 00 49 9e 00 00  ef e1 00 00 79 cd 00 00  |L...I.......y...|
00000040  d2 fc 00 00 27 19 01 00  8c 48 01 00 77 32 01 00  |....'....H..w2..|
00000050  88 66 01 00 1b 85 01 00  e4 b1 01 00 4d 9c 01 00  |.f..........M...|
00000060  3e cf 01 00 9f eb 01 00  31 02 02 00 b8 20 02 00  |>.......1.... ..|
00000070  39 51 02 00 18 38 02 00  77 71 02 00 89 92 02 00  |9Q...8..wq......|
00000080  73 ca 02 00 44 ad 02 00  d5 ea 02 00 bb 07 03 00  |s...D...........|
00000090  ee 28 03 00 c2 41 03 00  43 63 03 00 e3 93 03 00  |.(...A..Cc......|
000000a0  f4 7b 03 00 b7 d1 03 00  d7 b6 03 00 69 f4 03 00  |.{..........i...|
000000b0  3c 12 04 00 8c 2e 04 00  78 4a 04 00 fb 67 04 00  |<.......xJ...g..|

|-0-|-1-|-2-|-3-|-4-|-5-|-6-|-7-|-8-|-9-|-A-|-B-|-C-|-D-|-E-|-F-|
  ^   ^   ^   ^ |     nbytes    |   blocksize   |    cbytes     |
  |   |   |   |
  |   |   |   +--typesize
  |   |   +------flags
  |   +----------versionlz
  +--------------version

02 version 2
01 version lz 1
21 flags = byte shuffle 0x01, compressor 0x20 >> 5 = 0x01 = lz4
01 typesize = 1 byte
40 59 73 07 nbytes = 125000000 = 500 * 500 * 500
00 00 02 00 blocksize = 0x20000 = 131072
b4 02 69 00 cbytes = 0x6902b4 = 6881972

93 12 00 00 f8 0e 00 00  a3 14 00 00 2a 38 00 00


*/

#[test]
fn test_scroll1_zarr() {
    let file = "/tmp/25";
    let mut chunk = BloscChunk::<u8>::load(file).into_ctx();
    println!("Chunk: {:?}", chunk.chunk.header);
    let v0 = chunk.get(1000);
    println!("Value at 1000: {:?}", v0);

    let mut buf = vec![0; 131072];
    for i in 0..131072 {
        buf[i] = chunk.get(i);
    }
    // write to file
    let file = "/tmp/25-block0.raw";
    let file = File::create(file).unwrap();
    let mut writer = std::io::BufWriter::new(file);
    writer.write(&buf).unwrap();
}

use crate::volume::{PaintVolume, VoxelVolume};
use derive_more::Debug;
use egui::Color32;
use memmap::MmapOptions;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    fs::{File, OpenOptions},
    io::Write,
    ops::Index,
    rc::Rc,
    sync::Mutex,
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

/*
{
    "chunks": [
        500,
        500,
        500
    ],
    "compressor": {
        "blocksize": 0,
        "clevel": 5,
        "cname": "lz4",
        "id": "blosc",
        "shuffle": 1
    },
    "dtype": "|u1",
    "fill_value": 0,
    "filters": null,
    "order": "C",
    "shape": [
        4251,
        3145,
        3432
    ],
    "zarr_format": 2
}%

*/

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
}

pub struct ZarrArray<const N: usize, T> {
    path: String,
    def: ZarrArrayDef,
    phantom_t: std::marker::PhantomData<T>,
}

#[derive(Debug, Clone)]
struct BloscHeader {
    version: u8,
    version_lz: u8,
    flags: u8,
    typesize: usize,
    nbytes: usize,
    blocksize: usize,
    cbytes: usize,
}
impl BloscHeader {
    fn from_bytes(bytes: &[u8]) -> Self {
        BloscHeader {
            version: bytes[0],
            version_lz: bytes[1],
            flags: bytes[2],
            typesize: bytes[3] as usize,
            nbytes: u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize,
            blocksize: u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            cbytes: u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize,
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

struct BloscBlock {
    id: u16, // FIXME
    data: Vec<u8>,
}

struct BloscContext {
    chunk: Rc<BloscChunk<u8>>,
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
            if self.cache.len() > 20 {
                self.cache.clear();
            }

            let block_offset = self.chunk.offsets[block_idx] as usize;
            let block_compressed_length =
                u32::from_le_bytes(self.chunk.data[block_offset..block_offset + 4].try_into().unwrap()) as usize;
            let block_compressed_data = &self.chunk.data[block_offset + 4..block_offset + block_compressed_length + 4];

            let uncompressed = lz4_compression::decompress::decompress(&block_compressed_data).unwrap();
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
    fn into_ctx(self) -> BloscContext {
        BloscContext {
            chunk: Rc::new(self),
            cache: HashMap::new(),
            last_block_idx: usize::MAX,
            last_entry: None,
        }
    }
    fn get(&self, index: usize) -> u8 {
        let block_idx = index * self.header.typesize as usize / self.header.blocksize as usize;
        let idx = (index * self.header.typesize as usize) % self.header.blocksize as usize;
        let block_offset = self.offsets[block_idx] as usize;
        let block_compressed_length =
            u32::from_le_bytes(self.data[block_offset..block_offset + 4].try_into().unwrap()) as usize;
        let block_compressed_data = &self.data[block_offset + 4..block_offset + block_compressed_length + 4];

        dbg!(
            "Block: {:?} {:?} {:x} {}",
            index,
            idx,
            block_idx,
            block_offset,
            block_compressed_length
        );

        let uncompressed = lz4_compression::decompress::decompress(&block_compressed_data).unwrap();

        uncompressed[idx]
    }
}

impl<const N: usize, T> ZarrArray<N, T> {
    fn load_chunk(&self, chunk_no: [usize; N]) -> BloscChunk<T> {
        let chunk_path = self.chunk_path(chunk_no);
        let file = File::open(chunk_path.clone()).unwrap();
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

    fn chunk_path(&self, chunk_no: [usize; N]) -> String {
        format!(
            "{}/{}",
            self.path,
            chunk_no.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(".")
        )
    }
}

impl<const N: usize> ZarrArray<N, u8> {
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

    pub fn into_ctx(self) -> ZarrContext<N> {
        ZarrContext {
            array: Rc::new(self),
            cache: HashMap::new(),
            last_chunk_no: [usize::MAX; N],
            last_context: None,
        }
    }
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

        let chunk = self.load_chunk(chunk_no);

        println!("Chunk: {:?}", chunk);
        let idx = chunk_offset
            .iter()
            .zip(self.def.chunks.iter())
            //.rev() // FIXME: only if row-major
            .fold(0, |acc, (i, c)| acc * c + i);
        println!("Index for {:?}: {:?}", chunk_offset, idx);
        chunk.get(idx)
    }
}

struct ZarrCacheEntry<const N: usize> {
    chunk_no: [usize; N],
    ctx: BloscContext,
}

pub struct ZarrContext<const N: usize> {
    array: Rc<ZarrArray<N, u8>>,
    cache: HashMap<[usize; N], BloscContext>,
    last_chunk_no: [usize; N],
    last_context: Option<BloscContext>,
}
/* impl<const N: usize> ZarrContext<N> {
    fn get(&mut self, index: [usize; N]) -> u8 {
        let chunk_no = index
            .iter()
            .zip(self.array.def.chunks.iter())
            .map(|(i, c)| i / c)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let chunk_offset = index
            .iter()
            .zip(self.array.def.chunks.iter())
            .map(|(i, c)| i % c)
            .collect::<Vec<_>>();

        let idx = chunk_offset
            .iter()
            .zip(self.array.def.chunks.iter())
            //.rev() // FIXME: only if row-major
            .fold(0, |acc, (i, c)| acc * c + i);

        if self.last_chunk_context.as_ref().is_some_and(|e| e.chunk_no == chunk_no) {
            self.last_chunk_context.as_mut().unwrap().ctx.get(idx)
        } else {
            let chunk = self.array.load_chunk(chunk_no);
            let mut ctx = chunk.into_ctx();
            let res = ctx.get(idx);
            self.last_chunk_context = Some(ZarrCacheEntry { chunk_no, ctx });
            res
        }
    }
} */

impl ZarrContext<3> {
    fn get(&mut self, index: [usize; 3]) -> u8 {
        if index[0] > self.array.def.shape[0]
            || index[1] > self.array.def.shape[1]
            || index[2] > self.array.def.shape[2]
        {
            return 0;
        }
        let chunk_no = [index[0] / 500, index[1] / 500, index[2] / 500];
        let chunk_offset = [index[0] % 500, index[1] % 500, index[2] % 500];

        //let idx = chunk_offset[0] * 500 * 500 + chunk_offset[1] * 500 + chunk_offset[2];
        let idx = ((chunk_offset[0] * self.array.def.chunks[1]) + chunk_offset[1]) * self.array.def.chunks[2]
            + chunk_offset[2];

        if chunk_no == self.last_chunk_no {
            self.last_context.as_mut().unwrap().get(idx)
        } else if self.cache.contains_key(&chunk_no) {
            let mut last = self.cache.remove(&chunk_no).unwrap();
            if let Some(last) = self.last_context.take() {
                self.cache.insert(self.last_chunk_no, last);
            }
            let res = last.get(idx);
            self.last_chunk_no = chunk_no;
            self.last_context = Some(last);
            res
        } else {
            let chunk = self.array.load_chunk(chunk_no);
            let mut ctx = chunk.into_ctx();
            let res = ctx.get(idx);
            if let Some(last) = self.last_context.take() {
                self.cache.insert(self.last_chunk_no, last);
            }
            self.last_chunk_no = chunk_no;
            self.last_context = Some(ctx);
            res
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

                // x1961:5393 , y2135:5280, z7000:11249
                let x = -1961.0 + uvw[0];
                let y = -2135.0 + uvw[1];
                let z = -7000.0 + uvw[2];

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

struct SparsePointCloud {
    points: HashMap<[u16; 3], u8>,
}
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

    // use z-ordering
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
                let x = -1961.0 + uvw[0];
                let y = -2135.0 + uvw[1];
                let z = -7000.0 + uvw[2];

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

fn connected_components(array: &mut ZarrContext<3>, full: &FullMapVolume) {
    let shape = array.array.def.shape.clone();

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open("data/fiber-points-connected.map")
        .unwrap();

    file.set_len(8192 * 8192 * 8192 * 2).unwrap();
    let mut map = unsafe { MmapOptions::new().map_mut(&file).unwrap() };
    let mut read_map = unsafe { MmapOptions::new().map_mut(&file).unwrap() };

    //let mut id1 = 1u16;
    //let mut id2 = 3u16;
    let mut id = 2u8;

    let locked = Mutex::new((map, id));

    use rayon::prelude::*;

    (20..shape[0] as u16).into_par_iter().for_each(|z| {
        if z % 1 == 0 {
            //println!("z: {} id1: {} id2: {}", z, id1, id2);
            println!("z: {} id: {}", z, id);
        }
        for y in 0..shape[1] as u16 {
            for x in 0..shape[2] as u16 {
                let idx = index_of(x, y, z);
                //let idx = [z, y, x];
                let v = full.mmap[idx];
                //let v = array.get([z as usize, y as usize, x as usize]);
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
                    let mut work_list = vec![[x, y, z]];
                    let mut visited = HashSet::new();

                    while let Some([x, y, z]) = work_list.pop() {
                        let next_idx = index_of(x, y, z);
                        let v2 = full.mmap[next_idx];
                        //let v2 = array.get([z as usize, y as usize, x as usize]);
                        //println!("Checking at {:?}, found {}", [x, y, z], v2);
                        if v2 == v && !visited.contains(&next_idx) {
                            count += 1;

                            /* if count % 1000000 == 0 {
                                println!(
                                    "new_id {} with value {} had {} elements (at x: {:4} y: {:4} z: {:4}",
                                    new_id, v, count, x, y, z
                                );
                            } */

                            const MAX_DIST: i32 = 1;
                            // add neighbors to work_list
                            // for now just consider neighbors that share a full face of the voxel cube
                            for dx in -1..=MAX_DIST {
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
                            }
                            //println!("Worklist now has {} elements", work_list.len());
                        }

                        visited.insert(next_idx);
                    }

                    {
                        let (map, id) = &mut *locked.lock().unwrap();
                        let new_id = if visited.len() > 200000 {
                            *id += 1;
                            if *id & 0xff < 2 {
                                *id += 1;
                            }
                            *id
                        } else {
                            1
                        };

                        for idx in visited.iter() {
                            //map[idx * 2] = (new_id & 0xff) as u8;
                            //map[idx * 2 + 1] = ((new_id & 0xff00) >> 8) as u8;
                            map[*idx] = new_id;
                        }

                        if new_id > 1 {
                            println!(
                                "new_id {} starting at {:4}/{:4}/{:4} had {} elements",
                                new_id, x, y, z, count
                            );
                        }
                    }
                    //panic!();
                }
            }
        }
    });
}

fn write_points2(array: &mut ZarrContext<3>) -> SparsePointCloud {
    let shape = array.array.def.shape.clone();

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open("/tmp/fiber-points.map")
        .unwrap();

    file.set_len(8192 * 8192 * 8192).unwrap();
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
    let zarr: ZarrArray<3, u8> =
        ZarrArray::from_path("/home/johannes/tmp/pap/fiber-predictions/7000_11249_predictions.zarr");
    let mut zarr = zarr.into_ctx();

    //write_points2(&mut zarr);
    let full = FullMapVolume::new();
    connected_components(&mut zarr, &full);

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

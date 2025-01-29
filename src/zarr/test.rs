#![allow(unused)]
use super::{ZarrContext, ZarrContextBase};
use crate::{
    volume::{PaintVolume, VoxelVolume},
    zarr::{blosc::BloscChunk, ZarrArray},
};
use egui::{Color32, ColorImage, Image, Label, Sense, TextureHandle, WidgetText};
use egui_extras::{Column, TableBuilder};
use fxhash::{FxBuildHasher, FxHashMap, FxHashSet};
use image::{GenericImage, GenericImageView, Rgb};
use itertools::Itertools;
use memmap::MmapOptions;
use priority_queue::PriorityQueue;
use rayon::iter::IntoParallelRefIterator;
use std::{
    cmp::Reverse,
    collections::{BinaryHeap, VecDeque},
    fmt::{Debug, Display},
    fs::{File, OpenOptions},
    hash::Hash,
    io::{BufReader, BufWriter, Write},
    path::Path,
    str::FromStr,
    sync::{
        atomic::{AtomicU32, Ordering},
        Mutex,
    },
};

type HashMap<K, V> = FxHashMap<K, V>;
type HashSet<K> = FxHashSet<K>;

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
                if let Some(v) = v {
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
    #[allow(dead_code)]
    pub fn new() -> FullMapVolume {
        let file = File::open("data/fiber-points.map").unwrap();
        let map = unsafe { MmapOptions::new().map(&file) }.unwrap();

        FullMapVolume { mmap: map }
    }
}
impl PaintVolume for FullMapVolume {
    fn paint(
        &self,
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
    #[allow(dead_code)]
    pub fn new() -> ConnectedFullMapVolume {
        let file = File::open("data/fiber-points-connected.map").unwrap();
        let map = unsafe { MmapOptions::new().map(&file) }.unwrap();

        ConnectedFullMapVolume { mmap: map }
    }
}
impl PaintVolume for ConnectedFullMapVolume {
    fn paint(
        &self,
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

fn color_from_palette(idx: usize) -> Color32 {
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

    let color_id = idx % 60;
    let b = (color_id % 2) as f64 / 2.0 + 0.5;
    let h = (color_id / 2) as f64 / 30.0 * 360.0;
    from_hsb(h, 1.0, b)
}

// const CROP: [usize; 3] = [4400, 3300, 10550];
// const CROP_SIZE: [usize; 3] = [100, 100, 100];

/* const CROP: [usize; 3] = [2400, 3000, 9300];
const CROP_SIZE: [usize; 3] = [2000, 2000, 2000]; */

const CROP: [usize; 3] = [2400, 3000, 9300];
const CROP_SIZE: [usize; 3] = [1000, 1000, 5000];

/* const CROP: [usize; 3] = [2585, 3013, 9300];
const CROP_SIZE: [usize; 3] = [30, 45, 1]; */

/* const CROP: [usize; 3] = [2723, 3284, 9632];
const CROP_SIZE: [usize; 3] = [15, 60, 50]; */

pub struct ConnectedFullMapVolume2 {
    mmap: memmap::Mmap,
    ids: &'static [u32],
}
impl ConnectedFullMapVolume2 {
    #[allow(dead_code)]
    pub fn new() -> ConnectedFullMapVolume2 {
        let file = File::open("data/fiber-points-connected-new-255.map").unwrap();
        let map = unsafe { MmapOptions::new().map(&file) }.unwrap();
        let ids: &[u32] = unsafe { std::slice::from_raw_parts_mut(map.as_ptr() as *mut u32, map.len() / 4) };

        ConnectedFullMapVolume2 { mmap: map, ids }
    }
    fn index_of(x: usize, y: usize, z: usize) -> usize {
        ((z + 1 - CROP[2]) * CROP_SIZE[1] + (y + 1 - CROP[1])) * CROP_SIZE[0] + (x + 1 - CROP[0])
    }
    fn get_class(&self, x: usize, y: usize, z: usize) -> u32 {
        if x < CROP[0]
            || y < CROP[1]
            || z < CROP[2]
            || x >= CROP[0] + CROP_SIZE[0]
            || y >= CROP[1] + CROP_SIZE[1]
            || z >= CROP[2] + CROP_SIZE[2]
        {
            //println!("Out of bounds: {} {} {}", x, y, z);

            return 0;
        }

        self.ids[Self::index_of(x, y, z)]
    }
}
#[allow(unused)]
impl PaintVolume for ConnectedFullMapVolume2 {
    fn paint(
        &self,
        xyz: [i32; 3],
        u_coord: usize,
        v_coord: usize,
        plane_coord: usize,
        width: usize,
        height: usize,
        sfactor: u8,
        paint_zoom: u8,
        config: &crate::volume::DrawingConfig,
        buffer: &mut crate::volume::Image,
    ) {
        let fi32 = 1f64;
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

                let v = self.get_class(x as usize, y as usize, z as usize);
                if v != 0 {
                    //println!("painting at {} {} {} {}", x, y, z, v);
                    let color = color_from_palette(v as usize);
                    buffer.blend(im_u, im_v, color, 0.4);
                }
            }
        }
    }
}
#[allow(unused)]
impl VoxelVolume for ConnectedFullMapVolume2 {
    fn get(&self, xyz: [f64; 3], downsampling: i32) -> u8 {
        255
    }
    fn get_color(&self, xyz: [f64; 3], downsampling: i32) -> Color32 {
        let x = (xyz[0] * downsampling as f64) as usize;
        let y = (xyz[1] * downsampling as f64) as usize;
        let z = (xyz[2] * downsampling as f64) as usize;

        let v = self.get_class(x, y, z);

        color_from_palette(v as usize)
        //let idx = index_of(xyz[0] as u16, xyz[1] as u16, xyz[2] as u16);
    }
}

#[allow(unused)]
fn connected_components2(array: &ZarrContextBase<3>, target_class: u8) {
    let mut array = array.into_ctx();

    let pixels: usize = CROP_SIZE.iter().map(|v| v + 1).product();

    let shape = array.array.def.shape.clone();

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(format!("data/fiber-points-connected-new-{}.map", target_class))
        .unwrap();

    file.set_len(pixels as u64 * 4).unwrap();
    let mut map = unsafe { MmapOptions::new().map_mut(&file).unwrap() };
    //let read_map = unsafe { MmapOptions::new().map_mut(&file).unwrap() };

    let ids: &mut [u32] =
        unsafe { std::slice::from_raw_parts_mut(map.as_mut().as_mut_ptr() as *mut u32, map.len() / 4) };

    let mut next_id: u32 = 1u32;
    let mut classes: HashMap<u32, u32> = HashMap::default();

    fn index_of(x: usize, y: usize, z: usize) -> usize {
        ((z + 1 - CROP[2]) * (CROP_SIZE[1] + 1) + (y + 1 - CROP[1])) * (CROP_SIZE[0] + 1) + (x + 1 - CROP[0])
    }

    for z in CROP[2]..CROP[2] + CROP_SIZE[2] {
        if z % 100 == 0 {
            println!("z: {}", z);
        }
        for y in CROP[1]..CROP[1] + CROP_SIZE[1] {
            for x in CROP[0]..CROP[0] + CROP_SIZE[0] {
                let v = array.get([z as usize, y as usize, x as usize]).unwrap_or(0);
                if v == target_class {
                    //let idx = index_of(cx, cy, cz);
                    //println!("idx: {} x: {} y: {} z: {}", idx, x, y, z);

                    /* const NEIGHBOR_OFFSETS: [[usize; 3]; 3] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
                    for [dx, dy, dz] in NEIGHBOR_OFFSETS.iter() {
                        let nx = x - dx;
                        let ny = y - dy;
                        let nz = z - dz;

                        let nv = array.get([nz as usize, ny as usize, nx as usize]).unwrap_or(0);

                    } */
                    let nx = ids[index_of(x - 1, y, z)];
                    let ny = ids[index_of(x, y - 1, z)];
                    let nz = ids[index_of(x, y, z - 1)];

                    let idx = index_of(x, y, z);

                    if nx == 0 && ny == 0 && nz == 0 {
                        ids[idx] = next_id;
                        if next_id == u32::MAX {
                            panic!("Too many ids");
                        }
                        next_id += 1;
                    } else {
                        let mut set = HashSet::default();
                        set.insert(nx);
                        set.insert(ny);
                        set.insert(nz);
                        set.remove(&0);

                        if set.len() == 1 {
                            ids[idx] = set.iter().next().unwrap().clone();
                        } else {
                            let min_id = *set.iter().min().unwrap();
                            let min_id = *classes.get(&min_id).unwrap_or(&min_id);
                            ids[idx] = min_id;
                            set.iter().for_each(|&id| {
                                if min_id != id {
                                    classes.insert(id, min_id);
                                }
                            });
                        }
                    }
                    //println!("{} {} {} -> {} {} {} = {}", x, y, z, nx, ny, nz, ids[idx]);
                }
            }
        }
    }
    println!("Next id: {}", next_id);
    println!("Aliases: {}", classes.len());

    for i in 1..next_id {
        let mut a = i;
        while let Some(&new_i) = classes.get(&a) {
            if new_i == a {
                break;
            }
            a = new_i;
        }
        //println!("{} -> {}", i, a);
        classes.insert(i, a);
    }
    let vals = classes.values().cloned().collect::<HashSet<_>>();
    println!("Classes: {}", vals.len());
    println!("Resolving aliases");

    ids.iter_mut().for_each(|v| {
        if *v != 0 {
            *v = *classes.get(v).unwrap();
        }
    });
    let class_ids = vals
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i as u32))
        .collect::<HashMap<_, _>>();

    fn point_file(target_class: u8, id: u32) -> std::io::BufWriter<std::fs::File> {
        let file_name = format!("data/classes/class-{}/{:06}", target_class, id);
        std::fs::create_dir_all(format!("data/classes/class-{}", target_class)).unwrap();
        let file = File::create(file_name).unwrap();
        std::io::BufWriter::new(file)
    }
    let mut class_writers: HashMap<u32, std::io::BufWriter<std::fs::File>> =
        class_ids.values().map(|&v| (v, point_file(target_class, v))).collect();

    println!("Writing component files");

    for z in CROP[2]..CROP[2] + CROP_SIZE[2] {
        if z % 100 == 0 {
            println!("z: {}", z);
        }
        for y in CROP[1]..CROP[1] + CROP_SIZE[1] {
            for x in CROP[0]..CROP[0] + CROP_SIZE[0] {
                let idx = index_of(x, y, z);
                let v = ids[idx];
                if v != 0 {
                    let writer = class_writers.get_mut(class_ids.get(&v).unwrap()).unwrap();
                    writer.write((x as u16).to_le_bytes().as_ref()).unwrap();
                    writer.write((y as u16).to_le_bytes().as_ref()).unwrap();
                    writer.write((z as u16).to_le_bytes().as_ref()).unwrap();
                }
            }
        }
    }

    /* aliases.iter().for_each(|(a, b)| {
        println!("Alias: {} -> {}", a, b);
    }); */
}

#[test]
fn test_connected_components() {
    let zarr: ZarrArray<3, u8> =
        ZarrArray::from_path("/home/johannes/tmp/pap/bruniss/scrolls/s1/fibers/vt_regular.zarr");
    let mut zarr = zarr.into_ctx();
    connected_components2(&mut zarr, 255);
    //connected_components2(&mut zarr, 2);
}

pub struct PointCloudFile {
    pub id: u32,
    map: memmap::Mmap,
    pub num_elements: usize,
}
impl PointCloudFile {
    #[allow(dead_code)]
    pub fn new(id: u32, file_name: &str) -> PointCloudFile {
        let file = File::open(file_name).unwrap();
        let map = unsafe { MmapOptions::new().map(&file) }.unwrap();
        let num_elements = map.len() / 6;

        PointCloudFile { id, map, num_elements }
    }
    fn at_idx(&self, idx: usize) -> [u16; 3] {
        let idx = idx * 6;
        let x = u16::from_le_bytes([self.map[idx], self.map[idx + 1]]);
        let y = u16::from_le_bytes([self.map[idx + 2], self.map[idx + 3]]);
        let z = u16::from_le_bytes([self.map[idx + 4], self.map[idx + 5]]);
        [x, y, z]
    }
    fn iter(&self) -> impl Iterator<Item = [u16; 3]> + use<'_> {
        (0..self.num_elements).map(move |i| self.at_idx(i))
    }
}

pub struct PointCloudCollection {
    pub clouds: HashMap<u32, PointCloudFile>,
    pub grid: HashMap<[u16; 3], Vec<u32>>,
}
impl PointCloudCollection {
    fn load_from_dir(dir: &str) -> Self {
        let files = std::fs::read_dir(dir).unwrap();
        let clouds = files
            .map(|f| {
                let f = f.unwrap();
                let path = f.path();
                let file_name = path.to_str().unwrap();
                let id = path.file_name().unwrap().to_str().unwrap().parse::<u32>().unwrap();
                (id, PointCloudFile::new(id, file_name))
            })
            .sorted_by_key(|x| x.0)
            .collect::<HashMap<_, _>>();

        /* clouds.iter().for_each(|(id, cloud)| {
            println!("Class {}: {}", id, cloud.num_elements);
        }); */
        let mut grid: HashMap<[u16; 3], Vec<u32>> = HashMap::default();
        clouds.iter().for_each(|(id, cloud)| {
            let mut grids: HashSet<[u16; 3]> = HashSet::default();
            cloud.iter().for_each(|coords| {
                // create a grid of 64x64x64
                let grid_coords = coords.map(|x| x / 32 * 32);
                grids.insert(grid_coords);
                // overlay another grid of 64x64x64 but shifted by 32 in each direction for overlap
                let grid_coords = coords.map(|x| x / 32 * 32 + 16);
                grids.insert(grid_coords);
            });
            grids.into_iter().for_each(|grid_coords| {
                grid.entry(grid_coords).or_insert(vec![]).push(*id);
            });
        });
        /* println!("Populated grid cells: {}", grid.len());
        grid.iter().sorted_by_key(|(k, v)| v.len()).for_each(|(k, v)| {
            println!("{} {} {} -> {}", k[0], k[1], k[2], v.len());
        }); */

        PointCloudCollection { clouds, grid }
    }
}

fn collide(cloud1: &PointCloudFile, cloud2: &PointCloudFile) -> Option<[u16; 3]> {
    /* println!(
        "Colliding v{} ({}) h{} ({})",
        cloud1.id, cloud1.num_elements, cloud2.id, cloud2.num_elements
    ); */

    const GRID_SIZE: u16 = 16;
    // figure out all point pairs that are within 4 pixels of each other
    let mut grid: HashMap<[u16; 3], (bool, bool)> = HashMap::default();
    cloud1.iter().for_each(|coords| {
        let grid_coords = coords.map(|x| x / GRID_SIZE * GRID_SIZE);
        grid.entry(grid_coords).or_insert((false, false)).0 = true;
        //let grid_coords = coords.map(|x| x / 4 * 4 + 2);
        //grid.entry(grid_coords).or_insert((false, false)).0 = true;
    });
    cloud2.iter().for_each(|coords| {
        let grid_coords = coords.map(|x| x / GRID_SIZE * GRID_SIZE);
        grid.entry(grid_coords).or_insert((false, false)).1 = true;
        //let grid_coords = coords.map(|x| x / 4 * 4 + 2);
        //grid.entry(grid_coords).or_insert((false, false)).1 = true;
    });

    let colliding_cells = grid
        .iter()
        .filter_map(|(coords, (c1, c2))| if *c1 && *c2 { Some(*coords) } else { None })
        .collect::<Vec<_>>();

    let len = colliding_cells.len() as u32;
    if len == 0 {
        return None;
    }

    // actually find all points in the intersection of the two clouds
    let cloud1_points = cloud1
        .iter()
        .filter(|coords| {
            let grid_cooods = coords.map(|x| x / GRID_SIZE * GRID_SIZE);
            colliding_cells.contains(&grid_cooods)
        })
        .collect::<HashSet<_>>();
    let cloud2_points = cloud2
        .iter()
        .filter(|coords| {
            let grid_cooods = coords.map(|x| x / GRID_SIZE * GRID_SIZE);
            colliding_cells.contains(&grid_cooods)
        })
        .collect::<HashSet<_>>();

    let intersection = cloud1_points.intersection(&cloud2_points).collect::<Vec<_>>();
    if intersection.len() == 0 {
        return None;
    }
    /* println!(
        "Intersection of {} and {} has {} points",
        cloud1.id,
        cloud2.id,
        intersection.len()
    ); */
    let mut sum = [0, 0, 0];
    intersection.iter().for_each(|coords| {
        sum[0] += coords[0] as i32;
        sum[1] += coords[1] as i32;
        sum[2] += coords[2] as i32;
    });
    let centroid = [
        sum[0] / intersection.len() as i32,
        sum[1] / intersection.len() as i32,
        sum[2] / intersection.len() as i32,
    ];
    // we need an actual candidate that is included in both sets
    let medoid = *intersection
        .iter()
        .min_by_key(|coords| {
            let dx = (coords[0] as i32 - centroid[0]).abs();
            let dy = (coords[1] as i32 - centroid[1]).abs();
            let dz = (coords[2] as i32 - centroid[2]).abs();
            dx + dy + dz
        })
        .unwrap();
    Some(*medoid)
}

#[test]
fn analyze_fibers() {
    println!("Loading class 1");
    let class1 = PointCloudCollection::load_from_dir("data/classes/class-1");
    println!("Loading class 2");
    let class2 = PointCloudCollection::load_from_dir("data/classes/class-2");

    let all_grids = class1.grid.keys().chain(class2.grid.keys()).collect::<HashSet<_>>();
    let mut stats = all_grids
        .iter()
        .map(|grid| {
            let empty = vec![];
            let class1 = class1.grid.get(*grid).unwrap_or(&empty);
            let class2 = class2.grid.get(*grid).unwrap_or(&empty);
            let compares = class1
                .iter()
                .cloned()
                .cartesian_product(class2.iter().cloned())
                .collect::<Vec<_>>();

            (grid, class1.len(), class2.len(), class1.len() * class2.len(), compares)
        })
        .collect::<Vec<_>>();
    stats.sort_by_key(|x| x.3);
    /* stats.iter().for_each(|(grid, c1, c2, p)| {
        println!("{} {} {} -> {} * {} = {}", grid[0], grid[1], grid[2], c1, c2, p);
    }); */
    let total_compares = stats.iter().map(|x| x.3).sum::<usize>();
    println!("Total compares: {}", total_compares);
    let dedup_compares = stats.iter().flat_map(|x| x.4.clone()).collect::<HashSet<_>>();
    println!("Dedup compares: {}", dedup_compares.len());
    let pairs = dedup_compares
        .iter()
        .map(|(id1, id2)| {
            let cloud1 = class1.clouds.get(id1).unwrap();
            let cloud2 = class2.clouds.get(id2).unwrap();

            (
                id1,
                id2,
                cloud1.num_elements,
                cloud2.num_elements,
                cloud1.num_elements * cloud2.num_elements,
            )
        })
        .collect::<Vec<_>>();
    //println!("Total point pairs to consider: {}", pairs);
    use indicatif::{ParallelProgressIterator, ProgressIterator};
    use rayon::prelude::*;

    let mut colls = pairs
        .par_iter()
        .progress_count(pairs.len() as u64)
        .flat_map(|(id1, id2, c1, c2, p)| {
            let cross = collide(class1.clouds.get(id1).unwrap(), class2.clouds.get(id2).unwrap());
            match cross {
                Some(cross) => Some((*id1, *id2, c1, c2, p, cross)),
                None => None,
            }
        })
        .collect::<Vec<_>>();

    println!("Collisions: {}", colls.len());
    colls.sort_by_key(|c| c.1);

    // create a dot file for graphviz that connects all colliding points
    let mut colltxt = File::create("data/collisions").unwrap();
    let mut file = File::create("data/collisions.dot").unwrap();
    file.write(b"graph {\n").unwrap();
    colls.iter().for_each(|(id1, id2, c1, c2, p, [x, y, z])| {
        file.write(format!("h{} -- v{};\n", id2, id1).as_bytes()).unwrap();
        colltxt
            .write(format!("h{} v{} {} {} {}\n", id2, id1, x, y, z).as_bytes())
            .unwrap();
        //file.write(format!("{} [label=\"{}\"];\n", id1, c1).as_bytes()).unwrap();
        //file.write(format!("{} [label=\"{}\"];\n", id2, c2).as_bytes()).unwrap();
        //file.write(format!("{} [pos=\"{},{}!\"];\n", id1, x, y).as_bytes()).unwrap();
        //file.write(format!("{} [pos=\"{},{}!\"];\n", id2, x, y).as_bytes()).unwrap();
    });
    file.write(b"}\n").unwrap();
}

#[test]
fn analyze_collisions() {
    let file = File::open("data/collisions").unwrap();
    use std::io::BufRead;
    let mut reader = BufReader::new(file);
    let mut lines = reader.lines();
    let mut colls = vec![];
    while let Some(Ok(line)) = lines.next() {
        let mut parts = line.split_whitespace().collect::<Vec<_>>();
        // h<id1> v<id2> <x> <y> <z>
        let id1 = parts[0][1..].parse::<u32>().unwrap();
        let id2 = parts[1][1..].parse::<u32>().unwrap();
        let x = parts[2].parse::<u16>().unwrap();
        let y = parts[3].parse::<u16>().unwrap();
        let z = parts[4].parse::<u16>().unwrap();

        colls.push((id1, id2, x, y, z));
    }

    /* // group by id2 and count
    colls
        .iter()
        .sorted_by_key(|x| x.1)
        .chunk_by(|x| x.1)
        .into_iter()
        .map(|(id, group)| (id, group.count()))
        .sorted_by_key(|x| x.1)
        .chunk_by(|x| x.1)
        .into_iter()
        .map(|(count, group)| (count, group.count()))
        .for_each(|(count, num)| {
            println!("{}: {}", count, num);
        }); */

    // create new dot file that only contains edges where the vertical has rank 2
    /* let selected_vertical = colls
    .iter()
    .sorted_by_key(|x| x.1)
    .chunk_by(|x| x.1)
    .into_iter()
    .map(|(id, group)| (id, group.count()))
    .filter_map(|(id, count)| if count == 2 { Some(id) } else { None })
    .collect::<HashSet<_>>(); */

    /* let mut file = File::create("data/collisions-pruned.dot").unwrap();
    file.write(b"graph {\n").unwrap();
    colls.iter().for_each(|(id1, id2, x, y, z)| {
        if selected_vertical.contains(&id2) {
            file.write(format!("h{} -- v{};\n", id1, id2).as_bytes()).unwrap();
        }
    });
    file.write(b"}\n").unwrap(); */

    /* let filtered_colls = colls
    .iter()
    .filter(|(id1, id2, x, y, z)| selected_vertical.contains(id2))
    .collect::<Vec<_>>(); */
    /* let mut neighbor_map_h: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut neighbor_map_v: HashMap<u32, Vec<u32>> = HashMap::new();
    filtered_colls.iter().for_each(|(id1, id2, x, y, z)| {
        neighbor_map_h.entry(*id1).or_insert(vec![]).push(*id2);
        neighbor_map_v.entry(*id2).or_insert(vec![]).push(*id1);
    }); */

    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
    struct Collision {
        h_id: u32,
        v_id: u32,
        x: u16,
        y: u16,
        z: u16,
    }
    impl From<(u32, u32, u16, u16, u16)> for Collision {
        fn from(x: (u32, u32, u16, u16, u16)) -> Self {
            Self {
                h_id: x.0,
                v_id: x.1,
                x: x.2,
                y: x.3,
                z: x.4,
            }
        }
    }

    const DOWNSCALE: i32 = 2;

    struct Map {
        map: HashSet<[u16; 3]>,
    }
    impl Map {
        fn new(cloud: &PointCloudFile) -> Self {
            let map = cloud.iter().map(|x| x.map(|x| x / DOWNSCALE as u16)).collect();
            Self { map }
        }
        fn contains(&self, point: [u16; 3]) -> bool {
            self.map.contains(&point)
        }
        fn neighbors(&self, point: [u16; 3]) -> Vec<[u16; 3]> {
            let mut neighbors = vec![];
            let candidates = [
                // six directions, manhattan distance is good enough
                [DOWNSCALE, 0, 0],
                [-DOWNSCALE, 0, 0],
                [0, DOWNSCALE, 0],
                [0, -DOWNSCALE, 0],
                [0, 0, DOWNSCALE],
                [0, 0, -DOWNSCALE],
            ];
            for [dx, dy, dz] in candidates {
                let neighbor = [
                    (point[0] as i32 + dx) / DOWNSCALE as i32,
                    (point[1] as i32 + dy) / DOWNSCALE as i32,
                    (point[2] as i32 + dz) / DOWNSCALE as i32,
                ];
                if neighbor.iter().all(|x| *x >= 0 && *x < u16::MAX as i32) {
                    let neighbor = [neighbor[0] as u16, neighbor[1] as u16, neighbor[2] as u16];
                    if self.map.contains(&neighbor) {
                        neighbors.push(neighbor.map(|x| x as u16 * DOWNSCALE as u16));
                    }
                }
            }
            neighbors
        }
    }

    struct DistanceMap {
        distances: HashMap<[u16; 3], i32>,
    }
    impl DistanceMap {
        fn get(&self, point: &[u16; 3]) -> Option<i32> {
            self.distances
                .get(&point.map(|x| x as u16 / DOWNSCALE as u16 * DOWNSCALE as u16))
                .copied()
        }
    }

    fn dijkstra(start: [u16; 3], map: &Map) -> DistanceMap {
        let mut distances = HashMap::default();
        let mut queue: PriorityQueue<[u16; 3], Reverse<i32>> = PriorityQueue::new();
        queue.push(start, Reverse(0));
        distances.insert(start, 0);
        let mut step = 0;
        while let Some((p, Reverse(distance))) = queue.pop() {
            for neighbor in map.neighbors(p) {
                let new_distance = distance + 1;
                if new_distance < *distances.get(&neighbor).unwrap_or(&i32::MAX) {
                    distances.insert(neighbor, new_distance);
                    queue.push_increase(neighbor, Reverse(new_distance));
                }
            }
            step += 1;
            if step % 1000000 == 0 {
                println!("At step {}, queue_size: {}", step, queue.len());
            }
        }
        DistanceMap { distances }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
    struct CollisionPoint {
        horizontal_id: u32,
        vertical_id: u32,
    }
    impl CollisionPoint {
        fn new(horizontal_id: u32, vertical_id: u32) -> Self {
            Self {
                horizontal_id,
                vertical_id,
            }
        }
    }
    struct GlobalEdge {
        p1: CollisionPoint,
        p2: CollisionPoint,
        distance: u64,
    }

    trait Direction {
        type AlongId: Hash + Eq + Copy + Display + Ord + Debug + FromStr<Err: Debug>;
        type AcrossId: Hash + Eq + Copy + Display + Ord + Debug + FromStr<Err: Debug>;

        fn get_along_id(c: &Collision) -> Self::AlongId;
        fn get_across_id(c: &Collision) -> Self::AcrossId;
        fn along_id_prefix() -> &'static str;
        fn across_id_prefix() -> &'static str;
        fn create_collision_point(along_id: Self::AlongId, across_id: Self::AcrossId) -> CollisionPoint;
    }
    struct AlongHorizontal {}
    impl Direction for AlongHorizontal {
        type AlongId = u32;
        type AcrossId = u32;

        fn get_along_id(c: &Collision) -> Self::AlongId {
            c.h_id
        }
        fn get_across_id(c: &Collision) -> Self::AcrossId {
            c.v_id
        }
        fn along_id_prefix() -> &'static str {
            "h"
        }
        fn across_id_prefix() -> &'static str {
            "v"
        }
        fn create_collision_point(along_id: Self::AlongId, across_id: Self::AcrossId) -> CollisionPoint {
            CollisionPoint::new(along_id, across_id)
        }
    }
    struct AlongVertical {}
    impl Direction for AlongVertical {
        type AlongId = u32;
        type AcrossId = u32;

        fn get_along_id(c: &Collision) -> Self::AlongId {
            c.v_id
        }
        fn get_across_id(c: &Collision) -> Self::AcrossId {
            c.h_id
        }
        fn along_id_prefix() -> &'static str {
            "v"
        }
        fn across_id_prefix() -> &'static str {
            "h"
        }
        fn create_collision_point(along_id: Self::AlongId, across_id: Self::AcrossId) -> CollisionPoint {
            CollisionPoint::new(across_id, along_id)
        }
    }

    fn create_adjacency_matrix<D: Direction>(
        along_id: D::AlongId,
        collisions: &[&Collision],
        cloud: &PointCloudFile,
    ) -> HashMap<(D::AcrossId, D::AcrossId), u64> {
        println!(
            "At {}{} total collisions: {}",
            D::along_id_prefix(),
            along_id,
            collisions.len()
        );

        let map = Map::new(&cloud);
        println!("Map entries: {}", map.map.len());

        // calculate the geodesic adjacency matrix for all collision points on h22554
        let mut adjacency_matrix: HashMap<(D::AcrossId, D::AcrossId), u64> = HashMap::default();
        for start_coll @ Collision {
            x: start_x,
            y: start_y,
            z: start_z,
            ..
        } in collisions
        {
            let start_id_across = D::get_across_id(start_coll);
            println!(
                "At {}{} start_id_across: {}",
                D::across_id_prefix(),
                along_id,
                start_id_across
            );
            let start = [*start_x, *start_y, *start_z];
            // collision point might near the surface of the cloud but not contained in it
            let start = if !map.contains(start) {
                cloud
                    .iter()
                    .min_by_key(|x| {
                        let dx = (x[0] as i32 - start[0] as i32).abs();
                        let dy = (x[1] as i32 - start[1] as i32).abs();
                        let dz = (x[2] as i32 - start[2] as i32).abs();
                        dx + dy + dz
                    })
                    .unwrap()
            } else {
                start
            };
            let distances = dijkstra(start, &map);
            println!("distances: {}", distances.distances.len());

            for end_coll @ Collision {
                x: end_x,
                y: end_y,
                z: end_z,
                ..
            } in collisions
            {
                let end_id_across = D::get_across_id(end_coll);
                if start_id_across >= end_id_across {
                    continue;
                }
                let end = [*end_x, *end_y, *end_z];
                let end = if !map.contains(end) {
                    cloud
                        .iter()
                        .min_by_key(|x| {
                            let dx = (x[0] as i32 - end[0] as i32).abs();
                            let dy = (x[1] as i32 - end[1] as i32).abs();
                            let dz = (x[2] as i32 - end[2] as i32).abs();
                            dx + dy + dz
                        })
                        .unwrap()
                } else {
                    end
                };
                if let Some(distance) = distances.get(&end) {
                    adjacency_matrix.insert((start_id_across, end_id_across), distance as u64);
                } else {
                    println!("No distance found for {:?}", end_coll);
                }
            }
        }
        adjacency_matrix
    }

    fn create_or_cached_adjacency_matrix<D: Direction>(
        along_id: D::AlongId,
        collisions: &[&Collision],
        cloud: &PointCloudFile,
    ) -> HashMap<(D::AcrossId, D::AcrossId), u64> {
        let file_name = format!("data/fiber-graph-adjacency/{}{}.txt", D::along_id_prefix(), along_id);
        // simple format: lines with `<across_id> <across_id> <distance>`
        if Path::new(&file_name).exists() {
            let file = File::open(file_name).unwrap();
            let mut reader = BufReader::new(file);
            let mut adjacency_matrix = HashMap::default();
            for line in reader.lines() {
                let line = line.unwrap();
                let parts = line.split_whitespace().collect::<Vec<_>>();
                let across_id1 = parts[0].parse::<D::AcrossId>().unwrap();
                let across_id2 = parts[1].parse::<D::AcrossId>().unwrap();
                let distance = parts[2].parse::<u64>().unwrap();
                adjacency_matrix.insert((across_id1, across_id2), distance);
            }
            adjacency_matrix
        } else {
            let adjacency_matrix = create_adjacency_matrix::<D>(along_id, collisions, cloud);
            let tmp_file = format!("{}.tmp", file_name);
            let dir = Path::new(&file_name).parent().unwrap();
            std::fs::create_dir_all(dir).unwrap();
            let file = File::create(&tmp_file).unwrap();
            let mut writer = BufWriter::new(file);
            adjacency_matrix.iter().for_each(|((start, end), distance)| {
                writeln!(writer, "{} {} {}", start, end, distance).unwrap();
            });
            std::fs::rename(tmp_file, file_name).unwrap();
            adjacency_matrix
        }
    }

    #[derive(PartialEq, Eq, Hash)]
    struct Edge<T: Ord + Copy>(T, T, u64);
    impl<T: Ord + Copy> Edge<T> {
        fn new(id1: T, id2: T, distance: u64) -> Self {
            Self(id1.min(id2), id1.max(id2), distance)
        }
    }

    fn create_collision_graph<D: Direction>(
        along_id: D::AlongId,
        colls: &[Collision],
        cloud: &PointCloudFile,
    ) -> Vec<GlobalEdge> {
        let collisions = colls
            .iter()
            .filter(|c| D::get_along_id(c) == along_id)
            .collect::<Vec<_>>();
        let adjacency_matrix = create_or_cached_adjacency_matrix::<D>(along_id, &collisions, cloud);

        //println!("Adjacency matrix: {}", adjacency_matrix.len());
        /* adjacency_matrix.iter().for_each(|((start, end), distance)| {
            println!("{} {} {}", start.v_id, end.v_id, distance);
        }); */

        struct Adjacency<AcrossId: Hash + Eq + Copy + Ord> {
            matrix: HashMap<(AcrossId, AcrossId), u64>,
        }
        impl<AcrossId: Hash + Eq + Copy + Ord> Adjacency<AcrossId> {
            fn get_distance(&self, key1: &AcrossId, key2: &AcrossId) -> Option<u64> {
                let min = key1.min(key2);
                let max = key1.max(key2);
                self.neighbors(min).get(&max).copied()
            }
            fn neighbors(&self, key: &AcrossId) -> HashMap<AcrossId, u64> {
                self.matrix
                    .iter()
                    .filter(|(k, _)| k.0 == *key || k.1 == *key)
                    .map(|(k, v)| if k.0 == *key { (k.1, *v) } else { (k.0, *v) })
                    .collect()
            }
            fn neighbors2(&self, key: &AcrossId) -> [(AcrossId, u64); 2] {
                let mut neighbors = self.neighbors(key);
                let mut next2 = neighbors
                    .into_iter()
                    //.filter(|(_, dist)| *dist > 30) // FIXME: hacky way to ignore points that are too close
                    .sorted_by_key(|(id, dist)| *dist)
                    .take(2);
                [next2.next().unwrap(), next2.next().unwrap()]
            }
            fn neighbor1(&self, key: &AcrossId) -> (AcrossId, u64) {
                //self.neighbors2(key)[0]
                self.neighbors(key).into_iter().min_by_key(|(_, d)| *d).unwrap()
            }
            fn remove_node(&mut self, key: &AcrossId) {
                self.matrix.retain(|k, v| k.0 != *key && k.1 != *key);
            }
            fn remove_edge(&mut self, key1: &AcrossId, key2: &AcrossId) {
                self.matrix.remove(&(*key1.min(key2), *key1.max(key2)));
            }
        }
        let mut adjacency = Adjacency {
            matrix: adjacency_matrix,
        };

        if adjacency.matrix.is_empty() {
            return vec![];
        }
        /* let first_node: D::AcrossId = adjacency.matrix.iter().next().unwrap().0 .0;
        let most_distant = adjacency
            .neighbors(&first_node)
            .into_iter()
            .max_by_key(|(_, d)| *d)
            .unwrap();
        println!("Most distant: {:?}", &most_distant);

        let other_end = adjacency
            .neighbors(&most_distant.0)
            .into_iter()
            .max_by_key(|(_, d)| *d)
            .unwrap();
        println!("Other end: {:?}", other_end); */

        // step 0: create graph by taking only 2 nearest edges into account
        let mut edges = HashSet::default();
        let keys = collisions.iter().map(|x| D::get_across_id(x)).collect::<HashSet<_>>();
        keys.iter().for_each(|&key| {
            let num_neighbors = adjacency.neighbors(&key).len();
            if num_neighbors >= 2 {
                let [(n1, d1), (n2, d2)] = adjacency.neighbors2(&key);
                edges.insert(Edge::new(key, n1, d1));
                edges.insert(Edge::new(key, n2, d2));
            } else if num_neighbors == 1 {
                let (n1, d1) = adjacency.neighbor1(&key);
                edges.insert(Edge::new(key, n1, d1));
            }
        });

        // create pruned adjacency matrix
        let matrix = edges.into_iter().map(|Edge(e1, e2, d)| ((e1, e2), d)).collect();
        let mut adjacency = Adjacency { matrix };

        // steps:
        //  1. prune all nodes with more than 3 neighbors
        //  2. find and resolve all 3-cliques
        //      - if 1 node has rank 3, remove the edge of the clique that start at that node and is longest
        //      - if 2 nodes have rank 3, remove this edge between them
        //      - if all 3 nodes have rank 3 remove whole clique and start from the beginning?
        //  3. prune all remaining nodes with 3 neighbors

        // step 1
        let mut keys = keys.clone();
        //let mut removed = HashSet::default();

        for key in &keys {
            let num_neighbors = adjacency.neighbors(&key).len();
            if num_neighbors > 3 {
                adjacency.remove_node(key);
            }
        }

        // find all triangles
        let mut triangles: HashSet<[(D::AcrossId, usize); 3]> = HashSet::default();
        let sorted_keys = keys.iter().sorted().cloned().collect::<Vec<_>>();
        for u in &sorted_keys {
            let neighbors = adjacency.neighbors(u);
            for v in neighbors.keys() {
                if *v <= *u {
                    continue;
                }
                let neighbors2 = adjacency.neighbors(v);
                for w in neighbors2.keys() {
                    if *w <= *v {
                        continue;
                    }
                    if neighbors.keys().contains(w) {
                        let neighbors3 = adjacency.neighbors(w);
                        triangles.insert([(*u, neighbors.len()), (*v, neighbors2.len()), (*w, neighbors3.len())]);
                    }
                }
            }
        }
        println!("Triangles: {}", triangles.len());
        triangles.iter().for_each(|t| {
            println!(
                "{} ({}) {} ({}) {} ({})",
                t[0].0, t[0].1, t[1].0, t[1].1, t[2].0, t[2].1
            );
        });

        for t in triangles {
            let with_rank_3 = t.iter().filter(|(_, rank)| *rank == 3).collect::<Vec<_>>();

            match with_rank_3.len() {
                0 => {
                    // remove the longest edge of the triangle
                    let edges = [
                        (t[0].0, t[1].0, adjacency.get_distance(&t[0].0, &t[1].0).unwrap()),
                        (t[0].0, t[2].0, adjacency.get_distance(&t[0].0, &t[2].0).unwrap()),
                        (t[1].0, t[2].0, adjacency.get_distance(&t[1].0, &t[2].0).unwrap()),
                    ];
                    let longest_edge = edges.iter().max_by_key(|(_, _, d)| *d).unwrap();
                    adjacency.remove_edge(&longest_edge.0, &longest_edge.1);
                }
                1 => {
                    // ballon shape, will happen at every corner of the graph due to its construction
                    // just remove the longer of the two edges leading to the triangle
                    let single = with_rank_3[0].0;
                    let ns = adjacency.neighbors(&single);
                    let other_nodes = t.iter().filter(|(id, _)| *id != single).collect::<Vec<_>>();
                    let d1 = adjacency.get_distance(&single, &other_nodes[0].0).unwrap();
                    let d2 = adjacency.get_distance(&single, &other_nodes[1].0).unwrap();
                    if d1 > d2 {
                        adjacency.remove_edge(&single, &other_nodes[0].0);
                    } else {
                        adjacency.remove_edge(&single, &other_nodes[1].0);
                    }
                }
                2 => {
                    // one extra node, sitting "on the side", two options:
                    //  - remove the extra node (since it has non-linear geometry leading to the triangle)
                    //  - include the extra node in a linear path (<- do this for now)
                    adjacency.remove_edge(&with_rank_3[0].0, &with_rank_3[1].0);
                }
                3 => {
                    // remove triangle but keep nodes on their resp linear paths
                    adjacency.remove_edge(&with_rank_3[0].0, &with_rank_3[1].0);
                    adjacency.remove_edge(&with_rank_3[0].0, &with_rank_3[2].0);
                    adjacency.remove_edge(&with_rank_3[1].0, &with_rank_3[2].0);
                }
                _ => panic!("Invalid number of nodes with rank 3: {}", with_rank_3.len()),
            }
        }

        for key in &keys {
            let num_neighbors = adjacency.neighbors(&key).len();
            if num_neighbors >= 3 {
                adjacency.remove_node(key);
            }
        }

        /* let mut edges = HashSet::default();
        let keys = collisions.iter().map(|x| D::get_across_id(x)).collect::<HashSet<_>>();
        keys.into_iter().for_each(|key| {
            let num_neighbors = adjacency.neighbors(key).len();
            if (key == most_distant.0 || key == other_end.0) && num_neighbors >= 1 {
                let (n1, d) = adjacency.neighbor1(key);
                edges.insert(Edge::new(key, n1, d));
            } else if num_neighbors >= 2 {
                let [(n1, d1), (n2, d2)] = adjacency.neighbors2(key);
                edges.insert(Edge::new(key, n1, d1));
                edges.insert(Edge::new(key, n2, d2));
            }
        }); */

        /* println!("Edges: {}", edges.len());
        edges.iter().for_each(|edge| {
            println!("{} {}", edge.0, edge.1);
        }); */

        let mut edges = Vec::new();
        for ((v1, v2), distance) in adjacency.matrix.into_iter() {
            edges.push(GlobalEdge {
                p1: D::create_collision_point(along_id, v1),
                p2: D::create_collision_point(along_id, v2),
                distance,
            });
        }
        edges
    }

    fn load_edges_from_file(
        along_id: u32,
        along_prefix: &str,
        create_collision_point: impl Fn(u32, u32) -> CollisionPoint,
    ) -> Option<Vec<GlobalEdge>> {
        let file_name = format!("data/fiber-graph/{}/{}.txt", along_prefix, along_id);
        if !Path::new(&file_name).exists() {
            return None;
        }
        let file = File::open(file_name).unwrap();
        let mut reader = BufReader::new(file);
        let mut lines = reader.lines();
        let mut edges = vec![];
        while let Some(Ok(line)) = lines.next() {
            let parts = line.split_whitespace().collect::<Vec<_>>();
            let start_across_id = parts[0].parse::<u32>().unwrap();
            let end_across_id = parts[1].parse::<u32>().unwrap();
            let distance = parts[2].parse::<u64>().unwrap();
            edges.push(GlobalEdge {
                p1: create_collision_point(along_id, start_across_id),
                p2: create_collision_point(along_id, end_across_id),
                distance,
            });
        }
        Some(edges)
    }
    fn save_edges_to_file(
        edges: &[GlobalEdge],
        along_id: u32,
        along_prefix: &str,
        get_across_id: impl Fn(&CollisionPoint) -> u32,
    ) {
        let dir = format!("data/fiber-graph/{}", along_prefix);
        std::fs::create_dir_all(dir.clone()).unwrap();

        let file = File::create(format!("{}/{}.txt", dir, along_id)).unwrap();
        let mut writer = BufWriter::new(file);
        edges.iter().for_each(|edge| {
            writeln!(
                writer,
                "{} {} {}",
                get_across_id(&edge.p1),
                get_across_id(&edge.p2),
                edge.distance
            )
            .unwrap();
        });
    }

    fn create_horizontal_graph(horizontal_id: u32, colls: &[Collision]) -> Vec<GlobalEdge> {
        if let Some(edges) = load_edges_from_file(horizontal_id, "h", |h, v| CollisionPoint::new(h, v)) {
            return edges;
        }

        let cloud = PointCloudFile::new(horizontal_id, &format!("data/classes/class-2/{:06}", horizontal_id));
        let edges = create_collision_graph::<AlongHorizontal>(horizontal_id, &colls, &cloud);
        save_edges_to_file(&edges, horizontal_id, "h", |x| x.vertical_id);
        edges
    }
    fn create_vertical_graph(vertical_id: u32, colls: &[Collision]) -> Vec<GlobalEdge> {
        if let Some(edges) = load_edges_from_file(vertical_id, "v", |v, h| CollisionPoint::new(h, v)) {
            return edges;
        }

        let cloud = PointCloudFile::new(vertical_id, &format!("data/classes/class-1/{:06}", vertical_id));
        let edges = create_collision_graph::<AlongVertical>(vertical_id, &colls, &cloud);
        save_edges_to_file(&edges, vertical_id, "v", |x| x.horizontal_id);
        edges
    }

    fn write_dot_file(edges: &[GlobalEdge], file_name: &str) {
        // create parent dirs
        let dir = Path::new(file_name).parent().unwrap();
        std::fs::create_dir_all(dir).unwrap();
        let mut file = File::create(file_name).unwrap();
        file.write(b"graph {\n").unwrap();
        file.write(b"edge [len=2.0]\n").unwrap();
        edges.iter().for_each(|g| {
            file.write(
                format!(
                    "h{}_v{} -- h{}_v{} [label=\"{}\"]\n",
                    g.p1.horizontal_id, g.p1.vertical_id, g.p2.horizontal_id, g.p2.vertical_id, g.distance
                )
                .as_bytes(),
            )
            .unwrap();
        });
        file.write(b"}\n").unwrap();
    }

    //let h_candidates: HashSet<_> = vec![7317].into_iter().collect();
    //let horizontal_id = 7317;
    let colls: Vec<Collision> = colls.into_iter().map(|x| x.into()).collect::<Vec<_>>();
    let mut global_edges: Vec<GlobalEdge> = Vec::new();
    /* global_edges.extend(create_horizontal_graph(7317, &colls));
    global_edges.extend(create_horizontal_graph(672, &colls));
    global_edges.extend(create_vertical_graph(13594, &colls));
    global_edges.extend(create_vertical_graph(10281, &colls)); */

    let horizontal_blacklist: HashSet<_> = vec![345].into_iter().collect();
    let vertical_blacklist: HashSet<_> = vec![2131, 8168].into_iter().collect();

    let all_horizontal_ids = colls
        .iter()
        .map(|x| x.h_id)
        .filter(|x| !horizontal_blacklist.contains(x))
        .collect::<HashSet<_>>();
    let all_vertical_ids = colls
        .iter()
        .map(|x| x.v_id)
        .filter(|x| !vertical_blacklist.contains(x))
        .collect::<HashSet<_>>();

    use indicatif::{ParallelProgressIterator, ProgressIterator};
    use rayon::prelude::*;

    let num_horizontal_ids = all_horizontal_ids.len();
    let num_vertical_ids = all_vertical_ids.len();

    all_horizontal_ids
        .into_par_iter()
        .progress_count(num_horizontal_ids as u64)
        .map(|h| (h, create_horizontal_graph(h, &colls)))
        .collect::<Vec<_>>()
        .into_iter()
        .for_each(|(h, edges)| {
            write_dot_file(&edges, &format!("data/fiber-dot/h{}.dot", h));
            global_edges.extend(edges);
        });
    all_vertical_ids
        .into_par_iter()
        .progress_count(num_vertical_ids as u64)
        .map(|v| (v, create_vertical_graph(v, &colls)))
        .collect::<Vec<_>>()
        .into_iter()
        .for_each(|(v, edges)| {
            write_dot_file(&edges, &format!("data/fiber-dot/v{}.dot", v));
            global_edges.extend(edges);
        });

    // create dot file
    let mut file = File::create("data/global.dot").unwrap();
    file.write(b"graph {\n").unwrap();
    file.write(b"edge [len=2.0]\n").unwrap();
    global_edges.iter().for_each(|g| {
        // add label with distance
        file.write(
            format!(
                "h{}_v{} -- h{}_v{} [label=\"{}\", weight=-{}];\n",
                g.p1.horizontal_id, g.p1.vertical_id, g.p2.horizontal_id, g.p2.vertical_id, g.distance, g.distance
            )
            .as_bytes(),
        )
        .unwrap();
    });

    file.write(b"}\n").unwrap();

    let h_whitelist: HashSet<_> = vec![
        99, 351, 640, 673, 835, 934, 1049, 2063, 2380, 2970, 3036, 3158, 3177, 3260, 3637, 3791, 4336, 4505, 4685,
        4785, 4869, 4881, 5017, 5113, 5275, 5332, 5451, 6520, 8034,
    ]
    .into_iter()
    .collect();
    let v_whitelist: HashSet<_> = vec![
        581, 682, 921, 932, 1554, 3452, 4151, 4417, 4903, 5880, 6056, 6520, 6826, 8955, 9282,
    ]
    .into_iter()
    .collect();
    let h_blacklist: HashSet<u32> = vec![3181].into_iter().collect();
    let v_blacklist: HashSet<u32> = vec![1594].into_iter().collect();
    let ignore_edges: HashSet<_> = vec![((673, 651), (3177, 651))].into_iter().collect();

    let mut file = File::create("data/inc.dot").unwrap();
    file.write(b"graph {\n").unwrap();
    file.write(b"edge [len=2.0]\n").unwrap();
    global_edges
        .iter()
        .filter(|g| {
            h_whitelist.contains(&g.p1.horizontal_id)
                || h_whitelist.contains(&g.p2.horizontal_id)
                || v_whitelist.contains(&g.p1.vertical_id)
                || v_whitelist.contains(&g.p2.vertical_id)
        })
        .filter(|g| !v_blacklist.contains(&g.p1.vertical_id) && !v_blacklist.contains(&g.p2.vertical_id))
        .filter(|g| !h_blacklist.contains(&g.p1.horizontal_id) && !h_blacklist.contains(&g.p2.horizontal_id))
        .filter(|g| {
            !ignore_edges.contains(&(
                (g.p1.horizontal_id, g.p1.vertical_id),
                (g.p2.horizontal_id, g.p2.vertical_id),
            ))
        })
        .for_each(|g| {
            let is_vertical = g.p1.vertical_id == g.p2.vertical_id;
            let color = if is_vertical { "red" } else { "blue" };
            // add label with distance
            file.write(
                format!(
                    "h{}_v{} -- h{}_v{} [label=\"{}\", weight=-{}, color={}];\n",
                    g.p1.horizontal_id,
                    g.p1.vertical_id,
                    g.p2.horizontal_id,
                    g.p2.vertical_id,
                    g.distance,
                    g.distance,
                    color
                )
                .as_bytes(),
            )
            .unwrap();
        });

    file.write(b"}\n").unwrap();

    fn write_obj_from_grid(colls: &[Collision], grid: &HashMap<GridCoord, CollisionPoint>) {
        let position_map = colls
            .iter()
            .map(|c| ((c.h_id, c.v_id), (c.x, c.y, c.z)))
            .collect::<HashMap<_, _>>();

        let max_grid_x = grid.keys().map(|(x, _)| x).max().unwrap();
        let max_grid_y = grid.keys().map(|(_, y)| y).max().unwrap();

        let mut vertices = vec![];

        for ((x, y), point) in grid {
            let x = *x;
            let y = *y;
            //try to build two triangles with adjacent points, only if all points are in the grid
            let t1 = [(x, y), (x + 1, y), (x + 1, y + 1)];
            let t2 = [(x, y), (x + 1, y + 1), (x, y + 1)];
            if t1.iter().all(|p| grid.contains_key(p)) {
                let p1 = grid.get(&t1[0]).unwrap();
                let p2 = grid.get(&t1[1]).unwrap();
                let p3 = grid.get(&t1[2]).unwrap();
                let p1 = position_map.get(&(p1.horizontal_id, p1.vertical_id)).unwrap();
                let p2 = position_map.get(&(p2.horizontal_id, p2.vertical_id)).unwrap();
                let p3 = position_map.get(&(p3.horizontal_id, p3.vertical_id)).unwrap();

                vertices.push((p1, t1[0]));
                vertices.push((p2, t1[1]));
                vertices.push((p3, t1[2]));
            }
            if t2.iter().all(|p| grid.contains_key(p)) {
                let p1 = grid.get(&t2[0]).unwrap();
                let p2 = grid.get(&t2[1]).unwrap();
                let p3 = grid.get(&t2[2]).unwrap();
                let p1 = position_map.get(&(p1.horizontal_id, p1.vertical_id)).unwrap();
                let p2 = position_map.get(&(p2.horizontal_id, p2.vertical_id)).unwrap();
                let p3 = position_map.get(&(p3.horizontal_id, p3.vertical_id)).unwrap();

                vertices.push((p1, t2[0]));
                vertices.push((p2, t2[1]));
                vertices.push((p3, t2[2]));
            }
        }

        let mut file = File::create("data/vertices.obj").unwrap();
        for ((x, y, z), (u, v)) in vertices.iter() {
            file.write(format!("v {} {} {}\n", x, y, z).as_bytes()).unwrap();
            file.write(
                format!(
                    "vt {} {}\n",
                    *u as f64 / *max_grid_x as f64,
                    *v as f64 / *max_grid_y as f64
                )
                .as_bytes(),
            )
            .unwrap();
            file.write(format!("vn 0 0 0\n").as_bytes()).unwrap();
        }
        for i in 0..vertices.len() / 3 {
            file.write(
                format!(
                    "f {}/{}/{} {}/{}/{} {}/{}/{}\n",
                    i * 3 + 1,
                    i * 3 + 1,
                    i * 3 + 1,
                    i * 3 + 2,
                    i * 3 + 2,
                    i * 3 + 2,
                    i * 3 + 3,
                    i * 3 + 3,
                    i * 3 + 3
                )
                .as_bytes(),
            )
            .unwrap();
        }
    }

    type GridCoord = (i32, i32);
    struct Grid {
        grid: HashMap<GridCoord, CollisionPoint>,
        positions: HashMap<CollisionPoint, GridCoord>,
    }
    impl Grid {
        fn new() -> Self {
            Self {
                grid: HashMap::default(),
                positions: HashMap::default(),
            }
        }
        fn insert(&mut self, position: GridCoord, point: CollisionPoint) {
            self.grid.insert(position, point);
            self.positions.insert(point, position);
        }
        fn at_pos(&self, position: GridCoord) -> Option<CollisionPoint> {
            self.grid.get(&position).cloned()
        }
        fn get_position(&self, point: CollisionPoint) -> Option<GridCoord> {
            self.positions.get(&point).cloned()
        }
    }
    //let mut grid: HashMap<(u32, u32), CollisionPoint> = HashMap::default();
    let mut grid = Grid::new();

    // iteratively add points to grid by looking at graph
    // start with one square:
    // 1000,1000 h5275_v7449
    // 1001,1000 h5275_v761
    // 1000,1001 h4881_v7449
    // 1001,1001 h4881_v761

    let n1 = CollisionPoint::new(5275, 7449);
    let n2 = CollisionPoint::new(5275, 761);
    let n3 = CollisionPoint::new(4881, 7449);
    let n4 = CollisionPoint::new(4881, 761);

    let mut edges: HashSet<Edge<CollisionPoint>> = HashSet::default();
    grid.insert((0, 0), n1);
    grid.insert((1, 0), n2);
    grid.insert((0, 1), n3);
    grid.insert((1, 1), n4);

    let mut queue = VecDeque::from([(n1, (0, 0)), (n2, (1, 0)), (n3, (0, 1)), (n4, (1, 1))]);

    let mut horizontal_neighbors: HashMap<CollisionPoint, HashSet<CollisionPoint>> = HashMap::default();
    let mut vertical_neighbors: HashMap<CollisionPoint, HashSet<CollisionPoint>> = HashMap::default();
    global_edges.iter().for_each(|g| {
        let is_horizontal = g.p1.horizontal_id == g.p2.horizontal_id;
        if is_horizontal {
            horizontal_neighbors
                .entry(g.p1)
                .or_insert(HashSet::default())
                .insert(g.p2);
            horizontal_neighbors
                .entry(g.p2)
                .or_insert(HashSet::default())
                .insert(g.p1);
        } else {
            vertical_neighbors
                .entry(g.p1)
                .or_insert(HashSet::default())
                .insert(g.p2);
            vertical_neighbors
                .entry(g.p2)
                .or_insert(HashSet::default())
                .insert(g.p1);
        }
    });
    struct PathFinder {
        horizontal_neighbors: HashMap<CollisionPoint, HashSet<CollisionPoint>>,
        vertical_neighbors: HashMap<CollisionPoint, HashSet<CollisionPoint>>,
    }
    impl PathFinder {
        fn new(
            horizontal_neighbors: HashMap<CollisionPoint, HashSet<CollisionPoint>>,
            vertical_neighbors: HashMap<CollisionPoint, HashSet<CollisionPoint>>,
        ) -> Self {
            Self {
                horizontal_neighbors,
                vertical_neighbors,
            }
        }
        fn find_squares_at_horizontal_first(&self, point: CollisionPoint) -> Vec<[CollisionPoint; 4]> {
            let mut squares = vec![];
            for p1 in self.horizontal_neighbors.get(&point).unwrap_or(&HashSet::default()) {
                for p2 in self.vertical_neighbors.get(p1).unwrap_or(&HashSet::default()) {
                    for p3 in self.horizontal_neighbors.get(p2).unwrap_or(&HashSet::default()) {
                        for p4 in self.vertical_neighbors.get(p3).unwrap_or(&HashSet::default()) {
                            if p4 == &point {
                                squares.push([*p1, *p2, *p3, *p4]);
                            }
                        }
                    }
                }
            }
            squares
        }
    }

    let path_finder = PathFinder::new(horizontal_neighbors, vertical_neighbors);

    while let Some((current, (x, y))) = queue.pop_front() {
        println!("Processing {:?} at {:?}", current, (x, y));

        let grid_neighbors = [
            (x - 1, y),
            (x + 1, y),
            (x, y - 1),
            (x, y + 1),
            (x - 1, y - 1),
            (x - 1, y + 1),
            (x + 1, y - 1),
            (x + 1, y + 1),
        ];
        if grid_neighbors.iter().all(|(x, y)| grid.at_pos((*x, *y)).is_some()) {
            println!("Skipping {:?} at {:?}", current, (x, y));
            continue; // done for this point as whole neighborhood is already in grid
        }

        for [p1, p2, p3, p4] in path_finder.find_squares_at_horizontal_first(current) {
            //println!("Found square {:?}", [p1, p2, p3, p4]);
            let dx = if let Some(p1) = grid.get_position(p1) {
                p1.0 - x
            } else {
                // exactly one neighbor must be in the grid already, find out which one, and go in opposite direction
                x - [(x - 1, y), (x + 1, y)]
                    .iter()
                    .find(|(x, y)| grid.at_pos((*x, *y)).is_some())
                    .unwrap()
                    .0
            };

            let dy = if let Some(p3) = grid.get_position(p3) {
                p3.1 - y
            } else {
                y - [(x, y - 1), (x, y + 1)]
                    .iter()
                    .find(|(x, y)| grid.at_pos((*x, *y)).is_some())
                    .unwrap()
                    .1
            };

            // println!("dx: {}, dy: {}", dx, dy);

            // p1 -> (x+dx, y)
            // p2 -> (x+dx, y+dy)
            // p3 -> (x, y+dy)
            // p4 -> (x, y)
            if !grid.get_position(p1).is_some() {
                //println!("Adding p1 to grid at {:?}", (x + dx, y));
                grid.insert((x + dx, y), p1);
                queue.push_back((p1, (x + dx, y)));
            }
            if !grid.get_position(p2).is_some() {
                //println!("Adding p2 to grid at {:?}", (x + dx, y + dy));
                grid.insert((x + dx, y + dy), p2);
                queue.push_back((p2, (x + dx, y + dy)));
            }
            if !grid.get_position(p3).is_some() {
                //println!("Adding p3 to grid at {:?}", (x, y + dy));
                grid.insert((x, y + dy), p3);
                queue.push_back((p3, (x, y + dy)));
            }
        }
    }
    grid.grid.iter().for_each(|(pos, p)| {
        println!("{:?}: {:?}", pos, p);
    });

    write_obj_from_grid(&colls, &grid.grid);

    /*let h_whitelist: HashSet<_> = vec![
            3260, 351, 4785, 3036, 4881, 5275, 2970, 5332, 934, 5017, 3158, 3791, 3637, 3177, 3181, 5113, 4685, 1049,
            673, 4505, 2063,
        ]
        .into_iter()
        .collect();
        let v_whitelist: HashSet<_> = vec![8034, 5880, 7449, 761, 581, 6056, 4417, 932, 4353, 1554, 669]
            .into_iter()
            .collect();
        let v_blacklist: HashSet<u32> = vec![].into_iter().collect();

        // ignore these edges
        //h5275_v8034 -- h5332_v8034 [label="75", weight=-75, color=red];
        //h3181_v761 -- h3637_v761 [label="78", weight=-78, color=red];
        //h934_v761 -- h2970_v761 [label="84", weight=-84, color=red];
        //h934_v6056 -- h2970_v6056 [label="88", weight=-88, color=red];
        //h934_v5880 -- h2970_v5880 [label="76", weight=-76, color=red];
        //h3181_v581 -- h3637_v581 [label="80", weight=-80, color=red];
        //h934_v581 -- h2970_v581 [label="89", weight=-89, color=red];
        //h2970_v7449 -- h2970_v8034 [label="87", weight=-87, color=blue];
        //h934_v7449 -- h2970_v7449 [label="79", weight=-79, color=red];
        //h3181_v6056 -- h3637_v6056 [label="80", weight=-80, color=red];
        //h3158_v6056 -- h3791_v6056 [label="98", weight=-98, color=red];
        let ignore_edges: HashSet<_> = vec![
            ((5275, 8034), (5332, 8034)),
            ((3181, 761), (3637, 761)),
            ((934, 761), (2970, 761)),
            ((934, 6056), (2970, 6056)),
            ((934, 5880), (2970, 5880)),
            ((3181, 581), (3637, 581)),
            ((934, 581), (2970, 581)),
            ((2970, 7449), (2970, 8034)),
            ((934, 7449), (2970, 7449)),
            ((3181, 6056), (3637, 6056)),
            ((3158, 6056), (3791, 6056)),
        ]
        .into_iter()
        .collect();

        let selected_edges = global_edges
            .iter()
            .filter(|g| {
                h_whitelist.contains(&g.p1.horizontal_id)
                    && h_whitelist.contains(&g.p2.horizontal_id)
                    && v_whitelist.contains(&g.p1.vertical_id)
                    && v_whitelist.contains(&g.p2.vertical_id)
            })
            .filter(|g| {
                !ignore_edges.contains(&(
                    (g.p1.horizontal_id, g.p1.vertical_id),
                    (g.p2.horizontal_id, g.p2.vertical_id),
                ))
            })
            .collect::<Vec<_>>();

        let mut file = File::create("data/excl.dot").unwrap();
        file.write(b"graph {\n").unwrap();
        file.write(b"edge [len=2.0]\n").unwrap();

        selected_edges.iter().for_each(|&g| {
            let is_vertical = g.p1.vertical_id == g.p2.vertical_id;
            let color = if is_vertical { "red" } else { "blue" };
            // add label with distance
            file.write(
                format!(
                    "h{}_v{} -- h{}_v{} [label=\"{}\", weight=-{}, color={}];\n",
                    g.p1.horizontal_id,
                    g.p1.vertical_id,
                    g.p2.horizontal_id,
                    g.p2.vertical_id,
                    g.distance,
                    g.distance,
                    color
                )
                .as_bytes(),
            )
            .unwrap();
        });
        file.write(b"}\n").unwrap();

        let mut positions: HashMap<CollisionPoint, (u32, u32)> = HashMap::default();
        let start = CollisionPoint::new(2063, 6056);
        let mut queue: PriorityQueue<(CollisionPoint, (u32, u32)), Reverse<u32>> = PriorityQueue::new();
        queue.push((start, (0, 0)), Reverse(0));
        while let Some(((current, (x, y)), _)) = queue.pop() {
            if !positions.contains_key(&current) {
                positions.insert(current, (x, y));
                selected_edges
                    .iter()
                    .filter(|g| g.p1 == current || g.p2 == current)
                    .for_each(|g| {
                        let next = if g.p1 == current { g.p2 } else { g.p1 };
                        let is_horizontal = g.p1.horizontal_id == g.p2.horizontal_id;
                        let (dx, dy) = if is_horizontal { (1, 0) } else { (0, 1) };
                        queue.push((next, (x + dx, y + dy)), Reverse(x + dx + y + dy));
                    });
            }
        }
        positions
            .iter()
            .sorted_by_key(|(_, pos)| *pos)
            .for_each(|(point, (x, y))| {
                println!("{:?}: {:?}", (x, y), point);
            });

        let grid = positions
            .iter()
            .map(|(point, (x, y))| ((*x, *y), point))
            .collect::<HashMap<_, _>>();
    } */
}

struct PaintableImage {
    image: ColorImage,
}
impl GenericImageView for PaintableImage {
    type Pixel = Rgb<u8>;
    fn dimensions(&self) -> (u32, u32) {
        (self.image.size[0] as u32, self.image.size[1] as u32)
    }
    fn get_pixel(&self, x: u32, y: u32) -> Self::Pixel {
        //let data = self.data[y as usize * self.width + x as usize];
        let p = self.image[(x as usize, y as usize)];
        Rgb([p.r(), p.g(), p.b()])
    }
}
impl GenericImage for PaintableImage {
    fn get_pixel_mut(&mut self, x: u32, y: u32) -> &mut Self::Pixel {
        todo!()
    }
    fn put_pixel(&mut self, x: u32, y: u32, pixel: Self::Pixel) {
        self.image[(x as usize, y as usize)] = Color32::from_rgb(pixel[0], pixel[1], pixel[2]);
    }
    fn blend_pixel(&mut self, x: u32, y: u32, pixel: Self::Pixel) {
        todo!()
    }
}

pub(crate) struct CollisionPanel {
    collisions: Vec<(u32, u32, u16, u16, u16)>,
    neighbor_map_h: HashMap<u32, Vec<u32>>,
    neighbor_map_v: HashMap<u32, Vec<u32>>,
    selected_h: Option<u32>,
    texture: Option<TextureHandle>,
}
impl CollisionPanel {
    pub fn new() -> CollisionPanel {
        let file = File::open("data/collisions").unwrap();
        use std::io::BufRead;
        let mut reader = BufReader::new(file);
        let mut lines = reader.lines();
        let mut colls = vec![];
        while let Some(Ok(line)) = lines.next() {
            let mut parts = line.split_whitespace().collect::<Vec<_>>();
            // h<id1> v<id2> <x> <y> <z>
            let id1 = parts[0][1..].parse::<u32>().unwrap();
            let id2 = parts[1][1..].parse::<u32>().unwrap();
            let x = parts[2].parse::<u16>().unwrap();
            let y = parts[3].parse::<u16>().unwrap();
            let z = parts[4].parse::<u16>().unwrap();

            colls.push((id1, id2, x, y, z));
        }

        /* // group by id2 and count
        colls
            .iter()
            .sorted_by_key(|x| x.1)
            .chunk_by(|x| x.1)
            .into_iter()
            .map(|(id, group)| (id, group.count()))
            .sorted_by_key(|x| x.1)
            .chunk_by(|x| x.1)
            .into_iter()
            .map(|(count, group)| (count, group.count()))
            .for_each(|(count, num)| {
                println!("{}: {}", count, num);
            }); */

        // create new dot file that only contains edges where the vertical has rank 2
        /* let selected_vertical = colls
        .iter()
        .sorted_by_key(|x| x.1)
        .chunk_by(|x| x.1)
        .into_iter()
        .map(|(id, group)| (id, group.count()))
        .filter_map(|(id, count)| if count == 2 { Some(id) } else { None })
        .collect::<HashSet<_>>(); */

        //let filtered_colls = colls;

        let mut neighbor_map_h: HashMap<u32, Vec<u32>> = HashMap::default();
        let mut neighbor_map_v: HashMap<u32, Vec<u32>> = HashMap::default();
        colls.iter().for_each(|(id1, id2, x, y, z)| {
            neighbor_map_h.entry(*id1).or_insert(vec![]).push(*id2);
            neighbor_map_v.entry(*id2).or_insert(vec![]).push(*id1);
        });

        CollisionPanel {
            collisions: colls,
            neighbor_map_h,
            neighbor_map_v,
            selected_h: None,
            texture: None,
        }
    }
    pub fn draw(&mut self, ctx: &egui::Context) -> Option<[i32; 3]> {
        egui::Window::new("Collisions")
            .show(ctx, |ui| {
                egui::Grid::new("my_grid")
                    .num_columns(2)
                    .min_row_height(500.)
                    //.spacing([40.0, 4.0])
                    .show(ui, |ui| {
                        let available_height = ui.available_height();
                        let mut table = TableBuilder::new(ui)
                            //.vscroll(true)
                            .auto_shrink([false, true])
                            .resizable(true)
                            .max_scroll_height(available_height)
                            .column(Column::auto())
                            .column(Column::remainder() /* Column::initial(150.0) */);

                        table = table.sense(egui::Sense::click());

                        table
                            /* .header(20.0, |mut header| {
                                header.col(|ui| {
                                    ui.strong("ID");
                                });
                                header.col(|ui| {
                                    ui.strong("Num");
                                });
                            }) */
                            .body(|mut body| {
                                for (id, neighbors) in self.neighbor_map_h.iter().sorted_by_key(|x| x.1.len()).rev() {
                                    body.row(20., |mut row| {
                                        row.set_selected(self.selected_h == Some(*id));
                                        fn l(text: impl Into<WidgetText>) -> Label {
                                            Label::new(text).selectable(false)
                                        }
                                        use egui::Widget;
                                        row.col(|ui| {
                                            l(format!("h{}", id)).ui(ui);
                                        });
                                        row.col(|ui| {
                                            l(neighbors.len().to_string()).ui(ui);
                                        });
                                        if row.response().clicked() {
                                            println!("Selected {}", id);
                                            self.selected_h = Some(*id);
                                        }
                                    });
                                }
                            });

                        let col_image = ColorImage::new([500, 500], Color32::BLACK);
                        let mut im: PaintableImage = PaintableImage { image: col_image };

                        let colls = self
                            .collisions
                            .iter()
                            .filter(|(id1, id2, x, y, z)| self.selected_h == Some(*id1))
                            .collect::<Vec<_>>();

                        let min = colls
                            .iter()
                            .map(|(id1, id2, x, y, z)| [*x, *y, *z])
                            .fold([u16::MAX, u16::MAX, u16::MAX], |acc, v| {
                                [acc[0].min(v[0]), acc[1].min(v[1]), acc[2].min(v[2])]
                            });
                        let max = colls
                            .iter()
                            .map(|(id1, id2, x, y, z)| [*x, *y, *z])
                            .fold([0, 0, 0], |acc, v| {
                                [acc[0].max(v[0]), acc[1].max(v[1]), acc[2].max(v[2])]
                            });
                        let min = min.map(|x| x - 10);
                        let max = max.map(|x| x + 10);
                        let range = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];

                        let tex = if let Some(handle) = self.texture.as_mut() {
                            handle
                        } else {
                            let tex = ui.ctx().load_texture(
                                "image",
                                ColorImage::new([500, 500], Color32::BLACK),
                                Default::default(),
                            );
                            self.texture = Some(tex);
                            self.texture.as_ref().unwrap()
                        };
                        let image = Image::new(tex);
                        let image = image.sense(Sense::click_and_drag());

                        let i = ui.add(image);
                        i.interact(Sense::click_and_drag());

                        let hovered: Option<(u32, u32)> = if let Some(pos) = i.hover_pos() {
                            let pos = pos - i.rect.min;

                            let mx = (pos.x / 500.0 * range[0] as f32 + min[0] as f32) as u16;
                            let my = (pos.y / 500.0 * range[1] as f32 + min[1] as f32) as u16;

                            // find nearest point with max distance 10
                            colls
                                .iter()
                                .min_by_key(|(id1, id2, x, y, z)| {
                                    let dx = (mx as i32 - *x as i32).abs();
                                    let dy = (my as i32 - *y as i32).abs();
                                    dx + dy
                                })
                                .map(|(id1, id2, x, y, z)| (*id1, *id2))
                        } else {
                            None
                        };

                        let click_pos = if i.clicked() {
                            let pos = ui.input(|i| i.pointer.interact_pos().unwrap()) - i.rect.min;
                            println!("Clicked at {:?}", pos);
                            //let pos = pos - i.rect.min;

                            let mx = (pos.x / 500.0 * range[0] as f32 + min[0] as f32) as u16;
                            let my = (pos.y / 500.0 * range[1] as f32 + min[1] as f32) as u16;

                            // find nearest point with max distance 10
                            colls
                                .iter()
                                .min_by_key(|(id1, id2, x, y, z)| {
                                    let dx = (mx as i32 - *x as i32).abs();
                                    let dy = (my as i32 - *y as i32).abs();
                                    dx + dy
                                })
                                .map(|(id1, id2, x, y, z)| [*x as i32, *y as i32, *z as i32])
                        } else {
                            None
                        };

                        {
                            use imageproc::drawing::*;

                            colls.iter().for_each(|(id1, id2, x, y, z)| {
                                let lx = (x - min[0]) as f32 / range[0] as f32 * 500.0;
                                let ly = (y - min[1]) as f32 / range[1] as f32 * 500.0;

                                let color: Rgb<u8> = if let Some((hover_id1, hover_id2)) = hovered {
                                    if *id1 == hover_id1 && *id2 == hover_id2 {
                                        Rgb([0, 255, 0])
                                    } else {
                                        Rgb([255, 0, 0])
                                    }
                                } else {
                                    Rgb([255, 0, 0])
                                };

                                draw_filled_circle_mut(&mut im, (lx as i32, ly as i32), 2, color);
                            });
                        }

                        let handle = self.texture.as_mut().unwrap();
                        handle.set(im.image, Default::default());
                        click_pos
                    })
                    .inner
            })
            .map(|x| x.inner)
            .flatten()
            .flatten()
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
        let array = array.into_ctx();
        let mut work_list = vec![];
        let mut visited = HashSet::default();
        let mut selected = vec![];
        let mut selected_coords = vec![];

        for y in 0..shape[1] as u16 {
            for x in 0..shape[2] as u16 {
                let idx = index_of(x, y, z);
                //let idx = [z, y, x];
                //let v = full.mmap[idx];
                let v = array.get([z as usize, y as usize, x as usize]);
                let mut count = 0;
                if let Some(v) = v {
                    if read_map[idx] == 0 {
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
                            if let Some(v2) = v2 {
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
                                    // for now just consider neighbors that share a full face of the voxel cube ("6-connected")
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
                if let Some(v) = v {
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

    let zarr = zarr.into_ctx();
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
    for _ in 0..100000000 {
        sum += zarr.get(at0).unwrap_or(0);
        sum += zarr.get(at1).unwrap_or(0);
        sum += zarr.get(at2).unwrap_or(0);
        sum += zarr.get(at3).unwrap_or(0);
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

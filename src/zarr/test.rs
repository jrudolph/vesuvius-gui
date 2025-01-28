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
    collections::BinaryHeap,
    fmt::Debug,
    fmt::Display,
    fs::{File, OpenOptions},
    hash::Hash,
    io::{BufReader, Write},
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

// const CROP: [usize; 3] = [2400, 3000, 9300];
// const CROP_SIZE: [usize; 3] = [400, 1000, 1000];

/* const CROP: [usize; 3] = [2400, 3000, 9300];
const CROP_SIZE: [usize; 3] = [2000, 2000, 2000]; */

const CROP: [usize; 3] = [2400, 3000, 9300];
const CROP_SIZE: [usize; 3] = [2000, 2000, 2000];

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
        ((z + 1 - CROP[2]) * CROP_SIZE[1] + (y + 1 - CROP[1])) * CROP_SIZE[0] + (x + 1 - CROP[0])
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
    // figure out all point pairs that are within 4 pixels of each other
    let mut grid: HashMap<[u16; 3], (bool, bool)> = HashMap::default();
    cloud1.iter().for_each(|coords| {
        let grid_coords = coords.map(|x| x / 4 * 4);
        grid.entry(grid_coords).or_insert((false, false)).0 = true;
        let grid_coords = coords.map(|x| x / 4 * 4 + 2);
        grid.entry(grid_coords).or_insert((false, false)).0 = true;
    });
    cloud2.iter().for_each(|coords| {
        let grid_coords = coords.map(|x| x / 4 * 4);
        grid.entry(grid_coords).or_insert((false, false)).1 = true;
        let grid_coords = coords.map(|x| x / 4 * 4 + 2);
        grid.entry(grid_coords).or_insert((false, false)).1 = true;
    });

    let colliding_cells = grid
        .iter()
        .filter_map(|(coords, (c1, c2))| if *c1 && *c2 { Some(*coords) } else { None })
        .collect::<Vec<_>>();

    let len = colliding_cells.len() as u32;
    if len == 0 {
        return None;
    }

    // average grid_coords
    let mut sum = [0, 0, 0];
    colliding_cells.iter().for_each(|coords| {
        sum[0] += coords[0] as u32;
        sum[1] += coords[1] as u32;
        sum[2] += coords[2] as u32;
    });

    // average and center into the middle of the cell
    let avg = [
        (sum[0] / len + 2) as u16,
        (sum[1] / len + 2) as u16,
        (sum[2] / len + 2) as u16,
    ];
    let avg = avg.map(|x| x as u16);
    Some(avg)
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
    use indicatif::ParallelProgressIterator;
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

    fn create_collision_graph<
        AlongId: Hash + Eq + Copy + Display + Ord + Debug,
        AcrossId: Hash + Eq + Copy + Display + Ord + Debug,
    >(
        along_id: AlongId,
        colls: &[Collision],
        get_along_id: impl Fn(&Collision) -> AlongId,
        get_across_id: impl Fn(&Collision) -> AcrossId,
        cloud: &PointCloudFile,
    ) {
        let collisions = colls.iter().filter(|c| get_along_id(c) == along_id).collect::<Vec<_>>();

        println!("total collisions: {}", collisions.len());

        let map = Map::new(&cloud);
        println!("Map entries: {}", map.map.len());

        // calculate the geodesic adjacency matrix for all collision points on h22554
        let mut adjacency_matrix: HashMap<(AcrossId, AcrossId), u64> = HashMap::default();
        for start_coll @ Collision {
            x: start_x,
            y: start_y,
            z: start_z,
            ..
        } in &collisions
        {
            let start_id_across = get_across_id(start_coll);
            println!("At start_id_across: {}", start_id_across);
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
            } in &collisions
            {
                let end_id_across = get_across_id(end_coll);
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

        println!("Adjacency matrix: {}", adjacency_matrix.len());
        /* adjacency_matrix.iter().for_each(|((start, end), distance)| {
            println!("{} {} {}", start.v_id, end.v_id, distance);
        }); */

        struct Adjacency<AcrossId: Hash + Eq + Copy + Ord> {
            matrix: HashMap<(AcrossId, AcrossId), u64>,
        }
        impl<AcrossId: Hash + Eq + Copy + Ord> Adjacency<AcrossId> {
            fn get_distance(&self, key1: AcrossId, key2: AcrossId) -> Option<u64> {
                let min = key1.min(key2);
                let max = key1.max(key2);
                self.get(min).get(&max).copied()
            }
            fn get(&self, key: AcrossId) -> HashMap<AcrossId, u64> {
                self.matrix
                    .iter()
                    .filter(|(k, _)| k.0 == key || k.1 == key)
                    .map(|(k, v)| if k.0 == key { (k.1, *v) } else { (k.0, *v) })
                    .collect()
            }
            /* fn neighbors2(&self, key: u32) -> (u32, u32) {
                let mut neighbors = self.get(key);
                let mut next2 = neighbors
                    .into_iter()
                    //.filter(|(_, dist)| *dist > 30) // FIXME: hacky way to ignore points that are too close
                    .sorted_by_key(|(id, dist)| *dist)
                    .take(2);
                (next2.next().unwrap().0, next2.next().unwrap().0)
            }
            fn neighbor1(&self, key: u32) -> u32 {
                self.neighbors2(key).0
            } */
        }
        let adjacency = Adjacency {
            matrix: adjacency_matrix,
        };

        let first_node: AcrossId = adjacency.matrix.iter().next().unwrap().0 .0;
        let most_distant = adjacency.get(first_node).into_iter().max_by_key(|(_, d)| *d).unwrap();
        println!("Most distant: {:?}", &most_distant);

        let other_end = adjacency
            .get(most_distant.0)
            .into_iter()
            .max_by_key(|(_, d)| *d)
            .unwrap();
        println!("Other end: {:?}", other_end);

        #[derive(PartialEq, Eq, Hash)]
        struct Edge<AcrossId: Ord + Copy>(AcrossId, AcrossId);
        impl<AcrossId: Ord + Copy> Edge<AcrossId> {
            fn new(id1: AcrossId, id2: AcrossId) -> Self {
                Self(id1.min(id2), id1.max(id2))
            }
        }
        let mut edges = HashSet::default();
        let keys = collisions.iter().map(|x| get_across_id(x)).collect::<HashSet<_>>();
        keys.into_iter().for_each(|key| {
            /* if key == most_distant.0 || key == other_end.0 {
                edges.insert(Edge::new(key, adjacency.neighbor1(key)));
            } else {
                let (n1, n2) = adjacency.neighbors2(key);
                edges.insert(Edge::new(key, n1));
                edges.insert(Edge::new(key, n2));
            } */
            adjacency
                .get(key)
                .into_iter()
                .sorted_by_key(|(_, d)| *d)
                .take(3)
                .for_each(|(other, d)| {
                    edges.insert(Edge::new(key, other));
                });
        });

        println!("Edges: {}", edges.len());
        edges.iter().for_each(|edge| {
            println!("{} {}", edge.0, edge.1);
        });

        // create dot file
        let mut file = File::create(&format!("data/h{:06}.dot", along_id)).unwrap();
        file.write(b"graph {\n").unwrap();
        file.write(b"edge [len=2.0]\n").unwrap();
        edges.iter().for_each(|Edge(v1, v2)| {
            // add label with distance
            let distance = adjacency.get_distance(*v1, *v2).unwrap();
            file.write(
                format!(
                    "h{}_v{} -- h{}_v{} [label=\"{}\", weight=-{}];\n",
                    along_id, v1, along_id, v2, distance, distance
                )
                .as_bytes(),
            )
            .unwrap();
        });

        file.write(b"}\n").unwrap();
    }

    let h_candidates: HashSet<_> = vec![7317].into_iter().collect();
    let horizontal_id = 7317;
    let cloud = PointCloudFile::new(horizontal_id, &format!("data/classes/class-2/{:06}", horizontal_id));
    create_collision_graph(
        horizontal_id,
        &colls.into_iter().map(|x| x.into()).collect::<Vec<_>>(),
        |x| x.h_id,
        |x| x.v_id,
        &cloud,
    );

    // create dot file for vertical connections
    /* let vertical_connections = colls
        .iter()
        //.filter(|x| selected_vertical.contains(&x.1))
        .sorted_by_key(|x| x.1)
        .chunk_by(|x| x.1)
        .into_iter()
        .map(|(v, colls)| {
            let hs = colls.map(|x| x.0).collect::<HashSet<_>>();
            (v, hs)
        })
        .collect::<Vec<_>>();
    let mut file = File::create("data/verticals.dot").unwrap();
    file.write(b"graph {\n").unwrap();
    file.write(b"edge [len=2.0]\n").unwrap();
    for (v, hs) in vertical_connections {
        assert!(hs.len() == 2);
        let hs = hs.into_iter().collect::<Vec<_>>();
        let h1 = hs[0];
        let h2 = hs[1];

        if h_candidates.contains(&h1) || h_candidates.contains(&h2) {
            file.write(format!("h{}_v{} -- h{}_v{}\n", h1, v, h2, v).as_bytes())
                .unwrap();
        }
    }
    file.write(b"}\n").unwrap(); */

    // v19894 -> h22554

    /* let first_node = adjacency_matrix.iter().next().unwrap().0 .0;

    // find most distant node from first_node
    let most_distant = adjacency_matrix
        .iter()
        .filter(|(key, _)| key.0 == first_node || key.1 == first_node)
        .max_by_key(|(_, distance)| *distance)
        .unwrap()
        .0;
    println!("Most distant: {:?}", most_distant);
    let other_end = adjacency_matrix
        .iter()
        .filter(|(key, _)| key != most_distant && key.0 == most_distant.0 .1 || key.1 == most_distant.0 .1)
        .max_by_key(|(_, distance)| *distance)
        .unwrap()
        .0;
    println!("Other end: {:?}", other_end); */
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
        let selected_vertical = colls
            .iter()
            .sorted_by_key(|x| x.1)
            .chunk_by(|x| x.1)
            .into_iter()
            .map(|(id, group)| (id, group.count()))
            .filter_map(|(id, count)| if count == 2 { Some(id) } else { None })
            .collect::<HashSet<_>>();

        let filtered_colls = colls
            .iter()
            .filter(|(id1, id2, x, y, z)| selected_vertical.contains(id2))
            .collect::<Vec<_>>();

        let mut neighbor_map_h: HashMap<u32, Vec<u32>> = HashMap::default();
        let mut neighbor_map_v: HashMap<u32, Vec<u32>> = HashMap::default();
        filtered_colls.iter().for_each(|(id1, id2, x, y, z)| {
            neighbor_map_h.entry(*id1).or_insert(vec![]).push(*id2);
            neighbor_map_v.entry(*id2).or_insert(vec![]).push(*id1);
        });

        CollisionPanel {
            collisions: filtered_colls.into_iter().cloned().collect(),
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

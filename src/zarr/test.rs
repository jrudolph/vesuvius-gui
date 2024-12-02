use super::{ZarrContext, ZarrContextBase};
use crate::{
    volume::PaintVolume,
    zarr::{blosc::BloscChunk, ZarrArray},
};
use egui::Color32;
use memmap::MmapOptions;
use std::{
    collections::{HashMap, HashSet},
    fs::{File, OpenOptions},
    io::Write,
    sync::{
        atomic::{AtomicU32, Ordering},
        Mutex,
    },
};

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
    #[allow(dead_code)]
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
    #[allow(dead_code)]
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
    for _ in 0..100000000 {
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

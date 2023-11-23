use crate::downloader::*;
use crate::model::Quality;
use crate::volume::PaintVolume;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use super::DrawingConfig;

#[derive(Debug)]
enum TileState {
    Unknown,
    Missing,
    Loaded(memmap::Mmap),
    Downloading(Arc<Mutex<DownloadState>>),
    TryLater(SystemTime),
}

pub struct VolumeGrid64x4Mapped {
    data_dir: String,
    downloader: Downloader,
    data: HashMap<(usize, usize, usize, usize), TileState>,
}
impl VolumeGrid64x4Mapped {
    fn map_for(data_dir: &str, x: usize, y: usize, z: usize, quality: Quality) -> Option<TileState> {
        use memmap::MmapOptions;
        use std::fs::File;
        let file_name = format!(
            "{}/64-4/d{:02}/z{:03}/xyz-{:03}-{:03}-{:03}-b{:03}-d{:02}.bin",
            data_dir, quality.downsampling_factor, z, x, y, z, quality.bit_mask, quality.downsampling_factor
        );
        //println!("at {}", file_name);

        let file = File::open(file_name.clone()).ok()?;

        let map = unsafe { MmapOptions::new().map(&file) }.ok();
        map.filter(|m| {
            if m.len() == 64 * 64 * 64 {
                true
            } else {
                println!("file {} has wrong size {}", file_name, m.len());
                false
            }
        })
        .map(|x| TileState::Loaded(x))
    }
    fn try_loading_tile(&mut self, x: usize, y: usize, z: usize, quality: Quality) -> &TileState {
        let key = (x, y, z, quality.downsampling_factor as usize);
        if !self.data.contains_key(&key) {
            self.data.insert(key, TileState::Unknown);
        }
        let tile_state = self.data.get_mut(&key).unwrap();
        match tile_state {
            TileState::Unknown => {
                // println!("trying to load tile {}/{}/{} q{}", x, y, z, quality.downsampling_factor);
                if let Some(state) = Self::map_for(&self.data_dir, x, y, z, quality) {
                    *tile_state = state;
                } else {
                    let state = Arc::new(Mutex::new(DownloadState::Queuing));
                    self.downloader.queue((state.clone(), x, y, z, quality));
                    *tile_state = TileState::Downloading(state);
                }
            }
            TileState::TryLater(at) => {
                if at.elapsed().unwrap() > Duration::from_secs(10) {
                    println!(
                        "resetting tile {}/{}/{} q{} again",
                        x, y, z, quality.downsampling_factor
                    );
                    *tile_state = TileState::Unknown; // reset
                    self.try_loading_tile(x, y, z, quality);
                }
            }
            TileState::Downloading(state) => {
                match *state.clone().lock().unwrap() {
                    DownloadState::Done => {
                        if let Some(state) = Self::map_for(&self.data_dir, x, y, z, quality) {
                            *tile_state = state;
                        } else {
                            // set to missing permanently
                            println!(
                                "failed to load tile from map {}/{}/{} q{}",
                                x, y, z, quality.downsampling_factor
                            );
                            *tile_state = TileState::Missing;
                        }
                    }
                    DownloadState::Delayed => {
                        *tile_state = TileState::TryLater(SystemTime::now());
                    }
                    DownloadState::Failed => {
                        *tile_state = TileState::Missing;
                    }
                    DownloadState::Pruned => {
                        *tile_state = TileState::Unknown;
                    }
                    DownloadState::Queuing => {}
                    DownloadState::Downloading => {}
                };
            }
            _ => {}
        }
        self.data.get(&key).unwrap()
    }
    pub fn from_data_dir(data_dir: &str, downloader: Downloader) -> VolumeGrid64x4Mapped {
        if !std::path::Path::new(data_dir).exists() {
            panic!("Data directory {} does not exist", data_dir);
        }

        VolumeGrid64x4Mapped {
            data_dir: data_dir.to_string(),
            downloader,
            data: HashMap::new(),
        }
    }
}
impl PaintVolume for VolumeGrid64x4Mapped {
    fn paint(
        &mut self,
        xyz: [i32; 3],
        u_coord: usize,
        v_coord: usize,
        plane_coord: usize,
        canvas_width: usize,
        canvas_height: usize,
        _sfactor: u8,
        paint_zoom: u8,
        config: &DrawingConfig,
        buffer: &mut [u8],
    ) {
        let width = paint_zoom as usize * canvas_width;
        let height = paint_zoom as usize * canvas_height;

        let mask = config.bit_mask();
        let filters_active = config.filters_active();

        self.downloader.position(xyz[0], xyz[1], xyz[2], width, height);

        let center_u = canvas_width as i32 / 2;
        let center_v = canvas_height as i32 / 2;

        let sfactor = _sfactor as i32;
        let tilesize = 64 * sfactor as i32;
        let blocksize = 4 * sfactor as i32;

        let min_uc = xyz[u_coord] - width as i32 / 2;
        let max_uc = xyz[u_coord] + width as i32 / 2;
        let min_vc = xyz[v_coord] - height as i32 / 2;
        let max_vc = xyz[v_coord] + height as i32 / 2;
        let pc = xyz[plane_coord].max(0);

        let tile_min_uc = (min_uc / tilesize).max(0);
        let tile_max_uc = max_uc / tilesize;

        let tile_min_vc = (min_vc / tilesize).max(0);
        let tile_max_vc = max_vc / tilesize;

        let tile_pc = pc / tilesize;
        let tile_pc_off = pc % tilesize;
        let block_pc = tile_pc_off / blocksize;
        let block_pc_off = tile_pc_off % blocksize;

        for tile_uc in tile_min_uc..=tile_max_uc {
            for tile_vc in tile_min_vc..=tile_max_vc {
                let mut tile_i = [0; 3];
                tile_i[u_coord] = tile_uc as usize;
                tile_i[v_coord] = tile_vc as usize;
                tile_i[plane_coord] = tile_pc as usize;

                let state = self.try_loading_tile(
                    tile_i[0],
                    tile_i[1],
                    tile_i[2],
                    Quality {
                        bit_mask: 0xff,
                        downsampling_factor: sfactor as u8,
                    },
                );

                if let TileState::Loaded(tile) = state {
                    // iterate over blocks in tile
                    let min_tile_uc = (tile_uc * tilesize).max(min_uc) - tile_uc * tilesize;
                    let max_tile_uc = (tile_uc * tilesize + tilesize).min(max_uc) - tile_uc * tilesize;
                    let min_tile_vc = (tile_vc * tilesize).max(min_vc) - tile_vc * tilesize;
                    let max_tile_vc = (tile_vc * tilesize + tilesize).min(max_vc) - tile_vc * tilesize;

                    let min_block_uc = min_tile_uc / blocksize;
                    let max_block_uc = (max_tile_uc + blocksize - 1) / blocksize;
                    let min_block_vc = min_tile_vc / blocksize;
                    let max_block_vc = (max_tile_vc + (blocksize - 1)) / blocksize;

                    //println!("min_tile_x: {} max_tile_x: {} min_tile_y: {} max_tile_y: {}", min_tile_x, max_tile_x, min_tile_y, max_tile_y);
                    //println!("min_block_x: {} max_block_x: {} min_block_y: {} max_block_y: {}", min_block_x, max_block_x, min_block_y, max_block_y);

                    for block_vc in min_block_vc..max_block_vc {
                        for block_uc in min_block_uc..max_block_uc {
                            let mut block_i = [0; 3];
                            block_i[u_coord] = block_uc as usize;
                            block_i[v_coord] = block_vc as usize;
                            block_i[plane_coord] = block_pc as usize;
                            let boff = (block_i[2] << 8) + (block_i[1] << 4) + block_i[0];

                            // iterate over pixels in block
                            for vc in (0..blocksize).step_by(paint_zoom as usize) {
                                for uc in (0..blocksize).step_by(paint_zoom as usize) {
                                    let u = ((tile_uc * tilesize + block_uc * blocksize + uc) as i32 - min_uc)
                                        / paint_zoom as i32;
                                    let v = ((tile_vc * tilesize + block_vc * blocksize + vc) as i32 - min_vc)
                                        / paint_zoom as i32;
                                    if uc == 0 && vc == 0 {
                                        //println!("block_x: {} block_y: {}", block_x, block_y);
                                        //println!("u: {} v: {}", u, v);
                                    }
                                    let mut offs_i = [0; 3];
                                    //if (u / tilesize) % 2 == 0 {
                                    offs_i[u_coord] = uc as usize / sfactor as usize;
                                    offs_i[v_coord] = vc as usize / sfactor as usize;
                                    offs_i[plane_coord] = block_pc_off as usize / sfactor as usize;
                                    /* } else {
                                        let fac = 2;
                                        offs_i[u_coord] = (uc as usize) / fac * fac;
                                        offs_i[v_coord] = (vc as usize) / fac * fac;
                                        offs_i[plane_coord] = (block_pc_off as usize) / fac * fac;
                                    } */
                                    //let factor = quality.downsampling_factor as usize * quality.downsampling_factor as usize * quality.downsampling_factor as usize;

                                    if u >= 0 && u < canvas_width as i32 && v >= 0 && v < canvas_height as i32 {
                                        let off = boff * 64 + offs_i[2] * 16 + offs_i[1] * 4 + offs_i[0];
                                        if off > tile.len() {
                                            panic!("off: {} tile.len(): {}", off, tile.len());
                                        }
                                        let value = tile[off as usize];

                                        if u == center_u || v == center_v {
                                            buffer[v as usize * canvas_width + u as usize] = 0;
                                        } else if filters_active {
                                            let pluscon = ((value as i32 - config.threshold_min as i32).max(0) * 255
                                                / (255 - (config.threshold_min + config.threshold_max) as i32))
                                                .min(255)
                                                as u8;

                                            buffer[v as usize * canvas_width + u as usize] =
                                                (((pluscon & mask) as f32) / (mask as f32) * 255.0) as u8;
                                        } else {
                                            buffer[v as usize * canvas_width + u as usize] = value;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

pub struct PPMFile {
    pub width: usize,
    pub height: usize,
    map: memmap::Mmap,
}
impl PPMFile {
    pub fn new(file_name: &str, width: usize, height: usize) -> Option<Self> {
        use memmap::MmapOptions;
        use std::fs::File;

        let file = File::open(file_name).ok()?;
        let map = unsafe { MmapOptions::new().offset(73).map(&file) }.ok();

        map.map(|map| Self { width, height, map })
    }
    pub fn get(&self, u: usize, v: usize) -> [f64; 6] {
        let map = unsafe { std::slice::from_raw_parts(self.map.as_ptr() as *const f64, self.map.len() / 8) };
        let off = (v * self.width + u) * 6;
        [
            map[off + 0],
            map[off + 1],
            map[off + 2],
            map[off + 3],
            map[off + 4],
            map[off + 5],
        ]
    }
}

pub struct PPMVolume {
    volume: VolumeGrid64x4Mapped,
    ppm: PPMFile,
}
impl PPMVolume {
    pub fn new(ppm_file: &str, width: usize, height: usize, base_volume: VolumeGrid64x4Mapped) -> Self {
        let ppm = PPMFile::new(ppm_file, width, height).unwrap();

        Self {
            volume: base_volume,
            ppm,
        }
    }
}
impl PaintVolume for PPMVolume {
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
        config: &DrawingConfig,
        buffer: &mut [u8],
    ) {
        let sfactor = _sfactor as usize;
        /* if plane_coord != 2 {
            return;
        } */

        let mut last_tile: [usize; 4] = [0; 4];
        let mut last_state: &TileState = &TileState::Missing;

        for im_v in 0..height {
            for im_u in 0..width {
                let im_rel_u = (im_u as i32 - width as i32 / 2) * paint_zoom as i32;
                let im_rel_v = (im_v as i32 - height as i32 / 2) * paint_zoom as i32;

                let mut uvw: [i32; 3] = [0; 3];
                uvw[u_coord] = xyz[u_coord] + im_rel_u;
                uvw[v_coord] = xyz[v_coord] + im_rel_v;
                uvw[plane_coord] = xyz[plane_coord];

                /* if u == 300 && v == 300 {
                    println!("u: {} v: {} gu: {} gv: {}", u, v, gu, gv);
                } */
                if uvw[0] <= 0
                    || uvw[0] >= self.ppm.width as i32
                    || uvw[1] <= 0
                    || uvw[1] >= self.ppm.height as i32
                    || uvw[2].abs() > 30
                {
                    continue;
                }

                let [x0, y0, z0, nx, ny, nz] = self.ppm.get(uvw[0] as usize, uvw[1] as usize);

                if x0 == 0.0 && y0 == 0.0 && z0 == 0.0 {
                    continue;
                }

                let x = x0 + uvw[2] as f64 * nx;
                let y = y0 + uvw[2] as f64 * ny;
                let z = z0 + uvw[2] as f64 * nz;

                let tile = [
                    x.round() as usize / 64 / sfactor,
                    y.round() as usize / 64 / sfactor,
                    z.round() as usize / 64 / sfactor,
                    sfactor,
                ];
                let state = if tile == last_tile {
                    last_state
                } else {
                    last_tile = tile;
                    last_state = self.volume.try_loading_tile(
                        tile[0],
                        tile[1],
                        tile[2],
                        Quality {
                            bit_mask: 0xff,
                            downsampling_factor: tile[3] as u8,
                        },
                    );
                    last_state
                };

                /* if u == 300 && v == 300 {
                    println!("u: {} v: {} gu: {} gv: {}", u, v, gu, gv);
                    println!("x: {} y: {} z: {}", x, y, z);
                    println!(
                        "nx: {} ny: {} nz: {} len: {}",
                        nx,
                        ny,
                        nz,
                        (nx * nx + ny * ny + nz * nz).sqrt()
                    );
                    println!("state: {:?}", state);
                } */

                if let TileState::Loaded(tile) = state {
                    let tile_x = (x.round() as usize / sfactor) % 64;
                    let tile_y = (y.round() as usize / sfactor) as usize % 64;
                    let tile_z = (z.round() as usize / sfactor) as usize % 64;

                    let xblock = tile_x / 4;
                    let yblock = tile_y / 4;
                    let zblock = tile_z / 4;

                    let xoff = tile_x % 4;
                    let yoff = tile_y % 4;
                    let zoff = tile_z % 4;

                    let block = zblock * 256 + yblock * 16 + xblock;
                    let block_off = zoff * 16 + yoff * 4 + xoff;
                    let off = block * 64 + block_off;

                    /* if u == 300 && v == 300 {
                        println!("tile_x: {} tile_y: {} tile_z: {}", tile_x, tile_y, tile_z);
                        println!("tile_x: {:06b} tile_y: {:06b} tile_z: {:06b}", tile_x, tile_y, tile_z);
                        println!("xblock: {} yblock: {} zblock: {}", xblock, yblock, zblock);
                        println!("xoff: {} yoff: {} zoff: {}", xoff, yoff, zoff);
                        println!("block: {} block_off: {} off: {}", block, block_off, off);
                        println!("block: {:b} block_off: {:b} off: {:b}", block, block_off, off);
                    } */

                    if off > 0 && off < tile.len() {
                        buffer[im_v * width + im_u] = tile[off];
                    }
                }
            }
        }
    }
}

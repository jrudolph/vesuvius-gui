use crate::downloader::*;
use crate::model::Quality;
use crate::volume::PaintVolume;

use std::collections::HashMap;
use std::ops::Deref;
use std::rc::{Rc, Weak};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use super::{DrawingConfig, Image, VoxelVolume};

#[derive(Debug)]
pub(crate) enum TileState {
    Unknown,
    Missing,
    Loaded(memmap::Mmap),
    Downloading(Arc<Mutex<DownloadState>>),
    TryLater(SystemTime),
}

pub struct VolumeGrid64x4Mapped {
    data_dir: String,
    downloader: Downloader,
    data: HashMap<(usize, usize, usize, usize), Rc<TileState>>,
    last_tile_key: (usize, usize, usize, usize),
    last_tile: Weak<TileState>,
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
    pub(crate) fn try_loading_tile(&mut self, x: usize, y: usize, z: usize, quality: Quality) -> Rc<TileState> {
        let key = (x, y, z, quality.downsampling_factor as usize);
        if !self.data.contains_key(&key) {
            self.data.insert(key, TileState::Unknown.into());
        }
        let tile_state = Rc::get_mut(self.data.get_mut(&key).expect("expected tile state data"))
            .expect("concurrent access to tile state");
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
        self.data.get(&key).unwrap().clone()
    }
    pub fn from_data_dir(data_dir: &str, downloader: Downloader) -> VolumeGrid64x4Mapped {
        if !std::path::Path::new(data_dir).exists() {
            panic!("Data directory {} does not exist", data_dir);
        }

        VolumeGrid64x4Mapped {
            data_dir: data_dir.to_string(),
            downloader,
            data: HashMap::new(),
            last_tile_key: (0, 0, 0, 0),
            last_tile: Weak::new(),
        }
    }
}

impl VoxelVolume for VolumeGrid64x4Mapped {
    fn get(&mut self, xyz: [f64; 3], downsampling: i32) -> u8 {
        let x = xyz[0] as usize;
        let y = xyz[1] as usize;
        let z = xyz[2] as usize;

        let tile_x = x / 64;
        let tile_y = y / 64;
        let tile_z = z / 64;

        let key = (tile_x, tile_y, tile_z, downsampling as usize);
        if key != self.last_tile_key {
            self.last_tile_key = key;
            self.last_tile = Rc::downgrade(&self.try_loading_tile(
                tile_x,
                tile_y,
                tile_z,
                Quality {
                    downsampling_factor: downsampling as u8,
                    bit_mask: 0xff,
                },
            ));
        }

        if let Some(r) = self.last_tile.upgrade() {
            if let TileState::Loaded(tile) = r.deref() {
                let tx = x & 63;
                let ty = y & 63;
                let tz = z & 63;

                let bx = tx / 4;
                let by = ty / 4;
                let bz = tz / 4;

                let block = bz * 256 + by * 16 + bx;

                let off_x = tx & 3;
                let off_y = ty & 3;
                let off_z = tz & 3;

                let index = off_x + off_y * 4 + off_z * 16 + block * 64;

                tile[index]
            } else if let TileState::Downloading(_state) = r.deref() {
                /* match *_state.lock().unwrap() {
                    DownloadState::Downloading => 255,
                    DownloadState::Queuing => 160,
                    DownloadState::Delayed => 100,
                    _ => 0,
                } */
                0
            } else {
                0
            }
        } else {
            0
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
        buffer: &mut Image,
    ) {
        // drop last_tile, which we do not use for area painting and may get in the way of accessing tilestate otherwise
        self.last_tile_key = (0, 0, 0, 0);
        self.last_tile = Weak::new();

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

                if let TileState::Loaded(tile) = state.deref() {
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

                                        let value = if u == center_u || v == center_v {
                                            0
                                        } else if filters_active {
                                            let pluscon = ((value as i32 - config.threshold_min as i32).max(0) * 255
                                                / (255 - (config.threshold_min + config.threshold_max) as i32))
                                                .min(255)
                                                as u8;

                                            (((pluscon & mask) as f32) / (mask as f32) * 255.0) as u8
                                        } else {
                                            value
                                        };
                                        buffer.set_gray(u as usize, v as usize, value);
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

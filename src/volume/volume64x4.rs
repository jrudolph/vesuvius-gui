use crate::downloader::*;
use crate::model::Quality;
use crate::volume::PaintVolume;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

enum TileState {
    Unknown,
    Missing,
    //Exists,
    Loaded(memmap::Mmap),
    Downloading(Arc<Mutex<DownloadState>>),
    TryLater(SystemTime)
}

pub struct VolumeGrid64x4Mapped {
    data_dir: String,
    max_x: usize,
    max_y: usize,
    max_z: usize,
    downloader: Downloader,
    data: HashMap<(usize, usize, usize, usize), TileState>,
}
impl VolumeGrid64x4Mapped {
    fn map_for(data_dir: &str, x: usize, y: usize, z: usize, quality: Quality) -> Option<TileState> {
        use memmap::MmapOptions;
        use std::fs::File;
        let file_name = format!("{}/64-4/z{:03}/xyz-{:03}-{:03}-{:03}-b{:03}-d{:02}.bin", data_dir, z, x, y, z, quality.bit_mask, quality.downsampling_factor);
        //println!("at {}", file_name);

        let file = File::open(file_name.clone()).ok()?;
        
        let map = unsafe { MmapOptions::new().map(&file) }.ok();
        map
            .filter(|m| {
                if m.len() == 64*64*64 {
                    true
                } else {
                    println!("file {} has wrong size {}", file_name, m.len());
                    false
                }
            })
            .map(|x| TileState::Loaded(x))
    }
    fn get_tile_state(&self, x: usize, y: usize, z: usize, downsampling: u8) -> &TileState {
        let key = (x,y,z,downsampling as usize);
        self.data.get(&key).unwrap_or(&TileState::Unknown)
    }
    fn try_loading_tile(&mut self, x: usize, y: usize, z: usize, quality: Quality) {
        let key = (x,y,z,quality.downsampling_factor as usize);
        if !self.data.contains_key(&key) {
            self.data.insert(key, TileState::Unknown);
        }
        let tile_state = self.data.get_mut(&key).unwrap();
        match tile_state {
            TileState::Unknown => {
                if let Some(state) = Self::map_for(&self.data_dir, x, y, z, quality) {
                    *tile_state = state;
                } else {
                    let state = Arc::new(Mutex::new(DownloadState::Queuing));
                    self.downloader.queue((state.clone(), x, y, z, quality));
                    *tile_state = TileState::Downloading(state);
                } 
            },
            TileState::TryLater(at) => {
                if at.elapsed().unwrap() > Duration::from_secs(10) {
                    println!("resetting tile {}/{}/{} q{} again", x, y, z, quality.downsampling_factor);
                    *tile_state = TileState::Unknown; // reset
                    self.try_loading_tile(x, y, z, quality);
                }
            },
            TileState::Downloading(state) => {
                match *state.clone().lock().unwrap() {
                    DownloadState::Done => {
                        if let Some(state) = Self::map_for(&self.data_dir, x, y, z, quality) {
                            *tile_state = state;
                        } else {
                            // set to missing permanently
                            println!("failed to load tile from map {}/{}/{} q{}", x, y, z, quality.downsampling_factor);
                            *tile_state = TileState::Missing;
                        }
                    }
                    DownloadState::Delayed => {
                        *tile_state = TileState::TryLater(SystemTime::now());
                    }
                    DownloadState::Failed => {
                        *tile_state = TileState::Missing;
                    },
                    _ => {
                        *tile_state = TileState::Unknown;
                    }                
                };
            },
            _ => {}
        }
    }
    pub fn from_data_dir(data_dir: &str, max_x: usize, max_y: usize, max_z: usize, downloader: Downloader) -> VolumeGrid64x4Mapped {
        if !std::path::Path::new(data_dir).exists() {
            panic!("Data directory {} does not exist", data_dir);
        }

        VolumeGrid64x4Mapped {
            data_dir: data_dir.to_string(),
            max_x: max_x,
            max_y: max_y,
            max_z: max_z,
            downloader,
            data: HashMap::new(),
        }
    }
}
impl PaintVolume for VolumeGrid64x4Mapped {
    fn paint(&mut self, xyz: [i32; 3], u_coord: usize, v_coord: usize, plane_coord: usize, width: usize, height: usize, _sfactor: u8, buffer: &mut [u8]) {
        /* if plane_coord != 2 {
            return;
        } */
        self.downloader.position(xyz[0], xyz[1], xyz[2], width, height);

        let center_u = width as i32 / 2;
        let center_v = height as i32 / 2;

        let sfactor = _sfactor as i32; //Quality::Full.downsampling_factor as i32;
        let tilesize = 64 * sfactor;
        let blocksize = 4 * sfactor;

        let min_uc = (xyz[u_coord] - width as i32 / 2).max(0);
        let max_uc = xyz[u_coord] + width as i32 / 2;
        let min_vc = (xyz[v_coord] - height as i32 / 2).max(0);
        let max_vc = xyz[v_coord] + height as i32 / 2;
        let pc = xyz[plane_coord].max(0);

        let tile_min_uc = min_uc /tilesize;
        let uc = max_uc / tilesize;

        let tile_min_vc = min_vc / tilesize;
        let tile_max_vc = max_vc / tilesize;

        let tile_pc = pc / tilesize;
        let tile_pc_off = pc % tilesize;
        let block_pc = tile_pc_off / blocksize;
        let block_pc_off = tile_pc_off % blocksize;

        //println!("x: {} y: {} z: {}", xyz[0], xyz[1], xyz[2]);
        //println!("min_x: {} max_x: {} min_y: {} max_y: {} z: {}", min_x, max_x, min_y, max_y, z);
        //println!("tile_min_x: {} tile_max_x: {} tile_min_y: {} tile_max_y: {} tile_z: {} block_z: {} block_z_off: {}", tile_min_x, tile_max_x, tile_min_y, tile_max_y, tile_z, block_z, block_z_off);

        // iterate over all tiles
        for tile_uc in tile_min_uc..=uc {
            for tile_vc in tile_min_vc..=tile_max_vc {
                let mut tile_i = [0; 3];
                tile_i[u_coord] = tile_uc as usize;
                tile_i[v_coord] = tile_vc as usize;
                tile_i[plane_coord] = tile_pc as usize;

                if tile_i[0] >= self.max_x || tile_i[1] >= self.max_y || tile_i[2] >= self.max_z {
                    continue;
                }

                //println!("tile_x: {} tile_y: {}", tile_x, tile_y);
                if let TileState::Downloading(finished) = &self.get_tile_state(tile_i[0], tile_i[1], tile_i[2], sfactor as u8) {
                    if finished.lock().unwrap().needs_reload() {
                        self.try_loading_tile(tile_i[0], tile_i[1], tile_i[2], Quality { bit_mask: 0xff, downsampling_factor: sfactor as u8 });
                    }
                }
                if let TileState::TryLater(_) = &self.get_tile_state(tile_i[0], tile_i[1], tile_i[2], sfactor as u8) {
                    self.try_loading_tile(tile_i[0], tile_i[1], tile_i[2], Quality { bit_mask: 0xff, downsampling_factor: sfactor as u8 });
                }
                if let TileState::Unknown = &self.get_tile_state(tile_i[0], tile_i[1], tile_i[2], sfactor as u8) {
                    self.try_loading_tile(tile_i[0], tile_i[1], tile_i[2], Quality { bit_mask: 0xff, downsampling_factor: sfactor as u8 });
                } 
                /* match self.get_tile_state(tile_i[0], tile_i[1], tile_i[2], sfactor as u8) {
                    TileState::Missing => {},
                    TileState::Loaded(_) => {},
                    _ => {
                        self.try_loading_tile(tile_i[0], tile_i[1], tile_i[2], Quality { bit_mask: 0xff, downsampling_factor: sfactor as u8 });
                    }
                } */
                //self.try_loading_tile(tile_i[0], tile_i[1], tile_i[2], Quality { bit_mask: 0xff, downsampling_factor: sfactor as u8 });
                
                if let TileState::Loaded(tile) = self.get_tile_state(tile_i[0], tile_i[1], tile_i[2], sfactor as u8) {
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
                            for vc in 0..blocksize {
                                for uc in 0..blocksize {
                                    let u = (tile_uc * tilesize + block_uc * blocksize + uc) as i32 - min_uc;
                                    let v = (tile_vc * tilesize + block_vc * blocksize + vc) as i32 - min_vc;
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

                                    if u >= 0 && u < width as i32 && v >= 0 && v < height as i32 {
                                        let off = boff * 64 + offs_i[2] * 16 + offs_i[1] * 4 + offs_i[0];
                                        if off > tile.len() {
                                            panic!("off: {} tile.len(): {}", off, tile.len());
                                        }
                                        let mut value = tile[off as usize];

                                        //let pluscon = ((value as i32 - 70).max(0) * 255 / (255 - 100)).min(255) as u8;
                                        
                                        /* if (u / 128) % 2 == 0 */ {
                                        if u == center_u || v == center_v {
                                            value = value / 10;
                                        }

                                        buffer[v as usize * width + u as usize] = value;// & 0xf0;
                                        } /* else {
                                            buffer[v as usize * width + u as usize] = (value & 0xf0) + 0x08;//(tile[((off / 8) * 8) as usize] & 0xc0) + 0x20;
                                        } */
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
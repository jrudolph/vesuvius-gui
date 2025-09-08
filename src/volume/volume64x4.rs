use super::{DrawingConfig, Image, VoxelVolume};
use crate::downloader::*;
use crate::model::Quality;
use crate::volume::{PaintVolume, VoxelPaintVolume};
use dashmap::DashMap;
use libm::modf;
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

#[derive(Debug)]
pub(crate) enum TileState {
    Missing,
    Loaded(memmap::Mmap),
    Downloading(Arc<Mutex<DownloadState>>),
    TryLater(SystemTime),
}

struct TileStateEntry {
    state: Arc<TileState>,
    last_access: AtomicU64,
}

struct TileCache {
    cache: DashMap<(usize, usize, usize, usize), Option<TileStateEntry>>,
    access_counter: AtomicU64,
}

struct LocalState {
    last_tile_key: (usize, usize, usize, usize),
    last_tile: Option<Arc<TileState>>,
}
impl TileCache {
    fn new() -> Self {
        Self {
            cache: DashMap::new(),
            access_counter: AtomicU64::new(0),
        }
    }

    fn try_loading_tile(
        &self,
        volume: &VolumeGrid64x4Mapped,
        x: usize,
        y: usize,
        z: usize,
        quality: Quality,
    ) -> Option<Arc<TileState>> {
        let key = (x, y, z, quality.downsampling_factor as usize);
        let counter = self.access_counter.fetch_add(1, Ordering::Relaxed);

        let mut entry = self.cache.entry(key).or_insert_with(|| {
            // Try to load from disk first
            if let Some(state) = VolumeGrid64x4Mapped::map_for(&volume.data_dir, x, y, z, quality) {
                Some(TileStateEntry {
                    state: Arc::new(state),
                    last_access: AtomicU64::new(counter),
                })
            } else {
                // Queue for download
                let download_state = Arc::new(Mutex::new(DownloadState::Queuing));
                volume.downloader.queue((download_state.clone(), x, y, z, quality));
                Some(TileStateEntry {
                    state: Arc::new(TileState::Downloading(download_state)),
                    last_access: AtomicU64::new(counter),
                })
            }
        });

        if let Some(tile_entry) = entry.value_mut() {
            tile_entry.last_access.store(counter, Ordering::Relaxed);

            // Check if we need to update the state
            let current_state = tile_entry.state.clone();
            let new_state = match current_state.as_ref() {
                TileState::Downloading(download_state) => {
                    match *download_state.lock().unwrap() {
                        DownloadState::Done => {
                            if let Some(new_state) = VolumeGrid64x4Mapped::map_for(&volume.data_dir, x, y, z, quality) {
                                Some(Arc::new(new_state))
                            } else {
                                println!(
                                    "failed to load tile from map {}/{}/{} q{}",
                                    x, y, z, quality.downsampling_factor
                                );
                                Some(Arc::new(TileState::Missing))
                            }
                        }
                        DownloadState::Delayed => Some(Arc::new(TileState::TryLater(SystemTime::now()))),
                        DownloadState::Failed => Some(Arc::new(TileState::Missing)),
                        DownloadState::Pruned => {
                            // Reset to try loading again
                            if let Some(new_state) = VolumeGrid64x4Mapped::map_for(&volume.data_dir, x, y, z, quality) {
                                Some(Arc::new(new_state))
                            } else {
                                let download_state = Arc::new(Mutex::new(DownloadState::Queuing));
                                volume.downloader.queue((download_state.clone(), x, y, z, quality));
                                Some(Arc::new(TileState::Downloading(download_state)))
                            }
                        }
                        DownloadState::Queuing | DownloadState::Downloading => None,
                    }
                }
                TileState::TryLater(at) => {
                    if at.elapsed().unwrap() > Duration::from_secs(10) {
                        println!(
                            "resetting tile {}/{}/{} q{} again",
                            x, y, z, quality.downsampling_factor
                        );
                        // Reset to try loading again
                        if let Some(new_state) = VolumeGrid64x4Mapped::map_for(&volume.data_dir, x, y, z, quality) {
                            Some(Arc::new(new_state))
                        } else {
                            let download_state = Arc::new(Mutex::new(DownloadState::Queuing));
                            volume.downloader.queue((download_state.clone(), x, y, z, quality));
                            Some(Arc::new(TileState::Downloading(download_state)))
                        }
                    } else {
                        None
                    }
                }
                _ => None,
            };

            if let Some(new_state) = new_state {
                tile_entry.state = new_state;
            }

            Some(tile_entry.state.clone())
        } else {
            None
        }
    }
}

pub struct VolumeGrid64x4Mapped {
    data_dir: String,
    downloader: Arc<dyn Downloader>,
    tile_cache: Arc<TileCache>,
    local_state: RefCell<LocalState>,
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

    pub(crate) fn try_loading_tile(&self, x: usize, y: usize, z: usize, quality: Quality) -> Option<Arc<TileState>> {
        self.tile_cache.try_loading_tile(&self, x, y, z, quality)
    }

    pub fn from_data_dir(data_dir: &str, downloader: Arc<dyn Downloader>) -> VolumeGrid64x4Mapped {
        if !std::path::Path::new(data_dir).exists() {
            panic!("Data directory {} does not exist", data_dir);
        }

        VolumeGrid64x4Mapped {
            data_dir: data_dir.to_string(),
            downloader,
            tile_cache: Arc::new(TileCache::new()),
            local_state: RefCell::new(LocalState {
                last_tile_key: (0, 0, 0, 0),
                last_tile: None,
            }),
        }
    }
    fn tile_at(&self, x: usize, y: usize, z: usize, downsampling: usize) -> Option<Arc<TileState>> {
        let tile_x = x / 64;
        let tile_y = y / 64;
        let tile_z = z / 64;
        let key = (tile_x, tile_y, tile_z, downsampling as usize);

        // Check local cache first
        {
            let state = self.local_state.borrow();
            if key == state.last_tile_key {
                return state.last_tile.clone();
            }
        }

        // Load from shared cache
        let tile = self.try_loading_tile(
            tile_x,
            tile_y,
            tile_z,
            Quality {
                downsampling_factor: downsampling as u8,
                bit_mask: 0xff,
            },
        );

        // Update local cache
        {
            let mut state = self.local_state.borrow_mut();
            state.last_tile_key = key;
            state.last_tile = tile.clone();
        }

        tile
    }
    fn drop_last_cached(&self) {
        let mut state = self.local_state.borrow_mut();
        state.last_tile_key = (0, 0, 0, 0);
        state.last_tile = None;
    }
}

impl VoxelVolume for VolumeGrid64x4Mapped {
    fn get(&self, xyz: [f64; 3], downsampling: i32) -> u8 {
        let x = xyz[0] as usize;
        let y = xyz[1] as usize;
        let z = xyz[2] as usize;

        if let Some(r) = self.tile_at(x, y, z, downsampling as usize) {
            if let TileState::Loaded(tile) = r.as_ref() {
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
            } else if let TileState::Downloading(_state) = r.as_ref() {
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

    fn get_interpolated(&self, xyz: [f64; 3], downsampling: i32) -> u8 {
        let (dx, x0) = modf(xyz[0]);
        let x1 = x0 + 1.0;
        let (dy, y0) = modf(xyz[1]);
        let y1 = y0 + 1.0;
        let (dz, z0) = modf(xyz[2]);
        let z1 = z0 + 1.0;

        let fast_path = x0 as usize & 63 != 63 && y0 as usize & 63 != 63 && z0 as usize & 63 != 63;

        let (p000, p100, p010, p110, p001, p101, p011, p111) = if fast_path {
            let x = x0 as usize;
            let y = y0 as usize;
            let z = z0 as usize;

            if let Some(r) = self.tile_at(x, y, z, downsampling as usize) {
                if let TileState::Loaded(tile) = r.as_ref() {
                    let tx = x & 63;
                    let ty = y & 63;
                    let tz = z & 63;

                    let bx0 = tx / 4;
                    let bx1 = (tx + 1) / 4;
                    let by0 = ty / 4;
                    let by1 = (ty + 1) / 4;
                    let bz0 = tz / 4;
                    let bz1 = (tz + 1) / 4;

                    let block000 = bz0 * 256 + by0 * 16 + bx0;
                    let block100 = bz0 * 256 + by0 * 16 + bx1;
                    let block010 = bz0 * 256 + by1 * 16 + bx0;
                    let block110 = bz0 * 256 + by1 * 16 + bx1;
                    let block001 = bz1 * 256 + by0 * 16 + bx0;
                    let block101 = bz1 * 256 + by0 * 16 + bx1;
                    let block011 = bz1 * 256 + by1 * 16 + bx0;
                    let block111 = bz1 * 256 + by1 * 16 + bx1;

                    let off_x0 = tx & 3;
                    let off_x1 = (tx + 1) & 3;
                    let off_y0 = ty & 3;
                    let off_y1 = (ty + 1) & 3;
                    let off_z0 = tz & 3;
                    let off_z1 = (tz + 1) & 3;

                    let index000 = off_x0 + off_y0 * 4 + off_z0 * 16 + block000 * 64;
                    let index100 = off_x1 + off_y0 * 4 + off_z0 * 16 + block100 * 64;
                    let index010 = off_x0 + off_y1 * 4 + off_z0 * 16 + block010 * 64;
                    let index110 = off_x1 + off_y1 * 4 + off_z0 * 16 + block110 * 64;
                    let index001 = off_x0 + off_y0 * 4 + off_z1 * 16 + block001 * 64;
                    let index101 = off_x1 + off_y0 * 4 + off_z1 * 16 + block101 * 64;
                    let index011 = off_x0 + off_y1 * 4 + off_z1 * 16 + block011 * 64;
                    let index111 = off_x1 + off_y1 * 4 + off_z1 * 16 + block111 * 64;

                    /*
                    // we could avoid multiple bounds checks here, but I cannot measure a difference
                    unsafe {
                        (
                            *tile.get_unchecked(index000) as f64,
                            *tile.get_unchecked(index100) as f64,
                            *tile.get_unchecked(index010) as f64,
                            *tile.get_unchecked(index110) as f64,
                            *tile.get_unchecked(index001) as f64,
                            *tile.get_unchecked(index101) as f64,
                            *tile.get_unchecked(index011) as f64,
                            *tile.get_unchecked(index111) as f64,
                        )
                    }
                    */
                    (
                        tile[index000] as f64,
                        tile[index100] as f64,
                        tile[index010] as f64,
                        tile[index110] as f64,
                        tile[index001] as f64,
                        tile[index101] as f64,
                        tile[index011] as f64,
                        tile[index111] as f64,
                    )
                } else if let TileState::Downloading(_state) = r.as_ref() {
                    /* match *_state.lock().unwrap() {
                        DownloadState::Downloading => 255,
                        DownloadState::Queuing => 160,
                        DownloadState::Delayed => 100,
                        _ => 0,
                    } */
                    (0., 0., 0., 0., 0., 0., 0., 0.)
                } else {
                    (0., 0., 0., 0., 0., 0., 0., 0.)
                }
            } else {
                (0., 0., 0., 0., 0., 0., 0., 0.)
            }
        } else {
            let p000 = self.get([x0, y0, z0], downsampling) as f64;
            let p100 = self.get([x1, y0, z0], downsampling) as f64;
            let p010 = self.get([x0, y1, z0], downsampling) as f64;
            let p110 = self.get([x1, y1, z0], downsampling) as f64;
            let p001 = self.get([x0, y0, z1], downsampling) as f64;
            let p101 = self.get([x1, y0, z1], downsampling) as f64;
            let p011 = self.get([x0, y1, z1], downsampling) as f64;
            let p111 = self.get([x1, y1, z1], downsampling) as f64;
            (p000, p100, p010, p110, p001, p101, p011, p111)
        };

        let c00 = p000 * (1.0 - dx) + p100 * dx;
        let c10 = p010 * (1.0 - dx) + p110 * dx;
        let c01 = p001 * (1.0 - dx) + p101 * dx;
        let c11 = p011 * (1.0 - dx) + p111 * dx;

        let c0 = c00 * (1.0 - dy) + c10 * dy;
        let c1 = c01 * (1.0 - dy) + c11 * dy;

        let c = c0 * (1.0 - dz) + c1 * dz;

        c as u8
    }
}

impl PaintVolume for VolumeGrid64x4Mapped {
    fn paint(
        &self,
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
        self.drop_last_cached();

        let width = paint_zoom as usize * canvas_width;
        let height = paint_zoom as usize * canvas_height;

        let mask = config.bit_mask();
        let filters_active = config.filters_active();

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

                if let Some(state) = self.try_loading_tile(
                    tile_i[0],
                    tile_i[1],
                    tile_i[2],
                    Quality {
                        bit_mask: 0xff,
                        downsampling_factor: sfactor as u8,
                    },
                ) {
                    if let TileState::Loaded(tile) = state.as_ref() {
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

                                            if filters_active {
                                                let pluscon = ((value as i32 - config.threshold_min as i32).max(0)
                                                    * 255
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
    fn shared(&self) -> super::VolumeCons {
        let data_dir = self.data_dir.clone();
        let downloader = self.downloader.clone();
        let tile_cache = self.tile_cache.clone();
        Box::new(move || {
            VolumeGrid64x4Mapped {
                data_dir,
                downloader,
                tile_cache,
                local_state: RefCell::new(LocalState {
                    last_tile_key: (0, 0, 0, 0),
                    last_tile: None,
                }),
            }
            .into_volume()
        })
    }
}

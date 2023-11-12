use std::sync::{Arc, Mutex};

use egui::{ColorImage, CursorIcon, Image, PointerButton, Response, Ui};

trait World {
    fn get(&mut self, xyz: [i32; 3]) -> u8;
    fn paint(&mut self, xyz: [i32; 3], u_coord: usize, v_coord: usize, plane_coord: usize, width: usize, height: usize, buffer: &mut [u8]);
}

enum TileState {
    Unknown,
    Missing,
    Exists,
    Loaded(memmap::Mmap),
    Downloading(Arc<Mutex<bool>>),
}

struct VolumeGrid16x16x16Mapped {
    data_dir: String,
    max_x: usize,
    max_y: usize,
    max_z: usize,
    data: Vec<Vec<Vec<TileState>>>,
}
impl VolumeGrid16x16x16Mapped {
    fn map_for(data_dir: &str, x: usize, y: usize, z: usize) -> Option<memmap::Mmap> {
        use memmap::MmapOptions;
        use std::fs::File;
        let file_name = format!("{}/z{:03}/xyz-{:03}-{:03}-{:03}.bin", data_dir, z, x, y, z);
        //println!("at {}", file_name);

        let file = File::open(file_name).ok()?;
        unsafe { MmapOptions::new().map(&file) }.ok()
    }
    fn try_loading_tile(&mut self, x: usize, y: usize, z: usize) {
        let tile_state = &mut self.data[z][y][x];
        if let TileState::Unknown = tile_state {
            if let Some(mmap) = Self::map_for(&self.data_dir, x, y, z) {
                *tile_state = TileState::Loaded(mmap);
            } else {
                let finished = Arc::new(Mutex::new(false));
                let url = format!("https://vesuvius.virtual-void.net/tiles/scroll/332/volume/20231027191953/download/128-16?x={}&y={}&z={}", x, y, z);
                let mut request = ehttp::Request::get(url);
                request.headers.insert("Authorization".to_string(), "Basic blip".to_string());
                
                let inner = finished.clone();
                let dir = self.data_dir.clone();
                println!("downloading tile {}/{}/{}", x, y, z);
                ehttp::fetch(request, move |response| {
                    if let Ok(res) = response {
                        if res.status == 200 {
                            println!("got tile {}/{}/{}", x, y, z);
                            let bytes = res.bytes;
                            // save bytes to file
                            let file_name = format!("{}/z{:03}/xyz-{:03}-{:03}-{:03}.bin", dir, z, x, y, z);
                            std::fs::create_dir_all(format!("{}/z{:03}", dir, z)).unwrap();
                            std::fs::write(file_name, bytes).unwrap();
                        } else {
                            println!("failed to download tile {}/{}/{}: {}", x, y, z, res.status);
                            *inner.lock().unwrap() = true;
                        }
                    }       

                    *inner.lock().unwrap() = true;        
                });
                *tile_state = TileState::Downloading(finished);
            } 
        } else if let TileState::Downloading(finished) = tile_state {
            if *finished.lock().unwrap() {
                if let Some(mmap) = Self::map_for(&self.data_dir, x, y, z) {
                    *tile_state = TileState::Loaded(mmap);
                } else {
                    // set to missing permanently
                    *tile_state = TileState::Missing;
                }
            }
        }
    }
    pub fn from_data_dir(data_dir: &str, max_x: usize, max_y: usize, max_z: usize) -> VolumeGrid16x16x16Mapped {
        // find highest xyz values for files in data_dir named like this format: format!("{}/cell_yxz_{:03}_{:03}_{:03}.tif", data_dir, y, x, z);
        // use regex to match file names
        /* let mut max_x = 0;
        let mut max_y = 0;
        let mut max_z = 0;
        for entry in std::fs::read_dir(data_dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            let file_name = path.file_name().unwrap().to_str().unwrap();
            //home/johannes/git/self/_2023/vesuvius-browser/data/blocks/scroll1/20230205180739/128-16/z041/xyz-010-010-041.bin
            if let Some(captures) = regex::Regex::new(r"xyz-(\d+)-(\d+)-(\d+).bin")
                .unwrap()
                .captures(file_name)
            {
                println!("Found file: {}", file_name);
                let x = captures.get(1).unwrap().as_str().parse::<usize>().unwrap();
                let y = captures.get(2).unwrap().as_str().parse::<usize>().unwrap();
                let z = captures.get(3).unwrap().as_str().parse::<usize>().unwrap();
                if x > max_x {
                    max_x = x;
                }
                if y > max_y {
                    max_y = y;
                }
                if z > max_z {
                    max_z = z;
                }
            }
        } */
        if !std::path::Path::new(data_dir).exists() {
            panic!("Data directory {} does not exist", data_dir);
        }
        // map_for(data_dir, x, y, z).map_or(TileState::Missing, |x| TileState::Loaded(x))
        let data: Vec<Vec<Vec<TileState>>> = (0..=max_z)
            .map(|z| {
                (0..=max_y)
                    .map(|y| (0..=max_x)
                    .map(|x| TileState::Unknown).collect())
                    .collect()
            })
            .collect();

        /* // count number of slices found
        let slices_found = data.iter().flatten().flatten().flat_map(|x| {
            if let TileState::Loaded(_) = x {
                Some(())
            } else {
                None
            }
        }).count();
        println!("Found {} cells in {}", slices_found, data_dir);
        println!("max_x: {}, max_y: {}, max_z: {}", max_x, max_y, max_z); */

        VolumeGrid16x16x16Mapped {
            data_dir: data_dir.to_string(),
            max_x: max_x,
            max_y: max_y,
            max_z: max_z,
            data,
        }
    }
}
impl World for VolumeGrid16x16x16Mapped {
    fn get(&mut self, xyz: [i32; 3]) -> u8 {
        let x_tile = xyz[0] as usize >> 7;
        let y_tile = xyz[1] as usize >> 7;
        let z_tile = xyz[2] as usize >> 7;

        /* if xyz[0] % 100 == 0 && xyz[1] % 100 == 0 && xyz[2] % 100 == 0 {
            println!("x: {}, y: {}, z: {} x_tile: {} y_tile: {} z_tile: {}", xyz[0], xyz[1], xyz[2], x_tile, y_tile, z_tile);
        } */

        if xyz[0] < 0 || xyz[1] < 0 || xyz[2] < 0 || x_tile > self.max_x || y_tile > self.max_y || z_tile > self.max_z {
            /* if xyz[0] % 100 == 0 && xyz[1] % 100 == 0 && xyz[2] % 100 == 0 { 
                println!("out of bounds: {:?} x_tile: {} y_tile: {} z_tile: {} max_x: {}, max_y: {}, max_z: {}", xyz, x_tile, y_tile, z_tile, self.max_x, self.max_y, self.max_z);
            } */
                
            0
        } else {
            if let TileState::Downloading(finished) = &self.data[z_tile][y_tile][x_tile] {
                if *finished.lock().unwrap() {
                    self.try_loading_tile(x_tile, y_tile, z_tile);
                }
            }
            if let TileState::Unknown = &self.data[z_tile][y_tile][x_tile] {
                self.try_loading_tile(x_tile, y_tile, z_tile);
            }
            
            if let TileState::Loaded(tile) = &self.data[z_tile][y_tile][x_tile] {
                //println!("Found tile: {} {} {}", x_tile, y_tile, z_tile);
                let tx = (xyz[0] & 0x7f) as usize;
                let ty = (xyz[1] & 0x7f) as usize;
                let tz = (xyz[2] & 0x7f) as usize;

                let bx = tx >> 4;
                let by = ty >> 4;
                let bz = tz >> 4;

                let boff = (bz << 6) + (by << 3) + bx;
                //println!("bx: {}, by: {}, bz: {}, boff: {}", bx, by, bz, boff);

                let px = tx & 0xf;
                let py = ty & 0xf;
                let pz = tz & 0xf;

                let poff = pz * 256 + py * 16 + px;

                let off = boff * 4096 + poff;

                /* if xyz[0] % 100 == 0 && xyz[1] % 100 == 0 && xyz[2] % 100 == 0 {
                    println!("x: {} y: {} z: {} tile: {} {} {}  boff: {} poff: {} off: {}", xyz[0], xyz[1], xyz[2], x_tile, y_tile, z_tile, boff, poff, off);
                } */

                if off < tile.len() {
                    tile[off]
                } else {
                    println!("Buffer too small");
                    println!("x: {} y: {} z: {} tile: {} {} {}  boff: {} poff: {} off: {}", xyz[0], xyz[1], xyz[2], x_tile, y_tile, z_tile, boff, poff, off);
                    0
                }
            } else {
                /* if xyz[0] % 100 == 0 && xyz[1] % 100 == 0 && xyz[2] % 100 == 0 {
                    println!("tile not found: {:?} x_tile: {} y_tile: {} z_tile: {} max_x: {}, max_y: {}, max_z: {}", xyz, x_tile, y_tile, z_tile, self.max_x, self.max_y, self.max_z);
                } */

                0
            }
        }
    }
    fn paint(&mut self, xyz: [i32; 3], u_coord: usize, v_coord: usize, plane_coord: usize, width: usize, height: usize, buffer: &mut [u8]) {

    }
}

type V500 = [u8; 512*512*512];

struct VolumeGrid4x4x4 {
    max_x: usize,
    max_y: usize,
    max_z: usize,
    data: Vec<Vec<Vec<Option<Box<V500>>>>>,
}
impl World for VolumeGrid4x4x4 {
    fn get(&mut self, xyz: [i32; 3]) -> u8 {
        let x_tile = xyz[0] as usize / 500;
        let y_tile = xyz[1] as usize / 500;
        let z_tile = xyz[2] as usize / 500;

        if xyz[0] < 0 || xyz[1] < 0 || xyz[2] < 0 || x_tile > self.max_x || y_tile > self.max_y || z_tile > self.max_z {
            //println!("out of bounds: {:?} x_tile: {} y_tile: {} z_tile: {} max_x: {}, max_y: {}, max_z: {}", xyz, x_tile, y_tile, z_tile, self.max_x, self.max_y, self.max_z);
            0
        } else if let Some(tile) = &self.data[z_tile][y_tile][x_tile] {
            let tx = (xyz[0] % 500) as usize;
            let ty = (xyz[1] % 500) as usize;
            let tz = (xyz[2] % 500) as usize;

            let bx = tx >> 4;
            let by = ty >> 4;
            let bz = tz >> 4;

            let boff = (bz << 10) + (by << 5) + bx;
            //println!("bx: {}, by: {}, bz: {}, boff: {}", bx, by, bz, boff);

            let px = tx & 0xf;
            let py = ty & 0xf;
            let pz = tz & 0xf;

            let poff = pz * 256 + py * 16 + px;

            let off = boff * 4096 + poff;

            if off < tile.len() {
                tile[off]
            } else {
                0
            }
        } else {
            0
        }
    }
    fn paint(&mut self, xyz: [i32; 3], u_coord: usize, v_coord: usize, plane_coord: usize, width: usize, height: usize, buffer: &mut [u8]) {
        if plane_coord != 2 {
            return;
        }
        let min_x = xyz[0] - width as i32 / 2;
        let max_x = xyz[0] + width as i32 / 2;
        let min_y = xyz[1] - height as i32 / 2;
        let max_y = xyz[1] + height as i32 / 2;
        let z = xyz[2];

        let tile_min_x = min_x / 500;
        let tile_max_x = max_x / 500;

        let tile_min_y = min_y / 500;
        let tile_max_y = max_y / 500;

        let tile_z = z / 500;
        let tile_z_off = z % 500;
        let block_z = tile_z_off / 16;
        let block_z_off = tile_z_off % 16;

        //println!("x: {} y: {} z: {}", xyz[0], xyz[1], xyz[2]);
        //println!("min_x: {} max_x: {} min_y: {} max_y: {} z: {}", min_x, max_x, min_y, max_y, z);
        //println!("tile_min_x: {} tile_max_x: {} tile_min_y: {} tile_max_y: {} tile_z: {} block_z: {} block_z_off: {}", tile_min_x, tile_max_x, tile_min_y, tile_max_y, tile_z, block_z, block_z_off);

        // iterate over all tiles
        for tile_x in tile_min_x..=tile_max_x {
            for tile_y in tile_min_y..=tile_max_y {
                //println!("tile_x: {} tile_y: {}", tile_x, tile_y);
                
                if let Some(tile) = &mut self.data[tile_z as usize][tile_y as usize][tile_x as usize] {
                    // iterate over blocks in tile
                    let min_tile_x = (tile_x * 500).max(min_x) - tile_x * 500;
                    let max_tile_x = (tile_x * 500 + 500).min(max_x) - tile_x * 500;
                    let min_tile_y = (tile_y * 500).max(min_y) - tile_y * 500;
                    let max_tile_y = (tile_y * 500 + 500).min(max_y) - tile_y * 500;

                    let min_block_x = min_tile_x / 16;
                    let max_block_x = max_tile_x / 16;
                    let min_block_y = min_tile_y / 16;
                    let max_block_y = max_tile_y / 16;

                    //println!("min_tile_x: {} max_tile_x: {} min_tile_y: {} max_tile_y: {}", min_tile_x, max_tile_x, min_tile_y, max_tile_y);
                    //println!("min_block_x: {} max_block_x: {} min_block_y: {} max_block_y: {}", min_block_x, max_block_x, min_block_y, max_block_y);

                    for block_y in min_block_y..=max_block_y {
                        for block_x in min_block_x..=max_block_x {
                            let boff = (block_z << 10) + (block_y << 5) + block_x;
                            
                            // iterate over pixels in block
                            for y in 0..16 {
                                for x in 0..16 {
                                    let u = (tile_x * 500 + block_x * 16 + x) as i32 - min_x;
                                    let v = (tile_y * 500 + block_y * 16 + y) as i32 - min_y;
                                    if x == 0 && y == 0 {
                                        //println!("block_x: {} block_y: {}", block_x, block_y);
                                        //println!("u: {} v: {}", u, v);
                                    }

                                    if u >= 0 && u < width as i32 && v >= 0 && v < height as i32 {
                                        let off = boff * 4096 + block_z_off * 256 + y * 16 + x;
                                        buffer[v as usize * width + u as usize] = tile[off as usize];
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

struct MappedVolumeGrid {
    max_x: usize,
    max_y: usize,
    max_z: usize,
    data: Vec<Vec<Vec<Option<memmap::Mmap>>>>,
}
impl MappedVolumeGrid {
    pub fn from_data_dir(data_dir: &str) -> MappedVolumeGrid {
        use memmap::MmapOptions;
        use std::fs::File;

        // find highest xyz values for files in data_dir named like this format: format!("{}/cell_yxz_{:03}_{:03}_{:03}.tif", data_dir, y, x, z);
        // use regex to match file names
        let mut max_x = 0;
        let mut max_y = 0;
        let mut max_z = 0;
        for entry in std::fs::read_dir(data_dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            let file_name = path.file_name().unwrap().to_str().unwrap();
            if let Some(captures) = regex::Regex::new(r"cell_yxz_(\d+)_(\d+)_(\d+)\.tif")
                .unwrap()
                .captures(file_name)
            {
                //println!("Found file: {}", file_name);
                let x = captures.get(2).unwrap().as_str().parse::<usize>().unwrap();
                let y = captures.get(1).unwrap().as_str().parse::<usize>().unwrap();
                let z = captures.get(3).unwrap().as_str().parse::<usize>().unwrap();
                if x > max_x {
                    max_x = x;
                }
                if y > max_y {
                    max_y = y;
                }
                if z > max_z {
                    max_z = z;
                }
            }
        }
        fn map_for(data_dir: &str, x: usize, y: usize, z: usize) -> Option<memmap::Mmap> {
            let file_name = format!("{}/cell_yxz_{:03}_{:03}_{:03}.tif", data_dir, y, x, z);

            let file = File::open(file_name).ok()?;
            unsafe { MmapOptions::new().offset(8).map(&file) }.ok()
        }
        if !std::path::Path::new(data_dir).exists() {
            println!("Data directory {} does not exist", data_dir);
            return MappedVolumeGrid {
                max_x: 0,
                max_y: 0,
                max_z: 0,
                data: vec![],
            };
        }
        let data: Vec<Vec<Vec<Option<memmap::Mmap>>>> = (1..=max_z)
            .map(|z| {
                (1..=max_y)
                    .map(|y| (1..=max_x).map(|x| map_for(data_dir, x, y, z)).collect())
                    .collect()
            })
            .collect();

        // count number of slices found
        let slices_found = data.iter().flatten().flatten().flatten().count();
        println!("Found {} cells in {}", slices_found, data_dir);
        println!("max_x: {}, max_y: {}, max_z: {}", max_x, max_y, max_z);

        MappedVolumeGrid {
            max_x: max_x - 1,
            max_y: max_y - 1,
            max_z: max_z - 1,
            data,
        }
    }

    fn to_volume_grid(&self) -> VolumeGrid4x4x4 {
        fn data_for(mapped: &memmap::Mmap, x: usize, y: usize, z: usize) -> Box<V500> {
            let mut buffer = Box::new([0u8; 512 * 512 * 512]);
            let mut printed = false;
            for z in 0..500 {
                for y in 0..500 {
                    for x in 0..500 {
                        let tx = x;//(xyz[0] % 500) as usize;
                        let ty = y;//(xyz[1] % 500) as usize;
                        let tz = z;//(xyz[2] % 500) as usize;

                        let bx = tx >> 4;
                        let by = ty >> 4;
                        let bz = tz >> 4;

                        let boff = (bz << 10) + (by << 5) + bx;
                        //println!("bx: {}, by: {}, bz: {}, boff: {}", bx, by, bz, boff);

                        let px = tx & 0xf;
                        let py = ty & 0xf;
                        let pz = tz & 0xf;

                        let poff = pz * 256 + py * 16 + px;

                        let off = boff * 4096 + poff;
                        //let off = z * 500 * 500 + y * 500 + x;

                        let moff =
                            500147 * z + (y * 500 + x) * 2;
        
                        /* if x < 2 && y < 2 && z < 50 {
                            println!("x: {}, y: {}, z: {}, bx: {}, by: {}, bz: {}, px: {}, py: {}, pz: {}, boff: {}, poff: {}, off: {}, moff: {}", x, y, z, bx, by, bz, px, py, pz, boff, poff, off, moff);
                        } */
        
                        // off + 1 because we select the higher order bits of little endian 16 bit values
                        if moff + 1 >= mapped.len() {
                            ()
                        } else {
                            if (off >= buffer.len()) {
                                println!("out of bounds");
                                println!("x: {}, y: {}, z: {}, bx: {}, by: {}, bz: {}, px: {}, py: {}, pz: {}, boff: {}, poff: {}, off: {}, moff: {}", x, y, z, bx, by, bz, px, py, pz, boff, poff, off, moff);
                            }
                            if (buffer[off] != 0 && !printed) {
                                printed = true;
                                println!("Overwriting value at {} {} {}", x, y, z);
                                println!("x: {}, y: {}, z: {}, bx: {}, by: {}, bz: {}, px: {}, py: {}, pz: {}, boff: {}, poff: {}, off: {}, moff: {}", x, y, z, bx, by, bz, px, py, pz, boff, poff, off, moff);
                                panic!();
                            }
                            
                            buffer[off] = mapped[moff + 1];
                        }
                        }
                }
            }
            buffer
        }

        let data: Vec<Vec<Vec<Option<Box<V500>>>>> = 
        (0..=self.max_z).map(|z| {
                (0..=self.max_y).map(|y| 
                    (0..=self.max_x).map(|x| {
                        if let Some(map) = &self.data[z][y][x] {
                            println!("Converting cell {} {} {}", x, y, z);
                            Some(data_for(map, x, y, z))
                        } else {
                            None
                        }
                    }).collect())
                    .collect()
            })
            .collect();

        VolumeGrid4x4x4 { 
            max_x: self.max_x,
            max_y: self.max_y,
            max_z: self.max_z,
            data
        }
    }
}
impl World for MappedVolumeGrid {
    fn get(&mut self, xyz: [i32; 3]) -> u8 {
        let x_tile = xyz[0] as usize / 500;
        let y_tile = xyz[1] as usize / 500;
        let z_tile = xyz[2] as usize / 500;

        if xyz[0] < 0 || xyz[1] < 0 || xyz[2] < 0 || x_tile > self.max_x || y_tile > self.max_y || z_tile > self.max_z {
            //println!("out of bounds: {:?}", xyz);
            0
        } else if let Some(tile) = &self.data[z_tile][y_tile][x_tile] {
            let off =
                500147 * ((xyz[2] % 500) as usize) + ((xyz[1] % 500) as usize * 500 + (xyz[0] % 500) as usize) * 2;

            //println!("xyz: {:?}, off: {}, tile: {:?}", xyz, off, tile);

            // off + 1 because we select the higher order bits of little endian 16 bit values
            if off + 1 >= tile.len() {
                0
            } else {
                tile[off + 1]
            }
        } else {
            0
        }
    }
    fn paint(&mut self, xyz: [i32; 3], u_coord: usize, v_coord: usize, plane_coord: usize, width: usize, height: usize, buffer: &mut [u8]) {

    }
}

struct EmptyWorld {}
impl World for EmptyWorld {
    fn get(&mut self, _xyz: [i32; 3]) -> u8 { 0 }
    fn paint(&mut self, xyz: [i32; 3], u_coord: usize, v_coord: usize, plane_coord: usize, width: usize, height: usize, buffer: &mut [u8]) {

    }
}

#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct TemplateApp {
    coord: [i32; 3],
    zoom: f32,
    frame_width: usize,
    frame_height: usize,
    data_dir: String,
    #[serde(skip)]
    texture_xy: Option<egui::TextureHandle>,
    #[serde(skip)]
    texture_xz: Option<egui::TextureHandle>,
    #[serde(skip)]
    texture_yz: Option<egui::TextureHandle>,
    #[serde(skip)]
    world: Box<dyn World>,
}

impl Default for TemplateApp {
    fn default() -> Self {
        Self {
            coord: [2800, 2500, 10852],
            zoom: 1f32,
            frame_width: 500,
            frame_height: 500,
            data_dir: ".".to_string(),
            texture_xy: None,
            texture_xz: None,
            texture_yz: None,
            world: Box::new(EmptyWorld {}),
        }
    }
}

impl TemplateApp {
    /// Called once before the first frame.
    pub fn new(cc: &eframe::CreationContext<'_>, data_dir: Option<String>) -> Self {
        // This is also where you can customize the look and feel of egui using
        // `cc.egui_ctx.set_visuals` and `cc.egui_ctx.set_fonts`.
        let mut app: TemplateApp = if let Some(storage) = cc.storage {
            eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default()
        } else {
            Default::default()
        };

        app.load_data(&data_dir.unwrap_or_else(|| app.data_dir.clone()));

        app
    }
    fn load_data(&mut self, data_dir: &str) {
        self.world = Box::new(MappedVolumeGrid::from_data_dir(data_dir).to_volume_grid());
        //self.world = Box::new(VolumeGrid16x16x16Mapped::from_data_dir(data_dir ,78, 78, 200));
        self.data_dir = data_dir.to_string();
    }

    pub fn clear_textures(&mut self) {
        self.texture_xy = None;
        self.texture_xz = None;
        self.texture_yz = None;
    }

    fn add_scroll_handler(&mut self, image: &Response, ui: &Ui, v: fn(&mut Self) -> &mut i32) {
        if image.hovered() {
            let delta = ui.input(|i| i.scroll_delta);
            if delta.y != 0.0 {
                let delta = delta.y.signum() * 1.0;
                let m = v(self);
                *m += delta as i32;
                self.clear_textures();
            }
        }
    }
    fn x(&self) -> i32 { self.coord[0] }
    fn y(&self) -> i32 { self.coord[1] }
    fn z(&self) -> i32 { self.coord[2] }

    fn get_or_create_texture(
        &mut self,
        ui: &Ui,
        u_coord: usize,
        v_coord: usize,
        d_coord: usize,
        t: fn(&mut Self) -> &mut Option<egui::TextureHandle>,
    ) -> egui::TextureHandle {
        if let Some(texture) = t(self) {
            texture.clone()
        } else {
            let res = self.create_texture(ui, u_coord, v_coord, d_coord);
            *t(self) = Some(res.clone());
            res
        }
    }
    fn create_texture(&mut self, ui: &Ui, u_coord: usize, v_coord: usize, d_coord: usize) -> egui::TextureHandle {
        use std::time::Instant;
        let _start = Instant::now();

        let width = (self.frame_width as f32 / self.zoom) as usize;
        let height = (self.frame_height as f32 / self.zoom) as usize;
        let mut pixels = vec![0u8; width * height];

        //let q = 1;

        //let mut printed = false;
        let mut xyz: [i32; 3] = [0, 0, 0];
        xyz[d_coord] = self.coord[d_coord];

        self.world.paint(self.coord, u_coord, v_coord, d_coord, width, height, &mut pixels);
        /* for (i, p) in pixels.iter_mut().enumerate() {
            xyz[u_coord] = (i % width) as i32 + self.coord[u_coord] - (250 as f32 / self.zoom) as i32;
            xyz[v_coord] = (i / width) as i32 + self.coord[v_coord] - (250 as f32 / self.zoom) as i32;

            *p = self.world.get(xyz);
        } */
        
        let image = ColorImage::from_gray([width, height], &pixels);
        println!("Time elapsed before loading in ({}, {}, {}) is: {:?}", u_coord, v_coord, d_coord, _start.elapsed());
        // Load the texture only once.
        let res = ui.ctx().load_texture("my-image-xy", image, Default::default());

        let _duration = _start.elapsed();
        println!("Time elapsed in ({}, {}, {}) is: {:?}", u_coord, v_coord, d_coord, _duration);
        res
    }
}

impl eframe::App for TemplateApp {
    /// Called by the frame work to save state before shutdown.
    fn save(&mut self, storage: &mut dyn eframe::Storage) { eframe::set_value(storage, eframe::APP_KEY, self); }

    /// Called each time the UI needs repainting, which may be many times per second.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let x_sl = ui.add(
                egui::Slider::new(
                    &mut self.coord[0],
                    -10000..=10000, /* 0..=(self.img_width - self.frame_width - 1) */
                )
                .text("x"),
            );
            let y_sl = ui.add(
                egui::Slider::new(
                    &mut self.coord[1],
                    -10000..=10000, /* 0..=(self.img_height - self.frame_height - 1) */
                )
                .text("y"),
            );

            let _z_sl = ui.add(egui::Slider::new(&mut self.coord[2], 0..=25000).text("z"));
            let zoom_sl = ui.add(
                egui::Slider::new(&mut self.zoom, 0.1f32..=32f32)
                    .text("zoom")
                    .logarithmic(true),
            );
            if x_sl.changed() || y_sl.changed() || zoom_sl.changed() {
                self.clear_textures();
            }

            let texture_xy = &self.get_or_create_texture(ui, 0, 1, 2, |s| &mut s.texture_xy);
            let texture_xz = &self.get_or_create_texture(ui, 0, 2, 1, |s| &mut s.texture_xz);
            let texture_yz = &self.get_or_create_texture(ui, 2, 1, 0, |s| &mut s.texture_yz);

            // use remaining space for image
            //let size =ui.available_size();
            {
                //self.frame_width = size.x as usize;
                //self.frame_height = size.y as usize;

                let image = Image::new(texture_xy)
                    .max_height(500f32)
                    .max_width(500f32)
                    .fit_to_original_size(self.zoom);

                let image_xz = Image::new(texture_xz)
                    .max_height(500f32)
                    .max_width(500f32)
                    .fit_to_original_size(self.zoom);

                let image_yz = Image::new(texture_yz)
                    .max_height(500f32)
                    .max_width(500f32)
                    .fit_to_original_size(self.zoom);

                ui.horizontal(|ui| {
                    let im_xy = ui.add(image).interact(egui::Sense::drag());
                    let im_xz = ui.add(image_xz);
                    let im_yz = ui.add(image_yz);
                    self.add_scroll_handler(&im_xy, ui, |s| &mut s.coord[2]);
                    self.add_scroll_handler(&im_xz, ui, |s| &mut s.coord[1]);
                    self.add_scroll_handler(&im_yz, ui, |s| &mut s.coord[0]);
                    //let size2 = texture.size_vec2();

                    /* if im_xy.hovered() {
                        let delta = ui.input(|i| i.scroll_delta);
                        if delta.y != 0.0 {
                            let delta = delta.y.signum() * 1.0;
                            self.z() = (self.z() as i32 + delta as i32).max(0).min(15000) as usize;
                            self.clear_textures();
                        }
                    } */

                    if im_xy.dragged_by(PointerButton::Primary) {
                        let im2 = im_xy.on_hover_cursor(CursorIcon::Grabbing);
                        let delta = -im2.drag_delta() / self.zoom;
                        //println!("delta: {:?} orig delta: {:?}", delta, im2.drag_delta());
                        //let oldx = self.x();
                        //let oldy = self.y();

                        self.coord[0] += delta.x as i32;
                        self.coord[1] += delta.y as i32;
                        //println!("oldx: {}, oldy: {}, x: {}, y: {}", oldx, oldy, self.x(), self.y());
                        self.clear_textures();
                    } /* else if size2.x as usize != self.frame_width || size2.y as usize != self.frame_height {
                          println!("Reset because size changed from {:?} to {:?}", size2, size);
                          self.clear_textures();
                      }; */
                });
            };
        });
    }
}

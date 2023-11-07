use egui::{ColorImage, CursorIcon, Image, PointerButton, Response, Ui};

trait World {
    fn get(&self, xyz: [i32; 3]) -> u8;
}

struct MappedCells {
    max_x: usize,
    max_y: usize,
    max_z: usize,
    data: Vec<Vec<Vec<Option<memmap::Mmap>>>>,
}
impl MappedCells {
    pub fn from_data_dir(data_dir: &str) -> MappedCells {
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
            return MappedCells {
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

        MappedCells {
            max_x: max_x - 1,
            max_y: max_y - 1,
            max_z: max_z - 1,
            data,
        }
    }
}
impl World for MappedCells {
    fn get(&self, xyz: [i32; 3]) -> u8 {
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
}

struct EmptyWorld {}
impl World for EmptyWorld {
    fn get(&self, _xyz: [i32; 3]) -> u8 { 0 }
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
        self.world = Box::new(MappedCells::from_data_dir(data_dir));
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
    fn create_texture(&self, ui: &Ui, u_coord: usize, v_coord: usize, d_coord: usize) -> egui::TextureHandle {
        use std::time::Instant;
        let _start = Instant::now();

        let width = (self.frame_width as f32 / self.zoom) as usize;
        let height = (self.frame_height as f32 / self.zoom) as usize;
        let mut pixels = vec![0u8; width * height];

        //let q = 1;

        //let mut printed = false;
        let mut xyz: [i32; 3] = [0, 0, 0];
        xyz[d_coord] = self.coord[d_coord];

        for (i, p) in pixels.iter_mut().enumerate() {
            xyz[u_coord] = (i % width) as i32 + self.coord[u_coord] - (250_f32 / self.zoom) as i32;
            xyz[v_coord] = (i / width) as i32 + self.coord[v_coord] - (250_f32 / self.zoom) as i32;

            *p = self.world.get(xyz);
        }
        /* 4bit
        for (i, p) in pixels.iter_mut().enumerate() {
            let x = i % width + self.x();
            let y = i / width + self.y();
            let off = y * real_width + x;
            let v8 = self.data[off / 2];
            let v =
                if off % 2 == 0 {
                    v8 & 0xf << 4
                } else {
                    v8 & 0xf0
                };

            //let v = (v16 >> 8) as u8;

            *p = v;
        }
            */
        /* for (i, p) in pixels.iter_mut().enumerate() {
            let x = i % width + self.x();
            let y = i / width + self.y();
            let off = y * real_width + x;
            let v8 = self.data[off];
            *p = v8;
        } */
        let image = ColorImage::from_gray([width, height], &pixels);

        // Load the texture only once.
        let res = ui.ctx().load_texture("my-image-xy", image, Default::default());
        let _duration = _start.elapsed();
        //println!("Time elapsed in ({}, {}, {}) is: {:?}", u_coord, v_coord, d_coord, _duration);
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
                    &mut self.x(),
                    -10000..=10000, /* 0..=(self.img_width - self.frame_width - 1) */
                )
                .text("x"),
            );
            let y_sl = ui.add(
                egui::Slider::new(
                    &mut self.y(),
                    -10000..=10000, /* 0..=(self.img_height - self.frame_height - 1) */
                )
                .text("y"),
            );

            let _z_sl = ui.add(egui::Slider::new(&mut self.z(), 0..=14500).text("z"));
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

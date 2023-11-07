
use egui::{ColorImage, PointerButton, CursorIcon, Image, Response, Ui};

/// We derive Deserialize/Serialize so we can persist app state on shutdown.

pub struct TemplateApp {
    //#[serde(skip)] // This how you opt-out of serialization of a field
    x: i32,
    y: i32,
    z: i32,
    zoom: f32,
    img_width: usize,
    img_height: usize,
    frame_width: usize,
    frame_height: usize,
    texture_xy: Option<egui::TextureHandle>,
    texture_xz: Option<egui::TextureHandle>,
    texture_yz: Option<egui::TextureHandle>,
    data: Vec<Vec<Vec<Option<memmap::Mmap>>>>,
}

impl Default for TemplateApp {
    fn default() -> Self {
        use memmap::MmapOptions;
        use std::fs::File;

        //let file = File::open("/tmp/cell_yxz_006_007_022.tif").unwrap();
        //let mmap = unsafe { MmapOptions::new().offset(8).map(&file).unwrap() };

        fn map_for(x: usize, y: usize, z: usize) -> Option<memmap::Mmap> {
            let file_name = format!("/tmp/cell_yxz_{:03}_{:03}_{:03}.tif", y, x, z);
            
            let file = File::open(file_name).ok()?;
            unsafe { MmapOptions::new().offset(8).map(&file) }.ok()
        }
        let data =
            (1..=29).map( |z|
                (1..=16).map(|y|
                    (1..=17).map( |x|
                        map_for(x, y, z)
                    ).collect()
                ).collect()
            ).collect();

        Self {
            x: 2800,
            y: 2500,
            z: 10852,
            zoom: 1f32,
            img_width: 8096,
            img_height: 7888,
            frame_width: 500,
            frame_height: 500,
            texture_xy: None,
            texture_xz: None,
            texture_yz: None,
            data: data
        }
    }
}

impl TemplateApp {
    /// Called once before the first frame.
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // This is also where you can customize the look and feel of egui using
        // `cc.egui_ctx.set_visuals` and `cc.egui_ctx.set_fonts`.

        

        Default::default()
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
}

impl eframe::App for TemplateApp {
    /// Called by the frame work to save state before shutdown.
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        //eframe::set_value(storage, eframe::APP_KEY, self);
    }

    /// Called each time the UI needs repainting, which may be many times per second.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Put your widgets into a `SidePanel`, `TopPanel`, `CentralPanel`, `Window` or `Area`.
        // For inspiration and more examples, go to https://emilk.github.io/egui

        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            // The top panel is often a good place for a menu bar:

            egui::menu::bar(ui, |ui| {
                #[cfg(not(target_arch = "wasm32"))] // no File->Quit on web pages!
                {
                    ui.menu_button("File", |ui| {
                        if ui.button("Quit").clicked() {
                            _frame.close();
                        }
                    });
                    ui.add_space(16.0);
                }

                egui::widgets::global_dark_light_mode_buttons(ui);
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let x_sl = ui.add(egui::Slider::new(&mut self.x, -10000..=10000/* 0..=(self.img_width - self.frame_width - 1) */).text("x"));
            let y_sl = ui.add(egui::Slider::new(&mut self.y, -10000..=10000/* 0..=(self.img_height - self.frame_height - 1) */).text("y"));
            if x_sl.changed() || y_sl.changed() {
                self.clear_textures();
            }
            let _z_sl = ui.add(egui::Slider::new(&mut self.z, 0..=14500).text("z"));
            let _zoom_sl = ui.add(egui::Slider::new(&mut self.zoom, 0.1f32..=32f32).text("zoom").logarithmic(true));
            
            
            let texture_xy: &egui::TextureHandle = self.texture_xy.get_or_insert_with(|| {
                use std::time::Instant;
                let start = Instant::now();

                let width = (self.frame_width as f32 / self.zoom) as usize;
                let height = (self.frame_height as f32 / self.zoom) as usize;
                let mut pixels = vec![0u8; width * height];

                let q = 1;

                let mut printed = false;
                let z = self.z;
                
                for (i, p) in pixels.iter_mut().enumerate() {
                    let x = (i % width) as i32 + self.x - 250;
                    let y = (i / width) as i32 + self.y - 250;


                    let v = 
                        if x >= 0 && x < self.img_width as i32 && y >= 0 && y < self.img_height as i32 {
                            
                            if let Some(tile) = &self.data[(self.z / 500) as usize][(y / 500) as usize][(x / 500) as usize] {
                                let off = (((y % 500) as usize / q) * q * 500 + ((x % 500) as usize / q) * q) * 2 + 500147 * (z % 500) as usize;
                                if off + 1 >= tile.len() {
                                    if !printed {
                                        println!("x: {}, y: {}, z:{}, off: {}, len: {}", x, y, self.z, off, self.data.len());
                                        printed = true;
                                    }
                                    *p = 0;
                                    continue;
                                }
                                tile[off + 1]
                            } else {
                                0
                            }
                        } else {
                            0
                        };

                    *p = v;
                }
                /* 4bit
                for (i, p) in pixels.iter_mut().enumerate() {
                    let x = i % width + self.x;
                    let y = i / width + self.y;
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
                    let x = i % width + self.x;
                    let y = i / width + self.y;
                    let off = y * real_width + x;
                    let v8 = self.data[off];
                    *p = v8;                    
                } */
                let image = ColorImage::from_gray([width, height], &pixels);

                // Load the texture only once.
                let res = ui.ctx().load_texture(
                    "my-image-xy",
                    image,
                    Default::default()
                );
                let duration = start.elapsed();
                println!("Time elapsed in expensive_function() is: {:?}", duration);
                res
            });

            let texture_xz: &egui::TextureHandle = self.texture_xz.get_or_insert_with(|| {
                use std::time::Instant;
                let start = Instant::now();

                let width = (self.frame_width as f32 / self.zoom) as usize;
                let height = (self.frame_height as f32 / self.zoom) as usize;
                let mut pixels = vec![0u8; width * height];

                let q = 1;

                let mut printed = false;
                
                let y = self.y;

                for (i, p) in pixels.iter_mut().enumerate() {
                    let x = (i % width) as i32 + self.x - 250;
                    let z = (i / width) as i32 + (self.z as i32) - 250;
                    if i == 0 {
                        println!("x: {}, y: {}, z: {}", x, y, z);
                    }


                    let v = 
                        //if x >= 0 && x < self.img_width as i32 && z >= 0 && z < self.img_height as i32 {
                        if let Some(tile) = &self.data[(z / 500) as usize][(y / 500) as usize][(x / 500) as usize] {
                            let off = (((y % 500) as usize / q) * q * 500 + ((x % 500) as usize / q) * q) * 2 + 500147 * (z % 500) as usize;
                            if i == 0 {
                                println!("off: {}", off);
                            }
                            if off + 1 >= tile.len() {
                                if !printed {
                                    println!("x: {}, y: {}, z:{}, off: {}, len: {}", x, y, z, off, self.data.len());
                                    printed = true;
                                }
                                0
                            }
                            else {
                                tile[off + 1]
                            }
                        } else {
                            0
                        };
                        /* } else {
                            0
                        }; */

                    *p = v;
                }
                /* 4bit
                for (i, p) in pixels.iter_mut().enumerate() {
                    let x = i % width + self.x;
                    let y = i / width + self.y;
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
                    let x = i % width + self.x;
                    let y = i / width + self.y;
                    let off = y * real_width + x;
                    let v8 = self.data[off];
                    *p = v8;                    
                } */
                let image = ColorImage::from_gray([width, height], &pixels);

                // Load the texture only once.
                let res = ui.ctx().load_texture(
                    "my-image-xz",
                    image,
                    Default::default()
                );
                let duration = start.elapsed();
                println!("Time elapsed in expensive_function() is: {:?}", duration);
                res
            });

            let texture_yz: &egui::TextureHandle = self.texture_yz.get_or_insert_with(|| {
                use std::time::Instant;
                let start = Instant::now();

                let width = (self.frame_width as f32 / self.zoom) as usize;
                let height = (self.frame_height as f32 / self.zoom) as usize;
                let mut pixels = vec![0u8; width * height];

                let q = 1;

                let mut printed = false;
                
                let x = self.x;

                for (i, p) in pixels.iter_mut().enumerate() {
                    let y = (i % width) as i32 + (self.y as i32) - 250;
                    let z = (i / width) as i32 + (self.z as i32) - 250;
                    if i == 0 {
                        println!("x: {}, y: {}, z: {}", x, y, z);
                    }


                    let v = 
                        //if x >= 0 && x < self.img_width as i32 && z >= 0 && z < self.img_height as i32 {
                        if let Some(tile) = &self.data[(z / 500) as usize][(y / 500) as usize][(x / 500) as usize] {
                            let off = (((y % 500) as usize / q) * q * 500 + ((x % 500) as usize / q) * q) * 2 + 500147 * (z % 500) as usize;
                            if i == 0 {
                                println!("off: {}", off);
                            }
                            if off + 1 >= tile.len() {
                                if !printed {
                                    println!("x: {}, y: {}, z:{}, off: {}, len: {}", x, y, z, off, self.data.len());
                                    printed = true;
                                }
                                0
                            }
                            else {
                                tile[off + 1]
                            }
                        } else {
                            0
                        };
                        /* } else {
                            0
                        }; */

                    *p = v;
                }
                /* 4bit
                for (i, p) in pixels.iter_mut().enumerate() {
                    let x = i % width + self.x;
                    let y = i / width + self.y;
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
                    let x = i % width + self.x;
                    let y = i / width + self.y;
                    let off = y * real_width + x;
                    let v8 = self.data[off];
                    *p = v8;                    
                } */
                let image = ColorImage::from_gray([width, height], &pixels);

                // Load the texture only once.
                let res = ui.ctx().load_texture(
                    "my-image-yz",
                    image,
                    Default::default()
                );
                let duration = start.elapsed();
                println!("Time elapsed in expensive_function() is: {:?}", duration);
                res
            });
            // use remaining space for image
            //let size =ui.available_size();
            {
                //self.frame_width = size.x as usize;
                //self.frame_height = size.y as usize;
                
                let image =
                    Image::new(texture_xy)
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
                    let im_xy = ui.add(image)
                        .interact(egui::Sense::drag());
                    let im_xz = ui.add(image_xz);
                    let im_yz = ui.add(image_yz);
                    self.add_scroll_handler(&im_xy, &ui, |s| &mut s.z);
                    self.add_scroll_handler(&im_xz, &ui, |s| &mut s.y);
                    self.add_scroll_handler(&im_yz, &ui, |s| &mut s.x);
                                    //let size2 = texture.size_vec2();
                    
                    /* if im_xy.hovered() {
                        let delta = ui.input(|i| i.scroll_delta);
                        if delta.y != 0.0 {
                            let delta = delta.y.signum() * 1.0;
                            self.z = (self.z as i32 + delta as i32).max(0).min(15000) as usize;
                            self.clear_textures();
                        }
                    } */
                            
                    if im_xy.dragged_by(PointerButton::Primary) {
                        let im2 = im_xy.on_hover_cursor(CursorIcon::Grabbing);
                        let delta = -im2.drag_delta() / self.zoom;
                        //println!("delta: {:?} orig delta: {:?}", delta, im2.drag_delta());
                        //let oldx = self.x;
                        //let oldy = self.y;

                        self.x = self.x as i32 + delta.x as i32;
                        self.y = self.y as i32 + delta.y as i32;
                        //println!("oldx: {}, oldy: {}, x: {}, y: {}", oldx, oldy, self.x, self.y);
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


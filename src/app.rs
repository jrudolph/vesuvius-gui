
use egui::{ColorImage, PointerButton, CursorIcon};

/// We derive Deserialize/Serialize so we can persist app state on shutdown.

pub struct TemplateApp {
    // Example stuff:
    label: String,

    //#[serde(skip)] // This how you opt-out of serialization of a field
    x: usize,
    y: usize,
    img_width: usize,
    img_height: usize,
    frame_width: usize,
    frame_height: usize,
    texture: Option<egui::TextureHandle>,
    data: memmap::Mmap,
}

impl Default for TemplateApp {
    fn default() -> Self {
        use memmap::MmapOptions;
        use std::fs::File;

        let file = File::open(/* "/tmp/00870.tif" */"test-8bit.tif").unwrap();
        let mmap = unsafe { MmapOptions::new().offset(368).map(&file).unwrap() };

        Self {
            // Example stuff:
            label: "Hello World!".to_owned(),
            x: 100,
            y: 100,
            img_width: 9414,
            img_height: 9414,
            frame_width: 100,
            frame_height: 100,
            texture: None,
            data: mmap
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
            let x_sl = ui.add(egui::Slider::new(&mut self.x, 0..=(self.img_width - self.frame_width - 1)).text("x"));
            let y_sl = ui.add(egui::Slider::new(&mut self.y, 0..=(self.img_height - self.frame_height - 1)).text("y"));
            if x_sl.changed() || y_sl.changed() {
                self.texture = None;
            }
            
            let texture: &egui::TextureHandle = self.texture.get_or_insert_with(|| {
                fn view_as_u16(slice: &[u8]) -> &[u16] {
                    // Ensure that the slice length is a multiple of 2, as each u16 is 2 bytes.
                    //assert_eq!(slice.len() % 2, 0);
                
                    // Use pointer casting to reinterpret the slice as a slice of u16.
                    unsafe {
                        std::slice::from_raw_parts(slice.as_ptr() as *const u16, slice.len() / 2)
                    }
                }
                let real_width = self.img_width;
                use std::time::Instant;
                let start = Instant::now();

                let width = self.frame_width;
                let height = self.frame_height;
                let mut pixels = vec![0u8; width * height];
                /* 
                let data16 = view_as_u16(&self.data);
                for (i, p) in pixels.iter_mut().enumerate() {
                    let x = i % width + self.x;
                    let y = i / width + self.y;
                    let off = y * real_width + x;
                    let v16 = data16[off];
                    let v = (v16 >> 8) as u8;
                    
                    *p = v;
                }*/
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
                for (i, p) in pixels.iter_mut().enumerate() {
                    let x = i % width + self.x;
                    let y = i / width + self.y;
                    let off = y * real_width + x;
                    let v8 = self.data[off];
                    *p = v8;                    
                }
                let image = ColorImage::from_gray([width, height], &pixels);

                // Load the texture only once.
                let res = ui.ctx().load_texture(
                    "my-image",
                    image,
                    Default::default()
                );
                let duration = start.elapsed();
                println!("Time elapsed in expensive_function() is: {:?}", duration);
                res
            });
            // use remaining space for image
            let size =ui.available_size();
            {
                self.frame_width = size.x as usize;
                self.frame_height = size.y as usize;
                
                let im = 
                    ui.image((texture.id(), texture.size_vec2()))
                        .interact(egui::Sense::drag());
                if im.dragged_by(PointerButton::Primary) {
                    let im2 = im.on_hover_cursor(CursorIcon::Grabbing);
                    let delta = -im2.drag_delta();
                    
                    self.x = (self.x as i32 + delta.x as i32).max(0) as usize;
                    self.y = (self.y as i32 + delta.y as i32).max(0) as usize;
                    self.texture = None;
                } else if texture.size_vec2() != size {
                    self.texture = None;
                };
            };
        });
    }
}

fn powered_by_egui_and_eframe(ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 0.0;
        ui.label("Powered by ");
        ui.hyperlink_to("egui", "https://github.com/emilk/egui");
        ui.label(" and ");
        ui.hyperlink_to(
            "eframe",
            "https://github.com/emilk/egui/tree/master/crates/eframe",
        );
        ui.label(".");
    });
}

use crate::downloader::*;
use crate::volume::*;

use egui::{ColorImage, CursorIcon, Image, PointerButton, Response, Ui};

const ZOOM_RES_FACTOR: f32 = 1.5; // defines which resolution is used for which zoom level, 2 means only when zooming deeper than 2x the full resolution is pulled

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
    world: Box<dyn PaintVolume>,
}

impl Default for TemplateApp {
    fn default() -> Self {
        Self {
            coord: [2800, 2500, 10852],
            zoom: 1f32,
            frame_width: 1000,
            frame_height: 1000,
            data_dir: ".".to_string(),
            texture_xy: None,
            texture_xz: None,
            texture_yz: None,
            world: Box::new(EmptyVolume {}),
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
        app.frame_width = 1000;
        app.frame_height = 750;

        app.load_data(&data_dir.unwrap_or_else(|| app.data_dir.clone()));

        app
    }
    fn load_data(&mut self, data_dir: &str) {
        //self.world = Box::new(MappedVolumeGrid::from_data_dir(data_dir).to_volume_grid());
        self.world = Box::new(VolumeGrid64x4Mapped::from_data_dir(data_dir ,78, 78, 200, Downloader::new(data_dir, self.frame_width, self.frame_height)));
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
                let min_level = 1 << ((ZOOM_RES_FACTOR / self.zoom) as i32).min(4);
                let delta = delta.y.signum() * min_level as f32;
                let m = v(self);
                *m = (*m + delta as i32) / min_level as i32 * min_level as i32;
                self.clear_textures();
            }
        }
    }
    fn add_drag_handler(&mut self, image: &Response, ucoord: usize, vcoord: usize) {
        if image.dragged_by(PointerButton::Primary) {
            //let im2 = image.on_hover_cursor(CursorIcon::Grabbing);
            let delta = -image.drag_delta() / self.zoom;

            self.coord[ucoord] += delta.x as i32;
            self.coord[vcoord] += delta.y as i32;
            self.clear_textures();
        }
    }
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

        let min_level = (32 - ((ZOOM_RES_FACTOR / self.zoom) as u32).leading_zeros()).min(4).max(0);
        let max_level = (min_level + 2).min(4);
        for level in (min_level..=max_level).rev() {
            let sfactor = 1 << level as u8;
            //println!("level: {} factor: {}", level, sfactor);
            self.world.paint(self.coord, u_coord, v_coord, d_coord, width, height, sfactor, &mut pixels);
        }

        let image = ColorImage::from_gray([width, height], &pixels);
        //println!("Time elapsed before loading in ({}, {}, {}) is: {:?}", u_coord, v_coord, d_coord, _start.elapsed());
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
                egui::Slider::new(&mut self.zoom, 0.1f32..=6f32)
                    .text("zoom")
                    .logarithmic(true),
            );
            if x_sl.changed() || y_sl.changed() || zoom_sl.changed() {
                self.clear_textures();
            }

            ui.label(format!("FPS: {}", 1.0 / (_frame.info().cpu_usage.unwrap_or_default() + 1e-6)));

            let texture_xy = &self.get_or_create_texture(ui, 0, 1, 2, |s| &mut s.texture_xy);
            let texture_xz = &self.get_or_create_texture(ui, 0, 2, 1, |s| &mut s.texture_xz);
            let texture_yz = &self.get_or_create_texture(ui, 2, 1, 0, |s| &mut s.texture_yz);

            // use remaining space for image
            //let size =ui.available_size();
            {
                //self.frame_width = size.x as usize;
                //self.frame_height = size.y as usize;

                let image = Image::new(texture_xy)
                    .max_height(self.frame_height as f32)
                    .max_width(self.frame_width as f32)
                    .fit_to_original_size(self.zoom);

                let image_xz = Image::new(texture_xz)
                    .max_height(self.frame_height as f32)
                    .max_width(self.frame_width as f32)
                    .fit_to_original_size(self.zoom);

                let image_yz = Image::new(texture_yz)
                    .max_height(self.frame_height as f32)
                    .max_width(self.frame_width as f32)
                    .fit_to_original_size(self.zoom);

                ui.horizontal(|ui| {
                    let im_xy = ui.add(image).interact(egui::Sense::drag());
                    let im_xz = ui.add(image_xz).interact(egui::Sense::drag());
                    self.add_scroll_handler(&im_xy, ui, |s| &mut s.coord[2]);
                    self.add_scroll_handler(&im_xz, ui, |s| &mut s.coord[1]);
                    self.add_drag_handler(&im_xy, 0, 1);
                    self.add_drag_handler(&im_xz, 0, 2);
                });
                let im_yz = ui.add(image_yz).interact(egui::Sense::drag());
                self.add_scroll_handler(&im_yz, ui, |s| &mut s.coord[0]);
                self.add_drag_handler(&im_yz, 2, 1);
            };
        });
    }
}

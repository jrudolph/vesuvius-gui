use std::ops::Index;
use std::ops::RangeInclusive;
use std::sync::mpsc::Receiver;

use crate::downloader::*;
use crate::model::*;
use crate::volume::*;

use egui::Vec2;
use egui::{ColorImage, CursorIcon, Image, PointerButton, Response, Ui};

const ZOOM_RES_FACTOR: f32 = 1.3; // defines which resolution is used for which zoom level, 2 means only when zooming deeper than 2x the full resolution is pulled

#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct TemplateApp {
    volume_id: usize,
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
    #[serde(skip)]
    last_size: Vec2,
    #[serde(skip)]
    download_notifier: Option<Receiver<()>>,
    drawing_config: DrawingConfig,
}

impl Default for TemplateApp {
    fn default() -> Self {
        Self {
            volume_id: 0,
            coord: [2800, 2500, 10852],
            zoom: 1f32,
            frame_width: 1000,
            frame_height: 1000,
            data_dir: ".".to_string(),
            texture_xy: None,
            texture_xz: None,
            texture_yz: None,
            world: Box::new(EmptyVolume {}),
            last_size: Vec2::ZERO,
            download_notifier: None,
            drawing_config: Default::default(),
        }
    }
}

impl TemplateApp {
    const TILE_SERVER: &'static str = "https://vesuvius.virtual-void.net";

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
        if let Some(dir) = data_dir {
            app.data_dir = dir;
        }

        app.select_volume(app.volume_id);

        app
    }
    fn load_data_password(data_dir: &str) -> Option<String> {
        // load data password from <data_dir>/password.txt
        let mut password = String::new();
        let mut password_file = std::fs::File::open(format!("{}/password.txt", data_dir)).ok()?;
        std::io::Read::read_to_string(&mut password_file, &mut password).ok()?;
        password = password.trim().to_string();
        Some(password)
    }
    fn load_data(&mut self, volume: &'static VolumeReference, data_dir: &str) {
        let password = TemplateApp::load_data_password(data_dir);

        if !password.is_some() {
            panic!(
                "No password.txt found in data directory {}, attempting access with no password",
                data_dir
            );
        }

        let (sender, receiver) = std::sync::mpsc::channel();
        self.download_notifier = Some(receiver);

        let volume_dir = volume.sub_dir(data_dir);
        let downloader = Downloader::new(&volume_dir, Self::TILE_SERVER, volume, password, sender);
        self.world = Box::new(VolumeGrid64x4Mapped::from_data_dir(&volume_dir, downloader));
        self.data_dir = data_dir.to_string();
    }

    fn select_volume(&mut self, id: usize) {
        self.load_data(&VolumeReference::VOLUMES[id], &self.data_dir.to_string());
    }
    fn selected_volume(&self) -> &'static VolumeReference {
        &VolumeReference::VOLUMES[self.volume_id]
    }

    pub fn clear_textures(&mut self) {
        self.texture_xy = None;
        self.texture_xz = None;
        self.texture_yz = None;
    }

    fn add_scroll_handler(&mut self, image: &Response, ui: &Ui, v: fn(&mut Self) -> &mut i32) {
        if image.hovered() {
            let delta = ui.input(|i| i.scroll_delta);
            let zoom_delta = ui.input(|i| i.zoom_delta());
            if delta.y != 0.0 {
                let min_level = 1 << ((ZOOM_RES_FACTOR / self.zoom) as i32).min(4);
                let delta = delta.y.signum() * min_level as f32;
                let m = v(self);
                *m = (*m + delta as i32) / min_level as i32 * min_level as i32;
                self.clear_textures();
            } else if zoom_delta != 1.0 {
                self.zoom = (self.zoom * zoom_delta).max(0.1).min(6.0);
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

        let (scaling, paint_zoom) = if self.zoom >= 1.0 {
            (self.zoom, 1 as u8)
        } else {
            let next_smaller_pow_of_2 = 2.0f32.powf((self.zoom as f32).log2().floor());
            (
                self.zoom / next_smaller_pow_of_2,
                (1.0 / next_smaller_pow_of_2).round() as u8,
            )
        };
        //println!("scaling: {}, paint_zoom: {}", scaling, paint_zoom);

        let width = (self.frame_width as f32 / scaling) as usize;
        let height = (self.frame_height as f32 / scaling) as usize;
        let mut pixels = vec![0u8; width * height];

        //let q = 1;

        //let mut printed = false;
        let mut xyz: [i32; 3] = [0, 0, 0];
        xyz[d_coord] = self.coord[d_coord];

        let min_level = (32 - ((ZOOM_RES_FACTOR / self.zoom) as u32).leading_zeros())
            .min(4)
            .max(0);
        let max_level: u32 = (min_level + 1).min(4);
        /* let min_level = 0;
        let max_level = 0; */
        for level in (min_level..=max_level).rev() {
            let sfactor = 1 << level as u8;
            //println!("level: {} factor: {}", level, sfactor);
            self.world.paint(
                self.coord,
                u_coord,
                v_coord,
                d_coord,
                width,
                height,
                sfactor,
                paint_zoom,
                &self.drawing_config,
                &mut pixels,
            );
        }

        let image = ColorImage::from_gray([width, height], &pixels);
        //println!("Time elapsed before loading in ({}, {}, {}) is: {:?}", u_coord, v_coord, d_coord, _start.elapsed());
        // Load the texture only once.
        let res = ui.ctx().load_texture("my-image-xy", image, Default::default());

        let _duration = _start.elapsed();
        //println!("Time elapsed in ({}, {}, {}) is: {:?}", u_coord, v_coord, d_coord, _duration);
        res
    }

    fn controls(&mut self, _frame: &eframe::Frame, ui: &mut Ui) {
        ui.label("Volume");
        egui::ComboBox::from_id_source("Volume")
            .selected_text(self.selected_volume().label())
            .show_ui(ui, |ui| {
                // iterate over indices and values of VolumeReference::VOLUMES
                for (id, volume) in VolumeReference::VOLUMES.iter().enumerate() {
                    let res = ui.selectable_value(&mut self.volume_id, id, volume.label());
                    if res.changed() {
                        println!("Selected volume: {}", self.volume_id);
                        self.clear_textures();
                        self.select_volume(self.volume_id);
                    }
                }
            });

        ui.end_row();

        fn slider<T: emath::Numeric>(
            ui: &mut Ui,
            label: &str,
            value: &mut T,
            range: RangeInclusive<T>,
            logarithmic: bool,
        ) -> Response {
            ui.label(label);
            let slider = egui::Slider::new(value, range).clamp_to_range(true);
            let slider = if logarithmic { slider.logarithmic(true) } else { slider };
            let sl = ui.add(slider);
            ui.end_row();
            sl
        }
        let x_sl = slider(ui, "x", &mut self.coord[0], -1000..=10000, false);
        let y_sl = slider(ui, "y", &mut self.coord[1], -1000..=10000, false);
        let z_sl = slider(ui, "z", &mut self.coord[2], 0..=25000, false);
        let zoom_sl = slider(ui, "Zoom", &mut self.zoom, 0.1..=6.0, true);

        ui.collapsing("Filters", |ui| {
            let enable = ui.checkbox(&mut self.drawing_config.enable_filters, "Enable ('F')");
            ui.add_enabled_ui(self.drawing_config.enable_filters, |ui| {
                let min_sl = slider(
                    ui,
                    "Min",
                    &mut self.drawing_config.threshold_min,
                    0..=(254 - self.drawing_config.threshold_max),
                    false,
                );
                let max_sl = slider(
                    ui,
                    "Max",
                    &mut self.drawing_config.threshold_max,
                    0..=(254 - self.drawing_config.threshold_min),
                    false,
                );
                let bits_sl = slider(ui, "Mask Bits", &mut self.drawing_config.quant, 1..=8, false);
                let mask_sl = slider(ui, "Mask Shift", &mut self.drawing_config.mask_shift, 0..=7, false);
                ui.label("Mask");
                ui.label(format!("{:08b}", self.drawing_config.bit_mask()));
                ui.end_row();

                if min_sl.changed() || max_sl.changed() || bits_sl.changed() || mask_sl.changed() {
                    self.clear_textures();
                }
            });
            if enable.changed() {
                self.clear_textures();
            }
        });

        if x_sl.changed() || y_sl.changed() || z_sl.changed() || zoom_sl.changed() {
            self.clear_textures();
        }
    }

    fn try_recv_from_download_notifier(&mut self) -> bool {
        self.download_notifier.as_ref().is_some_and(|x| x.try_recv().is_ok())
    }
}

impl eframe::App for TemplateApp {
    /// Called by the frame work to save state before shutdown.
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, self);
    }

    /// Called each time the UI needs repainting, which may be many times per second.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.try_recv_from_download_notifier() {
            self.clear_textures();

            while self.try_recv_from_download_notifier() {} // clear queue
        }

        egui::Window::new("Controls").show(ctx, |ui| {
            ui.input(|i| {
                if i.key_pressed(egui::Key::F) {
                    self.drawing_config.enable_filters = !self.drawing_config.enable_filters;
                    self.clear_textures();
                }
            });

            egui::Grid::new("my_grid")
                .num_columns(2)
                .spacing([40.0, 4.0])
                .striped(true)
                .show(ui, |ui| {
                    self.controls(_frame, ui);
                });

            ui.separator();

            ui.horizontal(|ui| {
                ui.label("FPS");
                ui.label(format!(
                    "{}",
                    1.0 / (_frame.info().cpu_usage.unwrap_or_default() + 1e-6)
                ));
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let new_size = ui.available_size();
            if new_size != self.last_size {
                self.last_size = new_size;
                self.clear_textures();
            }

            let pane_scaling = if self.zoom >= 1.0 {
                self.zoom
            } else {
                let next_smaller_pow_of_2 = 2.0f32.powf((self.zoom as f32).log2().floor());
                self.zoom / next_smaller_pow_of_2
            };

            // use remaining space for image
            let size = ui.available_size();
            {
                let new_width = size.x as usize / 2 - 10;
                let new_height = size.y as usize / 2 - 10;

                self.frame_width = new_width;
                self.frame_height = new_height;

                let texture_xy = &self.get_or_create_texture(ui, 0, 1, 2, |s| &mut s.texture_xy);
                let texture_xz = &self.get_or_create_texture(ui, 0, 2, 1, |s| &mut s.texture_xz);
                let texture_yz = &self.get_or_create_texture(ui, 2, 1, 0, |s| &mut s.texture_yz);

                let image = Image::new(texture_xy)
                    .max_height(self.frame_height as f32)
                    .max_width(self.frame_width as f32)
                    .fit_to_original_size(pane_scaling);

                let image_xz = Image::new(texture_xz)
                    .max_height(self.frame_height as f32)
                    .max_width(self.frame_width as f32)
                    .fit_to_original_size(pane_scaling);

                let image_yz = Image::new(texture_yz)
                    .max_height(self.frame_height as f32)
                    .max_width(self.frame_width as f32)
                    .fit_to_original_size(pane_scaling);

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

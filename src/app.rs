use std::ops::RangeInclusive;
use std::sync::mpsc::Receiver;

use crate::downloader::*;
use crate::model::*;
use crate::volume::*;

use egui::Vec2;
use egui::{ColorImage, Image, PointerButton, Response, Ui};

const ZOOM_RES_FACTOR: f32 = 1.; // defines which resolution is used for which zoom level, 2 means only when zooming deeper than 2x the full resolution is pulled

#[derive(PartialEq, Eq)]
enum Mode {
    Blocks,
    Cells,
    Layers,
}

#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct TemplateApp {
    #[serde(skip)]
    is_authorized: bool,
    #[serde(skip)]
    credential_entry: (String, String),
    #[serde(skip)]
    last_login_failed: bool,
    volume_id: usize,
    coord: [i32; 3],
    trilinear_interpolation: bool,
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
    world: Box<dyn VoxelPaintVolume>,
    #[serde(skip)]
    last_size: Vec2,
    #[serde(skip)]
    download_notifier: Option<Receiver<()>>,
    drawing_config: DrawingConfig,
    #[serde(skip)]
    ranges: [RangeInclusive<i32>; 3],
    #[serde(skip)]
    ppm_file: Option<String>,
    #[serde(skip)]
    mode: Mode,
    #[serde(skip)]
    extra_resolutions: u32,
}

impl Default for TemplateApp {
    fn default() -> Self {
        Self {
            is_authorized: false,
            credential_entry: ("".to_string(), "".to_string()),
            last_login_failed: false,
            volume_id: 0,
            coord: [2800, 2500, 10852],
            trilinear_interpolation: false,
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
            ranges: [0..=10000, 0..=10000, 0..=15000],
            ppm_file: None,
            mode: Mode::Blocks,
            extra_resolutions: 1,
        }
    }
}

impl TemplateApp {
    const TILE_SERVER: &'static str = "https://vesuvius.virtual-void.net";

    /// Called once before the first frame.
    pub fn new(cc: &eframe::CreationContext<'_>, data_dir: Option<String>, ppm_file: Option<String>) -> Self {
        // This is also where you can customize the look and feel of egui using
        // `cc.egui_ctx.set_visuals` and `cc.egui_ctx.set_fonts`.
        let mut app: TemplateApp = if let Some(storage) = cc.storage {
            eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default()
        } else {
            Default::default()
        };
        if let Some(dir) = data_dir {
            app.data_dir = dir;
        }
        app.ppm_file = ppm_file;

        let contains_cell_files = std::fs::read_dir(&app.data_dir).unwrap().any(|entry| {
            let p = entry.unwrap().path();
            let name = p.file_name().unwrap().to_str().unwrap_or("");
            name.starts_with("cell_yxz_") && name.ends_with(".tif")
        });
        let contains_layer_files = std::fs::read_dir(&app.data_dir).unwrap().any(|entry| {
            let p = entry.unwrap().path();
            let name = p.file_name().unwrap().to_str().unwrap_or("");
            regex::Regex::new(r"(\d{5})\.tif").unwrap().captures(name).is_some()
        });

        let needs_authorization = !contains_cell_files && !contains_layer_files;

        if needs_authorization {
            let pass = Self::load_data_password(&app.data_dir);
            if Downloader::check_authorization(Self::TILE_SERVER, pass) {
                app.is_authorized = true;
            } else {
                app.is_authorized = false;
            }
        } else {
            app.is_authorized = true;
        }

        if app.is_authorized {
            if contains_cell_files {
                app.mode = Mode::Cells;
                app.load_from_cells();
                app.transform_volume();
            } else if contains_layer_files {
                app.mode = Mode::Layers;
                app.load_from_layers();
                app.transform_volume();
            } else {
                app.select_volume(app.volume_id);
            }
        }

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
    fn load_data(&mut self, volume: &'static dyn VolumeReference) {
        let password = TemplateApp::load_data_password(&self.data_dir);

        if !password.is_some() {
            panic!(
                "No password.txt found in data directory {}, attempting access with no password",
                self.data_dir
            );
        }

        let (sender, receiver) = std::sync::mpsc::channel();
        self.download_notifier = Some(receiver);

        let volume_dir = volume.sub_dir(&self.data_dir);

        self.world = {
            let downloader = Downloader::new(&volume_dir, Self::TILE_SERVER, volume, password, sender);
            let v = VolumeGrid64x4Mapped::from_data_dir(&volume_dir, downloader);
            Box::new(v)
        };

        self.transform_volume();
    }
    pub fn is_ppm_mode(&self) -> bool { self.ppm_file.is_some() }
    fn load_from_cells(&mut self) {
        let v = VolumeGrid500Mapped::from_data_dir(&self.data_dir);
        self.world = Box::new(v);
        self.extra_resolutions = 0;
    }
    fn load_from_layers(&mut self) {
        let v = LayersMappedVolume::from_data_dir(&self.data_dir);
        self.world = Box::new(v);
        self.extra_resolutions = 0;
    }

    fn transform_volume(&mut self) {
        if let Some(ppm_file) = &self.ppm_file {
            let old = std::mem::replace(&mut self.world, Box::new(EmptyVolume {}));
            let base = if self.trilinear_interpolation {
                Box::new(TrilinearInterpolatedVolume { base: old })
            } else {
                old
            };
            let ppm = PPMVolume::new(&ppm_file, base);
            let width = ppm.width() as i32;
            let height = ppm.height() as i32;
            println!("Loaded PPM volume with size {}x{}", width, height);

            self.world = Box::new(ppm);
            self.ranges = [0..=width, 0..=height, -30..=30];

            if self.coord[0] < 0 || self.coord[0] > width {}
            if !self.ranges[0].contains(&self.coord[0])
                || !self.ranges[1].contains(&self.coord[1])
                || !self.ranges[2].contains(&self.coord[2])
            {
                self.coord = [width / 2, height / 2, 0];
            }
        }
    }

    fn select_volume(&mut self, id: usize) {
        if self.ppm_file.is_some() {
            self.volume_id = 0;
            self.load_data(&FullVolumeReference::SCROLL1);
        } else {
            self.volume_id = id;
            self.load_data(<dyn VolumeReference>::VOLUMES[id]);
        }
    }
    fn selected_volume(&self) -> &'static dyn VolumeReference { <dyn VolumeReference>::VOLUMES[self.volume_id] }

    pub fn clear_textures(&mut self) {
        self.texture_xy = None;
        self.texture_xz = None;
        self.texture_yz = None;
    }

    fn add_scroll_handler(&mut self, image: &Response, ui: &Ui, coord: usize) {
        if image.hovered() {
            let delta = ui.input(|i| i.raw_scroll_delta);
            let zoom_delta = ui.input(|i| i.zoom_delta());
            if delta.y != 0.0 {
                let min_level = 1 << ((ZOOM_RES_FACTOR / self.zoom) as i32).min(4);
                let delta = delta.y.signum() * min_level as f32;
                let m = &mut self.coord[coord];
                *m = ((*m + delta as i32) / min_level as i32 * min_level as i32)
                    .clamp(*self.ranges[coord].start(), *self.ranges[coord].end());
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

            self.coord[ucoord] =
                (self.coord[ucoord] + delta.x as i32).clamp(*self.ranges[ucoord].start(), *self.ranges[ucoord].end());
            self.coord[vcoord] =
                (self.coord[vcoord] + delta.y as i32).clamp(*self.ranges[vcoord].start(), *self.ranges[vcoord].end());
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
        let max_level: u32 = (min_level + self.extra_resolutions).min(4);
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
        let res = ui
            .ctx()
            .load_texture(format!("{}{}{}", u_coord, v_coord, d_coord), image, Default::default());

        let _duration = _start.elapsed();
        //println!("Time elapsed in ({}, {}, {}) is: {:?}", u_coord, v_coord, d_coord, _duration);
        res
    }

    fn controls(&mut self, _frame: &eframe::Frame, ui: &mut Ui) {
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

        egui::Grid::new("my_grid")
            .num_columns(2)
            .spacing([40.0, 4.0])
            .show(ui, |ui| {
                ui.label("Volume");
                if self.is_ppm_mode() {
                    ui.label("Fixed to Scroll 1 in PPM mode");
                } else if self.mode == Mode::Cells {
                    ui.label(format!("Browsing cells in {}", self.data_dir));
                } else if self.mode == Mode::Layers {
                    ui.label(format!("Browsing layers in {}", self.data_dir));
                } else {
                    ui.add_enabled_ui(!self.is_ppm_mode(), |ui| {
                        egui::ComboBox::from_id_source("Volume")
                            .selected_text(self.selected_volume().label())
                            .show_ui(ui, |ui| {
                                // iterate over indices and values of VolumeReference::VOLUMES
                                for (id, volume) in <dyn VolumeReference>::VOLUMES.iter().enumerate() {
                                    let res = ui.selectable_value(&mut self.volume_id, id, volume.label());
                                    if res.changed() {
                                        println!("Selected volume: {}", self.volume_id);
                                        self.clear_textures();
                                        self.select_volume(self.volume_id);
                                        self.zoom = 0.25;
                                    }
                                }
                            });
                    });
                }
                ui.end_row();
                let x_sl = slider(ui, "x", &mut self.coord[0], self.ranges[0].clone(), false);
                let y_sl = slider(ui, "y", &mut self.coord[1], self.ranges[1].clone(), false);
                let z_sl = slider(ui, "z", &mut self.coord[2], self.ranges[2].clone(), false);
                let zoom_sl = slider(ui, "Zoom", &mut self.zoom, 0.1..=6.0, true);

                if self.is_ppm_mode() {
                    ui.label("Trilinear interpolation ('I')");
                    let c = ui.checkbox(&mut self.trilinear_interpolation, "");
                    if c.changed() {
                        self.load_data(self.selected_volume());
                        self.clear_textures();
                    }
                    ui.end_row()
                }

                if x_sl.changed() || y_sl.changed() || z_sl.changed() || zoom_sl.changed() {
                    self.clear_textures();
                }
            });

        ui.collapsing("Filters", |ui| {
            let enable = ui.checkbox(&mut self.drawing_config.enable_filters, "Enable ('F')");
            ui.add_enabled_ui(self.drawing_config.enable_filters, |ui| {
                egui::Grid::new("my_grid")
                    .num_columns(2)
                    .spacing([40.0, 4.0])
                    .show(ui, |ui| {
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
            });
            if enable.changed() {
                self.clear_textures();
            }
        });
    }

    fn try_recv_from_download_notifier(&mut self) -> bool {
        self.download_notifier.as_ref().is_some_and(|x| x.try_recv().is_ok())
    }

    fn update_main(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
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
                if i.key_pressed(egui::Key::I) {
                    self.trilinear_interpolation = !self.trilinear_interpolation;
                    self.load_data(self.selected_volume());
                    self.clear_textures();
                }
            });

            self.controls(_frame, ui);

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
                    self.add_scroll_handler(&im_xy, ui, 2);
                    self.add_scroll_handler(&im_xz, ui, 1);
                    self.add_drag_handler(&im_xy, 0, 1);
                    self.add_drag_handler(&im_xz, 0, 2);
                });
                let im_yz = ui.add(image_yz).interact(egui::Sense::drag());
                self.add_scroll_handler(&im_yz, ui, 0);
                self.add_drag_handler(&im_yz, 2, 1);
            };
        });
    }
    fn update_password_entry(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let mut should_try_auth = false;

        egui::Window::new("Login").show(ctx, |ui| {
            egui::Grid::new("login_grid")
                .num_columns(2)
                .spacing([40.0, 4.0])
                .show(ui, |ui| {
                    use egui::text::CCursor;
                    use egui::text::CCursorRange;
                    use egui::Id;

                    let user_id = Id::new("user");
                    let pass_id = Id::new("pass");

                    ui.label("Username");
                    let mut user_field = egui::TextEdit::singleline(&mut self.credential_entry.0)
                        .id(user_id)
                        .show(ui);

                    if user_field.response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                        if self.credential_entry.1.is_empty() {
                            ui.memory_mut(|m| m.request_focus(pass_id));
                        } else {
                            should_try_auth = true;
                        }
                    }
                    ui.end_row();

                    ui.label("Password");
                    let r = ui.add(
                        egui::TextEdit::singleline(&mut self.credential_entry.1)
                            .id(pass_id)
                            .password(true),
                    );
                    if r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                        if !self.credential_entry.0.is_empty() && !self.credential_entry.1.is_empty() {
                            should_try_auth = true;
                        }
                    }
                    ui.end_row();

                    ui.label("");
                    ui.label("Use same credentials as for the data server");
                    ui.end_row();

                    should_try_auth = should_try_auth || ui.button("Login").clicked();
                    if should_try_auth {
                        let credentials = format!("{}:{}", self.credential_entry.0, self.credential_entry.1);
                        if Downloader::check_authorization(Self::TILE_SERVER, Some(credentials.clone())) {
                            std::fs::write(format!("{}/password.txt", self.data_dir), credentials).unwrap();
                            self.is_authorized = true;
                            self.last_login_failed = false;
                            self.select_volume(self.volume_id);
                        } else {
                            self.last_login_failed = true;

                            ui.memory_mut(|m| m.request_focus(user_id));
                            user_field.state.set_ccursor_range(Some(CCursorRange::two(
                                CCursor::new(0),
                                CCursor::new(self.credential_entry.0.len()),
                            )));
                        }
                    }
                    if self.last_login_failed {
                        ui.colored_label(egui::Color32::RED, "Login failed.");
                    }

                    if self.credential_entry.0.is_empty() && self.credential_entry.1.is_empty() {
                        ui.memory_mut(|m| m.request_focus(user_id));
                    }
                });
        });
    }
}

impl eframe::App for TemplateApp {
    /// Called by the frame work to save state before shutdown.
    fn save(&mut self, storage: &mut dyn eframe::Storage) { eframe::set_value(storage, eframe::APP_KEY, self); }
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        if self.is_authorized {
            self.update_main(ctx, frame);
        } else {
            self.update_password_entry(ctx, frame);
        }
    }
}

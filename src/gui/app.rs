use std::ops::RangeInclusive;
use std::sync::mpsc::Receiver;

use crate::downloader::*;
use crate::model::*;
use crate::volume::*;

use directories::BaseDirs;
use egui::Vec2;
use egui::{ColorImage, Image, PointerButton, Response, Ui};
use std::cell::RefCell;
use std::mem;
use std::sync::Arc;

const ZOOM_RES_FACTOR: f32 = 1.3; // defines which resolution is used for which zoom level, 2 means only when zooming deeper than 2x the full resolution is pulled

#[derive(PartialEq, Eq)]
enum Mode {
    Blocks,
    Cells,
    Layers,
}

#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct SegmentMode {
    coord: [i32; 3],
    info: String,
    filename: String,
    #[serde(skip)]
    ranges: [RangeInclusive<i32>; 3],
    #[serde(skip)]
    world: Arc<RefCell<dyn VoxelPaintVolume>>,
    #[serde(skip)]
    texture_uv: Option<egui::TextureHandle>,
    #[serde(skip)]
    convert_to_world_coords: Box<dyn Fn([i32; 3]) -> [i32; 3]>,
}

impl Default for SegmentMode {
    fn default() -> Self {
        Self {
            coord: [0, 0, 0],
            info: "".to_string(),
            filename: "".to_string(),
            ranges: [0..=1000, 0..=1000, -40..=40],
            world: Arc::new(RefCell::new(EmptyVolume {})),
            texture_uv: None,
            convert_to_world_coords: Box::new(|x| x),
        }
    }
}

#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct TemplateApp {
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
    world: Arc<RefCell<dyn VoxelPaintVolume>>,
    #[serde(skip)]
    last_size: Vec2,
    #[serde(skip)]
    download_notifier: Option<Receiver<()>>,
    drawing_config: DrawingConfig,
    #[serde(skip)]
    ranges: [RangeInclusive<i32>; 3],
    /* #[serde(skip)]
    ppm_file: Option<String>,
    #[serde(skip)]
    obj_file: Option<String>, */
    #[serde(skip)]
    mode: Mode,
    #[serde(skip)]
    extra_resolutions: u32,
    //#[serde(skip)]
    segment_mode: Option<SegmentMode>,
}

impl Default for TemplateApp {
    fn default() -> Self {
        Self {
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
            world: Arc::new(RefCell::new(EmptyVolume {})),
            last_size: Vec2::ZERO,
            download_notifier: None,
            drawing_config: Default::default(),
            ranges: [0..=10000, 0..=10000, 0..=15000],
            //ppm_file: None,
            //obj_file: None,
            mode: Mode::Blocks,
            extra_resolutions: 1,
            segment_mode: None,
        }
    }
}

impl TemplateApp {
    const TILE_SERVER: &'static str = "https://vesuvius.virtual-void.net";
    //const TILE_SERVER: &'static str = "http://localhost:8095";

    /// Called once before the first frame.
    pub fn new(cc: &eframe::CreationContext<'_>, data_dir: Option<String>, segment_file: Option<String>) -> Self {
        // This is also where you can customize the look and feel of egui using
        // `cc.egui_ctx.set_visuals` and `cc.egui_ctx.set_fonts`.
        let mut app: TemplateApp = if let Some(storage) = cc.storage {
            eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default()
        } else {
            Default::default()
        };
        if let Some(dir) = data_dir {
            app.data_dir = dir;
        } else {
            let dir = BaseDirs::new().unwrap().cache_dir().join("vesuvius-gui");
            app.data_dir = dir.to_str().unwrap().to_string();
            println!("Using default data directory: {}", app.data_dir);

            // make sure dir exists
            std::fs::create_dir_all(&app.data_dir).unwrap();
        }

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

        if contains_cell_files {
            app.mode = Mode::Cells;
            app.load_from_cells();
        } else if contains_layer_files {
            app.mode = Mode::Layers;
            app.load_from_layers();
        } else {
            app.select_volume(app.volume_id);
        }

        if let Some(segment_file) = segment_file {
            app.setup_segment(&segment_file);
        }

        app
    }

    fn reload_segment(&mut self) {
        if let Some(segment_mode) = self.segment_mode.as_ref() {
            self.setup_segment(&segment_mode.filename.clone());
        }
    }

    fn setup_segment(&mut self, segment_file: &str) {
        if segment_file.ends_with(".ppm") {
            let mut segment: SegmentMode = self.segment_mode.take().unwrap_or_default();
            let old: Arc<RefCell<dyn VoxelPaintVolume>> = self.world.clone();
            let base = if self.trilinear_interpolation {
                Arc::new(RefCell::new(TrilinearInterpolatedVolume { base: old }))
            } else {
                old
            };
            let ppm = PPMVolume::new(segment_file, base);
            let width = ppm.width() as i32;
            let height = ppm.height() as i32;
            let ppm = Arc::new(RefCell::new(ppm));
            let ppm2 = ppm.clone();
            println!("Loaded PPM volume with size {}x{}", width, height);

            if segment.filename != segment_file {
                segment.coord = [width / 2, height / 2, 0];
                segment.filename = segment_file.to_string();
                segment.info = segment_file.to_string();
            }
            segment.ranges = [0..=width, 0..=height, -40..=40];
            segment.world = ppm;
            segment.convert_to_world_coords = Box::new(move |coord| ppm2.borrow().convert_to_world_coords(coord));

            self.segment_mode = Some(segment)
        } else if segment_file.ends_with(".obj") {
            let mut segment: SegmentMode = self.segment_mode.take().unwrap_or_default();
            let old: Arc<RefCell<dyn VoxelPaintVolume>> = self.world.clone();
            let base = if self.trilinear_interpolation {
                Arc::new(RefCell::new(TrilinearInterpolatedVolume { base: old }))
            } else {
                old
            };
            let obj_volume = ObjVolume::new(&segment_file, base);
            let width = obj_volume.width() as i32;
            let height = obj_volume.height() as i32;

            let volume = Arc::new(RefCell::new(obj_volume));
            let obj2 = volume.clone();
            println!("Loaded Obj volume with size {}x{}", width, height);

            if segment.filename != segment_file {
                segment.coord = [width / 2, height / 2, 0];
                segment.filename = segment_file.to_string();
                segment.info = segment_file.to_string();
            }
            segment.ranges = [0..=width, 0..=height, -40..=40];
            segment.world = volume;
            segment.convert_to_world_coords = Box::new(move |coords| obj2.borrow().convert_to_volume_coords(coords));

            self.segment_mode = Some(segment)
        }
    }

    fn load_data(&mut self, volume: &'static dyn VolumeReference) {
        let (sender, receiver) = std::sync::mpsc::channel();
        self.download_notifier = Some(receiver);

        let volume_dir = volume.sub_dir(&self.data_dir);

        self.world = {
            let downloader = Downloader::new(&volume_dir, Self::TILE_SERVER, volume, None, sender);
            let v = VolumeGrid64x4Mapped::from_data_dir(&volume_dir, downloader);
            Arc::new(RefCell::new(v))
        };
    }
    //pub fn is_ppm_mode(&self) -> bool { self.ppm_file.is_some() }
    //pub fn is_obj_mode(&self) -> bool { self.obj_file.is_some() }
    pub fn is_segment_mode(&self) -> bool { self.segment_mode.is_some() }
    fn load_from_cells(&mut self) {
        let v = VolumeGrid500Mapped::from_data_dir(&self.data_dir);
        self.world = Arc::new(RefCell::new(v));
        self.extra_resolutions = 0;
    }
    fn load_from_layers(&mut self) {
        let v = LayersMappedVolume::from_data_dir(&self.data_dir);
        self.world = Arc::new(RefCell::new(v));
        self.extra_resolutions = 0;
    }

    fn select_volume(&mut self, id: usize) {
        if self.is_segment_mode() {
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

        if let Some(segment_mode) = self.segment_mode.as_mut() {
            segment_mode.texture_uv = None;
            self.sync_coords();
        }
    }

    fn add_scroll_handler(&mut self, image: &Response, ui: &Ui, coord: usize, segment_pane: bool) {
        let (coords, ranges) = if segment_pane {
            let ranges = self.segment_mode.as_ref().unwrap().ranges.clone();
            (&mut self.segment_mode.as_mut().unwrap().coord, ranges)
        } else {
            (&mut self.coord, self.ranges.clone())
        };

        if image.hovered() {
            let delta = ui.input(|i| i.smooth_scroll_delta);
            let zoom_delta = ui.input(|i| i.zoom_delta());

            if zoom_delta != 1.0 {
                self.zoom = (self.zoom * zoom_delta).max(0.1).min(6.0);
                self.clear_textures();
            } else if delta.y != 0.0 {
                let min_level = 1 << ((ZOOM_RES_FACTOR / self.zoom) as i32).min(4);
                let delta = delta.y.signum() * min_level as f32;
                let m = &mut coords[coord];
                *m = ((*m + delta as i32) / min_level as i32 * min_level as i32)
                    .clamp(*ranges[coord].start(), *ranges[coord].end());
                self.clear_textures();
            }
        }
    }
    fn add_drag_handler(&mut self, image: &Response, ucoord: usize, vcoord: usize, segment_pane: bool) {
        let coords = if segment_pane {
            &mut self.segment_mode.as_mut().unwrap().coord
        } else {
            &mut self.coord
        };

        if image.dragged_by(PointerButton::Primary) {
            //let im2 = image.on_hover_cursor(CursorIcon::Grabbing);
            let delta = -image.drag_delta() / self.zoom;

            coords[ucoord] =
                (coords[ucoord] + delta.x as i32).clamp(*self.ranges[ucoord].start(), *self.ranges[ucoord].end());
            coords[vcoord] =
                (coords[vcoord] + delta.y as i32).clamp(*self.ranges[vcoord].start(), *self.ranges[vcoord].end());
            self.clear_textures();
        }
    }
    fn get_or_create_texture(
        &mut self,
        ui: &Ui,
        u_coord: usize,
        v_coord: usize,
        d_coord: usize,
        segment_pane: bool,
        t: fn(&mut Self) -> &mut Option<egui::TextureHandle>,
    ) -> egui::TextureHandle {
        if let Some(texture) = t(self) {
            texture.clone()
        } else {
            let res = self.create_texture(ui, u_coord, v_coord, d_coord, segment_pane);
            *t(self) = Some(res.clone());
            res
        }
    }
    fn create_texture(
        &mut self,
        ui: &Ui,
        u_coord: usize,
        v_coord: usize,
        d_coord: usize,
        segment_pane: bool,
    ) -> egui::TextureHandle {
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

        let (coords, world) = if !segment_pane {
            (self.coord, self.world.clone())
        } else {
            (
                self.segment_mode.as_ref().unwrap().coord,
                self.segment_mode.as_ref().unwrap().world.clone(),
            )
        };

        let mut xyz: [i32; 3] = [0, 0, 0];
        xyz[d_coord] = coords[d_coord];

        let min_level = (32 - ((ZOOM_RES_FACTOR / self.zoom) as u32).leading_zeros())
            .min(4)
            .max(0);
        let max_level: u32 = (min_level + self.extra_resolutions).min(4);
        /* let min_level = 0;
        let max_level = 0; */
        for level in (min_level..=max_level).rev() {
            let sfactor = 1 << level as u8;
            //println!("level: {} factor: {}", level, sfactor);
            world.borrow_mut().paint(
                coords,
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
        let texture_id = ui
            .ctx()
            .load_texture(format!("{}{}{}", u_coord, v_coord, d_coord), image, Default::default());

        let _duration = _start.elapsed();
        /* println!(
            "Time elapsed in segment: {segment_pane} ({}, {}, {}) is: {:?}",
            u_coord, v_coord, d_coord, _duration
        ); */
        texture_id
    }

    fn sync_coords(&mut self) {
        if let Some(segment_mode) = self.segment_mode.as_ref() {
            let res = (*segment_mode.convert_to_world_coords)(segment_mode.coord);
            if res[0] >= 0 && res[1] >= 0 && res[2] >= 0 {
                self.coord = res;
            }
        }
    }

    fn controls(&mut self, _frame: &eframe::Frame, ui: &mut Ui) {
        fn slider<T: emath::Numeric>(
            ui: &mut Ui,
            label: &str,
            value: &mut T,
            range: RangeInclusive<T>,
            logarithmic: bool,
            enabled: bool,
        ) -> Response {
            ui.label(label);
            let slider = egui::Slider::new(value, range).clamp_to_range(true);
            let slider = if logarithmic { slider.logarithmic(true) } else { slider };
            let sl = ui.add_enabled(enabled, slider);
            ui.end_row();
            sl
        }

        egui::Grid::new("my_grid")
            .num_columns(2)
            .spacing([40.0, 4.0])
            .show(ui, |ui| {
                ui.label("Volume");
                if self.is_segment_mode() {
                    ui.label("Fixed to Scroll 1 in segment mode");
                } else if self.mode == Mode::Cells {
                    ui.label(format!("Browsing cells in {}", self.data_dir));
                } else if self.mode == Mode::Layers {
                    ui.label(format!("Browsing layers in {}", self.data_dir));
                } else {
                    ui.add_enabled_ui(!self.is_segment_mode(), |ui| {
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
                let is_segment_mode = self.is_segment_mode();
                let x_sl = slider(
                    ui,
                    "x",
                    &mut self.coord[0],
                    self.ranges[0].clone(),
                    false,
                    !is_segment_mode,
                );
                let y_sl = slider(
                    ui,
                    "y",
                    &mut self.coord[1],
                    self.ranges[1].clone(),
                    false,
                    !is_segment_mode,
                );
                let z_sl = slider(
                    ui,
                    "z",
                    &mut self.coord[2],
                    self.ranges[2].clone(),
                    false,
                    !is_segment_mode,
                );

                let has_changed = if let Some(segment_mode) = self.segment_mode.as_mut() {
                    let u_sl = slider(
                        ui,
                        "u",
                        &mut segment_mode.coord[0],
                        segment_mode.ranges[0].clone(),
                        false,
                        true,
                    );
                    let v_sl = slider(
                        ui,
                        "v",
                        &mut segment_mode.coord[1],
                        segment_mode.ranges[1].clone(),
                        false,
                        true,
                    );
                    let w_sl = slider(
                        ui,
                        "w",
                        &mut segment_mode.coord[2],
                        segment_mode.ranges[2].clone(),
                        false,
                        true,
                    );

                    u_sl.changed() || v_sl.changed() || w_sl.changed()
                } else {
                    false
                };

                let zoom_sl = slider(ui, "Zoom", &mut self.zoom, 0.1..=6.0, true, true);

                if self.is_segment_mode() {
                    ui.label("Trilinear interpolation ('I')");
                    let c = ui.checkbox(&mut self.trilinear_interpolation, "");
                    if c.changed() {
                        self.reload_segment();
                        self.clear_textures();
                    }
                    ui.end_row()
                }

                if x_sl.changed() || y_sl.changed() || z_sl.changed() || zoom_sl.changed() || has_changed {
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
                            true,
                        );
                        let max_sl = slider(
                            ui,
                            "Max",
                            &mut self.drawing_config.threshold_max,
                            0..=(254 - self.drawing_config.threshold_min),
                            false,
                            true,
                        );
                        let bits_sl = slider(ui, "Mask Bits", &mut self.drawing_config.quant, 1..=8, false, true);
                        let mask_sl = slider(
                            ui,
                            "Mask Shift",
                            &mut self.drawing_config.mask_shift,
                            0..=7,
                            false,
                            true,
                        );
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
                    self.reload_segment();
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

                let texture_xy = &self.get_or_create_texture(ui, 0, 1, 2, false, |s| &mut s.texture_xy);
                let texture_xz = &self.get_or_create_texture(ui, 0, 2, 1, false, |s| &mut s.texture_xz);
                let texture_yz = &self.get_or_create_texture(ui, 2, 1, 0, false, |s| &mut s.texture_yz);

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
                    if !self.is_segment_mode() {
                        self.add_scroll_handler(&im_xy, ui, 2, false);
                        self.add_drag_handler(&im_xy, 0, 1, false);
                    }

                    let im_xz = ui.add(image_xz).interact(egui::Sense::drag());
                    if !self.is_segment_mode() {
                        self.add_scroll_handler(&im_xz, ui, 1, false);
                        self.add_drag_handler(&im_xz, 0, 2, false);
                    }
                });
                ui.horizontal(|ui| {
                    let im_yz = ui.add(image_yz).interact(egui::Sense::drag());
                    if !self.is_segment_mode() {
                        self.add_scroll_handler(&im_yz, ui, 0, false);
                        self.add_drag_handler(&im_yz, 2, 1, false);
                    }

                    if self.is_segment_mode() {
                        let texture_uv = &self.get_or_create_texture(ui, 0, 1, 2, true, |s| {
                            &mut s.segment_mode.as_mut().unwrap().texture_uv
                        });
                        let image_uv = Image::new(texture_uv)
                            .max_height(self.frame_height as f32)
                            .max_width(self.frame_width as f32)
                            .fit_to_original_size(pane_scaling);
                        let im_uv = ui.add(image_uv).interact(egui::Sense::drag());
                        self.add_scroll_handler(&im_uv, ui, 2, true);
                        self.add_drag_handler(&im_uv, 0, 1, true);
                    }
                });
            };
        });
    }
}

impl eframe::App for TemplateApp {
    /// Called by the frame work to save state before shutdown.
    fn save(&mut self, storage: &mut dyn eframe::Storage) { eframe::set_value(storage, eframe::APP_KEY, self); }
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) { self.update_main(ctx, frame); }
}

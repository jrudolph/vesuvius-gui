use crate::catalog::obj_repository::ObjRepository;
use crate::catalog::Catalog;
use crate::catalog::Segment;
use crate::gui::{PaneType, VolumePane};
use crate::model::*;
use crate::volume::*;
use crate::zarr::ZarrArray;
use directories::BaseDirs;
use egui::Color32;
use egui::Label;
use egui::RichText;
use egui::SliderClamping;
use egui::Stroke;
use egui::Vec2;
use egui::WidgetText;
use egui::{Response, Ui, Widget};
use egui_extras::Column;
use egui_extras::TableBuilder;
use std::ops::RangeInclusive;
use std::sync::mpsc::Receiver;
use std::sync::mpsc::Sender;
use std::sync::Arc;

#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct SegmentMode {
    coord: [i32; 3],
    info: String,
    filename: String,
    width: usize,
    height: usize,
    #[serde(skip)]
    ranges: [RangeInclusive<i32>; 3],
    #[serde(skip)]
    world: Volume,
    // This is the same reference as `world`. We need to add it just because upcasting between SurfaceVolume and VoxelPaintVolume is so hard.
    // TODO: remove when there's a better way to upcast
    #[serde(skip)]
    surface_volume: Arc<dyn SurfaceVolume>,
    #[serde(skip)]
    uv_pane: VolumePane,
    #[serde(skip)]
    convert_to_world_coords: Box<dyn Fn([i32; 3]) -> [i32; 3]>,
}

impl Default for SegmentMode {
    fn default() -> Self {
        Self {
            coord: [0, 0, 0],
            info: "".to_string(),
            filename: "".to_string(),
            width: 1000,
            height: 1000,
            ranges: [0..=1000, 0..=1000, -40..=40],
            world: EmptyVolume {}.into_volume(),
            surface_volume: Arc::new(EmptyVolume {}),
            uv_pane: VolumePane::new(PaneType::UV, true),
            convert_to_world_coords: Box::new(|x| x),
        }
    }
}

enum UINotification {
    ObjDownloadReady(Segment),
}

pub struct ObjFileConfig {
    pub obj_file: String,
    pub width: usize,
    pub height: usize,
}

pub struct VesuviusConfig {
    pub data_dir: Option<String>,
    pub obj_file: Option<ObjFileConfig>,
    pub overlay_dir: Option<String>,
    pub volume: Option<NewVolumeReference>,
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone, Copy, PartialEq, Eq)]
enum GuiLayout {
    Grid,
    XY,
    XZ,
    YZ,
    UV,
}

#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct TemplateApp {
    #[serde(skip)]
    last_login_failed: bool,
    volume_id: usize,
    coord: [i32; 3],
    zoom: f32,
    data_dir: String,
    #[serde(skip)]
    world: Volume,
    #[serde(skip)]
    download_notifier: Option<Receiver<(usize, usize, usize, Quality)>>,
    drawing_config: DrawingConfig,
    sync_coordinates: bool,
    show_overlay: bool,
    #[serde(skip)]
    ranges: [RangeInclusive<i32>; 3],
    #[serde(skip)]
    extra_resolutions: u32,
    #[serde(skip)]
    segment_mode: Option<SegmentMode>,
    #[serde(skip)]
    catalog: Catalog,
    #[serde(skip)]
    obj_repository: ObjRepository,
    #[serde(skip)]
    selected_segment: Option<Segment>,
    #[serde(skip)]
    downloading_segment: Option<Segment>,
    #[serde(skip)]
    notification_sender: Sender<UINotification>,
    #[serde(skip)]
    notification_receiver: Receiver<UINotification>,
    #[serde(skip)]
    overlay: Option<Box<dyn PaintVolume>>,
    catalog_panel_open: bool,
    layout: GuiLayout,
}

impl Default for TemplateApp {
    fn default() -> Self {
        let catalog = Catalog::default();
        let obj_repository = ObjRepository::new(&catalog);
        let (notification_sender, notification_receiver) = std::sync::mpsc::channel();
        Self {
            last_login_failed: false,
            volume_id: 0,
            coord: [2800, 2500, 10852],
            zoom: 1f32,
            data_dir: ".".to_string(),
            world: EmptyVolume {}.into_volume(),
            download_notifier: None,
            drawing_config: Default::default(),
            sync_coordinates: true,
            show_overlay: true,
            ranges: [0..=20000, 0..=20000, 0..=30000],
            extra_resolutions: 1,
            segment_mode: None,
            catalog,
            obj_repository,
            selected_segment: None,
            downloading_segment: None,
            notification_sender,
            notification_receiver,
            overlay: None,
            catalog_panel_open: true,
            layout: GuiLayout::Grid,
        }
    }
}

impl TemplateApp {
    /// Called once before the first frame.
    pub fn new(cc: &eframe::CreationContext<'_>, catalog: Catalog, config: VesuviusConfig) -> Self {
        // This is also where you can customize the look and feel of egui using
        // `cc.egui_ctx.set_visuals` and `cc.egui_ctx.set_fonts`.
        let mut app: TemplateApp = if let Some(storage) = cc.storage {
            eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default()
        } else {
            Default::default()
        };
        app.obj_repository = ObjRepository::new(&catalog);
        app.catalog = catalog;
        if let Some(dir) = config.data_dir {
            app.data_dir = dir;
        } else {
            let dir = BaseDirs::new().unwrap().cache_dir().join("vesuvius-gui");
            app.data_dir = dir.to_str().unwrap().to_string();
            println!("Using default data directory: {}", app.data_dir);

            // make sure dir exists
            std::fs::create_dir_all(&app.data_dir).unwrap();
        }

        if let Some(volume) = config.volume {
            app.load_volume(&volume);
        } else {
            app.select_volume(app.volume_id);
        }

        if let Some(ObjFileConfig {
            obj_file,
            width,
            height,
        }) = config.obj_file
        {
            app.setup_segment(&obj_file, width, height);
        }

        if let Some(segment_file) = config.overlay_dir {
            if segment_file.contains(".zarr") {
                app.overlay = Some({
                    if segment_file.starts_with("http") {
                        println!("Loading zarr from url: {}", segment_file);
                        Box::new(
                            ZarrArray::from_url_to_default_cache_dir(&segment_file)
                                .into_ctx()
                                .into_ctx(),
                        )
                        // TODO: autodetect or allow to choose whether to use ome-zarr or zarr
                        /* Box::new(OmeZarrContext::<FourColors>::from_url_to_default_cache_dir(
                            &segment_file,
                        )) */
                    } else {
                        Box::new(ZarrArray::from_path(&segment_file).into_ctx().into_ctx())
                    }
                });
            }
        }

        app
    }

    fn setup_segment(&mut self, segment_file: &str, width: usize, height: usize) {
        if segment_file.ends_with(".ppm") {
            let mut segment: SegmentMode = self.segment_mode.take().unwrap_or_default();
            let old = self.world.clone();
            let base = old;
            let ppm = PPMVolume::new(segment_file, base);
            let width = ppm.width() as i32;
            let height = ppm.height() as i32;
            let ppm = Arc::new(ppm);
            let ppm2 = ppm.clone();
            println!("Loaded PPM volume with size {}x{}", width, height);

            if segment.filename != segment_file {
                segment.coord = [width / 2, height / 2, 0];
                segment.filename = segment_file.to_string();
                segment.info = segment_file.to_string();
            }
            segment.width = width as usize;
            segment.height = height as usize;
            segment.ranges = [0..=width, 0..=height, -40..=40];
            segment.world = Volume::from_ref(ppm.clone());
            //segment.surface_volume = ppm;
            segment.convert_to_world_coords = Box::new(move |coord| ppm2.convert_to_world_coords(coord));

            self.segment_mode = Some(segment)
        } else if segment_file.ends_with(".obj") {
            let mut segment: SegmentMode = self.segment_mode.take().unwrap_or_default();
            let old = self.world.clone();
            let base = old;
            let obj_volume = ObjVolume::load_from_obj(&segment_file, base, width, height);
            let width = obj_volume.width() as i32;
            let height = obj_volume.height() as i32;

            let volume = Arc::new(obj_volume);
            let obj2 = volume.clone();
            println!("Loaded Obj volume with size {}x{}", width, height);

            if segment.filename != segment_file {
                segment.coord = [width / 2, height / 2, 0];
                segment.filename = segment_file.to_string();
                segment.info = segment_file.to_string();
            }
            segment.width = width as usize;
            segment.height = height as usize;
            segment.ranges = [0..=width, 0..=height, -40..=40];
            segment.world = Volume::from_ref(volume.clone());
            segment.surface_volume = volume;
            segment.convert_to_world_coords = Box::new(move |coords| obj2.convert_to_volume_coords(coords));

            self.segment_mode = Some(segment)
        }
    }

    fn load_volume(&mut self, volume: &NewVolumeReference) {
        let params = VolumeCreationParams {
            cache_dir: self.data_dir.clone(),
        };
        self.world = volume.volume(&params);
    }

    fn load_volume_by_ref(&mut self, volume_ref: &dyn VolumeReference) {
        let new_vol = NewVolumeReference::Volume64x4(volume_ref.owned());
        self.load_volume(&new_vol);
    }

    pub fn is_segment_mode(&self) -> bool {
        self.segment_mode.is_some()
    }

    fn select_volume(&mut self, id: usize) {
        if self.is_segment_mode() {
            self.volume_id = 0;
            self.load_volume_by_ref(&FullVolumeReference::SCROLL1);
        } else {
            self.volume_id = id;
            self.load_volume_by_ref(<dyn VolumeReference>::VOLUMES[id]);
        }
    }

    fn selected_volume(&self) -> &'static dyn VolumeReference {
        <dyn VolumeReference>::VOLUMES[self.volume_id]
    }

    fn sync_coords(&mut self) {
        if let Some(segment_mode) = self.segment_mode.as_ref() {
            if self.sync_coordinates {
                let res = (*segment_mode.convert_to_world_coords)(segment_mode.coord);
                if res[0] >= 0 && res[1] >= 0 && res[2] >= 0 {
                    self.coord = res;
                }
            }
        }
    }
    fn should_sync_coords(&self) -> bool {
        self.segment_mode.is_some() && self.sync_coordinates
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
            let slider = egui::Slider::new(value, range).clamping(SliderClamping::Always);
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
                    if ui.button("Unload segment").clicked() {
                        self.segment_mode = None;
                        if self.layout == GuiLayout::UV {
                            self.layout = GuiLayout::Grid;
                        }
                    }
                } else {
                    ui.add_enabled_ui(!self.is_segment_mode(), |ui| {
                        egui::ComboBox::from_id_salt("Volume")
                            .selected_text(self.selected_volume().label())
                            .show_ui(ui, |ui| {
                                // iterate over indices and values of VolumeReference::VOLUMES
                                for (id, volume) in <dyn VolumeReference>::VOLUMES.iter().enumerate() {
                                    let res = ui.selectable_value(&mut self.volume_id, id, volume.label());
                                    if res.changed() {
                                        println!("Selected volume: {}", self.volume_id);
                                        self.select_volume(self.volume_id);
                                        self.zoom = 0.25;
                                    }
                                }
                            });
                    });
                }
                ui.end_row();
                let sync_coordinates = self.should_sync_coords();
                slider(
                    ui,
                    "x",
                    &mut self.coord[0],
                    self.ranges[0].clone(),
                    false,
                    !sync_coordinates,
                );
                slider(
                    ui,
                    "y",
                    &mut self.coord[1],
                    self.ranges[1].clone(),
                    false,
                    !sync_coordinates,
                );
                slider(
                    ui,
                    "z",
                    &mut self.coord[2],
                    self.ranges[2].clone(),
                    false,
                    !sync_coordinates,
                );

                let mut has_changed = false;

                if let Some(segment_mode) = self.segment_mode.as_mut() {
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

                    has_changed = has_changed || u_sl.changed() || v_sl.changed() || w_sl.changed();
                }

                slider(ui, "Zoom", &mut self.zoom, 0.1..=6.0, true, true);

                fn cb<T: ToString>(ui: &mut Ui, label: T, value: &mut bool) -> Response {
                    ui.label(label.to_string());
                    let res = ui.checkbox(value, "");
                    ui.end_row();
                    res
                }

                if self.overlay.is_some() {
                    has_changed = has_changed || cb(ui, "Show overlay ('L')", &mut self.show_overlay).changed();
                }

                if self.is_segment_mode() {
                    let c = cb(
                        ui,
                        "Trilinear interpolation ('I')",
                        &mut self.drawing_config.trilinear_interpolation,
                    );
                    if c.changed() {
                        has_changed = true;
                    }

                    self.segment_mode.as_mut().unwrap();
                    has_changed = has_changed
                        || cb(
                            ui,
                            "Segment outlines ('O')",
                            &mut self.drawing_config.show_segment_outlines,
                        )
                        .changed();

                    has_changed = has_changed
                        || cb(
                            ui,
                            "Segment outline points ('P')",
                            &mut self.drawing_config.draw_outline_vertices,
                        )
                        .changed();

                    has_changed = has_changed || cb(ui, "Sync coordinates ('S')", &mut self.sync_coordinates).changed();

                    cb(ui, "XYZ outline ('X')", &mut self.drawing_config.draw_xyz_outlines);

                    ui.collapsing("Compositing", |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Mode");
                            // combo box
                            egui::ComboBox::from_id_salt("Compositing Mode")
                                .selected_text(self.drawing_config.compositing.mode.label())
                                .show_ui(ui, |ui| {
                                    for mode in CompositingMode::VALUES {
                                        ui.selectable_value(
                                            &mut self.drawing_config.compositing.mode,
                                            mode,
                                            mode.label(),
                                        );
                                    }
                                });
                            ui.end_row();
                        });

                        slider(
                            ui,
                            "Layers In Front",
                            &mut self.drawing_config.compositing.layers_in_front,
                            0..=100,
                            false,
                            true,
                        );

                        slider(
                            ui,
                            "Layers Behind",
                            &mut self.drawing_config.compositing.layers_behind,
                            0..=100,
                            false,
                            true,
                        );

                        if self.drawing_config.compositing.mode == CompositingMode::Alpha {
                            slider(
                                ui,
                                "Alpha Min",
                                &mut self.drawing_config.compositing.alpha_min,
                                0..=255,
                                false,
                                true,
                            );
                            slider(
                                ui,
                                "Alpha Max",
                                &mut self.drawing_config.compositing.alpha_max,
                                0..=255,
                                false,
                                true,
                            );
                            slider(
                                ui,
                                "Alpha Threshold",
                                &mut self.drawing_config.compositing.alpha_threshold,
                                0..=10000,
                                false,
                                true,
                            );
                            slider(
                                ui,
                                "Material",
                                &mut self.drawing_config.compositing.material,
                                0..=300,
                                false,
                                true,
                            );
                            cb(
                                ui,
                                "Reverse Direction",
                                &mut self.drawing_config.compositing.reverse_direction,
                            );
                        }
                    });
                }

                if has_changed {
                    self.sync_coords();
                }
            });

        ui.collapsing("Filters", |ui| {
            ui.checkbox(&mut self.drawing_config.enable_filters, "Enable ('F')");
            ui.add_enabled_ui(self.drawing_config.enable_filters, |ui| {
                egui::Grid::new("my_grid")
                    .num_columns(2)
                    .spacing([40.0, 4.0])
                    .show(ui, |ui| {
                        slider(
                            ui,
                            "Min",
                            &mut self.drawing_config.threshold_min,
                            0..=(254 - self.drawing_config.threshold_max),
                            false,
                            true,
                        );
                        slider(
                            ui,
                            "Max",
                            &mut self.drawing_config.threshold_max,
                            0..=(254 - self.drawing_config.threshold_min),
                            false,
                            true,
                        );
                        slider(ui, "Mask Bits", &mut self.drawing_config.quant, 1..=8, false, true);
                        slider(
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
                    });
            });
        });
    }

    fn try_recv_from_download_notifier(&mut self) -> bool {
        self.download_notifier.as_ref().is_some_and(|x| x.try_recv().is_ok())
    }

    fn update_main(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui_extras::install_image_loaders(ctx);

        if self.try_recv_from_download_notifier() {
            while self.try_recv_from_download_notifier() {} // clear queue
        }

        let mut switch_segment = None;
        for n in self.notification_receiver.try_iter() {
            match n {
                UINotification::ObjDownloadReady(segment) => {
                    if let Some(obj_file) = self.obj_repository.get(&segment) {
                        if self.downloading_segment.as_ref().map_or(false, |s| s == &segment) {
                            switch_segment = Some((segment, obj_file));
                        }
                    }
                }
            }
        }
        if let Some((segment, obj_file)) = switch_segment {
            self.load_volume_by_ref(&segment.volume_ref());
            self.setup_segment(obj_file.to_str().unwrap(), segment.width, segment.height);
            self.selected_segment = Some(segment);
            self.downloading_segment = None;
        }

        if self.catalog_panel_open {
            self.catalog_panel(ctx);
        }

        ctx.input(|i| {
            if i.key_pressed(egui::Key::F) {
                self.drawing_config.enable_filters = !self.drawing_config.enable_filters;
            }
            if self.overlay.is_some() && i.key_pressed(egui::Key::L) {
                self.show_overlay = !self.show_overlay;
            }
            if self.is_segment_mode() {
                if i.key_pressed(egui::Key::I) {
                    self.drawing_config.trilinear_interpolation = !self.drawing_config.trilinear_interpolation;
                }
                if i.key_pressed(egui::Key::O) {
                    self.drawing_config.show_segment_outlines = !self.drawing_config.show_segment_outlines;
                }
                if i.key_pressed(egui::Key::P) {
                    self.drawing_config.draw_outline_vertices = !self.drawing_config.draw_outline_vertices;
                }
                if i.key_pressed(egui::Key::S) {
                    self.sync_coordinates = !self.sync_coordinates;
                }
                if i.key_pressed(egui::Key::X) {
                    self.drawing_config.draw_xyz_outlines = !self.drawing_config.draw_xyz_outlines;
                }
                if i.key_pressed(egui::Key::A) {
                    if self.drawing_config.compositing.mode == CompositingMode::None {
                        self.drawing_config.compositing.mode = CompositingMode::Alpha;
                    } else {
                        self.drawing_config.compositing.mode = CompositingMode::None;
                    }
                }
                if i.key_pressed(egui::Key::J) {
                    let segment_mode = self.segment_mode.as_mut().unwrap();
                    segment_mode.coord[2] = (segment_mode.coord[2] - 1).max(*segment_mode.ranges[2].start());
                    self.sync_coords();
                }
                if i.key_pressed(egui::Key::K) {
                    let segment_mode = self.segment_mode.as_mut().unwrap();
                    segment_mode.coord[2] = (segment_mode.coord[2] + 1).min(*segment_mode.ranges[2].end());
                    self.sync_coords();
                }
                if i.key_pressed(egui::Key::C) {
                    self.catalog_panel_open = !self.catalog_panel_open;
                }
                if i.key_pressed(egui::Key::Num1) {
                    self.layout = GuiLayout::Grid;
                }
                if i.key_pressed(egui::Key::Num2) {
                    self.layout = GuiLayout::XY;
                }
                if i.key_pressed(egui::Key::Num3) {
                    self.layout = GuiLayout::XZ;
                }
                if i.key_pressed(egui::Key::Num4) {
                    self.layout = GuiLayout::YZ;
                }
                if i.key_pressed(egui::Key::Num5) {
                    self.layout = GuiLayout::UV;
                }
            }
        });

        egui::Window::new("Controls").show(ctx, |ui| {
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
            match self.layout {
                GuiLayout::Grid => {
                    let available_size = ui.available_size();
                    let cell_width = (available_size.x - 2.0) / 2.0; // Account for spacing
                    let cell_height = (available_size.y - 2.0) / 2.0; // Account for spacing
                    let cell_size = Vec2::new(cell_width, cell_height);

                    ui.vertical(|ui| {
                        ui.horizontal(|ui| {
                            self.render_pane(ui, cell_size, Self::XY_PANE);

                            ui.add_space(2.0);

                            self.render_pane(ui, cell_size, Self::XZ_PANE);
                        });

                        ui.add_space(2.0);

                        ui.horizontal(|ui| {
                            self.render_pane(ui, cell_size, Self::YZ_PANE);

                            ui.add_space(2.0);

                            self.render_uv_pane(ui, cell_size);
                        });
                    });
                }
                GuiLayout::XY => {
                    self.render_pane(ui, ui.available_size(), Self::XY_PANE);
                }
                GuiLayout::XZ => {
                    self.render_pane(ui, ui.available_size(), Self::XZ_PANE);
                }
                GuiLayout::YZ => {
                    self.render_pane(ui, ui.available_size(), Self::YZ_PANE);
                }
                GuiLayout::UV => {
                    if self.is_segment_mode() {
                        self.render_uv_pane(ui, ui.available_size());
                    } else {
                        ui.label("UV pane is only available in segment mode.");
                    }
                }
            }
        });
    }

    const XY_PANE: VolumePane = VolumePane::new(PaneType::XY, false);
    const XZ_PANE: VolumePane = VolumePane::new(PaneType::XZ, false);
    const YZ_PANE: VolumePane = VolumePane::new(PaneType::YZ, false);
    const UV_PANE: VolumePane = VolumePane::new(PaneType::UV, true);
    fn render_pane(&mut self, ui: &mut Ui, cell_size: Vec2, pane: VolumePane) {
        let segment_outlines_coord = if self.is_segment_mode() {
            Some(self.segment_mode.as_ref().unwrap().coord)
        } else {
            None
        };

        pane.render(
            ui,
            &mut self.coord,
            &self.world,
            self.segment_mode.as_ref().map(|s| s.surface_volume.clone()),
            &mut self.zoom,
            &self.drawing_config,
            self.extra_resolutions,
            segment_outlines_coord,
            &self.ranges,
            cell_size,
        );
    }
    fn render_uv_pane(&mut self, ui: &mut Ui, cell_size: Vec2) {
        if let Some(segment_mode) = self.segment_mode.as_mut() {
            if Self::UV_PANE.render(
                ui,
                &mut segment_mode.coord,
                &segment_mode.world,
                None,
                &mut self.zoom,
                &self.drawing_config,
                self.extra_resolutions,
                None,
                &segment_mode.ranges,
                cell_size,
            ) {
                if self.should_sync_coords() {
                    self.sync_coords();
                }
            }
        }
    }

    fn catalog_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("Catalog").show(ctx, |ui| {
            let selection = &mut ui.visuals_mut().selection;
            selection.stroke = Stroke::new(2.0, Color32::from_rgb(0x00, 0x00, 0x00));
            selection.bg_fill = Color32::from_rgb(0xcc, 0xcc, 0xcc);

            // Header
            ui.add_space(4.0);
            ui.vertical_centered(|ui| {
                ui.heading("📜 Catalog");
            });
            ui.separator();

            {
                let mut clicked = None;
                self.catalog.scrolls().iter().for_each(|scroll| {
                    egui::CollapsingHeader::new(scroll.label()).show(ui, |ui| {
                        let mut table = TableBuilder::new(ui)
                            .vscroll(true)
                            .column(Column::auto())
                            .column(Column::remainder().at_least(130.0) /* Column::initial(150.0) */)
                            .column(Column::auto())
                            .column(Column::auto())
                            .column(Column::auto());

                        table = table.sense(egui::Sense::click());

                        table
                            .header(20.0, |mut header| {
                                header.col(|ui| {
                                    ui.strong("Mask");
                                });
                                header.col(|ui| {
                                    ui.strong("ID");
                                });
                                header.col(|ui| {
                                    ui.strong("Width");
                                });
                                header.col(|ui| {
                                    ui.strong("Height");
                                });
                                header.col(|ui| {
                                    ui.strong("Area / cm²");
                                });
                            })
                            .body(|mut body| {
                                for segment in self.catalog.segments(&scroll) {
                                    body.row(20.0, |mut row| {
                                        row.set_selected(self.selected_segment.as_ref() == Some(segment));
                                        row
                                        .col(|ui| {
                                            let url = format!("https://vesuvius.virtual-void.net/scroll/{}/segment/{}/mask?ext=png&width=50&height=25", scroll.old_id, segment.id);
                                            ui.image(url);

                                            //ui.image(segment.urls.mask_url.clone());
                                        });
                                        fn l(text: impl Into<WidgetText>) -> Label {
                                            Label::new(text).selectable(false)
                                        }

                                        row.col(|ui| {
                                            let cached = self.obj_repository.is_cached(segment);
                                            let mut text = RichText::new(&segment.id);
                                            if cached {
                                                text = text.color(Color32::DARK_GREEN);
                                            } else if self.downloading_segment == Some(segment.clone()) {
                                                // current time millis
                                                let time = std::time::SystemTime::now()
                                                    .duration_since(std::time::UNIX_EPOCH)
                                                    .unwrap()
                                                    .as_millis() / 600;
                                                if time % 2 == 0 {
                                                    text = text.color(Color32::YELLOW);
                                                }
                                            }


                                            l(text).ui(ui);
                                        });
                                        row.col(|ui| {
                                            l(format!("{}", segment.width)).ui(ui);
                                        });
                                        row.col(|ui| {
                                            l(format!("{}", segment.height)).ui(ui);
                                        });
                                        row.col(|ui| {
                                            l(segment.area_cm2.map_or("".to_string(), |v| format!("{v:.1}"))).ui(ui);
                                        });

                                        if row.response().clicked() && segment.volume.is_some(){
                                            clicked = Some(segment.clone());
                                        }
                                    });
                                }
                            });
                    });
                });
                if let Some(segment) = clicked {
                    if let Some(obj_file) = self.obj_repository.get(&segment) {
                        self.load_volume_by_ref(&segment.volume_ref());
                        self.setup_segment(&obj_file.to_str().unwrap().to_string(), segment.width, segment.height);
                        self.selected_segment = Some(segment);
                    } else {
                        let sender = self.notification_sender.clone();
                        let segment = segment.clone();
                        self.downloading_segment = Some(segment.clone());
                        self.obj_repository.download(&segment, move |segment| {let _ =sender.send(UINotification::ObjDownloadReady(segment.clone()));});
                    }
                }
            }
            //);
        });
    }
}

impl eframe::App for TemplateApp {
    /// Called by the frame work to save state before shutdown.
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, self);
    }
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("op_bar")
            .frame(egui::Frame::none().inner_margin(4.0))
            .show(ctx, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.visuals_mut().button_frame = false;
                    ui.toggle_value(&mut self.catalog_panel_open, "📜 (C)atalog");

                    fn layout_button(
                        ui: &mut Ui,
                        field: &mut GuiLayout,
                        target_layout: GuiLayout,
                        label: &str,
                    ) -> Response {
                        let response = ui.selectable_value(field, target_layout, label);
                        if response.clicked() {
                            *field = target_layout;
                        }
                        response
                    }

                    ui.separator();
                    ui.label("Layout");

                    layout_button(ui, &mut self.layout, GuiLayout::Grid, "4x4 (1)");
                    layout_button(ui, &mut self.layout, GuiLayout::XY, "XY (2)");
                    layout_button(ui, &mut self.layout, GuiLayout::XZ, "XZ (3)");
                    layout_button(ui, &mut self.layout, GuiLayout::YZ, "YZ (4)");
                    if self.is_segment_mode() {
                        layout_button(ui, &mut self.layout, GuiLayout::UV, "UV (5)");
                    }
                });
            });

        self.update_main(ctx, frame);
    }
}

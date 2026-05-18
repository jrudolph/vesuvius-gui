use crate::gui::{FrameBudget, PaneType, VolumePane, UV_PANE_BUDGET_FRACTION};
use directories::BaseDirs;
use egui::CollapsingHeader;
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
use std::time::Duration;
use vesuvius_atlas_rs::{AtlasMetadata, AtlasSample};
use vesuvius_rs::catalog::obj_repository::ObjRepository;
use vesuvius_rs::catalog::Catalog;
use vesuvius_rs::catalog::Segment;
use vesuvius_rs::model::*;
use vesuvius_rs::volume::*;
use vesuvius_zarr::{OmeZarrContext, ZarrArray};

pub(crate) const ZOOM_MIN: f32 = 0.01;
pub(crate) const ZOOM_MAX: f32 = 8.0;

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
    obj_volume: Option<Arc<ObjVolume>>,
    #[serde(skip)]
    uv_pane: VolumePane,
    #[serde(skip)]
    convert_to_world_coords: Box<dyn Fn([i32; 3]) -> [i32; 3]>,
    #[serde(skip)]
    segment_id: Option<String>,
    #[serde(skip)]
    sample_id: Option<String>,
    #[serde(skip)]
    current_base_volume_id: Option<String>,
    #[serde(skip)]
    available_volumes: Vec<String>,
    /// Last affine transform applied to the segment's mesh (via setup_segment).
    /// Stored so we can rebuild the obj base in-place (e.g. on overlay toggle)
    /// without losing the transform.
    #[serde(skip)]
    last_transform: Option<AffineTransform>,
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
            obj_volume: None,
            uv_pane: VolumePane::new(PaneType::UV, true),
            convert_to_world_coords: Box::new(|x| x),
            segment_id: None,
            sample_id: None,
            current_base_volume_id: None,
            available_volumes: Vec::new(),
            last_transform: None,
        }
    }
}

enum UINotification {
    ObjDownloadReady(Segment),
    AtlasObjDownloadReady(String, String, String),
}

pub struct ObjFileConfig {
    pub obj_file: String,
    pub width: usize,
    pub height: usize,
    pub transform: Option<AffineTransform>,
    pub projection: ProjectionKind,
}

pub struct VesuviusConfig {
    pub data_dir: Option<String>,
    pub obj_file: Option<ObjFileConfig>,
    pub overlay_dir: Option<String>,
    pub overlay_coloring: Option<OverlayColoring>,
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
    atlas: Option<AtlasMetadata>,
    #[serde(skip)]
    obj_repository: ObjRepository,
    #[serde(skip)]
    selected_segment: Option<Segment>,
    #[serde(skip)]
    downloading_segment: Option<Segment>,
    #[serde(skip)]
    downloading_atlas_segment: Option<(String, String)>, // (sample_id, segment_id)
    #[serde(skip)]
    notification_sender: Sender<UINotification>,
    #[serde(skip)]
    notification_receiver: Receiver<UINotification>,
    #[serde(skip)]
    overlay: Option<Volume>,
    /// Raw inner overlay (the underlying zarr/ome-zarr `Volume`, before wrapping
    /// in `OverlayPaintVolume`). Kept so we can rebuild `overlay` with a new
    /// coloring without re-loading from disk/network.
    #[serde(skip)]
    overlay_inner: Option<Volume>,
    overlay_coloring: OverlayColoring,
    catalog_panel_open: bool,
    layout: GuiLayout,
    #[serde(skip)]
    pending_volume_switch: Option<String>,
    target_fps: u32,
}

impl Default for TemplateApp {
    fn default() -> Self {
        let catalog = Catalog::default();
        let atlas = None;
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
            ranges: [0..=50000, 0..=50000, 0..=100000],
            extra_resolutions: 0,
            segment_mode: None,
            catalog,
            atlas,
            obj_repository,
            selected_segment: None,
            downloading_segment: None,
            downloading_atlas_segment: None,
            notification_sender,
            notification_receiver,
            overlay: None,
            overlay_inner: None,
            overlay_coloring: OverlayColoring::default(),
            catalog_panel_open: true,
            layout: GuiLayout::Grid,
            pending_volume_switch: None,
            target_fps: 20,
        }
    }
}

impl TemplateApp {
    fn atlas_obj_cache_path(sample_id: &str, segment_id: &str) -> std::path::PathBuf {
        let dir = BaseDirs::new().unwrap().cache_dir().join("vesuvius-gui");
        dir.join(format!("atlas-segments/{}/{}.obj", sample_id, segment_id))
    }

    fn load_atlas_segment(&mut self, sample_id: &str, segment_id: &str) {
        self.load_atlas_segment_with_volume(sample_id, segment_id, None);
    }

    fn load_atlas_segment_with_volume(&mut self, sample_id: &str, segment_id: &str, target_volume_id: Option<&str>) {
        let mut volume_url_opt = None;
        let mut segment_info = None;
        let mut available_volumes = Vec::new();
        let mut current_volume_id = None;
        let mut transform: Option<AffineTransform> = None;
        let mut xyz_transform: Option<AffineTransform> = None;

        if let Some(atlas) = &self.atlas {
            if let Some(atlas_sample) = atlas.get_sample(sample_id) {
                if let Some(segment) = atlas_sample.get_segment(segment_id) {
                    let source_volume_id = self
                        .segment_mode
                        .as_ref()
                        .and_then(|sm| sm.current_base_volume_id.as_ref())
                        .map(|s| s.as_str())
                        .unwrap_or(&segment.original_volume_id);

                    let volume_id = target_volume_id.unwrap_or(&segment.original_volume_id);
                    current_volume_id = Some(volume_id.to_string());

                    if volume_id != &segment.original_volume_id {
                        transform = atlas_sample.get_transform(&segment.original_volume_id, volume_id);
                    }

                    if let Some(volume) = atlas_sample.get_volume(volume_id) {
                        volume_url_opt = volume.get_ome_zarr_url();
                    }

                    let coord_scale_transform = if source_volume_id != volume_id {
                        atlas_sample.get_transform(source_volume_id, volume_id)
                    } else {
                        None
                    };

                    if target_volume_id.is_some() && source_volume_id != volume_id {
                        xyz_transform = atlas_sample.get_transform(source_volume_id, volume_id);
                    }

                    let (width, height) = if target_volume_id.is_some() {
                        self.segment_mode
                            .as_ref()
                            .map(|seg_mode| (seg_mode.width, seg_mode.height))
                            .unwrap_or((segment.properties.width, segment.properties.height))
                    } else {
                        (segment.properties.width, segment.properties.height)
                    };

                    segment_info = Some((width, height, coord_scale_transform));

                    available_volumes = atlas_sample
                        .get_volumes_for_segment(segment_id)
                        .into_iter()
                        .map(|(vol_id, _, _)| vol_id)
                        .collect();
                }
            }
        }

        if let Some(volume_url) = volume_url_opt {
            if let Ok(vol_ref) = NewVolumeReference::from_url(volume_url) {
                self.load_volume(&vol_ref);
            }
        }

        if let Some((width, height, coord_scale_transform)) = segment_info {
            let obj_cache_path = Self::atlas_obj_cache_path(sample_id, segment_id);
            if obj_cache_path.exists() {
                self.setup_segment(
                    obj_cache_path.to_str().unwrap(),
                    width,
                    height,
                    transform.as_ref(),
                    coord_scale_transform.as_ref(),
                    ProjectionKind::None,
                    Some((sample_id.to_string(), segment_id.to_string())),
                );

                if let Some(segment_mode) = self.segment_mode.as_mut() {
                    segment_mode.segment_id = Some(segment_id.to_string());
                    segment_mode.sample_id = Some(sample_id.to_string());
                    segment_mode.current_base_volume_id = current_volume_id;
                    segment_mode.available_volumes = available_volumes;
                }

                if let Some(xyz_tf) = xyz_transform {
                    self.coord = xyz_tf.transform_point(self.coord);
                }
            }
        }
    }

    /// Called once before the first frame.
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        catalog: Catalog,
        atlas: Option<AtlasMetadata>,
        config: VesuviusConfig,
    ) -> Self {
        // This is also where you can customize the look and feel of egui using
        // `cc.egui_ctx.set_visuals` and `cc.egui_ctx.set_fonts`.
        let mut app: TemplateApp = if let Some(storage) = cc.storage {
            eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default()
        } else {
            Default::default()
        };
        app.obj_repository = ObjRepository::new(&catalog);
        app.catalog = catalog;
        app.atlas = atlas;
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
            transform,
            projection,
        }) = config.obj_file
        {
            app.setup_segment(&obj_file, width, height, transform.as_ref(), None, projection, None);
        }

        if let Some(coloring) = config.overlay_coloring {
            app.overlay_coloring = coloring;
        }

        if let Some(segment_file) = config.overlay_dir {
            if segment_file.contains(".zarr") {
                let inner: Volume = if segment_file.contains(".ome.zarr") {
                    if segment_file.starts_with("http") {
                        log::info!("Loading ome-zarr overlay from url: {}", segment_file);
                        OmeZarrPaintVolume::<GrayScale>::new(OmeZarrContext::from_url_to_default_cache_dir(
                            &segment_file,
                        ))
                        .into_volume()
                    } else {
                        log::info!("Loading ome-zarr overlay from path: {}", segment_file);
                        OmeZarrPaintVolume::<GrayScale>::new(OmeZarrContext::from_path(&segment_file)).into_volume()
                    }
                } else if segment_file.starts_with("http") {
                    log::info!("Loading zarr overlay from url: {}", segment_file);
                    ZarrArray::from_url_to_default_cache_dir(&segment_file)
                        .into_ctx()
                        .into_ctx()
                        .into_volume()
                } else {
                    log::info!("Loading zarr overlay from path: {}", segment_file);
                    ZarrArray::<3, u8>::from_path_auto(&segment_file)
                        .into_ctx()
                        .into_ctx()
                        .into_volume()
                };
                app.overlay = Some(OverlayPaintVolume::new(inner.clone(), app.overlay_coloring).into_volume());
                app.overlay_inner = Some(inner);
            }
        }

        app
    }

    fn setup_segment(
        &mut self,
        segment_file: &str,
        width: usize,
        height: usize,
        transform: Option<&AffineTransform>,
        coord_scale_transform: Option<&AffineTransform>,
        projection: ProjectionKind,
        atlas_metadata: Option<(String, String)>,
    ) {
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
            segment.obj_volume = None;
            segment.convert_to_world_coords = Box::new(move |coord| ppm2.convert_to_world_coords(coord));

            if let Some((sample_id, segment_id)) = atlas_metadata {
                segment.sample_id = Some(sample_id);
                segment.segment_id = Some(segment_id);
            } else {
                segment.sample_id = None;
                segment.segment_id = None;
                segment.current_base_volume_id = None;
                segment.available_volumes.clear();
            }

            self.segment_mode = Some(segment)
        } else if segment_file.ends_with(".obj") {
            let mut segment: SegmentMode = self.segment_mode.take().unwrap_or_default();
            let base = self.world.clone();
            let base = if let (Some(overlay), true) = (self.overlay.as_ref(), self.show_overlay) {
                OverlayVolume::new(base, overlay.clone(), self.overlay_coloring).into_volume()
            } else {
                base
            };
            let transform_owned = transform.cloned();
            let coord_scale_transform_owned = coord_scale_transform.cloned();

            let is_reload = segment.filename == segment_file;
            let old_coord = segment.coord;
            let old_zoom = self.zoom;

            let scale = coord_scale_transform_owned
                .as_ref()
                .map(|t| t.scale_factor())
                .unwrap_or(1.0);

            let scaled_width = (width as f64 * scale) as usize;
            let scaled_height = (height as f64 * scale) as usize;

            let obj_volume = match (is_reload, projection, segment.obj_volume.as_ref()) {
                (true, ProjectionKind::None, Some(prev)) => {
                    log::info!("ObjVolume::with_base reusing parsed mesh for {}", segment_file);
                    prev.with_base(base, scaled_width, scaled_height, &transform_owned)
                }
                _ => ObjVolume::load_from_obj(
                    segment_file,
                    base,
                    scaled_width,
                    scaled_height,
                    &transform_owned,
                    projection,
                ),
            };
            let width = obj_volume.width() as i32;
            let height = obj_volume.height() as i32;

            let volume = Arc::new(obj_volume);
            let obj2 = volume.clone();

            if is_reload {
                segment.coord = [
                    (old_coord[0] as f64 * scale) as i32,
                    (old_coord[1] as f64 * scale) as i32,
                    (old_coord[2] as f64 * scale) as i32,
                ];
                self.zoom = (old_zoom as f64 / scale) as f32;
            } else {
                segment.coord = [width / 2, height / 2, 0];
                segment.filename = segment_file.to_string();
                segment.info = segment_file.to_string();
            }

            segment.width = width as usize;
            segment.height = height as usize;
            segment.ranges = [0..=width, 0..=height, -40..=40];
            segment.world = Volume::from_ref(volume.clone());
            segment.surface_volume = volume.clone();
            segment.obj_volume = Some(volume);
            segment.convert_to_world_coords = Box::new(move |coords| obj2.convert_to_volume_coords(coords));
            segment.last_transform = transform_owned.clone();

            if let Some((sample_id, segment_id)) = atlas_metadata {
                segment.sample_id = Some(sample_id);
                segment.segment_id = Some(segment_id);
            } else {
                segment.sample_id = None;
                segment.segment_id = None;
                segment.current_base_volume_id = None;
                segment.available_volumes.clear();
            }

            self.segment_mode = Some(segment)
        }
    }

    /// Rebuild `self.overlay` from `self.overlay_inner` + current coloring, producing
    /// a fresh Arc so the tile cache invalidates.
    fn rebuild_overlay(&mut self) {
        if let Some(inner) = self.overlay_inner.as_ref() {
            self.overlay = Some(OverlayPaintVolume::new(inner.clone(), self.overlay_coloring).into_volume());
        }
    }

    /// Rebuild the current segment's ObjVolume against the current world + overlay state.
    /// Used when the user toggles overlay visibility or changes overlay coloring without
    /// touching the segment file. Uses ObjVolume::with_base which preserves the parsed mesh.
    fn rebuild_segment_base(&mut self) {
        let Some(mut segment) = self.segment_mode.take() else {
            return;
        };
        let Some(prev) = segment.obj_volume.as_ref() else {
            self.segment_mode = Some(segment);
            return;
        };

        let base = self.world.clone();
        let base = if let (Some(overlay), true) = (self.overlay.as_ref(), self.show_overlay) {
            OverlayVolume::new(base, overlay.clone(), self.overlay_coloring).into_volume()
        } else {
            base
        };

        let obj_volume = prev.with_base(base, segment.width, segment.height, &segment.last_transform);
        let volume = Arc::new(obj_volume);
        let obj2 = volume.clone();
        segment.world = Volume::from_ref(volume.clone());
        segment.surface_volume = volume.clone();
        segment.obj_volume = Some(volume);
        segment.convert_to_world_coords = Box::new(move |coords| obj2.convert_to_volume_coords(coords));
        self.segment_mode = Some(segment);
    }

    fn load_volume(&mut self, volume: &NewVolumeReference) {
        let params = VolumeCreationParams {
            cache_dir: self.data_dir.clone(),
        };
        self.world = volume.volume(&params);
    }

    fn load_volume_by_ref(&mut self, volume_ref: &dyn VolumeReference) {
        let id = volume_ref.id();
        let new_vol = vesuvius_rs::remap_config::RemapConfig::get()
            .volume_override_url(&id)
            .and_then(|url| NewVolumeReference::from_url(url).ok())
            .unwrap_or_else(|| NewVolumeReference::Volume64x4(volume_ref.owned()));
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

    fn get_atlas_context(&self) -> Option<(&str, &str, &AtlasSample)> {
        self.segment_mode
            .as_ref()
            .and_then(|sm| sm.sample_id.as_ref().zip(sm.segment_id.as_ref()))
            .and_then(|(sample_id, segment_id)| {
                self.atlas
                    .as_ref()
                    .and_then(|atlas| atlas.get_sample(sample_id))
                    .map(|atlas_sample| (sample_id.as_str(), segment_id.as_str(), atlas_sample))
            })
    }

    fn format_volume_label(&self, atlas_sample: &AtlasSample, volume_id: &str, shortcut_num: Option<usize>) -> String {
        atlas_sample
            .get_volume(volume_id)
            .and_then(|volume| volume.properties.as_ref())
            .and_then(|props| {
                let base = format!(
                    "{:.2}µm, {}keV, {:.2}m",
                    props.pixel_size_um?,
                    props.energy_kev?,
                    props.detector_distance_mm? / 1000.0
                );
                Some(if let Some(num) = shortcut_num {
                    format!("{:.2}µm (^{})", props.pixel_size_um?, num)
                } else {
                    base
                })
            })
            .unwrap_or_else(|| volume_id.to_string())
    }

    fn get_sorted_volumes(&self, atlas_sample: &AtlasSample) -> Vec<String> {
        if let Some(segment_mode) = self.segment_mode.as_ref() {
            let mut sorted_volumes = segment_mode.available_volumes.clone();
            sorted_volumes.sort_by(|a, b| {
                let props_a = atlas_sample.get_volume(a).and_then(|v| v.properties.as_ref());
                let props_b = atlas_sample.get_volume(b).and_then(|v| v.properties.as_ref());
                match (props_a, props_b) {
                    (Some(pa), Some(pb)) => pa
                        .pixel_size_um
                        .partial_cmp(&pb.pixel_size_um)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| {
                            pa.energy_kev
                                .partial_cmp(&pb.energy_kev)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .then_with(|| {
                            pa.detector_distance_mm
                                .partial_cmp(&pb.detector_distance_mm)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        }),
                    _ => std::cmp::Ordering::Equal,
                }
            });
            sorted_volumes
        } else {
            Vec::new()
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
            let slider = egui::Slider::new(value, range).clamping(SliderClamping::Always);
            let slider = if logarithmic { slider.logarithmic(true) } else { slider };
            let sl = ui.add_enabled(enabled, slider);
            ui.end_row();
            sl
        }

        let mut overlay_ui_state_changed = false;
        let mut overlay_coloring_changed = false;

        egui::Grid::new("my_grid")
            .num_columns(2)
            .spacing([40.0, 4.0])
            .show(ui, |ui| {
                if self.is_segment_mode() {
                    if let Some((sample_id, segment_id, atlas_sample)) = self.get_atlas_context() {
                        ui.label("Segment");
                        ui.label(format!("{}/{}", sample_id, segment_id));
                        ui.end_row();

                        if let Some(segment_mode) = self.segment_mode.as_ref() {
                            if !segment_mode.available_volumes.is_empty() {
                                ui.label("Base Volume");

                                let current_volume_id = segment_mode.current_base_volume_id.clone().unwrap_or_default();
                                let mut selected_volume_id = current_volume_id.clone();

                                let current_display = self.format_volume_label(atlas_sample, &current_volume_id, None);

                                egui::ComboBox::from_id_salt("BaseVolume")
                                    .selected_text(current_display)
                                    .show_ui(ui, |ui| {
                                        for volume_id in &segment_mode.available_volumes {
                                            let label = self.format_volume_label(atlas_sample, volume_id, None);
                                            let label_with_id = if &label != volume_id {
                                                format!("{} ({})", label, volume_id)
                                            } else {
                                                label
                                            };
                                            ui.selectable_value(
                                                &mut selected_volume_id,
                                                volume_id.clone(),
                                                label_with_id,
                                            );
                                        }
                                    });

                                if selected_volume_id != current_volume_id {
                                    self.pending_volume_switch = Some(selected_volume_id.clone());
                                }

                                ui.end_row();
                            } else {
                                ui.label("Volume");
                                ui.label("(segment mode)");
                                ui.end_row();
                            }
                        }
                    } else if self.segment_mode.is_some() {
                        ui.label("Volume");
                        ui.label("(segment mode)");
                        ui.end_row();

                        ui.label("");
                        if ui.button("Unload segment").clicked() {
                            self.segment_mode = None;
                            if self.layout == GuiLayout::UV {
                                self.layout = GuiLayout::Grid;
                            }
                        }
                        ui.end_row();
                    }
                } else {
                    ui.label("Volume");
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
                    ui.end_row();
                }
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

                slider(ui, "Zoom", &mut self.zoom, ZOOM_MIN..=ZOOM_MAX, true, true);

                fn cb<T: ToString>(ui: &mut Ui, label: T, value: &mut bool) -> Response {
                    ui.label(label.to_string());
                    let res = ui.checkbox(value, "");
                    ui.end_row();
                    res
                }

                if self.overlay.is_some() {
                    let show_changed = cb(ui, "Show overlay ('L')", &mut self.show_overlay).changed();
                    if show_changed {
                        overlay_ui_state_changed = true;
                        has_changed = true;
                    }
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

                    let mut header = CollapsingHeader::new("Compositing");
                    if self.drawing_config.compositing.mode != CompositingMode::None {
                        header = header.open(Some(true));
                    }
                    header.show(ui, |ui| {
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

                        if self.drawing_config.compositing.mode == CompositingMode::Alpha
                            || self.drawing_config.compositing.mode == CompositingMode::AlphaHeightMap
                        {
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
                                "Opacity",
                                &mut self.drawing_config.compositing.opacity,
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

        if self.overlay.is_some() {
            ui.collapsing("Overlay coloring", |ui| {
                let label = match self.overlay_coloring {
                    OverlayColoring::FourColors { .. } => "FourColors",
                    OverlayColoring::Boolean { .. } => "Boolean",
                    OverlayColoring::Hue { .. } => "Hue",
                };
                egui::ComboBox::from_id_salt("OverlayColoringMode")
                    .selected_text(label)
                    .show_ui(ui, |ui| {
                        let current_alpha = match self.overlay_coloring {
                            OverlayColoring::FourColors { alpha }
                            | OverlayColoring::Boolean { alpha, .. }
                            | OverlayColoring::Hue { alpha, .. } => alpha,
                        };
                        if ui
                            .selectable_label(
                                matches!(self.overlay_coloring, OverlayColoring::FourColors { .. }),
                                "FourColors",
                            )
                            .clicked()
                        {
                            self.overlay_coloring = OverlayColoring::FourColors { alpha: current_alpha };
                            overlay_coloring_changed = true;
                        }
                        if ui
                            .selectable_label(
                                matches!(self.overlay_coloring, OverlayColoring::Boolean { .. }),
                                "Boolean",
                            )
                            .clicked()
                        {
                            self.overlay_coloring = OverlayColoring::Boolean {
                                color: [255, 0, 255],
                                alpha: current_alpha,
                            };
                            overlay_coloring_changed = true;
                        }
                        if ui
                            .selectable_label(matches!(self.overlay_coloring, OverlayColoring::Hue { .. }), "Hue")
                            .clicked()
                        {
                            self.overlay_coloring = OverlayColoring::Hue {
                                hue_deg: 0.0,
                                alpha: current_alpha,
                            };
                            overlay_coloring_changed = true;
                        }
                    });
                match &mut self.overlay_coloring {
                    OverlayColoring::FourColors { alpha } => {
                        if ui.add(egui::Slider::new(alpha, 0.0..=1.0).text("Alpha")).changed() {
                            overlay_coloring_changed = true;
                        }
                    }
                    OverlayColoring::Boolean { color, alpha } => {
                        ui.horizontal(|ui| {
                            ui.label("Color");
                            let mut c = egui::Color32::from_rgb(color[0], color[1], color[2]);
                            if ui.color_edit_button_srgba(&mut c).changed() {
                                *color = [c.r(), c.g(), c.b()];
                                overlay_coloring_changed = true;
                            }
                        });
                        if ui.add(egui::Slider::new(alpha, 0.0..=1.0).text("Alpha")).changed() {
                            overlay_coloring_changed = true;
                        }
                    }
                    OverlayColoring::Hue { hue_deg, alpha } => {
                        if ui
                            .add(egui::Slider::new(hue_deg, 0.0..=360.0).text("Hue (deg)"))
                            .changed()
                        {
                            overlay_coloring_changed = true;
                        }
                        if ui.add(egui::Slider::new(alpha, 0.0..=1.0).text("Alpha")).changed() {
                            overlay_coloring_changed = true;
                        }
                    }
                }
            });
        }

        if overlay_coloring_changed {
            self.rebuild_overlay();
        }
        if overlay_coloring_changed || overlay_ui_state_changed {
            if self.is_segment_mode() {
                self.rebuild_segment_base();
            }
        }

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
        let notifications: Vec<_> = self.notification_receiver.try_iter().collect();
        for n in notifications {
            match n {
                UINotification::ObjDownloadReady(segment) => {
                    if let Some(obj_file) = self.obj_repository.get(&segment) {
                        if self.downloading_segment.as_ref().map_or(false, |s| s == &segment) {
                            switch_segment = Some((segment, obj_file));
                        }
                    }
                }
                UINotification::AtlasObjDownloadReady(sample_id, segment_id, _volume_id) => {
                    self.load_atlas_segment(&sample_id, &segment_id);
                    self.downloading_atlas_segment = None;
                }
            }
        }
        if let Some((segment, obj_file)) = switch_segment {
            self.load_volume_by_ref(&segment.volume_ref());
            self.setup_segment(
                obj_file.to_str().unwrap(),
                segment.width,
                segment.height,
                None,
                None,
                ProjectionKind::None,
                None,
            );
            self.selected_segment = Some(segment);
            self.downloading_segment = None;
        }

        if let Some(new_volume_id) = self.pending_volume_switch.take() {
            let segment_info = self.segment_mode.as_ref().and_then(|segment_mode| {
                segment_mode
                    .sample_id
                    .as_ref()
                    .zip(segment_mode.segment_id.as_ref())
                    .map(|(sample_id, segment_id)| (sample_id.clone(), segment_id.clone()))
            });

            if let Some((sample_id, segment_id)) = segment_info {
                self.load_atlas_segment_with_volume(&sample_id, &segment_id, Some(&new_volume_id));
            }
        }

        if self.catalog_panel_open {
            self.catalog_panel(ctx);
        }

        let mut overlay_state_changed = false;
        if !ctx.wants_keyboard_input() {
            ctx.input(|i| {
                if i.key_pressed(egui::Key::F) {
                    self.drawing_config.enable_filters = !self.drawing_config.enable_filters;
                }
                if self.overlay.is_some() && i.key_pressed(egui::Key::L) {
                    self.show_overlay = !self.show_overlay;
                    overlay_state_changed = true;
                }
                if i.key_pressed(egui::Key::C) {
                    self.catalog_panel_open = !self.catalog_panel_open;
                }
                if i.key_pressed(egui::Key::Num1) && !i.modifiers.ctrl {
                    self.layout = GuiLayout::Grid;
                }
                if i.key_pressed(egui::Key::Num2) && !i.modifiers.ctrl {
                    self.layout = GuiLayout::XY;
                }
                if i.key_pressed(egui::Key::Num3) && !i.modifiers.ctrl {
                    self.layout = GuiLayout::XZ;
                }
                if i.key_pressed(egui::Key::Num4) && !i.modifiers.ctrl {
                    self.layout = GuiLayout::YZ;
                }

                if let Some((_, _, atlas_sample)) = self.get_atlas_context() {
                    let sorted_volumes = self.get_sorted_volumes(atlas_sample);
                    if !sorted_volumes.is_empty() {
                        let keys = [
                            egui::Key::Num1,
                            egui::Key::Num2,
                            egui::Key::Num3,
                            egui::Key::Num4,
                            egui::Key::Num5,
                            egui::Key::Num6,
                            egui::Key::Num7,
                            egui::Key::Num8,
                            egui::Key::Num9,
                        ];

                        for (idx, volume_id) in sorted_volumes.iter().enumerate().take(9) {
                            if i.modifiers.ctrl && i.key_pressed(keys[idx]) {
                                self.pending_volume_switch = Some(volume_id.clone());
                            }
                        }
                    }
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
                        if self.drawing_config.compositing.mode != CompositingMode::Alpha {
                            self.drawing_config.compositing.mode = CompositingMode::Alpha;
                        } else {
                            self.drawing_config.compositing.mode = CompositingMode::None;
                        }
                    }
                    if i.key_pressed(egui::Key::M) {
                        if self.drawing_config.compositing.mode != CompositingMode::Max {
                            self.drawing_config.compositing.mode = CompositingMode::Max;
                        } else {
                            self.drawing_config.compositing.mode = CompositingMode::None;
                        }
                    }
                    if i.key_pressed(egui::Key::H) {
                        if self.drawing_config.compositing.mode != CompositingMode::AlphaHeightMap {
                            self.drawing_config.compositing.mode = CompositingMode::AlphaHeightMap;
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
                    if i.key_pressed(egui::Key::Num0) {
                        let segment_mode = self.segment_mode.as_mut().unwrap();
                        segment_mode.coord[2] = 0;
                        self.sync_coords();
                    }
                    if i.key_pressed(egui::Key::Num5) {
                        self.layout = GuiLayout::UV;
                    }
                }
            });
        }

        if overlay_state_changed && self.is_segment_mode() {
            self.rebuild_segment_base();
        }

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

            ui.horizontal(|ui| {
                let tooltip = "Caps the wall-clock time spent waiting for tile renders each frame \
                    (budget = 1s / target FPS).\n\n\
                    Higher FPS → smaller budget → snappy panning/zooming, but tiles flash blank \
                    for a few frames before they're ready.\n\
                    Lower FPS → larger budget → tiles appear immediately without flashing, but the \
                    UI thread can stutter while waiting on slow tile loads.";
                ui.label("Target").on_hover_text(tooltip);
                ui.add(egui::Slider::new(&mut self.target_fps, 1..=60).suffix(" FPS"))
                    .on_hover_text(tooltip);
            });
        });

        let frame_target = Duration::from_secs_f64(1.0 / self.target_fps.max(1) as f64);
        let budget = FrameBudget::new(frame_target);
        let segment_mode = self.is_segment_mode();
        let (xs_share, uv_share) = match self.layout {
            GuiLayout::Grid if segment_mode => (
                frame_target.mul_f32((1.0 - UV_PANE_BUDGET_FRACTION) / 3.0),
                frame_target.mul_f32(UV_PANE_BUDGET_FRACTION),
            ),
            GuiLayout::Grid => (frame_target.div_f32(3.0), Duration::ZERO),
            GuiLayout::XY | GuiLayout::XZ | GuiLayout::YZ => (frame_target, Duration::ZERO),
            GuiLayout::UV => (Duration::ZERO, frame_target),
        };

        egui::CentralPanel::default().show(ctx, |ui| {
            match self.layout {
                GuiLayout::Grid => {
                    let available_size = ui.available_size();
                    let cell_width = (available_size.x - 2.0) / 2.0; // Account for spacing
                    let cell_height = (available_size.y - 2.0) / 2.0; // Account for spacing
                    let cell_size = Vec2::new(cell_width, cell_height);

                    ui.vertical(|ui| {
                        ui.horizontal(|ui| {
                            self.render_pane(ui, cell_size, Self::XY_PANE, &budget, xs_share);

                            ui.add_space(2.0);

                            self.render_pane(ui, cell_size, Self::XZ_PANE, &budget, xs_share);
                        });

                        ui.add_space(2.0);

                        ui.horizontal(|ui| {
                            self.render_pane(ui, cell_size, Self::YZ_PANE, &budget, xs_share);

                            ui.add_space(2.0);

                            self.render_uv_pane(ui, cell_size, &budget, uv_share);
                        });
                    });
                }
                GuiLayout::XY => {
                    self.render_pane(ui, ui.available_size(), Self::XY_PANE, &budget, xs_share);
                }
                GuiLayout::XZ => {
                    self.render_pane(ui, ui.available_size(), Self::XZ_PANE, &budget, xs_share);
                }
                GuiLayout::YZ => {
                    self.render_pane(ui, ui.available_size(), Self::YZ_PANE, &budget, xs_share);
                }
                GuiLayout::UV => {
                    if self.is_segment_mode() {
                        self.render_uv_pane(ui, ui.available_size(), &budget, uv_share);
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
    fn render_pane(
        &mut self,
        ui: &mut Ui,
        cell_size: Vec2,
        pane: VolumePane,
        budget: &FrameBudget,
        pane_share: Duration,
    ) {
        let segment_outlines_coord = if self.is_segment_mode() {
            Some(self.segment_mode.as_ref().unwrap().coord)
        } else {
            None
        };

        let overlay = if self.show_overlay { self.overlay.as_ref() } else { None };

        pane.render(
            ui,
            &mut self.coord,
            &self.world,
            overlay,
            self.segment_mode.as_ref().map(|s| s.surface_volume.clone()),
            &mut self.zoom,
            &self.drawing_config,
            self.extra_resolutions,
            segment_outlines_coord,
            &self.ranges,
            cell_size,
            budget,
            pane_share,
        );
    }
    fn render_uv_pane(&mut self, ui: &mut Ui, cell_size: Vec2, budget: &FrameBudget, pane_share: Duration) {
        if let Some(segment_mode) = self.segment_mode.as_mut() {
            if Self::UV_PANE.render(
                ui,
                &mut segment_mode.coord,
                &segment_mode.world,
                None,
                None,
                &mut self.zoom,
                &self.drawing_config,
                self.extra_resolutions,
                None,
                &segment_mode.ranges,
                cell_size,
                budget,
                pane_share,
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

            egui::ScrollArea::vertical().show(ui, |ui| {
                let mut clicked = None;
                self.catalog.scrolls().iter().for_each(|scroll| {
                    egui::CollapsingHeader::new(scroll.label()).show(ui, |ui| {
                        let mut table = TableBuilder::new(ui)
                            .vscroll(false)
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

                let mut atlas_clicked = None;
                if let Some(atlas) = &self.atlas {
                    let mut samples: Vec<_> = atlas.samples.iter().collect();
                    samples.sort_by_key(|(id, _)| *id);
                    for (sample_id, atlas_sample) in samples {
                        egui::CollapsingHeader::new(format!("Atlas: {}", sample_id)).show(ui, |ui| {
                            let mut table = TableBuilder::new(ui)
                                .vscroll(false)
                                .column(Column::auto())
                                .column(Column::remainder().at_least(130.0))
                                .column(Column::auto())
                                .column(Column::auto())
                                .column(Column::auto());

                            table = table.sense(egui::Sense::click());

                            table
                                .header(20.0, |mut header| {
                                    header.col(|_ui| {});
                                    header.col(|ui| {
                                        ui.strong("ID");
                                    });
                                    header.col(|ui| {
                                        ui.strong("Width");
                                    });
                                    header.col(|ui| {
                                        ui.strong("Height");
                                    });
                                    header.col(|_ui| {});
                                })
                                .body(|mut body| {
                                    let mut segments: Vec<_> = atlas_sample.segments.iter().collect();
                                    segments.sort_by_key(|(id, _)| *id);
                                    for (segment_id, segment) in segments {
                                        body.row(20.0, |mut row| {
                                            row.col(|_ui| {});
                                            fn l(text: impl Into<WidgetText>) -> Label {
                                                Label::new(text).selectable(false)
                                            }
                                            row.col(|ui| {
                                                let obj_cache_path = Self::atlas_obj_cache_path(&sample_id, &segment_id);
                                                let cached = obj_cache_path.exists();
                                                let mut text = RichText::new(segment_id.as_str());
                                                if cached {
                                                    text = text.color(Color32::DARK_GREEN);
                                                } else if self.downloading_atlas_segment == Some((sample_id.clone(), segment_id.clone())) {
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
                                                l(format!("{}", segment.properties.width)).ui(ui);
                                            });
                                            row.col(|ui| {
                                                l(format!("{}", segment.properties.height)).ui(ui);
                                            });
                                            row.col(|_ui| {});

                                            if row.response().clicked() {
                                                atlas_clicked = Some((sample_id.clone(), segment_id.clone(), segment.clone()));
                                            }
                                        });
                                    }
                                });
                        });
                    }
                }

                if let Some((sample_id, segment_id, segment)) = atlas_clicked {
                    let obj_cache_path = Self::atlas_obj_cache_path(&sample_id, &segment_id);
                    if obj_cache_path.exists() {
                        self.load_atlas_segment(&sample_id, &segment_id);
                    } else if let Some(obj_url) = segment.get_obj_url() {
                        log::info!("Downloading atlas obj from {}", obj_url);
                        self.downloading_atlas_segment = Some((sample_id.clone(), segment_id.clone()));
                        let sender = self.notification_sender.clone();
                        let volume_id = segment.original_volume_id.clone();
                        ehttp::fetch(ehttp::Request::get(&obj_url), move |response| {
                            if let Ok(response) = response {
                                let obj_file = Self::atlas_obj_cache_path(&sample_id, &segment_id);
                                std::fs::create_dir_all(&obj_file.parent().unwrap()).unwrap();
                                let mut file = std::fs::File::create(&obj_file).unwrap();
                                let bytes = response.bytes;
                                log::info!("Downloaded {} bytes to {}", bytes.len(), obj_file.display());
                                std::io::copy(&mut std::io::Cursor::new(bytes), &mut file).unwrap();
                                let _ = sender.send(UINotification::AtlasObjDownloadReady(sample_id, segment_id, volume_id));
                            }
                        });
                    }
                }

                if let Some(segment) = clicked {
                    if let Some(obj_file) = self.obj_repository.get(&segment) {
                        self.load_volume_by_ref(&segment.volume_ref());
                        self.setup_segment(&obj_file.to_str().unwrap().to_string(), segment.width, segment.height, None, None, ProjectionKind::None, None);
                        self.selected_segment = Some(segment);
                    } else {
                        let sender = self.notification_sender.clone();
                        let segment = segment.clone();
                        self.downloading_segment = Some(segment.clone());
                        self.obj_repository.download(&segment, move |segment| {let _ =sender.send(UINotification::ObjDownloadReady(segment.clone()));});
                    }
                }
            });
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
            .frame(egui::Frame::NONE.inner_margin(4.0))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
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

                    let volume_buttons = self.get_atlas_context().map(|(_, _, atlas_sample)| {
                        (
                            self.get_sorted_volumes(atlas_sample)
                                .into_iter()
                                .take(9)
                                .enumerate()
                                .map(|(idx, vol_id)| {
                                    let label = self.format_volume_label(atlas_sample, &vol_id, Some(idx + 1));
                                    (vol_id, label)
                                })
                                .collect::<Vec<_>>(),
                            self.segment_mode
                                .as_ref()
                                .and_then(|sm| sm.current_base_volume_id.clone()),
                        )
                    });

                    if let Some((volume_labels, current_volume_id)) = volume_buttons {
                        if !volume_labels.is_empty() {
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                let current_volume_id = current_volume_id.unwrap_or_default();

                                for (volume_id, label) in volume_labels.iter().rev() {
                                    if ui.selectable_label(volume_id == &current_volume_id, label).clicked() {
                                        self.pending_volume_switch = Some(volume_id.clone());
                                    }
                                }

                                ui.separator();
                                ui.label("Volume");
                            });
                        }
                    }
                });
            });

        self.update_main(ctx, frame);
    }
}

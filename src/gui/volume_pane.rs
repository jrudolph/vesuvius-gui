use crate::volume::{DrawingConfig, PaintVolume, SurfaceVolume, Volume};
use egui::cache::FramePublisher;
use egui::{Image, PointerButton, Response, Ui, Vec2};
use std::ops::RangeInclusive;
use std::rc::Rc;

const ZOOM_RES_FACTOR: f32 = 1.3;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TextureCacheKey {
    pane_type: PaneType,
    coord: [i32; 3],
    zoom_level: u32, // zoom * 1000 as u32 for discrete levels
    frame_size: (usize, usize),
    drawing_config: DrawingConfig,
    segment_outlines_coord: Option<[i32; 3]>,
    extra_resolutions: u32,
    volume_id: usize,
    surface_volume_id: Option<usize>,
}

impl TextureCacheKey {
    fn new(
        pane_type: PaneType,
        coord: [i32; 3],
        zoom: f32,
        frame_size: (usize, usize),
        drawing_config: &DrawingConfig,
        segment_outlines_coord: Option<[i32; 3]>,
        extra_resolutions: u32,
        world: &Volume,
        surface_volume: Option<&Rc<dyn SurfaceVolume>>,
    ) -> Self {
        let zoom_level = (zoom * 1000.0) as u32;
        let volume_id = world as *const Volume as usize;
        let surface_volume_id = surface_volume.map(|sv| {
            let ptr: *const dyn SurfaceVolume = sv.as_ref();
            ptr as *const () as usize
        });

        Self {
            pane_type,
            coord,
            zoom_level,
            frame_size,
            drawing_config: drawing_config.clone(),
            segment_outlines_coord,
            extra_resolutions,
            volume_id,
            surface_volume_id,
        }
    }
}

type TextureCache = FramePublisher<TextureCacheKey, egui::TextureHandle>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PaneType {
    XY, // u=0, v=1, d=2
    XZ, // u=0, v=2, d=1
    YZ, // u=2, v=1, d=0
    UV, // u=0, v=1, d=2 (for segment mode)
}

impl PaneType {
    pub fn coordinates(&self) -> (usize, usize, usize) {
        match self {
            PaneType::XY => (0, 1, 2),
            PaneType::XZ => (0, 2, 1),
            PaneType::YZ => (2, 1, 0),
            PaneType::UV => (0, 1, 2),
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            PaneType::XY => "XY",
            PaneType::XZ => "XZ",
            PaneType::YZ => "YZ",
            PaneType::UV => "UV",
        }
    }
}

pub struct VolumePane {
    pane_type: PaneType,
    is_segment_pane: bool,
}

impl VolumePane {
    pub fn new(pane_type: PaneType, is_segment_pane: bool) -> Self {
        Self {
            pane_type,
            is_segment_pane,
        }
    }

    pub fn render(
        &mut self,
        ui: &mut Ui,
        coord: &mut [i32; 3],
        world: &Volume,
        surface_volume: Option<&Rc<dyn SurfaceVolume>>,
        zoom: &mut f32,
        drawing_config: &DrawingConfig,
        extra_resolutions: u32,
        segment_outlines_coord: Option<[i32; 3]>,
        ranges: &[RangeInclusive<i32>; 3],
        should_sync_coords: bool,
        cell_size: Vec2,
    ) -> bool {
        // Allocate exact size for this pane
        let (rect, _) = ui.allocate_exact_size(cell_size, egui::Sense::hover());
        let ui = &mut ui.new_child(egui::UiBuilder::new().max_rect(rect));

        let frame_width = cell_size.x as usize;
        let frame_height = cell_size.y as usize;

        // Get or create texture
        let texture = self.get_or_create_texture(
            ui,
            *coord,
            world,
            surface_volume,
            *zoom,
            frame_width,
            frame_height,
            drawing_config,
            extra_resolutions,
            segment_outlines_coord,
        );

        // Calculate scaling for image display
        let pane_scaling = if *zoom >= 1.0 {
            *zoom
        } else {
            let next_smaller_pow_of_2 = 2.0f32.powf((*zoom as f32).log2().floor());
            *zoom / next_smaller_pow_of_2
        };

        // Create and display image using available space but keeping scaling
        let image = Image::new(&texture)
            .max_height(frame_height as f32)
            .max_width(frame_width as f32)
            .fit_to_original_size(pane_scaling);

        let response = ui.add(image).interact(egui::Sense::drag());

        // Handle interactions and return whether textures need clearing
        let mut needs_clear = false;

        if self.handle_scroll(&response, ui, coord, ranges, zoom) {
            needs_clear = true;
        }

        if !should_sync_coords && self.handle_drag(&response, coord, ranges, *zoom) {
            needs_clear = true;
        }

        needs_clear
    }

    pub fn handle_scroll(
        &mut self,
        response: &Response,
        ui: &Ui,
        coord: &mut [i32; 3],
        ranges: &[RangeInclusive<i32>; 3],
        zoom: &mut f32,
    ) -> bool {
        let (_, _, d_coord) = self.pane_type.coordinates();
        let mut changed = false;

        if response.hovered() {
            let delta = ui.input(|i| i.smooth_scroll_delta);
            let zoom_delta = ui.input(|i| i.zoom_delta());

            if zoom_delta != 1.0 {
                *zoom = (*zoom * zoom_delta).max(0.1).min(6.0);
                changed = true;
            } else if delta.y != 0.0 {
                let min_level = 1 << ((ZOOM_RES_FACTOR / *zoom) as i32).min(4);
                let delta = delta.y.signum() * min_level as f32;
                let m = &mut coord[d_coord];
                *m = ((*m + delta as i32) / min_level as i32 * min_level as i32)
                    .clamp(*ranges[d_coord].start(), *ranges[d_coord].end());
                changed = true;
            }
        }

        changed
    }

    pub fn handle_drag(
        &mut self,
        response: &Response,
        coord: &mut [i32; 3],
        ranges: &[RangeInclusive<i32>; 3],
        zoom: f32,
    ) -> bool {
        let (u_coord, v_coord, _) = self.pane_type.coordinates();
        let mut changed = false;

        if response.dragged_by(PointerButton::Primary) {
            let delta = -response.drag_delta() / zoom;
            coord[u_coord] = (coord[u_coord] + delta.x as i32).clamp(*ranges[u_coord].start(), *ranges[u_coord].end());
            coord[v_coord] = (coord[v_coord] + delta.y as i32).clamp(*ranges[v_coord].start(), *ranges[v_coord].end());
            changed = true;
        }

        changed
    }

    fn get_or_create_texture(
        &self,
        ui: &Ui,
        coord: [i32; 3],
        world: &Volume,
        surface_volume: Option<&Rc<dyn SurfaceVolume>>,
        zoom: f32,
        frame_width: usize,
        frame_height: usize,
        drawing_config: &DrawingConfig,
        extra_resolutions: u32,
        segment_outlines_coord: Option<[i32; 3]>,
    ) -> egui::TextureHandle {
        let cache_key = TextureCacheKey::new(
            self.pane_type,
            coord,
            zoom,
            (frame_width, frame_height),
            drawing_config,
            segment_outlines_coord,
            extra_resolutions,
            world,
            surface_volume,
        );

        // Check if texture exists in cache first
        let cached_texture = ui.memory_mut(|mem| {
            let cache: &mut TextureCache = mem.caches.cache::<TextureCache>();
            cache.get(&cache_key).cloned()
        });

        if let Some(texture) = cached_texture {
            texture
        } else {
            // Create texture outside of memory lock to avoid deadlock
            let texture = self.create_texture(
                ui,
                coord,
                world,
                surface_volume,
                zoom,
                frame_width,
                frame_height,
                drawing_config,
                extra_resolutions,
                segment_outlines_coord,
            );

            // Store in cache
            ui.memory_mut(|mem| {
                let cache: &mut TextureCache = mem.caches.cache::<TextureCache>();
                cache.set(cache_key, texture.clone());
            });

            texture
        }
    }

    fn create_texture(
        &self,
        ui: &Ui,
        coord: [i32; 3],
        world: &Volume,
        surface_volume: Option<&Rc<dyn SurfaceVolume>>,
        zoom: f32,
        frame_width: usize,
        frame_height: usize,
        drawing_config: &DrawingConfig,
        extra_resolutions: u32,
        segment_outlines_coord: Option<[i32; 3]>,
    ) -> egui::TextureHandle {
        use std::time::Instant;
        let _start = Instant::now();

        let (u_coord, v_coord, d_coord) = self.pane_type.coordinates();

        let (scaling, paint_zoom) = if zoom >= 1.0 {
            (zoom, 1 as u8)
        } else {
            let next_smaller_pow_of_2 = 2.0f32.powf((zoom as f32).log2().floor());
            (
                zoom / next_smaller_pow_of_2,
                (1.0 / next_smaller_pow_of_2).round() as u8,
            )
        };

        let width = (frame_width as f32 / scaling) as usize;
        let height = (frame_height as f32 / scaling) as usize;
        let mut image = crate::volume::Image::new(width, height);

        let min_level = (32 - ((ZOOM_RES_FACTOR / zoom) as u32).leading_zeros()).min(4).max(0);
        let max_level: u32 = (min_level + extra_resolutions).min(4);

        for level in (min_level..=max_level).rev() {
            let sfactor = 1 << level as u8;
            world.paint(
                coord,
                u_coord,
                v_coord,
                d_coord,
                width,
                height,
                sfactor,
                paint_zoom,
                drawing_config,
                &mut image,
            );
        }

        // Add overlay if enabled and not in segment pane
        // TODO: Re-enable overlay support after fixing borrow checker issues
        // if !self.is_segment_pane && show_overlay {
        //     if let Some(zarr) = overlay {
        //         zarr.paint(
        //             coord,
        //             u_coord,
        //             v_coord,
        //             d_coord,
        //             width,
        //             height,
        //             1 << min_level as u8,
        //             paint_zoom,
        //             drawing_config,
        //             &mut image,
        //         );
        //     }
        // }

        // Add segment outlines if configured
        if let (Some(surface_vol), Some(outlines_coord)) = (surface_volume, segment_outlines_coord) {
            if !self.is_segment_pane && drawing_config.show_segment_outlines {
                surface_vol.paint_plane_intersection(
                    coord,
                    u_coord,
                    v_coord,
                    d_coord,
                    width,
                    height,
                    1,
                    paint_zoom,
                    Some(outlines_coord),
                    drawing_config,
                    &mut image,
                );
            }
        }

        let image: egui::ColorImage = image.into();
        ui.ctx().load_texture(
            format!("{}_{}{}{}", self.pane_type.label(), u_coord, v_coord, d_coord),
            image,
            Default::default(),
        )
    }
}

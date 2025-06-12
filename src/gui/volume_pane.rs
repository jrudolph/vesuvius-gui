use crate::volume::{DrawingConfig, PaintVolume, SurfaceVolume, Volume};
use egui::{Image, PointerButton, Response, Ui};
use std::ops::RangeInclusive;
use std::rc::Rc;

const ZOOM_RES_FACTOR: f32 = 1.3;

#[derive(Debug, Clone, Copy, PartialEq)]
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
    texture: Option<egui::TextureHandle>,
    is_segment_pane: bool,
}

impl VolumePane {
    pub fn new(pane_type: PaneType, is_segment_pane: bool) -> Self {
        Self {
            pane_type,
            texture: None,
            is_segment_pane,
        }
    }

    pub fn clear_texture(&mut self) {
        self.texture = None;
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
    ) -> bool {
        // Determine available space for this pane
        let available_size = ui.available_size();
        let frame_width = available_size.x as usize;
        let frame_height = available_size.y as usize;

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
        &mut self,
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
        if let Some(texture) = &self.texture {
            texture.clone()
        } else {
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
            self.texture = Some(texture.clone());
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

use crate::volume::{DrawingConfig, PaintVolume, SurfaceVolume, Volume};
use egui::cache::FramePublisher;
use egui::{PointerButton, Response, Ui, Vec2};
use std::ops::RangeInclusive;
use std::rc::Rc;

const ZOOM_RES_FACTOR: f32 = 1.3;
const TILE_SIZE: usize = 512;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TileCacheKey {
    pane_type: PaneType,
    tile_x: i32,
    tile_y: i32,
    coord: [i32; 3],
    zoom_level: u32, // zoom * 1000 as u32 for discrete levels
    drawing_config: DrawingConfig,
    segment_outlines_coord: Option<[i32; 3]>,
    extra_resolutions: u32,
    volume_id: usize,
    surface_volume_id: Option<usize>,
}

impl TileCacheKey {
    fn new(
        pane_type: PaneType,
        tile_x: i32,
        tile_y: i32,
        coord: [i32; 3],
        zoom: f32,
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

        // For tile caching, we only need the depth coordinate (d_coord)
        // since tile_x, tile_y already encode the spatial position
        let (_, _, d_coord) = pane_type.coordinates();
        let mut cache_coord = [0; 3];
        cache_coord[d_coord] = coord[d_coord];

        Self {
            pane_type,
            tile_x,
            tile_y,
            coord: cache_coord,
            zoom_level,
            drawing_config: drawing_config.clone(),
            segment_outlines_coord,
            extra_resolutions,
            volume_id,
            surface_volume_id,
        }
    }
}

type TileCache = FramePublisher<TileCacheKey, egui::TextureHandle>;

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
    pub const fn new(pane_type: PaneType, is_segment_pane: bool) -> Self {
        Self {
            pane_type,
            is_segment_pane,
        }
    }

    fn calculate_visible_tiles(
        &self,
        coord: [i32; 3],
        zoom: f32,
        frame_width: usize,
        frame_height: usize,
    ) -> Vec<(i32, i32, egui::Rect)> {
        let (u_coord, v_coord, _) = self.pane_type.coordinates();
        
        // Calculate world space viewport dimensions
        let world_width = frame_width as f32 / zoom;
        let world_height = frame_height as f32 / zoom;
        
        // Calculate viewport bounds in world coordinates
        let viewport_left = coord[u_coord] as f32 - world_width / 2.0;
        let viewport_right = coord[u_coord] as f32 + world_width / 2.0;
        let viewport_top = coord[v_coord] as f32 - world_height / 2.0;
        let viewport_bottom = coord[v_coord] as f32 + world_height / 2.0;
        
        // Calculate tile range
        let start_tile_x = (viewport_left / TILE_SIZE as f32).floor() as i32;
        let end_tile_x = (viewport_right / TILE_SIZE as f32).ceil() as i32;
        let start_tile_y = (viewport_top / TILE_SIZE as f32).floor() as i32;
        let end_tile_y = (viewport_bottom / TILE_SIZE as f32).ceil() as i32;
        
        // Generate tile list with screen positions
        let mut tiles = Vec::new();
        for tile_y in start_tile_y..end_tile_y {
            for tile_x in start_tile_x..end_tile_x {
                let screen_rect = self.calculate_tile_screen_rect(
                    tile_x, tile_y, coord, zoom, frame_width, frame_height
                );
                tiles.push((tile_x, tile_y, screen_rect));
            }
        }
        tiles
    }

    fn calculate_tile_screen_rect(
        &self,
        tile_x: i32,
        tile_y: i32,
        coord: [i32; 3],
        zoom: f32,
        frame_width: usize,
        frame_height: usize,
    ) -> egui::Rect {
        let (u_coord, v_coord, _) = self.pane_type.coordinates();
        
        // Tile bounds in world coordinates
        let tile_world_left = tile_x as f32 * TILE_SIZE as f32;
        let tile_world_right = (tile_x + 1) as f32 * TILE_SIZE as f32;
        let tile_world_top = tile_y as f32 * TILE_SIZE as f32;
        let tile_world_bottom = (tile_y + 1) as f32 * TILE_SIZE as f32;
        
        // Convert to screen coordinates relative to viewport center
        let center_x = frame_width as f32 / 2.0;
        let center_y = frame_height as f32 / 2.0;
        
        let screen_left = center_x + (tile_world_left - coord[u_coord] as f32) * zoom;
        let screen_right = center_x + (tile_world_right - coord[u_coord] as f32) * zoom;
        let screen_top = center_y + (tile_world_top - coord[v_coord] as f32) * zoom;
        let screen_bottom = center_y + (tile_world_bottom - coord[v_coord] as f32) * zoom;
        
        egui::Rect::from_min_max(
            egui::pos2(screen_left, screen_top),
            egui::pos2(screen_right, screen_bottom),
        )
    }

    pub fn render(
        &self,
        ui: &mut Ui,
        coord: &mut [i32; 3],
        world: &Volume,
        surface_volume: Option<&Rc<dyn SurfaceVolume>>,
        zoom: &mut f32,
        drawing_config: &DrawingConfig,
        extra_resolutions: u32,
        segment_outlines_coord: Option<[i32; 3]>,
        ranges: &[RangeInclusive<i32>; 3],
        cell_size: Vec2,
    ) -> bool {
        // Allocate exact size for this pane
        let (rect, _) = ui.allocate_exact_size(cell_size, egui::Sense::hover());
        let ui = &mut ui.new_child(egui::UiBuilder::new().max_rect(rect));

        let frame_width = cell_size.x as usize;
        let frame_height = cell_size.y as usize;

        // Get or create tiles
        let tiles = self.get_or_create_tiles(
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

        // Use a canvas to paint all tiles
        let (response, painter) = ui.allocate_painter(cell_size, egui::Sense::drag());
        
        // Paint all tiles on the canvas
        for (texture, tile_rect) in tiles {
            painter.image(
                texture.id(),
                tile_rect,
                egui::Rect::from_min_size(egui::Pos2::ZERO, egui::Vec2::splat(1.0)),
                egui::Color32::WHITE,
            );
        }

        // Handle interactions and return whether textures need clearing
        let mut interaction_happened = false;

        if self.handle_scroll(&response, ui, coord, ranges, zoom) {
            interaction_happened = true;
        }

        if self.handle_drag(&response, coord, ranges, *zoom) {
            interaction_happened = true;
        }

        interaction_happened
    }

    pub fn handle_scroll(
        &self,
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
        &self,
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

    fn get_or_create_tiles(
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
    ) -> Vec<(egui::TextureHandle, egui::Rect)> {
        let visible_tiles = self.calculate_visible_tiles(coord, zoom, frame_width, frame_height);
        
        visible_tiles
            .into_iter()
            .map(|(tile_x, tile_y, tile_rect)| {
                let texture = self.get_or_create_tile(
                    ui,
                    tile_x,
                    tile_y,
                    coord,
                    world,
                    surface_volume,
                    zoom,
                    drawing_config,
                    extra_resolutions,
                    segment_outlines_coord,
                );
                (texture, tile_rect)
            })
            .collect()
    }

    fn get_or_create_tile(
        &self,
        ui: &Ui,
        tile_x: i32,
        tile_y: i32,
        coord: [i32; 3],
        world: &Volume,
        surface_volume: Option<&Rc<dyn SurfaceVolume>>,
        zoom: f32,
        drawing_config: &DrawingConfig,
        extra_resolutions: u32,
        segment_outlines_coord: Option<[i32; 3]>,
    ) -> egui::TextureHandle {
        let cache_key = TileCacheKey::new(
            self.pane_type,
            tile_x,
            tile_y,
            coord,
            zoom,
            drawing_config,
            segment_outlines_coord,
            extra_resolutions,
            world,
            surface_volume,
        );

        // Check if tile exists in cache first
        let cached_texture = ui.memory_mut(|mem| {
            let cache: &mut TileCache = mem.caches.cache::<TileCache>();
            cache.get(&cache_key).cloned()
        });

        if let Some(texture) = cached_texture {
            ui.memory_mut(|mem| {
                let cache: &mut TileCache = mem.caches.cache::<TileCache>();
                cache.set(cache_key, texture.clone());
            });
            texture
        } else {
            // Create tile outside of memory lock to avoid deadlock
            let texture = self.create_tile(
                ui,
                tile_x,
                tile_y,
                coord,
                world,
                surface_volume,
                zoom,
                drawing_config,
                extra_resolutions,
                segment_outlines_coord,
            );

            // Store in cache
            ui.memory_mut(|mem| {
                let cache: &mut TileCache = mem.caches.cache::<TileCache>();
                cache.set(cache_key, texture.clone());
            });

            texture
        }
    }

    fn create_tile(
        &self,
        ui: &Ui,
        tile_x: i32,
        tile_y: i32,
        coord: [i32; 3],
        world: &Volume,
        surface_volume: Option<&Rc<dyn SurfaceVolume>>,
        zoom: f32,
        drawing_config: &DrawingConfig,
        extra_resolutions: u32,
        segment_outlines_coord: Option<[i32; 3]>,
    ) -> egui::TextureHandle {
        use std::time::Instant;
        let _start = Instant::now();

        let (u_coord, v_coord, d_coord) = self.pane_type.coordinates();

        // For tiles, always use fixed TILE_SIZE regardless of scaling
        // The scaling is handled in the painting process via paint_zoom
        let tile_width = TILE_SIZE;
        let tile_height = TILE_SIZE;
        let paint_zoom = 1u8;
        let mut image = crate::volume::Image::new(tile_width, tile_height);

        // Calculate world coordinates for this tile
        // Each tile always covers a TILE_SIZE x TILE_SIZE area in world space
        // regardless of the actual rendered tile size (for consistent tiling)
        let tile_world_u = tile_x as f32 * TILE_SIZE as f32;
        let tile_world_v = tile_y as f32 * TILE_SIZE as f32;
        
        // Set tile center in world coordinates  
        let mut tile_coord = coord;
        tile_coord[u_coord] = (tile_world_u + TILE_SIZE as f32 / 2.0) as i32;
        tile_coord[v_coord] = (tile_world_v + TILE_SIZE as f32 / 2.0) as i32;

        let min_level = (32 - ((ZOOM_RES_FACTOR / zoom) as u32).leading_zeros()).min(4).max(0);
        let max_level: u32 = (min_level + extra_resolutions).min(4);

        for level in (min_level..=max_level).rev() {
            let sfactor = 1 << level as u8;
            world.paint(
                tile_coord,
                u_coord,
                v_coord,
                d_coord,
                tile_width,
                tile_height,
                sfactor,
                paint_zoom,
                drawing_config,
                &mut image,
            );
        }

        // Add segment outlines if configured
        if let (Some(surface_vol), Some(outlines_coord)) = (surface_volume, segment_outlines_coord) {
            if !self.is_segment_pane && drawing_config.show_segment_outlines {
                surface_vol.paint_plane_intersection(
                    tile_coord,
                    u_coord,
                    v_coord,
                    d_coord,
                    tile_width,
                    tile_height,
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
            format!("{}_{}_{}_{}", self.pane_type.label(), tile_x, tile_y, coord[d_coord]),
            image,
            Default::default(),
        )
    }
}

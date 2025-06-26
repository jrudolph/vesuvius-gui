use crate::volume::{DrawingConfig, PaintVolume, SurfaceVolume, Volume, VolumeCons};
use egui::cache::FramePublisher;
use egui::{ColorImage, PointerButton, Response, Ui, Vec2};
use futures::FutureExt;
use std::ops::{Deref, RangeInclusive};
use std::pin::Pin;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};
use tokio::runtime::Handle;
use tokio::task::JoinHandle;

const ZOOM_RES_FACTOR: f32 = 1.3;
const TILE_SIZE: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TileCacheKey {
    pane_type: PaneType,
    tile_x: i32,
    tile_y: i32,
    coord: [i32; 3],
    paint_zoom: u8, // The actual paint_zoom level used for rendering
    drawing_config: DrawingConfig,
    segment_outlines_coord: Option<[i32; 3]>,
    extra_resolutions: u32,
    volume_id: usize,
    //surface_volume_id: Option<usize>,
}

impl TileCacheKey {
    fn new(
        pane_type: PaneType,
        tile_x: i32,
        tile_y: i32,
        coord: [i32; 3],
        paint_zoom: u8,
        drawing_config: &DrawingConfig,
        segment_outlines_coord: Option<[i32; 3]>,
        extra_resolutions: u32,
        world: &Volume,
        //surface_volume: Option<&Arc<dyn SurfaceVolume + Send + Sync>>,
    ) -> Self {
        let volume_id = world as *const Volume as usize;
        /* let surface_volume_id = surface_volume.map(|sv| {
            let ptr: *const dyn SurfaceVolume = sv.as_ref();
            ptr as *const () as usize
        }); */

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
            paint_zoom,
            drawing_config: drawing_config.clone(),
            segment_outlines_coord,
            extra_resolutions,
            volume_id,
            //surface_volume_id,
        }
    }
}

struct HandleHolder {
    handle: Option<JoinHandle<egui::ColorImage>>,
}
impl HandleHolder {
    fn is_finished(&self) -> bool {
        self.handle.as_ref().map_or(false, |h| h.is_finished())
    }
    fn now_or_never(mut self) -> Option<Result<egui::ColorImage, tokio::task::JoinError>> {
        self.handle.take().unwrap().now_or_never()
    }
}
impl Drop for HandleHolder {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            //handle.abort();
        }
    }
}

#[derive(Clone)]
enum AsyncTexture {
    Loading(Arc<Mutex<Pin<Box<dyn futures::Future<Output = Arc<ColorImage>> + Send + Sync>>>>),
    Ready(egui::TextureHandle),
}

type TileCache = FramePublisher<TileCacheKey, AsyncTexture>;

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

        // Calculate paint_zoom to determine effective tile size
        let paint_zoom = if zoom >= 1.0 {
            1u8
        } else {
            let downsample_factor = (1.0 / zoom).ceil() as u8;
            downsample_factor.clamp(1, 8)
        };

        // When paint_zoom > 1, the effective tile size in world coordinates is larger
        let effective_tile_size = TILE_SIZE as f32 * paint_zoom as f32;

        // Calculate world space viewport dimensions
        let world_width = frame_width as f32 / zoom;
        let world_height = frame_height as f32 / zoom;

        // Calculate viewport bounds in world coordinates
        let viewport_left = coord[u_coord] as f32 - world_width / 2.0;
        let viewport_right = coord[u_coord] as f32 + world_width / 2.0;
        let viewport_top = coord[v_coord] as f32 - world_height / 2.0;
        let viewport_bottom = coord[v_coord] as f32 + world_height / 2.0;

        #[cfg(debug_assertions)]
        {
            println!("Pane {:?}: u_coord={}, v_coord={}", self.pane_type, u_coord, v_coord);
            println!("  coord=[{},{},{}]", coord[0], coord[1], coord[2]);
            println!(
                "  viewport: left={:.1}, right={:.1}, top={:.1}, bottom={:.1}",
                viewport_left, viewport_right, viewport_top, viewport_bottom
            );
            println!("  effective_tile_size={:.1}", effective_tile_size);
        }

        // Calculate tile range using effective tile size
        let start_tile_x = (viewport_left / effective_tile_size).floor() as i32;
        let end_tile_x = (viewport_right / effective_tile_size).ceil() as i32;
        let start_tile_y = (viewport_top / effective_tile_size).floor() as i32;
        let end_tile_y = (viewport_bottom / effective_tile_size).ceil() as i32;

        // Generate tile list with screen positions
        let mut tiles = Vec::new();
        for tile_y in start_tile_y - 1..end_tile_y + 1 {
            for tile_x in start_tile_x - 1..end_tile_x + 1 {
                let screen_rect =
                    self.calculate_tile_screen_rect(tile_x, tile_y, coord, zoom, frame_width, frame_height);
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

        // Calculate paint_zoom to account for scaling
        let paint_zoom = if zoom >= 1.0 {
            1u8
        } else {
            let downsample_factor = (1.0 / zoom).ceil() as u8;
            downsample_factor.clamp(1, 8)
        };

        // When paint_zoom > 1, the effective tile size in world coordinates is larger
        let effective_tile_size = TILE_SIZE as f32 * paint_zoom as f32;

        // Tile bounds in world coordinates using effective tile size
        let tile_world_left = tile_x as f32 * effective_tile_size;
        let tile_world_right = (tile_x + 1) as f32 * effective_tile_size;
        let tile_world_top = tile_y as f32 * effective_tile_size;
        let tile_world_bottom = (tile_y + 1) as f32 * effective_tile_size;

        // Convert to screen coordinates relative to the pane's viewport center
        // The painter uses coordinates relative to the allocated UI area (0,0 to frame_width,frame_height)
        let center_x = frame_width as f32 / 2.0;
        let center_y = frame_height as f32 / 2.0;

        // Calculate screen position relative to current view center
        let screen_left = center_x + (tile_world_left - coord[u_coord] as f32) * zoom;
        let screen_right = center_x + (tile_world_right - coord[u_coord] as f32) * zoom;
        let screen_top = center_y + (tile_world_top - coord[v_coord] as f32) * zoom;
        let screen_bottom = center_y + (tile_world_bottom - coord[v_coord] as f32) * zoom;

        // Ensure coordinates are within reasonable bounds for the pane
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
        //surface_volume: Option<&Arc<dyn SurfaceVolume + Send + Sync>>,
        zoom: &mut f32,
        drawing_config: &DrawingConfig,
        extra_resolutions: u32,
        segment_outlines_coord: Option<[i32; 3]>,
        ranges: &[RangeInclusive<i32>; 3],
        cell_size: Vec2,
    ) -> bool {
        let frame_width = cell_size.x as usize;
        let frame_height = cell_size.y as usize;

        // Get or create tiles
        let tiles = self.get_or_create_tiles(
            ui,
            *coord,
            world,
            //surface_volume,
            *zoom,
            frame_width,
            frame_height,
            drawing_config,
            extra_resolutions,
            segment_outlines_coord,
        );

        // Allocate space for this pane using the proper egui pattern
        let (response, painter) = ui.allocate_painter(cell_size, egui::Sense::drag());

        // Paint all tiles on the allocated space - tiles should use response.rect coordinate system
        for (texture, tile_rect) in tiles {
            // Adjust tile_rect to be relative to response.rect
            let adjusted_rect =
                egui::Rect::from_min_size(response.rect.min + tile_rect.min.to_vec2(), tile_rect.size());

            painter.image(
                texture.id(),
                adjusted_rect,
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
        //surface_volume: Option<&Arc<dyn SurfaceVolume + Send + Sync>>,
        zoom: f32,
        frame_width: usize,
        frame_height: usize,
        drawing_config: &DrawingConfig,
        extra_resolutions: u32,
        segment_outlines_coord: Option<[i32; 3]>,
    ) -> Vec<(egui::TextureHandle, egui::Rect)> {
        let visible_tiles = self.calculate_visible_tiles(coord, zoom, frame_width, frame_height);
        let mut ready_tiles = Vec::new();

        for (tile_x, tile_y, tile_rect) in visible_tiles {
            if let Some(texture) = self.get_or_create_tile_async(
                ui,
                tile_x,
                tile_y,
                coord,
                world,
                //surface_volume,
                zoom,
                drawing_config,
                extra_resolutions,
                segment_outlines_coord,
            ) {
                ready_tiles.push((texture, tile_rect));
            }
        }

        ready_tiles
    }

    fn get_or_create_tile_async(
        &self,
        ui: &Ui,
        tile_x: i32,
        tile_y: i32,
        coord: [i32; 3],
        world: &Volume,
        //surface_volume: Option<&Arc<dyn SurfaceVolume + Send + Sync>>,
        zoom: f32,
        drawing_config: &DrawingConfig,
        extra_resolutions: u32,
        segment_outlines_coord: Option<[i32; 3]>,
    ) -> Option<egui::TextureHandle> {
        // Calculate paint_zoom for cache key (same logic as in create_tile)
        let paint_zoom = if zoom >= 1.0 {
            1u8
        } else {
            // For zoom < 1.0, use integer downsampling
            let downsample_factor = (1.0 / zoom).ceil() as u8;
            downsample_factor.clamp(1, 8) // Reasonable limits
        };

        let cache_key = TileCacheKey::new(
            self.pane_type,
            tile_x,
            tile_y,
            coord,
            paint_zoom,
            drawing_config,
            segment_outlines_coord,
            extra_resolutions,
            world,
            //surface_volume,
        );

        // Check if tile exists in cache
        let cached_value = ui.memory_mut(|mem| {
            let cache: &mut TileCache = mem.caches.cache::<TileCache>();

            // Clone the cached value to avoid borrow conflicts
            cache.get(&cache_key).cloned()
        });
        fn set(ui: &Ui, key: TileCacheKey, value: AsyncTexture) {
            ui.memory_mut(|mem| {
                let cache: &mut TileCache = mem.caches.cache::<TileCache>();
                cache.set(key, value);
            });
        }

        match cached_value {
            Some(AsyncTexture::Ready(texture)) => {
                // Texture is ready, return it and refresh cache
                set(ui, cache_key, AsyncTexture::Ready(texture.clone()));
                Some(texture)
            }
            Some(AsyncTexture::Loading(future_mutex)) => {
                let mut future_guard = future_mutex.lock().unwrap();

                // Create a waker that will work properly with tokio futures
                let waker = futures::task::noop_waker();
                let mut context = Context::from_waker(&waker);

                // Poll the future in a blocking context to get accurate results
                let poll_result = tokio::task::block_in_place(|| {
                    /* Handle::current().block_on(async {
                        Poll::Ready(future_guard.as_mut().await)
                    }) */
                    future_guard.as_mut().poll(&mut context)
                });

                match poll_result {
                    Poll::Ready(image) => {
                        //println!("Tile ({}, {}) for pane {:?} is ready", tile_x, tile_y, self.pane_type);
                        let texture = ui.ctx().load_texture(
                            format!(
                                "{}_{}_{}_{}",
                                self.pane_type.label(),
                                tile_x,
                                tile_y,
                                self.pane_type.coordinates().2
                            ),
                            image.as_ref().clone(),
                            Default::default(),
                        );
                        // Store the ready texture and return it
                        drop(future_guard); // Release lock before cache operation
                        set(ui, cache_key, AsyncTexture::Ready(texture.clone()));
                        Some(texture)
                    }
                    Poll::Pending => {
                        /* println!(
                            "Tile ({}, {}) for pane {:?} is still loading",
                            tile_x, tile_y, self.pane_type
                        ); */
                        // Still loading, refresh cache entry to keep it alive and request repaint
                        let future_clone = future_mutex.clone();
                        drop(future_guard); // Release lock before cache operation
                        set(ui, cache_key, AsyncTexture::Loading(future_clone));
                        ui.ctx().request_repaint();
                        None
                    }
                }

                /* if handle.is_finished() {
                    /* // Take ownership of the handle to consume it
                    if let Some(owned_handle) = handle_guard.take() {
                        // Release the lock before calling now_or_never
                        drop(handle_guard);

                        match owned_handle.now_or_never() {
                            Some(Ok(image)) => {
                                let texture = ui.ctx().load_texture(
                                    format!(
                                        "{}_{}_{}_{}",
                                        self.pane_type.label(),
                                        tile_x,
                                        tile_y,
                                        self.pane_type.coordinates().2
                                    ),
                                    image,
                                    Default::default(),
                                );
                                // Store the ready texture and return it
                                set(ui, cache_key, AsyncTexture::Ready(texture.clone()));
                                Some(texture)
                            }
                            Some(Err(_)) => {
                                // Task failed, don't cache anything, let it retry
                                println!("Error loading tile ({}, {}): task failed", tile_x, tile_y);
                                None
                            }
                            None => {
                                println!("Error loading tile ({}, {}): task not finished", tile_x, tile_y);
                                // This shouldn't happen if is_finished() was true
                                ui.ctx().request_repaint();
                                None
                            }
                        }
                    } else {
                        None
                    } */
                } else {
                    // Still loading, refresh cache entry to keep it alive and request repaint
                    let handle_mutex_clone = handle_mutex.clone();
                    drop(handle_guard); // Release the lock
                    set(ui, cache_key, AsyncTexture::Loading(handle_mutex_clone));
                    ui.ctx().request_repaint();
                    None
                } */
                /* } else {
                    // Handle was already taken, this shouldn't happen normally
                    None
                } */
            }
            None => {
                // Start async rendering
                let handle = self.create_tile_async(
                    tile_x,
                    tile_y,
                    coord,
                    world.shared(),
                    //surface_volume.cloned(),
                    zoom,
                    drawing_config.clone(),
                    extra_resolutions,
                    segment_outlines_coord,
                );

                set(ui, cache_key, AsyncTexture::Loading(handle));
                // Request repaint to check again later
                ui.ctx().request_repaint();
                None
            }
        }
    }

    fn create_tile_async(
        &self,
        tile_x: i32,
        tile_y: i32,
        coord: [i32; 3],
        world: VolumeCons,
        //surface_volume: Option<Arc<dyn SurfaceVolume + Send + Sync>>,
        zoom: f32,
        drawing_config: DrawingConfig,
        extra_resolutions: u32,
        segment_outlines_coord: Option<[i32; 3]>,
    ) -> Arc<Mutex<Pin<Box<dyn futures::Future<Output = Arc<ColorImage>> + Send + Sync>>>> {
        let pane_type = self.pane_type;
        let is_segment_pane = self.is_segment_pane;

        let handle = tokio::task::spawn_blocking(move || {
            /* println!(
                "Creating tile ({}, {}) for pane {:?} at coord {:?} with zoom {:.2}",
                tile_x, tile_y, pane_type, coord, zoom
            ); */
            let volume_pane = VolumePane::new(pane_type, is_segment_pane);
            let image = volume_pane.create_tile_sync(
                tile_x,
                tile_y,
                coord,
                &world(),
                //surface_volume, //.as_ref().map(|sv| sv),
                zoom,
                &drawing_config,
                extra_resolutions,
                segment_outlines_coord,
            );
            /* println!(
                "Finished creating tile ({}, {}) for pane {:?} at coord {:?} with zoom {:.2}",
                tile_x, tile_y, pane_type, coord, zoom
            ); */
            Arc::new(image)
        });

        // Map the JoinError to a default error image and box the future
        let future: Pin<Box<dyn futures::Future<Output = Arc<ColorImage>> + Send + Sync>> = Box::pin(async move {
            match handle.await {
                Ok(image) => image,
                Err(_join_error) => {
                    println!("Error loading tile ({}, {}): task failed", tile_x, tile_y);
                    // Return a simple error image
                    Arc::new(egui::ColorImage::example())
                }
            }
        });

        Arc::new(Mutex::new(future))
    }

    fn create_tile_sync(
        &self,
        tile_x: i32,
        tile_y: i32,
        coord: [i32; 3],
        world: &Volume,
        //surface_volume: Option<Arc<dyn SurfaceVolume + Send + Sync>>,
        zoom: f32,
        drawing_config: &DrawingConfig,
        extra_resolutions: u32,
        segment_outlines_coord: Option<[i32; 3]>,
    ) -> egui::ColorImage {
        use std::time::Instant;
        let _start = Instant::now();

        let (u_coord, v_coord, d_coord) = self.pane_type.coordinates();

        // Use integer paint zoom levels like the original code
        let paint_zoom = if zoom >= 1.0 {
            1u8
        } else {
            // For zoom < 1.0, use integer downsampling
            let downsample_factor = (1.0 / zoom).ceil() as u8;
            downsample_factor.clamp(1, 8) // Reasonable limits
        };

        // Always use fixed tile size - let paint_zoom handle the scaling
        let tile_width = TILE_SIZE;
        let tile_height = TILE_SIZE;
        let mut image = crate::volume::Image::new(tile_width, tile_height);

        // Calculate world coordinates for this tile
        // When paint_zoom > 1, each tile covers a larger world area
        let effective_tile_size = TILE_SIZE as f32 * paint_zoom as f32;

        // tile_x corresponds to u_coord, tile_y corresponds to v_coord
        let tile_world_u = tile_x as f32 * effective_tile_size;
        let tile_world_v = tile_y as f32 * effective_tile_size;

        // Set tile center in world coordinates for this pane's coordinate system
        let mut tile_coord = coord;
        tile_coord[u_coord] = (tile_world_u + effective_tile_size / 2.0) as i32;
        tile_coord[v_coord] = (tile_world_v + effective_tile_size / 2.0) as i32;

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
        /* if let (Some(surface_vol), Some(outlines_coord)) = (surface_volume, segment_outlines_coord) {
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
        } */

        let image: egui::ColorImage = image.into();
        image
    }
}

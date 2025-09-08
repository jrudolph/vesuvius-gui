use crate::volume::{PaintVolume, VoxelPaintVolume, VoxelVolume};

use super::{DrawingConfig, Image, SurfaceVolume};

pub struct EmptyVolume {}
impl VoxelVolume for EmptyVolume {
    fn get(&self, _xyz: [f64; 3], _downsampling: i32) -> u8 {
        0
    }
}

impl PaintVolume for EmptyVolume {
    fn paint(
        &self,
        _xyz: [i32; 3],
        _u_coord: usize,
        _v_coord: usize,
        _plane_coord: usize,
        _width: usize,
        _height: usize,
        _sfactor: u8,
        _paint_zoom: u8,
        _config: &DrawingConfig,
        _buffer: &mut Image,
    ) {
    }
    fn shared(&self) -> super::VolumeCons {
        Box::new(|| EmptyVolume {}.into_volume())
    }
}

impl SurfaceVolume for EmptyVolume {
    fn paint_plane_intersection(
        &self,
        _xyz: [i32; 3],
        _u_coord: usize,
        _v_coord: usize,
        _plane_coord: usize,
        _width: usize,
        _height: usize,
        _sfactor: u8,
        _paint_zoom: u8,
        _highlight_uv_section: Option<[i32; 3]>,
        _config: &DrawingConfig,
        _buffer: &mut Image,
    ) {
    }
}

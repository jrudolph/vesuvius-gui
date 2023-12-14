use crate::volume::{PaintVolume, VoxelVolume};

use super::DrawingConfig;

pub struct EmptyVolume {}
impl VoxelVolume for EmptyVolume {
    fn get(&mut self, _xyz: [f64; 3], _downsampling: i32) -> u8 {
        0
    }
}

impl PaintVolume for EmptyVolume {
    fn paint(
        &mut self,
        _xyz: [i32; 3],
        _u_coord: usize,
        _v_coord: usize,
        _plane_coord: usize,
        _width: usize,
        _height: usize,
        _sfactor: u8,
        _paint_zoom: u8,
        _config: &DrawingConfig,
        _buffer: &mut [u8],
    ) {
    }
}

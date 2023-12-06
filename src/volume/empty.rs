use crate::volume::{PaintVolume, VoxelVolume};

use super::DrawingConfig;

pub struct EmptyVolume {}
impl VoxelVolume for EmptyVolume {
    fn get(&mut self, _xyz: [i32; 3]) -> u8 {
        0
    }
}

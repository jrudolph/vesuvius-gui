use crate::volume::{PaintVolume, VoxelVolume};

pub struct EmptyWorld {}
impl VoxelVolume for EmptyWorld {
    fn get(&mut self, _xyz: [i32; 3]) -> u8 { 0 }
}
impl PaintVolume for EmptyWorld {
    fn paint(&mut self, xyz: [i32; 3], u_coord: usize, v_coord: usize, plane_coord: usize, width: usize, height: usize, sfactor: u8, buffer: &mut [u8]) {}
}
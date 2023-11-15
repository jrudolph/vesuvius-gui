mod empty;
mod volume64x4;

pub use empty::EmptyVolume;
pub use volume64x4::VolumeGrid64x4Mapped;

pub trait VoxelVolume {
    fn get(&mut self, xyz: [i32; 3]) -> u8;
}

pub trait PaintVolume {
    fn paint(
        &mut self,
        xyz: [i32; 3],
        u_coord: usize,
        v_coord: usize,
        plane_coord: usize,
        width: usize,
        height: usize,
        sfactor: u8,
        buffer: &mut [u8],
    );
}

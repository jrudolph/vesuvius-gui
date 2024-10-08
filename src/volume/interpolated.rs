use super::{AutoPaintVolume, VoxelPaintVolume, VoxelVolume};
use libm::modf;
use std::{cell::RefCell, sync::Arc};

pub struct TrilinearInterpolatedVolume {
    pub base: Arc<RefCell<dyn VoxelPaintVolume>>,
}

impl VoxelVolume for TrilinearInterpolatedVolume {
    fn get(&mut self, xyz: [f64; 3], downsampling: i32) -> u8 {
        let (dx, x0) = modf(xyz[0]);
        let x1 = x0 + 1.0;
        let (dy, y0) = modf(xyz[1]);
        let y1 = y0 + 1.0;
        let (dz, z0) = modf(xyz[2]);
        let z1 = z0 + 1.0;

        let mut base = self.base.borrow_mut();

        let c00 =
            base.get([x0, y0, z0], downsampling) as f64 * (1.0 - dx) + base.get([x1, y0, z0], downsampling) as f64 * dx;
        let c10 =
            base.get([x0, y1, z0], downsampling) as f64 * (1.0 - dx) + base.get([x1, y1, z0], downsampling) as f64 * dx;
        let c01 =
            base.get([x0, y0, z1], downsampling) as f64 * (1.0 - dx) + base.get([x1, y0, z1], downsampling) as f64 * dx;
        let c11 =
            base.get([x0, y1, z1], downsampling) as f64 * (1.0 - dx) + base.get([x1, y1, z1], downsampling) as f64 * dx;

        let c0 = c00 * (1.0 - dy) + c10 * dy;
        let c1 = c01 * (1.0 - dy) + c11 * dy;

        let c = c0 * (1.0 - dz) + c1 * dz;

        c as u8
    }
}
impl AutoPaintVolume for TrilinearInterpolatedVolume {}

use super::{DrawingConfig, PaintVolume, VoxelVolume};

// marker trait for volumes that do not want to provide a specific PaintVolume implementation
pub trait AutoPaintVolume {}

impl<T: VoxelVolume + AutoPaintVolume> PaintVolume for T {
    fn paint(
        &mut self,
        xyz: [i32; 3],
        u_coord: usize,
        v_coord: usize,
        plane_coord: usize,
        width: usize,
        height: usize,
        _sfactor: u8,
        paint_zoom: u8,
        _config: &DrawingConfig,
        buffer: &mut [u8],
    ) {
        let fi32 = _sfactor as i32;

        for im_v in 0..height {
            for im_u in 0..width {
                let im_rel_u = (im_u as i32 - width as i32 / 2) * paint_zoom as i32;
                let im_rel_v = (im_v as i32 - height as i32 / 2) * paint_zoom as i32;

                let mut uvw: [i32; 3] = [0; 3];
                uvw[u_coord] = (xyz[u_coord] + im_rel_u) / fi32;
                uvw[v_coord] = (xyz[v_coord] + im_rel_v) / fi32;
                uvw[plane_coord] = (xyz[plane_coord]) / fi32;

                let v = self.get(uvw, fi32);
                buffer[im_v * width + im_u] = v;
            }
        }
    }
}

use super::{PaintVolume, Volume, VoxelVolume};
use egui::Color32;

pub struct OverlayVolume {
    first: Volume,
    second: Volume,
    alpha: f32,
}
impl OverlayVolume {
    pub fn new(first: Volume, second: Volume, alpha: f32) -> Self {
        Self { first, second, alpha }
    }
}

impl VoxelVolume for OverlayVolume {
    fn get(&mut self, xyz: [f64; 3], downsampling: i32) -> u8 {
        let first = self.first.get(xyz, downsampling) as f32;
        let second = self.second.get(xyz, downsampling) as f32;

        (first * (1.0 - self.alpha) + second * self.alpha) as u8
    }
    fn get_interpolated(&mut self, xyz: [f64; 3], downsampling: i32) -> u8 {
        let first = self.first.get_interpolated(xyz, downsampling) as f32;
        let second = self.second.get_interpolated(xyz, downsampling) as f32;

        (first * (1.0 - self.alpha) + second * self.alpha) as u8
    }
    fn get_color(&mut self, xyz: [f64; 3], downsampling: i32) -> egui::Color32 {
        let first = self.first.get_color(xyz, downsampling);
        let second = self.second.get_color(xyz, downsampling);

        Color32::from_rgb(
            (first.r() as f32 * (1.0 - self.alpha) + second.r() as f32 * self.alpha) as u8,
            (first.g() as f32 * (1.0 - self.alpha) + second.g() as f32 * self.alpha) as u8,
            (first.b() as f32 * (1.0 - self.alpha) + second.b() as f32 * self.alpha) as u8,
        )
    }
    fn get_color_interpolated(&mut self, xyz: [f64; 3], downsampling: i32) -> Color32 {
        let first = self.first.get_color_interpolated(xyz, downsampling);
        let second = self.second.get_color_interpolated(xyz, downsampling);

        Color32::from_rgb(
            (first.r() as f32 * (1.0 - self.alpha) + second.r() as f32 * self.alpha) as u8,
            (first.g() as f32 * (1.0 - self.alpha) + second.g() as f32 * self.alpha) as u8,
            (first.b() as f32 * (1.0 - self.alpha) + second.b() as f32 * self.alpha) as u8,
        )
    }
}
impl PaintVolume for OverlayVolume {
    #[allow(unused)]
    fn paint(
        &mut self,
        xyz: [i32; 3],
        u_coord: usize,
        v_coord: usize,
        plane_coord: usize,
        width: usize,
        height: usize,
        sfactor: u8,
        paint_zoom: u8,
        config: &super::DrawingConfig,
        buffer: &mut super::Image,
    ) {
        todo!()
    }
}

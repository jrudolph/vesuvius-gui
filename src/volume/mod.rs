mod empty;
mod generic;
mod grid500;
mod interpolated;
mod layers;
mod objvolume;
mod ppmvolume;
mod volume64x4;

use egui::{Color32, ColorImage};
pub use empty::EmptyVolume;
pub use generic::AutoPaintVolume;
pub use grid500::VolumeGrid500Mapped;
pub use interpolated::TrilinearInterpolatedVolume;
pub use layers::LayersMappedVolume;
pub use objvolume::{ObjFile, ObjVolume};
pub use ppmvolume::PPMVolume;
pub use volume64x4::VolumeGrid64x4Mapped;

#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct DrawingConfig {
    pub enable_filters: bool,
    pub threshold_min: u8,
    pub threshold_max: u8,
    pub quant: u8,
    pub mask_shift: u8,
    pub draw_xyz_outlines: bool,
    pub draw_outline_vertices: bool,
}
impl DrawingConfig {
    pub fn filters_active(&self) -> bool {
        self.enable_filters
            && (self.threshold_min > 0 || self.threshold_max > 0 || self.quant < 8 || self.mask_shift > 0)
    }
    pub fn bit_mask(&self) -> u8 {
        (match self.quant {
            8 => 0xff,
            7 => 0xfe,
            6 => 0xfc,
            5 => 0xf8,
            4 => 0xf0,
            3 => 0xe0,
            2 => 0xc0,
            1 => 0x80,
            _ => 0xff,
        }) >> self.mask_shift
    }
}
impl Default for DrawingConfig {
    fn default() -> Self {
        Self {
            enable_filters: false,
            threshold_min: 0,
            threshold_max: 0,
            quant: 0xff,
            mask_shift: 0,
            draw_xyz_outlines: false,
            draw_outline_vertices: false,
        }
    }
}

pub trait VoxelVolume {
    fn get(&mut self, xyz: [f64; 3], downsampling: i32) -> u8;
}

pub struct Image {
    width: usize,
    height: usize,
    pub data: Vec<Color32>,
}
impl Image {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![Color32::BLACK; width * height],
        }
    }

    pub fn set(&mut self, x: usize, y: usize, value: Color32) {
        self.data[y * self.width + x] = value;
    }
    pub fn set_rgb(&mut self, x: usize, y: usize, r: u8, g: u8, b: u8) {
        self.set(x, y, Color32::from_rgb(r, g, b));
    }
    pub fn set_gray(&mut self, x: usize, y: usize, value: u8) {
        self.set(x, y, Color32::from_gray(value));
    }
}
impl From<Image> for ColorImage {
    fn from(value: Image) -> Self {
        ColorImage {
            size: [value.width, value.height],
            pixels: value.data,
        }
    }
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
        paint_zoom: u8,
        config: &DrawingConfig,
        buffer: &mut Image,
    );
}

pub trait VoxelPaintVolume: PaintVolume + VoxelVolume {}
impl<T: PaintVolume + VoxelVolume> VoxelPaintVolume for T {}

pub trait SurfaceVolume: PaintVolume + VoxelVolume {
    fn paint_plane_intersection(
        &self,
        xyz: [i32; 3],
        u_coord: usize,
        v_coord: usize,
        plane_coord: usize,
        width: usize,
        height: usize,
        sfactor: u8,
        paint_zoom: u8,
        highlight_uv_section: Option<[i32; 3]>,
        config: &DrawingConfig,
        buffer: &mut Image,
    );
}

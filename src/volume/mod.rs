mod empty;
mod volume64x4;

pub use empty::EmptyVolume;
pub use volume64x4::VolumeGrid64x4Mapped;

#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct DrawingConfig {
    pub threshold_min: u8,
    pub threshold_max: u8,
    pub quant: u8,
}
impl DrawingConfig {
    pub fn bit_mask(&self) -> u8 {
        match self.quant {
            8 => 0xff,
            7 => 0xfe,
            6 => 0xfc,
            5 => 0xf8,
            4 => 0xf0,
            3 => 0xe0,
            2 => 0xc0,
            1 => 0x80,
            _ => 0xff,
        }
    }
}
impl Default for DrawingConfig {
    fn default() -> Self {
        Self {
            threshold_min: 0,
            threshold_max: 0,
            quant: 0xff,
        }
    }
}

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
        paint_zoom: u8,
        config: &DrawingConfig,
        buffer: &mut [u8],
    );
}

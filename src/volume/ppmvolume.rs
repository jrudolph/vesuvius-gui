use memmap::MmapOptions;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, Seek, SeekFrom};

use super::{AutoPaintVolume, VoxelPaintVolume, VoxelVolume};
use libm::modf;
use std::cell::RefCell;
use std::sync::Arc;

pub struct PPMFile {
    pub width: usize,
    pub height: usize,
    map: memmap::Mmap,
}
impl PPMFile {
    pub fn new(file_name: &str) -> Option<Self> {
        let file = File::open(file_name).ok()?;
        let mut reader = std::io::BufReader::new(&file);

        let mut header_map = HashMap::new();
        let mut line = String::new();

        let mut end_pos = 0;
        while let Ok(len) = reader.read_line(&mut line) {
            if line.starts_with("<>\n") {
                end_pos = reader.seek(SeekFrom::Current(0)).ok()?;
                break;
            }
            let mut split = line[..len].split(": ");
            let key = split.next()?.trim().to_string();
            let value = split.next()?.trim().to_string();
            header_map.insert(key, value);
            line.clear();
        }
        let width = header_map.get("width")?.parse::<usize>().ok()?;
        let height = header_map.get("height")?.parse::<usize>().ok()?;

        let map = unsafe { MmapOptions::new().offset(end_pos as u64).map(&file) }.ok();

        map.map(|map| Self { width, height, map })
    }
    pub fn get(&self, u: usize, v: usize) -> [f64; 6] {
        let map = unsafe { std::slice::from_raw_parts(self.map.as_ptr() as *const f64, self.map.len() / 8) };
        let off = (v * self.width + u) * 6;
        [
            map[off + 0],
            map[off + 1],
            map[off + 2],
            map[off + 3],
            map[off + 4],
            map[off + 5],
        ]
    }
}

pub struct PPMVolume {
    volume: Arc<RefCell<dyn VoxelPaintVolume>>,
    ppm: PPMFile,
    /// should the volume use bilinear interpolation
    interpolate: bool,
}
impl PPMVolume {
    pub fn new(ppm_file: &str, base_volume: Arc<RefCell<dyn VoxelPaintVolume>>) -> Self {
        let ppm = PPMFile::new(ppm_file).unwrap();

        Self {
            volume: base_volume,
            ppm,
            interpolate: false,
        }
    }
    pub fn enable_bilinear_interpolation(&mut self) { self.interpolate = true; }
    pub fn convert_to_world_coords(&self, coord: [i32; 3]) -> [i32; 3] {
        let xyz = self.ppm.get(coord[0] as usize, coord[1] as usize);
        [
            (xyz[0] + coord[2] as f64 * xyz[3]) as i32,
            (xyz[1] + coord[2] as f64 * xyz[4]) as i32,
            (xyz[2] + coord[2] as f64 * xyz[5]) as i32,
        ]
    }
    pub fn width(&self) -> usize { self.ppm.width }
    pub fn height(&self) -> usize { self.ppm.height }
}

impl VoxelVolume for PPMVolume {
    fn get(&mut self, xyz: [f64; 3], downsampling: i32) -> u8 {
        let uvw: [i32; 3] = [
            xyz[0] as i32 * downsampling,
            xyz[1] as i32 * downsampling,
            xyz[2] as i32 * downsampling,
        ];

        if uvw[0] <= 0
            || uvw[0] >= self.ppm.width as i32
            || uvw[1] <= 0
            || uvw[1] >= self.ppm.height as i32
            || uvw[2].abs() > 45
        {
            return 0;
        }

        let [x0, y0, z0, nx, ny, nz] = if self.interpolate {
            let (du, u0) = modf(xyz[0]);
            let u1 = u0 + 1.0;
            let (dv, v0) = modf(xyz[1]);
            let v1 = v0 + 1.0;

            let [x00, y00, z00, nx00, ny00, nz00] = self.ppm.get(u0 as usize, v0 as usize);
            let [x10, y10, z10, nx10, ny10, nz10] = self.ppm.get(u1 as usize, v0 as usize);
            let [x01, y01, z01, nx01, ny01, nz01] = self.ppm.get(u0 as usize, v1 as usize);
            let [x11, y11, z11, nx11, ny11, nz11] = self.ppm.get(u1 as usize, v1 as usize);

            fn interpolate(x00: f64, x10: f64, x01: f64, x11: f64, dx: f64, dy: f64) -> f64 {
                x00 * (1.0 - dx) * (1.0 - dy) + x10 * dx * (1.0 - dy) + x01 * (1.0 - dx) * dy + x11 * dx * dy
            }
            [
                interpolate(x00, x10, x01, x11, du, dv),
                interpolate(y00, y10, y01, y11, du, dv),
                interpolate(z00, z10, z01, z11, du, dv),
                interpolate(nx00, nx10, nx01, nx11, du, dv),
                interpolate(ny00, ny10, ny01, ny11, du, dv),
                interpolate(nz00, nz10, nz01, nz11, du, dv),
            ]
        } else {
            self.ppm.get(uvw[0] as usize, uvw[1] as usize)
        };

        if x0 == 0.0 && y0 == 0.0 && z0 == 0.0 {
            return 0;
        }

        let x = x0 + uvw[2] as f64 * nx;
        let y = y0 + uvw[2] as f64 * ny;
        let z = z0 + uvw[2] as f64 * nz;
        self.volume.borrow_mut().get(
            [
                x / downsampling as f64,
                y / downsampling as f64,
                z / downsampling as f64,
            ],
            downsampling,
        )
    }
}
impl AutoPaintVolume for PPMVolume {}

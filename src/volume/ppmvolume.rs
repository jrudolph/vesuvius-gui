use memmap::MmapOptions;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, Seek, SeekFrom};

use super::{AutoPaintVolume, VoxelPaintVolume, VoxelVolume};

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
    volume: Box<dyn VoxelPaintVolume>,
    ppm: PPMFile,
}
impl PPMVolume {
    pub fn new(ppm_file: &str, base_volume: Box<dyn VoxelPaintVolume>) -> Self {
        let ppm = PPMFile::new(ppm_file).unwrap();

        Self {
            volume: base_volume,
            ppm,
        }
    }
    pub fn width(&self) -> usize {
        self.ppm.width
    }
    pub fn height(&self) -> usize {
        self.ppm.height
    }
}

impl VoxelVolume for PPMVolume {
    fn get(&mut self, xyz: [i32; 3], downsampling: i32) -> u8 {
        let uvw: [i32; 3] = [xyz[0] * downsampling, xyz[1] * downsampling, xyz[2] * downsampling];

        if uvw[0] <= 0
            || uvw[0] >= self.ppm.width as i32
            || uvw[1] <= 0
            || uvw[1] >= self.ppm.height as i32
            || uvw[2].abs() > 30
        {
            return 0;
        }

        let [x0, y0, z0, nx, ny, nz] = self.ppm.get(uvw[0] as usize, uvw[1] as usize);

        if x0 == 0.0 && y0 == 0.0 && z0 == 0.0 {
            return 0;
        }

        let x = x0 + uvw[2] as f64 * nx;
        let y = y0 + uvw[2] as f64 * ny;
        let z = z0 + uvw[2] as f64 * nz;
        self.volume.get(
            [
                x as i32 / downsampling,
                y as i32 / downsampling,
                z as i32 / downsampling,
            ],
            downsampling,
        )
    }
}
impl AutoPaintVolume for PPMVolume {}

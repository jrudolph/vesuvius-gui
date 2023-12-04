use memmap::MmapOptions;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, Seek, SeekFrom};

use crate::model::Quality;

use super::volume64x4::TileState;
use super::{DrawingConfig, PaintVolume, VolumeGrid64x4Mapped};

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
    volume: VolumeGrid64x4Mapped,
    ppm: PPMFile,
}
impl PPMVolume {
    pub fn new(ppm_file: &str, base_volume: VolumeGrid64x4Mapped) -> Self {
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
impl PaintVolume for PPMVolume {
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
        config: &DrawingConfig,
        buffer: &mut [u8],
    ) {
        let sfactor = _sfactor as usize;
        /* if plane_coord != 2 {
            return;
        } */

        let mut last_tile: [usize; 4] = [0; 4];
        let mut last_state: &TileState = &TileState::Missing;

        for im_v in 0..height {
            for im_u in 0..width {
                let im_rel_u = (im_u as i32 - width as i32 / 2) * paint_zoom as i32;
                let im_rel_v = (im_v as i32 - height as i32 / 2) * paint_zoom as i32;

                let mut uvw: [i32; 3] = [0; 3];
                uvw[u_coord] = xyz[u_coord] + im_rel_u;
                uvw[v_coord] = xyz[v_coord] + im_rel_v;
                uvw[plane_coord] = xyz[plane_coord];

                /* if u == 300 && v == 300 {
                    println!("u: {} v: {} gu: {} gv: {}", u, v, gu, gv);
                } */
                if uvw[0] <= 0
                    || uvw[0] >= self.ppm.width as i32
                    || uvw[1] <= 0
                    || uvw[1] >= self.ppm.height as i32
                    || uvw[2].abs() > 30
                {
                    continue;
                }

                let [x0, y0, z0, nx, ny, nz] = self.ppm.get(uvw[0] as usize, uvw[1] as usize);

                if x0 == 0.0 && y0 == 0.0 && z0 == 0.0 {
                    continue;
                }

                let x = x0 + uvw[2] as f64 * nx;
                let y = y0 + uvw[2] as f64 * ny;
                let z = z0 + uvw[2] as f64 * nz;

                let tile = [
                    x.round() as usize / 64 / sfactor,
                    y.round() as usize / 64 / sfactor,
                    z.round() as usize / 64 / sfactor,
                    sfactor,
                ];
                let state = if tile == last_tile {
                    last_state
                } else {
                    last_tile = tile;
                    last_state = self.volume.try_loading_tile(
                        tile[0],
                        tile[1],
                        tile[2],
                        Quality {
                            bit_mask: 0xff,
                            downsampling_factor: tile[3] as u8,
                        },
                    );
                    last_state
                };

                /* if u == 300 && v == 300 {
                    println!("u: {} v: {} gu: {} gv: {}", u, v, gu, gv);
                    println!("x: {} y: {} z: {}", x, y, z);
                    println!(
                        "nx: {} ny: {} nz: {} len: {}",
                        nx,
                        ny,
                        nz,
                        (nx * nx + ny * ny + nz * nz).sqrt()
                    );
                    println!("state: {:?}", state);
                } */

                if let TileState::Loaded(tile) = state {
                    let tile_x = (x.round() as usize / sfactor) % 64;
                    let tile_y = (y.round() as usize / sfactor) as usize % 64;
                    let tile_z = (z.round() as usize / sfactor) as usize % 64;

                    let xblock = tile_x / 4;
                    let yblock = tile_y / 4;
                    let zblock = tile_z / 4;

                    let xoff = tile_x % 4;
                    let yoff = tile_y % 4;
                    let zoff = tile_z % 4;

                    let block = zblock * 256 + yblock * 16 + xblock;
                    let block_off = zoff * 16 + yoff * 4 + xoff;
                    let off = block * 64 + block_off;

                    /* if u == 300 && v == 300 {
                        println!("tile_x: {} tile_y: {} tile_z: {}", tile_x, tile_y, tile_z);
                        println!("tile_x: {:06b} tile_y: {:06b} tile_z: {:06b}", tile_x, tile_y, tile_z);
                        println!("xblock: {} yblock: {} zblock: {}", xblock, yblock, zblock);
                        println!("xoff: {} yoff: {} zoff: {}", xoff, yoff, zoff);
                        println!("block: {} block_off: {} off: {}", block, block_off, off);
                        println!("block: {:b} block_off: {:b} off: {:b}", block, block_off, off);
                    } */

                    if off > 0 && off < tile.len() {
                        buffer[im_v * width + im_u] = tile[off];
                    }
                }
            }
        }
    }
}

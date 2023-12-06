use super::VoxelVolume;

pub struct VolumeGrid500Mapped {
    max_x: usize,
    max_y: usize,
    max_z: usize,
    data: Vec<Vec<Vec<Option<memmap::Mmap>>>>,
}
impl VolumeGrid500Mapped {
    pub fn from_data_dir(data_dir: &str) -> Self {
        use memmap::MmapOptions;
        use std::fs::File;

        // find highest xyz values for files in data_dir named like this format: format!("{}/cell_yxz_{:03}_{:03}_{:03}.tif", data_dir, y, x, z);
        // use regex to match file names
        let mut max_x = 0;
        let mut max_y = 0;
        let mut max_z = 0;
        for entry in std::fs::read_dir(data_dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            let file_name = path.file_name().unwrap().to_str().unwrap();
            if let Some(captures) = regex::Regex::new(r"cell_yxz_(\d+)_(\d+)_(\d+)\.tif")
                .unwrap()
                .captures(file_name)
            {
                //println!("Found file: {}", file_name);
                let x = captures.get(2).unwrap().as_str().parse::<usize>().unwrap();
                let y = captures.get(1).unwrap().as_str().parse::<usize>().unwrap();
                let z = captures.get(3).unwrap().as_str().parse::<usize>().unwrap();
                if x > max_x {
                    max_x = x;
                }
                if y > max_y {
                    max_y = y;
                }
                if z > max_z {
                    max_z = z;
                }
            }
        }
        fn map_for(data_dir: &str, x: usize, y: usize, z: usize) -> Option<memmap::Mmap> {
            let file_name = format!("{}/cell_yxz_{:03}_{:03}_{:03}.tif", data_dir, y, x, z);

            let file = File::open(file_name).ok()?;
            unsafe { MmapOptions::new().offset(8).map(&file) }.ok()
        }
        if !std::path::Path::new(data_dir).exists() {
            println!("Data directory {} does not exist", data_dir);
            return Self {
                max_x: 0,
                max_y: 0,
                max_z: 0,
                data: vec![],
            };
        }
        let data: Vec<Vec<Vec<Option<memmap::Mmap>>>> = (1..=max_z)
            .map(|z| {
                (1..=max_y)
                    .map(|y| (1..=max_x).map(|x| map_for(data_dir, x, y, z)).collect())
                    .collect()
            })
            .collect();

        // count number of slices found
        let slices_found = data.iter().flatten().flatten().flatten().count();
        println!("Found {} cells in {}", slices_found, data_dir);
        println!("max_x: {}, max_y: {}, max_z: {}", max_x, max_y, max_z);

        Self {
            max_x: max_x - 1,
            max_y: max_y - 1,
            max_z: max_z - 1,
            data,
        }
    }
}
impl VoxelVolume for VolumeGrid500Mapped {
    fn get(&mut self, xyz: [i32; 3]) -> u8 {
        let x_tile = xyz[0] as usize / 500;
        let y_tile = xyz[1] as usize / 500;
        let z_tile = xyz[2] as usize / 500;

        if xyz[0] < 0 || xyz[1] < 0 || xyz[2] < 0 || x_tile > self.max_x || y_tile > self.max_y || z_tile > self.max_z {
            //println!("out of bounds: {:?}", xyz);
            0
        } else if let Some(tile) = &self.data[z_tile][y_tile][x_tile] {
            let off =
                500147 * ((xyz[2] % 500) as usize) + ((xyz[1] % 500) as usize * 500 + (xyz[0] % 500) as usize) * 2;

            //println!("xyz: {:?}, off: {}, tile: {:?}", xyz, off, tile);

            // off + 1 because we select the higher order bits of little endian 16 bit values
            if off + 1 >= tile.len() {
                0
            } else {
                tile[off + 1]
            }
        } else {
            0
        }
    }
}

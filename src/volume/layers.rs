use std::io::{Read, Seek};

use super::{AutoPaintVolume, VoxelVolume};

struct Layer {
    data: memmap::Mmap,
    width: usize,
    height: usize,
}
impl Layer {
    fn get(&self, x: usize, y: usize) -> u8 {
        let off = (y * self.width + x) * 2;

        // off + 1 because we select the higher order bits of little endian 16 bit values
        if off + 1 >= self.data.len() {
            0
        } else {
            self.data[off + 1]
        }
    }
}

pub struct LayersMappedVolume {
    max_x: usize,
    max_y: usize,
    max_z: usize,
    data: Vec<Option<Layer>>,
}
impl LayersMappedVolume {
    pub fn from_data_dir(data_dir: &str) -> Self {
        use memmap::MmapOptions;
        use std::fs::File;

        // find highest xyz values for files in data_dir named like this format: format!("{}/cell_yxz_{:03}_{:03}_{:03}.tif", data_dir, y, x, z);
        // use regex to match file names
        let mut max_z = 0;
        for entry in std::fs::read_dir(data_dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            let file_name = path.file_name().unwrap().to_str().unwrap();
            if let Some(captures) = regex::Regex::new(r"(\d{5})\.tif").unwrap().captures(file_name) {
                //println!("Found file: {}", file_name);
                let z = captures.get(1).unwrap().as_str().parse::<usize>().unwrap();
                if z > max_z {
                    max_z = z;
                }
            }
        }
        fn layer_for(data_dir: &str, z: usize) -> Option<Layer> {
            let file_name = format!("{}/{:05}.tif", data_dir, z);

            let file = File::open(file_name.clone()).ok()?;
            let mut decoder = tiff::decoder::Decoder::new(file).ok()?;

            use tiff::tags::Tag;
            fn assume<T: Seek + Read>(decoder: &mut tiff::decoder::Decoder<T>, tag: Tag, value: u32) -> Option<()> {
                if let Ok(v) = decoder.get_tag_u32(tag) {
                    if v != value {
                        println!("Expected {:?} to be {} but was {}", tag, value, v);
                        None
                    } else {
                        Some(())
                    }
                } else {
                    // ok if not found
                    Some(())
                }
            }

            let width = decoder.get_tag_u32(Tag::ImageWidth).ok()? as usize;
            let height = decoder.get_tag_u32(Tag::ImageLength).ok()? as usize;

            assume(&mut decoder, Tag::BitsPerSample, 16)?;
            assume(&mut decoder, Tag::Compression, 1 /* None */)?;
            assume(&mut decoder, Tag::SamplesPerPixel, 1)?;
            // PlanarConfiguration does not matter for grayscale
            // assume(&mut decoder, Tag::PlanarConfiguration, 2 /* Planar */)?;

            // get offsets
            let offsets = decoder.get_tag_u32_vec(Tag::StripOffsets).ok()?;
            if offsets.len() != 1 {
                println!("Expected 1 strip offset: {}", offsets.len());
                return None;
            }
            let first_offset = offsets[0] as usize;

            let file = File::open(file_name).ok()?;
            if let Ok(mmap) = unsafe { MmapOptions::new().offset(first_offset as u64).map(&file) } {
                Some(Layer {
                    data: mmap,
                    width,
                    height,
                })
            } else {
                None
            }
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
        let data: Vec<Option<Layer>> = (0..=max_z).map(|z| layer_for(data_dir, z)).collect();

        // count number of slices found
        let layers_found = data.iter().flatten().count();
        let max_x = data.iter().flatten().map(|l| l.width).min().unwrap_or(0);
        let max_y = data.iter().flatten().map(|l| l.height).min().unwrap_or(0);
        println!("Found {} layers in {}", layers_found, data_dir);
        println!("max_x: {}, max_y: {}, max_z: {}", max_x, max_y, max_z);

        Self {
            max_x: max_x - 1,
            max_y: max_y - 1,
            max_z: max_z - 1,
            data,
        }
    }
}
impl VoxelVolume for LayersMappedVolume {
    fn get(&mut self, _xyz: [f64; 3], downsampling: i32) -> u8 {
        let xyz = [
            _xyz[0] as i32 * downsampling,
            _xyz[1] as i32 * downsampling,
            _xyz[2] as i32 * downsampling,
        ];

        if xyz[0] < 0
            || xyz[1] < 0
            || xyz[2] < 0
            || xyz[0] > self.max_x as i32
            || xyz[1] > self.max_y as i32
            || xyz[2] > self.max_z as i32
        {
            //println!("out of bounds: {:?}", xyz);
            0
        } else if let Some(layer) = &self.data[xyz[2] as usize] {
            layer.get(xyz[0] as usize, xyz[1] as usize)
        } else {
            0
        }
    }
}

impl AutoPaintVolume for LayersMappedVolume {}

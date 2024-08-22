use std::sync::Arc;
use vesuvius_gui::downloader::Downloader;
use vesuvius_gui::model::VolumeReference;
use vesuvius_gui::model::{self, FullVolumeReference};
use vesuvius_gui::volume::{self, PPMVolume, TrilinearInterpolatedVolume, VoxelVolume};

fn main() {
    const TILE_SERVER: &'static str = "https://vesuvius.virtual-void.net";
    // TODO: use proper command line argument parsing
    let data_dir = std::env::args().nth(1).unwrap();
    let ppm = std::env::args().nth(2).unwrap();

    //self.download_notifier = Some(receiver);
    let volume: &'static FullVolumeReference = &model::FullVolumeReference::FRAGMENT_PHerc1667Cr01Fr03;

    let volume_dir = volume.sub_dir(&data_dir);

    fn create_world(volume: &'static FullVolumeReference, volume_dir: String, ppm: String) -> Box<PPMVolume> {
        let world = {
            let (sender, _receiver) = std::sync::mpsc::channel();
            let downloader = Downloader::new(&volume_dir, TILE_SERVER, volume, None, sender);
            let v = volume::VolumeGrid64x4Mapped::from_data_dir(&volume_dir, downloader);
            Box::new(v)
        };
        let mut world = transform_volume(&ppm, world, true);
        world.enable_bilinear_interpolation();
        world
    }

    // 3.24um original resolution
    // want to rescale to 7.91um

    let factor = 7.91f64 / 3.24f64;

    let mid_w = 32;

    use rayon::prelude::*;

    #[derive(Clone)]
    struct WorldSetup {
        volume_dir: String,
        ppm: String,
    }
    let setup = WorldSetup {
        volume_dir: volume_dir.clone(),
        ppm: ppm.clone(),
    };
    let setup = Arc::new(setup);

    let w_range = 15..=49;

    (w_range).into_par_iter().for_each(move |w| {
        let mut world = {
            let setup = setup.clone();
            create_world(&volume, setup.volume_dir.clone(), setup.ppm.clone())
        };
        let width = ((world.width() as f64) / factor) as usize;
        let height = ((world.height() as f64) / factor) as usize;
        println!("Rescaling layer w:{} to {}x{}", w, width, height);

        let mut buf = vec![0u8; width * height];
        for v in 0..height {
            if v % 500 == 0 {
                println!("Layer z:{} v:{} / {}", w, v, height);
            }
            for u in 0..width {
                let value = world.get([u as f64 * factor, v as f64 * factor, ((w - mid_w) as f64) * factor], 1);
                buf[v * width + u] = value;
            }
        }
        let image = image::GrayImage::from_raw(width as u32, height as u32, buf).unwrap();
        image.save(format!("rescaled-layers/{:02}.png", w)).unwrap();
    });
}

fn transform_volume(
    ppm_file: &str,
    world: Box<dyn volume::VoxelPaintVolume>,
    trilinear_interpolation: bool,
) -> Box<PPMVolume> {
    //let old = std::mem::replace(&mut self.world, Box::new(EmptyVolume {}));
    let base = if trilinear_interpolation {
        Box::new(TrilinearInterpolatedVolume { base: world })
    } else {
        world
    };
    let ppm = PPMVolume::new(&ppm_file, base);
    let width = ppm.width() as i32;
    let height = ppm.height() as i32;
    println!("Loaded PPM volume with size {}x{}", width, height);

    Box::new(ppm)
}

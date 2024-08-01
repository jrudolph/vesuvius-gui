use std::sync::Arc;
use vesuvius_gui::downloader::Downloader;
use vesuvius_gui::gui::TemplateApp;
use vesuvius_gui::model::VolumeReference;
use vesuvius_gui::model::{self, FullVolumeReference};
use vesuvius_gui::volume::{self, PPMVolume, TrilinearInterpolatedVolume, VoxelVolume};

fn main() {
    const TILE_SERVER: &'static str = "https://vesuvius.virtual-void.net";
    // TODO: use proper command line argument parsing
    let data_dir = std::env::args().nth(1).unwrap();
    let ppm = std::env::args().nth(2).unwrap();

    let password = TemplateApp::load_data_password(&data_dir).unwrap();

    //self.download_notifier = Some(receiver);
    let volume: &'static FullVolumeReference = &model::FullVolumeReference::FRAGMENT_PHerc1667Cr01Fr03;

    let volume_dir = volume.sub_dir(&data_dir);

    fn create_world(
        volume: &'static FullVolumeReference,
        password: String,
        volume_dir: String,
        ppm: String,
    ) -> Box<PPMVolume> {
        let world = {
            let (sender, _receiver) = std::sync::mpsc::channel();
            let downloader = Downloader::new(&volume_dir, TILE_SERVER, volume, Some(password), sender);
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

    let mid_z = 32;

    use rayon::prelude::*;

    #[derive(Clone)]
    struct WorldSetup {
        password: String,
        volume_dir: String,
        ppm: String,
    }
    let setup = WorldSetup {
        password: password.clone(),
        volume_dir: volume_dir.clone(),
        ppm: ppm.clone(),
    };
    let setup = Arc::new(setup);

    //for z in 15..=49 {
    (15..=49).into_par_iter().for_each(move |z| {
        let mut world = {
            let setup = setup.clone();
            create_world(
                &volume,
                setup.password.clone(),
                setup.volume_dir.clone(),
                setup.ppm.clone(),
            )
        };
        let width = ((world.width() as f64) / factor) as usize;
        let height = ((world.height() as f64) / factor) as usize;
        println!("Rescaling layer z:{} to {}x{}", z, width, height);

        let mut buf = vec![0u8; width * height];
        for y in 0..height {
            if y % 500 == 0 {
                println!("Layer z:{} v:{} / {}", z, y, height);
            }
            for x in 0..width {
                let v = world.get([x as f64 * factor, y as f64 * factor, ((z - mid_z) as f64) * factor], 1);
                buf[y * width + x] = v;
            }
        }
        let image = image::GrayImage::from_raw(width as u32, height as u32, buf).unwrap();
        image.save(format!("rescaled-layers/{:02}.png", z)).unwrap();
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

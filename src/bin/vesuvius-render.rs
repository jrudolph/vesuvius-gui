use directories::BaseDirs;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::iter::IntoParallelIterator;
use serde::de;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};
use vesuvius_gui::downloader::{DownloadState, DownloadTask, Downloader, SimpleDownloader};
use vesuvius_gui::model::{self, FullVolumeReference};
use vesuvius_gui::model::{Quality, VolumeReference};
use vesuvius_gui::volume::{
    self, DrawingConfig, Image, ObjVolume, PPMVolume, PaintVolume, TrilinearInterpolatedVolume, VoxelPaintVolume,
    VoxelVolume,
};
use wavefront_obj::obj;

struct DummyDownloader {
    requested_tiles: HashSet<(usize, usize, usize)>,
}
impl DummyDownloader {
    fn new() -> Self {
        Self {
            requested_tiles: HashSet::new(),
        }
    }
}

struct WrappedDummy {
    inner: Arc<RefCell<DummyDownloader>>,
}
impl WrappedDummy {
    fn new(inner: Arc<RefCell<DummyDownloader>>) -> Self {
        Self { inner }
    }
}
impl Downloader for WrappedDummy {
    fn queue(
        &mut self,
        (
            _,
            x,
            y,
            z,
            Quality {
                bit_mask,
                downsampling_factor,
            },
        ): DownloadTask,
    ) {
        assert!(bit_mask == 255 && downsampling_factor == 1);
        self.inner.borrow_mut().requested_tiles.insert((x, y, z));
    }
}

#[derive(Clone)]
struct DummyVolume {
    requested_tiles: Arc<RefCell<HashSet<(usize, usize, usize)>>>,
    last_requested: Arc<RefCell<(usize, usize, usize)>>,
}
impl DummyVolume {
    fn new() -> Self {
        Self {
            requested_tiles: Arc::new(RefCell::new(HashSet::new())),
            last_requested: Arc::new(RefCell::new((0, 0, 0))), //FIXME: use max,max,max
        }
    }
}
impl PaintVolume for DummyVolume {
    fn paint(
        &mut self,
        _xyz: [i32; 3],
        _u_coord: usize,
        _v_coord: usize,
        _plane_coord: usize,
        _width: usize,
        _height: usize,
        _sfactor: u8,
        _paint_zoom: u8,
        _config: &DrawingConfig,
        _buffer: &mut Image,
    ) {
        panic!();
    }
}
impl VoxelVolume for DummyVolume {
    fn get(&mut self, xyz: [f64; 3], _downsampling: i32) -> u8 {
        let xyz2 = ((xyz[0] as usize) >> 6, (xyz[1] as usize) >> 6, (xyz[2] as usize) >> 6);
        //println!("{:?} -> {:?}", xyz, xyz2);
        let mut last = self.last_requested.borrow_mut();
        if *last != xyz2 {
            *last = xyz2;
            self.requested_tiles.borrow_mut().insert(xyz2);
        }
        0
    }
}

#[derive(Clone)]
struct DummyVolume2 {
    requested_tiles: BTreeSet<(usize, usize, usize)>,
    last_requested: (usize, usize, usize),
}
impl DummyVolume2 {
    fn new() -> Self {
        Self {
            requested_tiles: BTreeSet::new(),
            last_requested: (0, 0, 0),
        }
    }
}
impl PaintVolume for DummyVolume2 {
    fn paint(
        &mut self,
        _xyz: [i32; 3],
        _u_coord: usize,
        _v_coord: usize,
        _plane_coord: usize,
        _width: usize,
        _height: usize,
        _sfactor: u8,
        _paint_zoom: u8,
        _config: &DrawingConfig,
        _buffer: &mut Image,
    ) {
        panic!();
    }
}
impl VoxelVolume for DummyVolume2 {
    fn get(&mut self, xyz: [f64; 3], _downsampling: i32) -> u8 {
        let xyz2 = ((xyz[0] as usize) >> 6, (xyz[1] as usize) >> 6, (xyz[2] as usize) >> 6);

        if self.last_requested != xyz2 {
            self.last_requested = xyz2;
            self.requested_tiles.insert(xyz2);
        }
        0
    }
}

fn main() {
    const TILE_SERVER: &'static str = "https://vesuvius.virtual-void.net";

    let obj_file = "/tmp/20231031143852.obj";
    let width = 13577;
    let height = 10620;

    let w_range/* : std::ops::RangeInclusive<usize> */ = 25..=41; //25..=41; //32..=32; //15..=49;
    let mid_w = 32 as i32;

    let start = std::time::Instant::now();

    // for better parallelism, tile the whole widthxheight into 16 tiles
    let tiles: Vec<(usize, usize, usize)> = (0..width)
        .step_by(width / 4)
        .flat_map(|u| {
            let cloned_range = w_range.clone();
            (0..height)
                .step_by(height / 4)
                .flat_map(move |v| cloned_range.clone().into_iter().map(move |w| (u, v, w)))
        })
        .collect();
    let tile_width = width / 4;
    let tile_height = height / 4;

    use indicatif::ParallelProgressIterator;
    use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
    use rayon::prelude::*;

    let obj_file = Arc::new(ObjVolume::load_obj(obj_file));

    let style =
        ProgressStyle::with_template("[{elapsed_precise}] {bar:80.cyan/blue} {pos}/{len} ({eta}) {msg}").unwrap();
    let requested_tiles = tiles
        .into_par_iter()
        .progress_with_style(style.clone())
        .with_message("Scanning tiles to download...")
        .flat_map(|(u, v, w)| {
            let dummy = Arc::new(RefCell::new(DummyVolume2::new()));
            //let world = Arc::new(RefCell::new(dummy.clone()));
            let trilinear_interpolation = true;
            let base: Arc<RefCell<dyn VoxelPaintVolume + 'static>> = if trilinear_interpolation {
                Arc::new(RefCell::new(TrilinearInterpolatedVolume { base: dummy.clone() }))
            } else {
                dummy.clone()
            };
            let world = Arc::new(RefCell::new(ObjVolume::new(obj_file.clone(), base, width, height)));
            let mut world = world.borrow_mut();

            let mut image = Image::new(width / 4, height / 4);
            let xyz = [
                u as i32 + tile_width as i32 / 2,
                v as i32 + tile_height as i32 / 2,
                w as i32 - mid_w,
            ];
            world.paint(
                xyz,
                0,
                1,
                2,
                tile_width,
                tile_height,
                1,
                1,
                &DrawingConfig::default(),
                &mut image,
            );
            let res = dummy.borrow().requested_tiles.clone();
            //println!("Tile: {},{} [{:?}]-> {:?}", u, v, xyz, res.len());
            res
        })
        .collect::<HashSet<_>>();

    let end = start.elapsed();
    println!("Time needed for finding out files to download: {:?}", end);

    //let requested_tiles = requested_tiles.into_iter().collect::<HashSet<_>>();
    //&dummy.requested_tiles.borrow();
    println!("Total tiles: {}", requested_tiles.len());

    let dir = BaseDirs::new().unwrap().cache_dir().join("vesuvius-gui");

    let requested_tiles = requested_tiles
        .iter()
        .filter(|(x, y, z)| {
            let file_name = format!(
                "{}/64-4/d01/z{:03}/xyz-{:03}-{:03}-{:03}-b255-d01.bin",
                dir.to_str().unwrap(),
                z,
                x,
                y,
                z
            );
            !std::path::Path::new(&file_name).exists()
        })
        .collect::<Vec<_>>();

    // total size = 64^3 * requested_tiles.len()
    println!(
        "Requested tiles: {} total download size: {} MB",
        requested_tiles.len(),
        requested_tiles.len() * 64 * 64 * 64 / 1024 / 1024
    );

    let (sender, receiver) = std::sync::mpsc::channel();
    let mut downloader = SimpleDownloader::new(
        dir.to_str().unwrap(),
        TILE_SERVER,
        &FullVolumeReference::SCROLL1,
        None,
        sender,
        false,
    );

    // queue max 16 tiles to downloader and only schedule new when one is finished
    let dstyle = ProgressStyle::with_template("[{elapsed_precise}] {bar:80.cyan/blue} {decimal_bytes}/{decimal_total_bytes} ({decimal_bytes_per_sec}) ({eta})").unwrap();
    let bar = ProgressBar::new(requested_tiles.len() as u64 * 64 * 64 * 64).with_style(dstyle);
    let queue = requested_tiles.iter().cloned().collect::<Vec<_>>();
    let mut iter = queue.iter();
    let mut running = 0;
    let mut finished = 0;
    while finished < requested_tiles.len() {
        while running < 32 {
            if let Some((x, y, z)) = iter.next() {
                let state = Arc::new(Mutex::new(DownloadState::Queuing));
                downloader.queue((
                    state.clone(),
                    *x,
                    *y,
                    *z,
                    Quality {
                        bit_mask: 255,
                        downsampling_factor: 1,
                    },
                ));
                running += 1;
            } else {
                println!("No more tiles to download, finished: {}", finished);
                std::thread::sleep(std::time::Duration::from_millis(1000));
                //break;
            }
        }
        while running > 0 && receiver.try_recv().is_ok() {
            finished += 1;
            running -= 1;
            bar.inc(64 * 64 * 64);
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    bar.finish();

    println!("Finished downloading all tiles");

    w_range
        .into_iter()
        .collect::<Vec<_>>()
        .into_par_iter()
        .progress_with_style(style.clone())
        .with_message("Rendering layers")
        .for_each(move |w| {
            let (sender, receiver) = std::sync::mpsc::channel();
            let downloader = SimpleDownloader::new(
                dir.to_str().unwrap(),
                TILE_SERVER,
                &FullVolumeReference::SCROLL1,
                None,
                sender,
                false,
            );
            let v = volume::VolumeGrid64x4Mapped::from_data_dir(dir.to_str().unwrap(), Box::new(downloader));
            let world = Arc::new(RefCell::new(v));
            let trilinear_interpolation = true;
            let base: Arc<RefCell<dyn VoxelPaintVolume + 'static>> = if trilinear_interpolation {
                Arc::new(RefCell::new(TrilinearInterpolatedVolume { base: world }))
            } else {
                world
            };
            let obj = Arc::new(RefCell::new(ObjVolume::new(obj_file.clone(), base, width, height)));
            let mut world = obj.borrow_mut();
            //let mut buf = vec![0u8; width * height];
            /* for v in 0..height {
                if v % 500 == 0 {
                    println!("Layer z:{} v:{} / {}", w, v, height);
                }
                for u in 0..width {
                    let value = world.get([u as f64, v as f64, ((w - mid_w) as f64)], 1);
                    buf[v * width + u] = value;
                }
            } */
            /* let image = image::GrayImage::from_raw(width as u32, height as u32, buf).unwrap();
            image.save(format!("rescaled-layers/{:02}.png", w)).unwrap(); */
            let mut image = Image::new(width, height);
            world.paint(
                [width as i32 / 2, height as i32 / 2, w as i32 - mid_w as i32],
                0,
                1,
                2,
                width,
                height,
                1,
                1,
                &DrawingConfig::default(),
                &mut image,
            );
            let data = image.data.iter().map(|c| c.r()).collect::<Vec<_>>();
            let image = image::GrayImage::from_raw(width as u32, height as u32, data).unwrap();
            image.save(format!("test_layers/{:02}.png", w)).unwrap();
        });
}

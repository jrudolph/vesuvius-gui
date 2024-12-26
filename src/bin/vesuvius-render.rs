#![allow(dead_code, unused)]
use anyhow::{anyhow, Result};
use async_recursion::async_recursion;
use directories::BaseDirs;
use futures::{stream, Stream, StreamExt};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::cell::RefCell;
use std::collections::{BTreeSet, HashSet};
use std::future::{Future, IntoFuture};
use std::ops::RangeInclusive;
use std::sync::{Arc, Mutex};
use vesuvius_gui::downloader::{DownloadState as DS, Downloader, SimpleDownloader};
use vesuvius_gui::gui::ObjFileConfig;
use vesuvius_gui::model::Quality;
use vesuvius_gui::model::{FullVolumeReference, VolumeReference};
use vesuvius_gui::volume::{
    self, DrawingConfig, Image, ObjFile, ObjVolume, PaintVolume, TrilinearInterpolatedVolume, VoxelPaintVolume,
    VoxelVolume,
};

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

#[derive(Clone)]
struct RenderParams {
    obj_file: String,
    width: usize,
    height: usize,
    tile_size: usize,
    w_range: RangeInclusive<usize>,
    target_dir: String,
}

#[derive(Debug, Copy, Clone)]
struct UVTile {
    u: usize,
    v: usize,
    w: usize,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct VolumeChunk {
    x: usize,
    y: usize,
    z: usize,
}
impl From<(usize, usize, usize)> for VolumeChunk {
    fn from((x, y, z): (usize, usize, usize)) -> Self {
        VolumeChunk { x, y, z }
    }
}

const MID_W: i32 = 32;

struct DownloadState {
    downloaded: BTreeSet<VolumeChunk>,
}
impl DownloadState {
    fn new() -> Self {
        DownloadState {
            downloaded: BTreeSet::new(),
        }
    }
}

#[derive(Clone, Debug)]
struct DownloadSettings {
    tile_server_base: String,
    volume_base_path: String,
    cache_dir: String,
    retries: u8,
    concurrent_downloads: usize,
}

struct AsyncDownloader {
    semaphore: tokio::sync::Semaphore,
    settings: DownloadSettings,
}
impl AsyncDownloader {
    async fn download_chunk(&self, chunk: VolumeChunk) -> Result<()> {
        self.download_attempt(chunk, self.settings.retries).await
    }

    #[async_recursion]
    async fn download_attempt(&self, chunk: VolumeChunk, retries: u8) -> Result<()> {
        if (retries == 0) {
            return Err(anyhow!("Failed to download tile"));
        }
        //println!("Queueing Downloading chunk: {:?}", chunk);

        let permit = self.semaphore.acquire().await.unwrap();
        let url = self.url_for(chunk);
        let mut request = ehttp::Request::get(url.clone());

        //println!("Downloading chunk: {:?} by request to {:?}", chunk, &request);
        let response = ehttp::fetch_async(request).await;
        //println!("Finished downloading chunk: {:?}, response: {:?}", chunk, &response);
        drop(permit);
        if let Ok(res) = response {
            if res.status == 200 {
                let VolumeChunk { x, y, z } = chunk;
                let bytes = res.bytes;
                let file_name = format!(
                    "{}/64-4/d{:02}/z{:03}/xyz-{:03}-{:03}-{:03}-b{:03}-d{:02}.bin",
                    self.settings.cache_dir, 1, z, x, y, z, 255, 1
                );
                std::fs::create_dir_all(format!("{}/64-4/d{:02}/z{:03}", self.settings.cache_dir, 1, z)).unwrap();
                let tmp_file = format!("{}.tmp", file_name);
                std::fs::write(&tmp_file, bytes).unwrap();
                std::fs::rename(tmp_file, file_name).unwrap();
            } else if res.status == 420 {
                // retry in 10 seconds
                tokio::time::sleep(std::time::Duration::from_secs(10)).await;
                return self.download_attempt(chunk, retries - 1).await;
            } else {
                todo!();
            }
        } else {
            return Err(anyhow!("Failed to download tile"));
        }

        Ok(())
    }

    fn url_for(&self, VolumeChunk { x, y, z }: VolumeChunk) -> String {
        format!(
            "{}/tiles/{}download/64-4?x={}&y={}&z={}&bitmask={}&downsampling={}",
            self.settings.tile_server_base, self.settings.volume_base_path, x, y, z, 255, 1
        )
    }
}

#[derive(Clone)]
struct Rendering {
    params: RenderParams,
    obj: Arc<ObjFile>,
    download_state: Arc<Mutex<DownloadState>>,
    downloader: Arc<AsyncDownloader>,
}

const TILE_SERVER: &'static str = "https://vesuvius.virtual-void.net";

impl Rendering {
    fn new(params: RenderParams) -> Self {
        let obj = Arc::new(ObjVolume::load_obj(&params.obj_file));
        // FIXME: make configurable
        let volume = FullVolumeReference::SCROLL1;

        let settings = DownloadSettings {
            tile_server_base: TILE_SERVER.to_string(),
            volume_base_path: volume.url_path_base(),
            cache_dir: BaseDirs::new()
                .unwrap()
                .cache_dir()
                .join("vesuvius-gui")
                .to_str()
                .unwrap()
                .to_string(),
            retries: 20,
            concurrent_downloads: 32,
        };

        Self {
            params,
            obj,
            download_state: Arc::new(Mutex::new(DownloadState::new())),
            downloader: Arc::new(AsyncDownloader {
                semaphore: tokio::sync::Semaphore::new(settings.concurrent_downloads),
                settings,
            }),
        }
    }
    async fn run(&self) -> Result<()> {
        let multi = MultiProgress::new();

        let count_style = ProgressStyle::with_template(
            "{spinner} {msg:25} {bar:80.cyan/blue} [{elapsed_precise}] ({eta:>4}) {pos}/{len}",
        )
        .unwrap()
        .tick_chars("→↘↓↙←↖↑↗");

        let tiles = self.uv_tiles();

        let map_bar = ProgressBar::new(tiles.len() as u64)
            .with_style(count_style.clone())
            .with_message("Mapping segment");
        multi.add(map_bar.clone());

        let dstyle =
    ProgressStyle::with_template("{spinner} {msg:25} {bar:80.cyan/blue} [{elapsed_precise}] ({eta:>4}) {decimal_bytes}/{decimal_total_bytes} ({decimal_bytes_per_sec})")
    .unwrap().tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈");

        let download_bar = ProgressBar::new(0)
            .with_style(dstyle)
            .with_message("Downloading chunks");
        multi.add(download_bar.clone());

        /* let render_bar = ProgressBar::new(tiles.len() as u64)
            .with_style(count_style.clone().tick_chars("▪▫▨▧▦▩"))
            .with_message("Rendering tiles");
        multi.add(render_bar.clone()); */

        let layers_bar = ProgressBar::new(self.params.w_range.clone().count() as u64)
            .with_style(count_style.tick_chars("⌷⌸⌹⌺"))
            .with_message("Saving layers");
        multi.add(layers_bar.clone());

        let buf_size = 4096;

        stream::iter(tiles)
            .map(move |tile| {
                let map_bar = map_bar.clone();
                async move {
                    let self_clone = self.clone();
                    let chunks = tokio::task::spawn_blocking(move || self_clone.chunks_for(&tile))
                        .await
                        .unwrap();
                    map_bar.inc(1);
                    (tile, chunks)
                }
            })
            .buffered(buf_size)
            .map(|(tile, chunks)| {
                let download_bar = download_bar.clone();
                async move {
                    self.download_all_chunks(chunks, download_bar).await;
                    tile
                }
            })
            .buffered(buf_size) // needs ordering because we deduplicate downloads here
            /* .map(|tile| {
                let render_bar = render_bar.clone();
                async move {
                    self.render_tile(&tile).await;
                    render_bar.inc(1);
                    tile
                }
            })
            .buffered(buf_size) */
            .map(|tile @ UVTile { u, v, w }| {
                let self_clone = self.clone();
                let layers_bar = layers_bar.clone();
                let tile_size = self.params.tile_size;
                let is_last = u + tile_size > self.params.width && v + tile_size > self.params.height;

                async move {
                    let res =
                        tokio::task::spawn_blocking(move || if is_last { self_clone.render_layer(w) } else { Ok(()) })
                            .await
                            .unwrap()
                            .unwrap();
                    if is_last {
                        layers_bar.inc(1);
                    }
                    res
                }
            })
            .buffered(buf_size)
            .for_each(futures::future::ready)
            .await;

        Ok(())
    }
    fn uv_tiles(&self) -> Vec<UVTile> {
        let mut res = Vec::new();
        for w in self.params.w_range.clone() {
            for u in (0..self.params.width).step_by(self.params.tile_size) {
                for v in (0..self.params.height).step_by(self.params.tile_size) {
                    let tile = UVTile { u, v, w };
                    res.push(tile);
                }
            }
        }
        res
    }
    fn chunks_for(&self, UVTile { u, v, w }: &UVTile) -> BTreeSet<VolumeChunk> {
        let dummy = Arc::new(RefCell::new(DummyVolume2::new()));
        //let world = Arc::new(RefCell::new(dummy.clone()));
        let trilinear_interpolation = true;
        let base: Arc<RefCell<dyn VoxelPaintVolume + 'static>> = if trilinear_interpolation {
            Arc::new(RefCell::new(TrilinearInterpolatedVolume { base: dummy.clone() }))
        } else {
            dummy.clone()
        };
        let width = self.params.width;
        let height = self.params.height;
        let tile_width = self.params.tile_size;
        let tile_height = self.params.tile_size;
        let world = Arc::new(RefCell::new(ObjVolume::new(self.obj.clone(), base, width, height)));
        let mut world = world.borrow_mut();

        let mut image = Image::new(tile_width, tile_height);
        let xyz = [
            *u as i32 + tile_width as i32 / 2,
            *v as i32 + tile_height as i32 / 2,
            *w as i32 - MID_W,
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
        res.into_iter().map(Into::into).collect()
    }
    async fn download_all_chunks(&self, mut chunks: BTreeSet<VolumeChunk>, bar: ProgressBar) -> Result<()> {
        let dir = BaseDirs::new().unwrap().cache_dir().join("vesuvius-gui");
        // FIXME: proper base path missing

        let filtered = {
            let mut downloaded = self.download_state.lock().unwrap();
            let res = chunks
                .difference(&downloaded.downloaded)
                .cloned()
                .collect::<BTreeSet<_>>();
            downloaded.downloaded.append(&mut chunks);
            res
        };
        let requested_tiles = filtered
            .into_iter()
            .filter(|VolumeChunk { x, y, z }| {
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
        let old_len = bar.length().unwrap_or(0);
        bar.set_length(old_len + requested_tiles.len() as u64 * 64 * 64 * 64);

        stream::iter(requested_tiles)
            .map(|chunk| {
                let bar = bar.clone();
                let c = self.clone();
                async move {
                    c.downloader.download_chunk(chunk).await.unwrap();
                    bar.inc(64 * 64 * 64);
                }
            })
            .buffered(32)
            .collect::<Vec<_>>()
            .await;

        Ok(())
    }
    async fn render_tile(&self, uv_tile: &UVTile) -> Result<()> {
        //std::thread::sleep(std::time::Duration::from_millis(1000));
        Ok(())
    }
    fn render_layer(&self, w: usize) -> Result<()> {
        let width = self.params.width;
        let height = self.params.height;

        struct PanicDownloader {}
        impl Downloader for PanicDownloader {
            fn queue(&mut self, task: (Arc<Mutex<DS>>, usize, usize, usize, Quality)) {
                panic!("All files should be downloaded already but got {:?}", task);
            }
        }
        let dir = BaseDirs::new()
            .unwrap()
            .cache_dir()
            .join("vesuvius-gui")
            .to_str()
            .unwrap()
            .to_string();

        let v = volume::VolumeGrid64x4Mapped::from_data_dir(&dir, Box::new(PanicDownloader {}));
        let world = Arc::new(RefCell::new(v));
        let trilinear_interpolation = true;
        let base: Arc<RefCell<dyn VoxelPaintVolume + 'static>> = if trilinear_interpolation {
            Arc::new(RefCell::new(TrilinearInterpolatedVolume { base: world }))
        } else {
            world
        };
        let obj = Arc::new(RefCell::new(ObjVolume::new(self.obj.clone(), base, width, height)));
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
            [width as i32 / 2, height as i32 / 2, w as i32 - MID_W as i32],
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
        image.save(format!("{}/{:02}.png", self.params.target_dir, w)).unwrap();

        Ok(())
    }
}

#[tokio::main]
async fn main() {
    /* let params = RenderParams {
        obj_file: "/tmp/20231031143852.obj".to_string(),
        width: 13577,
        height: 10620,
        tile_size: 4096,
        w_range: 25..=41,
        target_dir: "/tmp".to_string(),
    }; */

    let params = RenderParams {
        obj_file: "/home/johannes/tmp/pap/20230827161847.obj".to_string(),
        width: 5048,
        height: 9163,
        tile_size: 4096,
        w_range: 25..=41,
        target_dir: "/tmp".to_string(),
    };

    let rendering = Rendering::new(params);
    rendering.run().await.unwrap();
}

fn main3() {
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
    use rayon::prelude::*;

    let obj_file = Arc::new(ObjVolume::load_obj(obj_file));

    let style =
        ProgressStyle::with_template("{spinner} {msg:30} {bar:80.cyan/blue} {pos}/{len} [{elapsed_precise}] ({eta}) ")
            .unwrap()
            .tick_chars("→↘↓↙←↖↑↗");
    let requested_tiles = tiles
        .into_par_iter()
        .progress_with_style(style.clone())
        .with_message("Mapping segment...")
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
    // FIXME: proper base path missing

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
    let dstyle =
    ProgressStyle::with_template("{spinner} {msg:30} {bar:80.cyan/blue} [{elapsed_precise}] ({eta}) {decimal_bytes}/{decimal_total_bytes} ({decimal_bytes_per_sec})")
    .unwrap().tick_chars("→↘↓↙←↖↑↗");

    let bar = ProgressBar::new(requested_tiles.len() as u64 * 64 * 64 * 64)
        .with_style(dstyle)
        .with_message("Downloading tiles");
    let queue = requested_tiles.iter().cloned().collect::<Vec<_>>();
    let mut iter = queue.iter();
    let mut running = 0;
    let mut finished = 0;
    while finished < requested_tiles.len() {
        while running < 32 {
            if let Some((x, y, z)) = iter.next() {
                let state = Arc::new(Mutex::new(DS::Queuing));
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
                //println!("[{},{}] Queued downloading tile: {},{},{}", running, finished, x, y, z);
            } else {
                /* println!(
                    "No more tiles to download, running: {} finished: {}/{}",
                    running,
                    finished,
                    requested_tiles.len()
                ); */
                std::thread::sleep(std::time::Duration::from_millis(1000));
                break;
            }
        }
        while running > 0 && receiver.try_recv().is_ok() {
            finished += 1;
            running -= 1;
            /* println!(
                "[{},{}] Finished downloading tile: {},{},{}",
                running, finished, x, y, z
            ); */

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
            let (sender, _) = std::sync::mpsc::channel();
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

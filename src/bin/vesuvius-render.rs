#![allow(dead_code, unused)]
use anyhow::{anyhow, Result};
use async_recursion::async_recursion;
use clap::Parser;
use directories::BaseDirs;
use futures::{stream, StreamExt};
use image::Luma;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::ops::RangeInclusive;
use std::sync::{Arc, Mutex};
use vesuvius_gui::downloader::{DownloadState as DS, Downloader};
use vesuvius_gui::model::Quality;
use vesuvius_gui::model::{FullVolumeReference, VolumeReference};
use vesuvius_gui::volume::{
    self, DrawingConfig, Image, ObjFile, ObjVolume, PaintVolume, TrilinearInterpolatedVolume, VoxelPaintVolume,
    VoxelVolume,
};

/// Vesuvius Renderer, a tool to render segments from obj files
#[derive(Parser, Debug)]
#[command(about, long_about = None)]
pub struct Args {
    /// Provide segment file to render
    #[clap(long)]
    obj: String,

    /// Width of the segment file when browsing obj files
    #[clap(long)]
    width: u32,
    /// Height of the segment file when browsing obj files
    #[clap(long)]
    height: u32,

    /// The target directory to save the rendered images
    #[clap(long)]
    target_dir: String,

    /// Output layer id that corresponds to the segment surface (default 32)
    #[clap(long)]
    middle_layer: Option<u8>,

    /// Minimum layer id to render (default 25)
    #[clap(long)]
    min_layer: Option<u8>,

    /// Maximum layer id to render (default 41)
    #[clap(long)]
    max_layer: Option<u8>,

    /// File extension / image format to use for layers (default png)
    #[clap(long)]
    target_format: Option<String>,

    /// The id of a volume to render against, otherwise Scroll 1A is used
    #[clap(short, long)]
    volume: Option<String>,

    /// Override the data directory. By default, a directory in the user's cache is used
    #[clap(short, long)]
    data_directory: Option<String>,

    /// The tile size to split a segment into (for ergonomic reasons) (default 1024)
    #[clap(long)]
    tile_size: Option<u32>,

    /// The number of concurrent downloads to use (default 64)
    #[clap(long)]
    concurrent_downloads: Option<u8>,

    /// The number of retries to use for downloads (default 20)
    #[clap(long)]
    retries: Option<u8>,
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

#[derive(Clone)]
struct RenderParams {
    obj_file: String,
    width: usize,
    height: usize,
    tile_size: usize,
    w_range: RangeInclusive<usize>,
    mid_layer: usize,
    target_dir: String,
    target_format: String,
}
impl From<&Args> for RenderParams {
    fn from(args: &Args) -> Self {
        Self {
            obj_file: args.obj.clone(),
            width: args.width as usize,
            height: args.height as usize,
            tile_size: args.tile_size.unwrap_or(1024) as usize,
            w_range: args.min_layer.unwrap_or(25) as usize..=args.max_layer.unwrap_or(41) as usize,
            mid_layer: args.middle_layer.unwrap_or(32) as usize,
            target_dir: args.target_dir.clone(),
            target_format: args.target_format.clone().unwrap_or("png".to_string()),
        }
    }
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
impl TryFrom<&Args> for DownloadSettings {
    type Error = anyhow::Error;
    fn try_from(args: &Args) -> std::result::Result<Self, Self::Error> {
        let vol: &'static dyn VolumeReference = if let Some(vol_id) = args.volume.clone() {
            vol_id.try_into().map_err(|e| anyhow!("Cannot find volume: {}", e))?
        } else {
            &FullVolumeReference::SCROLL1
        };
        let cache_dir = BaseDirs::new().unwrap().cache_dir().join("vesuvius-gui");
        let download_dir = vol.sub_dir(cache_dir.to_str().unwrap());

        Ok(Self {
            tile_server_base: TILE_SERVER.to_string(),
            volume_base_path: vol.url_path_base(),
            cache_dir: download_dir,
            retries: args.retries.unwrap_or(20),
            concurrent_downloads: args.concurrent_downloads.unwrap_or(32) as usize,
        })
    }
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
        if retries == 0 {
            return Err(anyhow!("Failed to download tile"));
        }
        //println!("Queueing Downloading chunk: {:?}", chunk);

        let permit = self.semaphore.acquire().await.unwrap();
        let url = self.url_for(chunk);
        let request = ehttp::Request::get(url.clone());

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
    fn new(params: RenderParams, download_settings: DownloadSettings) -> Self {
        let obj = Arc::new(ObjVolume::load_obj(&params.obj_file));
        // FIXME: make configurable
        let volume = FullVolumeReference::SCROLL1;

        Self {
            params,
            obj,
            download_state: Arc::new(Mutex::new(DownloadState::new())),
            downloader: Arc::new(AsyncDownloader {
                semaphore: tokio::sync::Semaphore::new(download_settings.concurrent_downloads),
                settings: download_settings,
            }),
        }
    }
    async fn run(&self, multi: &MultiProgress) -> Result<()> {
        let count_style = ProgressStyle::with_template(
            "{spinner} {msg:25} {bar:80.cyan/blue} [{elapsed_precise}] ({eta:>4}) {pos}/{len}",
        )
        .unwrap()
        .tick_chars("→↘↓↙←↖↑↗");

        let tiles = self.uv_tiles();
        let tiles_per_layer = tiles.len() as u64 / self.params.w_range.clone().count() as u64;
        println!("Tiles per layer: {}", tiles_per_layer);

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
        download_bar.tick();

        let render_bar = ProgressBar::new(tiles.len() as u64)
            .with_style(count_style.clone().tick_chars("▪▫▨▧▦▩"))
            .with_message("Rendering tiles");
        multi.add(render_bar.clone());
        render_bar.tick();

        let layers_bar = ProgressBar::new(self.params.w_range.clone().count() as u64)
            .with_style(count_style.tick_chars("⌷⌸⌹⌺"))
            .with_message("Saving layers");
        multi.add(layers_bar.clone());
        layers_bar.tick();

        let buf_size = 1024;

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
            .map(|tile| {
                let self_clone = self.clone();
                let render_bar = render_bar.clone();
                async move {
                    let tile_image = tokio::task::spawn_blocking(move || self_clone.render_tile(&tile).unwrap())
                        .await
                        .unwrap();
                    render_bar.inc(1);
                    (tile, tile_image)
                }
            })
            .buffered(tiles_per_layer as usize)
            .chunks(tiles_per_layer as usize)
            .map(|tiles| {
                let self_clone = self.clone();
                let layers_bar = layers_bar.clone();
                async move {
                    let res = tokio::task::spawn_blocking(move || self_clone.render_layer_from_tiles(tiles))
                        .await
                        .unwrap();
                    layers_bar.inc(1);
                    res
                }
            })
            .buffer_unordered(buf_size)
            .collect::<Vec<_>>()
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
        let dir = self.downloader.settings.cache_dir.clone();
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
                let file_name = format!("{}/64-4/d01/z{:03}/xyz-{:03}-{:03}-{:03}-b255-d01.bin", dir, z, x, y, z);
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

    fn render_layer_from_tiles(&self, tiles: Vec<(UVTile, Image)>) -> Result<()> {
        let width = self.params.width;
        let height = self.params.height;
        let tile_size = self.params.tile_size;
        let w = tiles[0].0.w;
        let mut image = image::GrayImage::new(width as u32, height as u32);

        // copy in all the tile images
        for (UVTile { u, v, w: _ }, tile_image) in tiles {
            let tile_data = tile_image.data;
            // blit into the right area of image
            for lu in 0..tile_size {
                for lv in 0..tile_size {
                    let gu = u + lu;
                    let gv = v + lv;
                    if gu < width && gv < height {
                        // edge tiles may spill over boundaries of target image
                        image.put_pixel(gu as u32, gv as u32, Luma([tile_data[lv * tile_size + lu].r() as u8]));
                    }
                }
            }
        }

        image
            .save(format!(
                "{}/{:02}.{}",
                self.params.target_dir, w, self.params.target_format
            ))
            .unwrap();

        Ok(())
    }
    fn render_tile(&self, UVTile { u, v, w }: &UVTile) -> Result<Image> {
        let paint_width = self.params.tile_size;
        let paint_height = self.params.tile_size;

        struct PanicDownloader {}
        impl Downloader for PanicDownloader {
            fn queue(&mut self, task: (Arc<Mutex<DS>>, usize, usize, usize, Quality)) {
                panic!("All files should be downloaded already but got {:?}", task);
            }
        }
        let dir = self.downloader.settings.cache_dir.clone();

        let vol = volume::VolumeGrid64x4Mapped::from_data_dir(&dir, Box::new(PanicDownloader {}));
        let world = Arc::new(RefCell::new(vol));
        let trilinear_interpolation = true;
        let base: Arc<RefCell<dyn VoxelPaintVolume + 'static>> = if trilinear_interpolation {
            Arc::new(RefCell::new(TrilinearInterpolatedVolume { base: world }))
        } else {
            world
        };
        let obj = Arc::new(RefCell::new(ObjVolume::new(
            self.obj.clone(),
            base,
            self.params.width,
            self.params.height,
        )));
        let mut world = obj.borrow_mut();

        let mut image = Image::new(paint_width, paint_height);
        world.paint(
            [
                *u as i32 + paint_width as i32 / 2,
                *v as i32 + paint_height as i32 / 2,
                *w as i32 - MID_W as i32,
            ],
            0,
            1,
            2,
            paint_width,
            paint_height,
            1,
            1,
            &DrawingConfig::default(),
            &mut image,
        );
        Ok(image)
    }
}

fn main() -> Result<()> {
    let result = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .max_blocking_threads(num_cpus::get())
        .build()
        .unwrap()
        .block_on(main_run());

    result
}

async fn main_run() -> Result<()> {
    let args = Args::parse();

    let multi = MultiProgress::new();
    monitor_runtime_stats(&multi).await;

    let params = (&args).into();
    let settings = (&args).try_into()?;
    /* let params = RenderParams {
        obj_file: "/tmp/20231031143852.obj".to_string(),
        width: 13577,
        height: 10620,
        tile_size: 4096,
        w_range: 25..=41,
        target_dir: "/tmp".to_string(),
    }; */

    /* let params = RenderParams {
        obj_file: "/tmp/mesh_window_350414_400414_flatboi.obj".to_string(),
        width: 40174,
        height: 16604,
        tile_size: 4096,
        w_range: 30..=36,
        target_dir: "/tmp".to_string(),
    };
    // /tmp/mesh_window_900414_950414_flatboi.obj
    // 15312	16097
    let params = RenderParams {
        obj_file: "/tmp/mesh_window_900414_950414_flatboi.obj".to_string(),
        width: 15312,
        height: 16097,
        tile_size: 1024,
        w_range: 25..=41,
        target_dir: "/tmp".to_string(),
    };

    // /tmp/20231221180251.obj
    //12975	9893
    let params = RenderParams {
        obj_file: "/tmp/20231221180251.obj".to_string(),
        width: 12975,
        height: 9893,
        tile_size: 2048,
        w_range: 0..=63,
        target_dir: "/tmp".to_string(),
    }; */

    /*let params = RenderParams {
        obj_file: "/tmp/20230827161847.obj".to_string(),
        width: 5048,
        height: 9163,
        tile_size: 1024,
        w_range: 0..=63,
        target_dir: "/tmp".to_string(),
    }; */

    let rendering = Rendering::new(params, settings);
    rendering.run(&multi).await?;
    Ok(())
}

async fn monitor_runtime_stats(multi: &MultiProgress) {
    let bar = ProgressBar::new(0).with_style(ProgressStyle::with_template("{msg}").unwrap());
    multi.add(bar.clone());
    tokio::spawn(async move {
        loop {
            let metrics = tokio::runtime::Handle::current().metrics();

            // Print all available metrics
            /* println!("=== Runtime Metrics ===");
            println!("Workers: {}", metrics.num_workers());
            println!("Blocking threads: {}", metrics.num_blocking_threads());
            println!("Active tasks: {}", metrics.num_alive_tasks());
            println!("Idle blocking threads: {}", metrics.num_idle_blocking_threads()); */

            let bar_line = format!(
                "Workers: {} Active tasks: {} Running blocking threads: {} Blocking threads: {} Idle blocking threads: {}",
                metrics.num_workers(),
                metrics.num_alive_tasks(),
                metrics.num_blocking_threads() - metrics.num_idle_blocking_threads(),
                metrics.num_blocking_threads(),
                metrics.num_idle_blocking_threads()
            );
            bar.set_message(bar_line);

            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }
    });
}

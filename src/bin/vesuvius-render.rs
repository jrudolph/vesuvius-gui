use anyhow::{anyhow, Result};
use async_recursion::async_recursion;
use clap::Parser;
use directories::BaseDirs;
use futures::{stream, StreamExt};
use image::Luma;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use reqwest::blocking::Client;
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::ops::RangeInclusive;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use vesuvius_gui::downloader::{DownloadState as DS, Downloader};
use vesuvius_gui::model::Quality;
use vesuvius_gui::model::{FullVolumeReference, VolumeReference};
use vesuvius_gui::volume::{
    self, DrawingConfig, Image, ObjFile, ObjVolume, PaintVolume, Volume, VoxelPaintVolume, VoxelVolume,
};

#[derive(Clone, Debug)]
pub struct Crop {
    pub top: usize,
    pub left: usize,
    pub width: usize,
    pub height: usize,
}
#[derive(Clone)]
struct CropParser;
impl clap::builder::TypedValueParser for CropParser {
    type Value = Crop;

    fn parse_ref(
        &self,
        _cmd: &clap::Command,
        _arg: Option<&clap::Arg>,
        value: &std::ffi::OsStr,
    ) -> std::result::Result<Self::Value, clap::Error> {
        // parse a value like 0+0-0x0 with a regexp

        let re = regex::Regex::new(r"(\d+)\+(\d+)-(\d+)x(\d+)").unwrap();
        let captures = re.captures(value.to_str().unwrap()).ok_or(clap::Error::raw(
            clap::error::ErrorKind::ValueValidation,
            "--crop argument could not be parsed. Use '--crop <left>+<top>-<width>x<height>', e.g. '--crop 1000+1000-500x500'.",
        ))?;
        let left = captures.get(1).unwrap().as_str().parse().unwrap();
        let top = captures.get(2).unwrap().as_str().parse().unwrap();
        let width = captures.get(3).unwrap().as_str().parse().unwrap();
        let height = captures.get(4).unwrap().as_str().parse().unwrap();
        Ok(Crop {
            top,
            left,
            width,
            height,
        })
    }
}

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

    /// Crop a specific region from the segment. The format is <left>+<top>-<width>x<height>.
    #[clap(long, value_parser = CropParser)]
    crop: Option<Crop>,

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

    /// Internal stream buffer size (default 1024)
    /// This limits the amount of internal work to buffer before backpressuring
    /// and continue working on output.
    #[clap(long)]
    stream_buffer_size: Option<usize>,

    /// CPU-bound worker threads to use (default number of cores/threads)
    #[clap(long)]
    worker_threads: Option<usize>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let threads = args.worker_threads.unwrap_or(num_cpus::get());

    let result = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .max_blocking_threads(threads)
        .build()
        .unwrap()
        .block_on(main_run(args));

    result
}

async fn main_run(args: Args) -> Result<()> {
    let multi = MultiProgress::new();
    monitor_runtime_stats(&multi).await;

    let params = (&args).into();
    let settings = (&args).try_into()?;

    let rendering = Rendering::new(params, settings);
    rendering.run(&multi).await?;
    Ok(())
}

#[derive(Clone)]
struct RenderParams {
    obj_file: String,
    width: usize,
    height: usize,
    tile_size: usize,
    w_range: RangeInclusive<usize>,
    crop: Option<Crop>,
    mid_layer: usize,
    target_dir: String,
    target_format: String,
    stream_buffer_size: usize,
}
impl RenderParams {
    fn render_left(&self) -> usize {
        self.crop.as_ref().map(|c| c.left).unwrap_or(0)
    }
    fn render_top(&self) -> usize {
        self.crop.as_ref().map(|c| c.top).unwrap_or(0)
    }
    fn render_width(&self) -> usize {
        self.crop.as_ref().map(|c| c.width).unwrap_or(self.width)
    }
    fn render_height(&self) -> usize {
        self.crop.as_ref().map(|c| c.height).unwrap_or(self.height)
    }
}
impl From<&Args> for RenderParams {
    fn from(args: &Args) -> Self {
        Self {
            obj_file: args.obj.clone(),
            width: args.width as usize,
            height: args.height as usize,
            tile_size: args.tile_size.unwrap_or(1024) as usize,
            w_range: args.min_layer.unwrap_or(25) as usize..=args.max_layer.unwrap_or(41) as usize,
            crop: args.crop.clone(),
            mid_layer: args.middle_layer.unwrap_or(32) as usize,
            target_dir: args.target_dir.clone(),
            target_format: args.target_format.clone().unwrap_or("png".to_string()),
            stream_buffer_size: args.stream_buffer_size.unwrap_or(1024),
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
        std::fs::create_dir_all(&self.params.target_dir)?;

        let count_style = ProgressStyle::with_template(
            "{spinner} {msg:25} {bar:80.cyan/blue} [{elapsed_precise}] ({eta:>4}) {pos}/{len}",
        )
        .unwrap()
        .tick_chars("→↘↓↙←↖↑↗");

        let tiles = self.uv_tiles();
        let tiles_per_layer = tiles.len() as u64 / self.params.w_range.clone().count() as u64;
        //println!("Tiles per layer: {}", tiles_per_layer);

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

        let buf_size = self.params.stream_buffer_size;

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
                    // FIXME: handle download error more gracefully
                    self.download_all_chunks(chunks, download_bar).await.unwrap();
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
        let top = self.params.crop.as_ref().map(|c| c.top).unwrap_or(0);
        let left = self.params.crop.as_ref().map(|c| c.left).unwrap_or(0);
        let width = self.params.render_width();
        let height = self.params.render_height();

        let mut res = Vec::new();
        for w in self.params.w_range.clone() {
            for v in (top..top + height).step_by(self.params.tile_size) {
                for u in (left..left + width).step_by(self.params.tile_size) {
                    let tile = UVTile { u, v, w };
                    res.push(tile);
                }
            }
        }
        res
    }
    fn chunks_for(&self, UVTile { u, v, w }: &UVTile) -> BTreeSet<VolumeChunk> {
        let dummy = Rc::new(TileCollectingVolume::new());
        let width = self.params.width;
        let height = self.params.height;
        let tile_width = self.params.tile_size;
        let tile_height = self.params.tile_size;
        let world = ObjVolume::new(self.obj.clone(), Volume::from_ref(Arc::new(dummy.as_ref().clone())), width, height).into_volume();

        let mut image = Image::new(tile_width, tile_height);
        let xyz = [
            *u as i32 + tile_width as i32 / 2,
            *v as i32 + tile_height as i32 / 2,
            *w as i32 - self.params.mid_layer as i32,
        ];
        let mut config = DrawingConfig::default();
        config.trilinear_interpolation = true;
        world.paint(xyz, 0, 1, 2, tile_width, tile_height, 1, 1, &config, &mut image);
        let res = dummy.state.replace(Default::default()).requested_tiles;
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
        let width = self.params.render_width();
        let height = self.params.render_height();
        let left = self.params.render_left();
        let top = self.params.render_top();

        let tile_size = self.params.tile_size;
        let w = tiles[0].0.w;
        let mut image = image::GrayImage::new(width as u32, height as u32);

        // copy in all the tile images
        for (UVTile { u, v, w: _ }, tile_image) in tiles {
            let tile_data = tile_image.data;
            // blit into the right area of image
            for lu in 0..tile_size {
                for lv in 0..tile_size {
                    let gu = u + lu - left;
                    let gv = v + lv - top;
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
            fn queue(&self, task: (Arc<Mutex<DS>>, usize, usize, usize, Quality)) {
                panic!("All files should be downloaded already but got {:?}", task);
            }
        }
        let dir = self.downloader.settings.cache_dir.clone();

        let vol = volume::VolumeGrid64x4Mapped::from_data_dir(&dir, Arc::new(PanicDownloader {}));
        let world = ObjVolume::new(
            self.obj.clone(),
            vol.into_volume(),
            self.params.width,
            self.params.height,
        )
        .into_volume();
        let mut config = DrawingConfig::default();
        config.trilinear_interpolation = true;

        let mut image = Image::new(paint_width, paint_height);
        world.paint(
            [
                *u as i32 + paint_width as i32 / 2,
                *v as i32 + paint_height as i32 / 2,
                *w as i32 - self.params.mid_layer as i32,
            ],
            0,
            1,
            2,
            paint_width,
            paint_height,
            1,
            1,
            &config,
            &mut image,
        );
        Ok(image)
    }
}

async fn monitor_runtime_stats(multi: &MultiProgress) {
    let count_style = ProgressStyle::with_template(
        "{spinner} {msg:25} {bar:80.cyan/blue} [{elapsed_precise}] ({eta:>4}) {pos}/{len}",
    )
    .unwrap();
    let bar = ProgressBar::new(0)
        .with_style(count_style)
        .with_message("CPU threads active");

    multi.add(bar.clone());
    tokio::spawn(async move {
        loop {
            let metrics = tokio::runtime::Handle::current().metrics();

            bar.set_length(metrics.num_blocking_threads() as u64);
            bar.set_position((metrics.num_blocking_threads() - metrics.num_idle_blocking_threads()) as u64);

            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }
    });
}

#[derive(Clone)]
struct TileCollectingVolumeState {
    requested_tiles: BTreeSet<(usize, usize, usize)>,
    last_requested: (usize, usize, usize),
}
impl Default for TileCollectingVolumeState {
    fn default() -> Self {
        Self {
            requested_tiles: BTreeSet::new(),
            last_requested: (0, 0, 0),
        }
    }
}

/// A VoxelVolume implementation that just collects needed tiles
#[derive(Clone)]
struct TileCollectingVolume {
    state: RefCell<TileCollectingVolumeState>,
}
impl TileCollectingVolume {
    fn new() -> Self {
        Self {
            state: TileCollectingVolumeState::default().into(),
        }
    }
    fn add_tile(&self, tile: (usize, usize, usize)) {
        let mut state = self.state.borrow_mut();
        if state.last_requested != tile {
            state.last_requested = tile;
            state.requested_tiles.insert(tile);
        }
    }
}
impl PaintVolume for TileCollectingVolume {
    fn paint(
        &self,
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
    fn shared(&self) -> super::VolumeCons {
        panic!();
    }
}
impl VoxelVolume for TileCollectingVolume {
    fn get(&self, xyz: [f64; 3], _downsampling: i32) -> u8 {
        let tile: (usize, usize, usize) = ((xyz[0] as usize) >> 6, (xyz[1] as usize) >> 6, (xyz[2] as usize) >> 6);

        self.add_tile(tile);
        0
    }
    fn get_interpolated(&self, xyz: [f64; 3], downsampling: i32) -> u8 {
        let x = xyz[0] as usize;
        let y = xyz[1] as usize;
        let z = xyz[2] as usize;

        if x & 63 == 63 || y & 63 == 63 || z & 63 == 63 {
            // slow path, call default trilinear interpolation
            self.get_interpolated_slow(xyz, downsampling);
        } else {
            self.add_tile(((xyz[0] as usize) >> 6, (xyz[1] as usize) >> 6, (xyz[2] as usize) >> 6));
        }
        0
    }
}

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

        let cache_dir = if let Some(dir) = args.data_directory.clone() {
            dir
        } else {
            BaseDirs::new()
                .unwrap()
                .cache_dir()
                .join("vesuvius-gui")
                .to_str()
                .unwrap()
                .to_string()
        };
        let download_dir = vol.sub_dir(&cache_dir);

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

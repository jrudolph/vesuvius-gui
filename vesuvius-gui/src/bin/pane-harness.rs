//! Headless, scripted run of a single `VolumePane` against a real volume —
//! the full download → cache → paint → tile path without a window. See
//! `vesuvius_gui::harness`.
//!
//! Writes `timeline.csv` (one row per frame), `net.jsonl` (the downloader's
//! per-request netlog, unless VESUVIUS_NET_LOG is already set), `summary.txt`
//! and any `png` snapshots to `--out`.

use clap::Parser;
use std::path::PathBuf;
use std::time::Duration;
use vesuvius_gui::gui::PaneType;
use vesuvius_gui::harness::{parse_script, run_script, HarnessOptions, HarnessWorld, PaneHarness};

#[derive(Parser, Debug)]
#[command(about, long_about = None)]
struct Args {
    /// URL or local path of a zarr/ome-zarr volume (as for the GUI's -v)
    #[clap(short, long)]
    volume: String,

    /// tifxyz segment directory to render on top of the volume
    #[clap(long)]
    tifxyz: Option<String>,

    /// Pane to render: uv (default with --tifxyz), uw, vw, xy (default otherwise), xz, yz
    #[clap(long)]
    pane: Option<String>,

    /// Pane size in pixels
    #[clap(long, default_value = "1600x1000")]
    size: String,

    /// Target frame rate (also sets the per-frame tile poll budget)
    #[clap(long, default_value_t = 20)]
    fps: u32,

    #[clap(long, default_value_t = 1.0)]
    zoom: f32,

    /// Start coordinate "a,b,c" (segment u,v,w or volume x,y,z). Default: center.
    #[clap(long)]
    coord: Option<String>,

    /// Start with an empty cache in <out>/cache (cold run)
    #[clap(long, conflicts_with = "cache_dir")]
    cold: bool,

    /// Cache base directory (default: the GUI's)
    #[clap(long)]
    cache_dir: Option<PathBuf>,

    /// Output directory
    #[clap(long, default_value = "pane-harness-out")]
    out: PathBuf,

    /// Only render frames when egui asks for a repaint (idle window) instead of continuously
    #[clap(long)]
    honor_repaint: bool,

    /// Scenario, `;`-separated: settle [timeout] [quiet] | wait <s> | frames <n> | pan <dx> <dy> |
    /// drag <dx> <dy> <s> | zoom <z> | goto <a> <b> <c> | png <name>
    #[clap(long, default_value = "settle 300; png settled")]
    script: String,
}

fn parse_pane(s: &str) -> Result<PaneType, String> {
    Ok(match s.to_ascii_lowercase().as_str() {
        "uv" => PaneType::UV,
        "uw" => PaneType::UW,
        "vw" => PaneType::VW,
        "xy" => PaneType::XY,
        "xz" => PaneType::XZ,
        "yz" => PaneType::YZ,
        other => return Err(format!("unknown pane `{}`", other)),
    })
}

fn run(args: Args) -> Result<(), String> {
    let steps = parse_script(&args.script)?;
    let (width, height) = args
        .size
        .split_once('x')
        .and_then(|(w, h)| Some((w.parse().ok()?, h.parse().ok()?)))
        .ok_or_else(|| format!("bad --size `{}`, expected WxH", args.size))?;
    let coord = args
        .coord
        .as_deref()
        .map(|c| {
            let v: Vec<i32> = c.split(',').map(|x| x.trim().parse::<i32>()).collect::<Result<_, _>>().map_err(|e| e.to_string())?;
            <[i32; 3]>::try_from(v).map_err(|_| format!("bad --coord `{}`, expected a,b,c", c))
        })
        .transpose()?;

    std::fs::create_dir_all(&args.out).map_err(|e| e.to_string())?;
    let cache_dir = if args.cold {
        let dir = args.out.join("cache");
        if dir.exists() {
            std::fs::remove_dir_all(&dir).map_err(|e| e.to_string())?;
        }
        Some(dir)
    } else {
        args.cache_dir.clone()
    };

    let mut world = HarnessWorld::open_volume(&args.volume, cache_dir.as_deref())?;
    if let Some(dir) = &args.tifxyz {
        world = world.with_tifxyz(dir)?;
    }
    let pane = match &args.pane {
        Some(p) => parse_pane(p)?,
        None if world.is_segment => PaneType::UV,
        None => PaneType::XY,
    };
    let segment_pane = matches!(pane, PaneType::UV | PaneType::UW | PaneType::VW);
    if segment_pane != world.is_segment {
        return Err(format!("pane {:?} needs {}", pane, if segment_pane { "--tifxyz" } else { "no --tifxyz" }));
    }

    let opts = HarnessOptions {
        pane,
        width,
        height,
        fps: args.fps,
        zoom: args.zoom,
        coord,
        honor_repaint: args.honor_repaint,
        ..Default::default()
    };
    let mut harness = PaneHarness::new(world, opts);
    println!(
        "pane {:?} {}x{} at {:?} zoom {} ({} fps){}",
        pane,
        width,
        height,
        harness.coord(),
        harness.zoom(),
        args.fps,
        if args.cold { ", cold cache" } else { "" }
    );

    let mut summary = String::new();
    let result = run_script(&mut harness, &steps, &args.out, |r| {
        println!("{}", r);
        summary.push_str(&format!("{}\n", r));
    });
    harness
        .write_timeline_csv(args.out.join("timeline.csv"))
        .map_err(|e| e.to_string())?;
    std::fs::write(args.out.join("summary.txt"), &summary).map_err(|e| e.to_string())?;
    println!("wrote {}", args.out.display());
    result.map(|_| ())
}

fn main() {
    let args = Args::parse();
    // Before any thread exists: the netlog sink reads this once.
    if std::env::var_os("VESUVIUS_NET_LOG").is_none() {
        let _ = std::fs::create_dir_all(&args.out);
        let path = args.out.join("net.jsonl");
        let _ = std::fs::remove_file(&path);
        std::env::set_var("VESUVIUS_NET_LOG", path);
    }
    env_logger::init();

    // Same runtime flavour as the GUI's #[tokio::main]: tiles render on
    // spawn_blocking and are polled with block_in_place from this thread.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    let result = runtime.block_on(async move { run(args) });
    // Let in-flight shard writes land before exiting.
    vesuvius_rs::cache::UnifiedCache::shutdown_all();
    runtime.shutdown_timeout(Duration::from_secs(5));
    if let Err(e) = result {
        eprintln!("error: {}", e);
        std::process::exit(1);
    }
}

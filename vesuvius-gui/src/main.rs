use vesuvius_gui::gui::{ObjFileConfig, TemplateApp, VesuviusConfig};
use vesuvius_atlas_rs::load_atlas_from_directory;
use vesuvius_rs::cache::UnifiedCache;
use vesuvius_rs::catalog::load_catalog;

use clap::Parser;
use vesuvius_rs::model::{NewVolumeReference, VolumeReference};
use vesuvius_rs::volume::{AffineTransform, BlendMode, OverlayColoring, ProjectionKind};
use vesuvius_zarr::base_cache_dir;

/// Vesuvius GUI, an app to visualize and explore 3D data of the Vesuvius Challenge (https://scrollprize.org)
#[derive(Parser, Debug)]
#[command(about, long_about = None)]
pub struct Args {
    /// Override the data directory. By default, a directory in the user's cache is used
    #[clap(short, long)]
    data_directory: Option<String>,

    /// Browse segment from obj file. You need to also provide --width and --height. Provide the --volume if the segment does not target Scroll 1a / 20230205180739
    #[clap(long, conflicts_with = "tifxyz")]
    obj: Option<String>,

    /// Browse segment from a vc3d tifxyz directory (containing meta.json + x/y/z.tif). Grid dimensions are read from the TIFFs, so --width/--height are not required.
    #[clap(long, conflicts_with = "obj")]
    tifxyz: Option<String>,

    /// Width of the segment file when browsing obj files
    #[clap(long)]
    width: Option<usize>,
    /// Height of the segment file when browsing obj files
    #[clap(long)]
    height: Option<usize>,

    /// Transform to apply to the obj file (to map between different scans). You can either supply a filename to a transform json file
    /// (as defined in https://github.com/ScrollPrize/villa/blob/main/foundation/volume-registration/transform_schema.json) or supply
    /// a 4x3 affine transformation matrix as a json array string directly
    #[clap(long)]
    transform: Option<String>,

    /// Use orthographic projection along the Y axis (top-down view) when loading obj files (discarding existing texture coordinates).
    #[clap(long, default_value_t = false)]
    ortho_xz: bool,

    /// Invert the transform before applying it
    #[clap(long)]
    invert_transform: bool,

    /// A directory that contains data to overlay. Only zarr arrays are currently supported
    #[clap(short, long)]
    overlay: Option<String>,

    /// Coloring of the overlay volume. Forms (trailing `:MODE` is optional, MODE ∈ {alpha, multiply}, default alpha):
    ///   - `four-colors[:ALPHA[:MODE]]`     (default ALPHA=0.4) values 1-4 → red/green/yellow/blue
    ///   - `boolean:#RRGGBB[:ALPHA[:MODE]]` (default ALPHA=0.4) value 255 → given color
    ///   - `hue:DEG[:ALPHA[:MODE]]`         (default ALPHA=0.4) value → HSV(DEG, 1, 1), strength ∝ value
    /// Multiply mode preserves the grayscale brightness and only shifts the hue.
    #[clap(long, value_parser = parse_overlay_coloring)]
    overlay_coloring: Option<OverlayColoring>,

    /// The id of a volume to open, URL to a zarr/ome-zarr volume, or local path to zarr/ome-zarr directory
    #[clap(short, long)]
    volume: Option<Option<String>>,

    /// Path to vesuvius-atlas data directory
    #[clap(long)]
    atlas: Option<String>,
}

impl TryFrom<Args> for VesuviusConfig {
    type Error = String;

    fn try_from(args: Args) -> Result<Self, Self::Error> {
        let v = args.volume.clone();
        if let Some(None) = v {
            return Err(format!(
                "Volumes:\n{}",
                <dyn VolumeReference>::VOLUMES
                    .iter()
                    .map(|v| format!("{} -> {}", v.id(), v.label()))
                    .collect::<Vec<_>>()
                    .join("\n")
            ));
        }
        let v = v.map(|x| x.unwrap());
        let volume = if let Some(vol_str) = v.clone() {
            // Try to parse as URL first
            if vol_str.starts_with("http") {
                Some(NewVolumeReference::from_url(vol_str).map_err(|e| e.to_string())?)
            } else if std::path::Path::new(&vol_str).exists() {
                // Try to parse as local path
                Some(NewVolumeReference::from_path(vol_str).map_err(|e| e.to_string())?)
            } else {
                // Try to find in static volumes
                if let Some(static_vol) = <dyn VolumeReference>::VOLUMES.iter().find(|v| v.id() == vol_str) {
                    Some(NewVolumeReference::Volume64x4(static_vol.owned()))
                } else {
                    return Err(format!(
                        "Error: Volume {} not found. Use one of:\n{}\n\nOr provide:\n- HTTP URL to zarr/ome-zarr volume\n- Local filesystem path to zarr/ome-zarr directory",
                        vol_str,
                        <dyn VolumeReference>::VOLUMES
                            .iter()
                            .map(|v| format!("{} -> {}", v.id(), v.label()))
                            .collect::<Vec<_>>()
                            .join("\n")
                    ));
                }
            }
        } else {
            None
        };
        let transform_opt = if let Some(transform) = args.transform.as_ref() {
            let mut t = AffineTransform::from_json_array_or_path(transform).map_err(|e| e.to_string())?;
            if args.invert_transform {
                t = t.invert().map_err(|e| e.to_string())?;
            }
            Some(t)
        } else {
            None
        };

        let obj_file = if let Some(obj_file) = args.obj {
            let projection = if args.ortho_xz {
                ProjectionKind::OrthographicXZ
            } else {
                ProjectionKind::None
            };

            if let (Some(width), Some(height)) = (args.width, args.height) {
                Some(ObjFileConfig {
                    obj_file,
                    width,
                    height,
                    transform: transform_opt.clone(),
                    projection,
                })
            } else {
                return Err("Error: You need to provide --width and --height when using --obj".to_string());
            }
        } else {
            None
        };

        let tifxyz_dir = args.tifxyz.map(|dir| (dir, transform_opt));

        Ok(VesuviusConfig {
            data_dir: args.data_directory,
            obj_file,
            tifxyz_dir,
            overlay_dir: args.overlay,
            overlay_coloring: args.overlay_coloring,
            volume,
        })
    }
}

fn parse_overlay_coloring(s: &str) -> Result<OverlayColoring, String> {
    let mut parts = s.split(':');
    let kind = parts.next().ok_or("empty overlay coloring spec".to_string())?;
    match kind {
        "four-colors" => {
            let alpha = parts.next().map(parse_alpha).transpose()?.unwrap_or(0.4);
            let mode = parts.next().map(parse_blend_mode).transpose()?.unwrap_or_default();
            Ok(OverlayColoring::FourColors { alpha, mode })
        }
        "boolean" => {
            let color_hex = parts.next().ok_or("boolean: needs #RRGGBB color".to_string())?;
            let color = parse_hex_color(color_hex)?;
            let alpha = parts.next().map(parse_alpha).transpose()?.unwrap_or(0.4);
            let mode = parts.next().map(parse_blend_mode).transpose()?.unwrap_or_default();
            Ok(OverlayColoring::Boolean { color, alpha, mode })
        }
        "hue" => {
            let hue_deg = parts
                .next()
                .ok_or("hue: needs DEG".to_string())?
                .parse::<f32>()
                .map_err(|e| format!("invalid hue degrees: {}", e))?;
            let alpha = parts.next().map(parse_alpha).transpose()?.unwrap_or(0.4);
            let mode = parts.next().map(parse_blend_mode).transpose()?.unwrap_or_default();
            Ok(OverlayColoring::Hue { hue_deg, alpha, mode })
        }
        other => Err(format!(
            "unknown overlay coloring `{}` (expected four-colors / boolean / hue)",
            other
        )),
    }
}

fn parse_blend_mode(s: &str) -> Result<BlendMode, String> {
    match s {
        "alpha" => Ok(BlendMode::Alpha),
        "multiply" => Ok(BlendMode::Multiply),
        other => Err(format!("unknown blend mode `{}` (expected alpha / multiply)", other)),
    }
}

fn parse_alpha(s: &str) -> Result<f32, String> {
    s.parse::<f32>()
        .map_err(|e| format!("invalid alpha `{}`: {}", s, e))
        .and_then(|a| {
            if (0.0..=1.0).contains(&a) {
                Ok(a)
            } else {
                Err(format!("alpha must be in [0, 1], got {}", a))
            }
        })
}

fn parse_hex_color(s: &str) -> Result<[u8; 3], String> {
    let hex = s.strip_prefix('#').unwrap_or(s);
    if hex.len() != 6 {
        return Err(format!("expected #RRGGBB, got `{}`", s));
    }
    let r = u8::from_str_radix(&hex[0..2], 16).map_err(|e| e.to_string())?;
    let g = u8::from_str_radix(&hex[2..4], 16).map_err(|e| e.to_string())?;
    let b = u8::from_str_radix(&hex[4..6], 16).map_err(|e| e.to_string())?;
    Ok([r, g, b])
}

// When compiling natively:
#[cfg(not(target_arch = "wasm32"))]
#[tokio::main]
async fn main() -> eframe::Result<()> {
    let args = Args::parse();
    let catalog = load_catalog();

    let atlas_path = args.atlas.clone().or_else(|| std::env::var("VESUVIUS_ATLAS").ok());
    let atlas = atlas_path.and_then(|path| {
        load_atlas_from_directory(&path)
            .map_err(|e| eprintln!("Warning: Failed to load atlas from {}: {}", path, e))
            .ok()
    });

    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).

    // Bring the unified chunk cache online before anything that might
    // open a volume. `for_cache_dir` runs the disk survey (seeds the
    // epoch histogram from on-disk sidecars); `run_startup_maintenance`
    // then synchronously evicts down to the low-water mark if the
    // survey shows we're over high water or low on free space, so the
    // app starts in a known good state instead of racing the
    // background watchdog through its first 30 s.
    UnifiedCache::for_cache_dir(base_cache_dir()).run_startup_maintenance();

    let native_options = Default::default();

    let config = args.try_into();
    match config {
        Ok(config) => eframe::run_native(
            "vesuvius-gui",
            native_options,
            Box::new(|cc| Ok(Box::new(TemplateApp::new(cc, catalog, atlas, config)))),
        ),
        Err(e) => {
            eprintln!("{}", e);
            std::process::exit(1);
        }
    }
}

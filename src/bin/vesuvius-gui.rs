use vesuvius_gui::catalog::load_catalog;
use vesuvius_gui::gui::{ObjFileConfig, TemplateApp, VesuviusConfig};

use clap::Parser;
use vesuvius_gui::model::{NewVolumeReference, VolumeReference};
use vesuvius_gui::volume::{AffineTransform, ProjectionKind};

/// Vesuvius GUI, an app to visualize and explore 3D data of the Vesuvius Challenge (https://scrollprize.org)
#[derive(Parser, Debug)]
#[command(about, long_about = None)]
pub struct Args {
    /// Override the data directory. By default, a directory in the user's cache is used
    #[clap(short, long)]
    data_directory: Option<String>,

    /// Browse segment from obj file. You need to also provide --width and --height. Provide the --volume if the segment does not target Scroll 1a / 20230205180739
    #[clap(long)]
    obj: Option<String>,

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

    /// The id of a volume to open, URL to a zarr/ome-zarr volume, or local path to zarr/ome-zarr directory
    #[clap(short, long)]
    volume: Option<Option<String>>,
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
        let obj_file = if let Some(obj_file) = args.obj {
            let transform = if let Some(transform) = args.transform {
                let mut t = AffineTransform::from_json_array_or_path(&transform).map_err(|e| e.to_string())?;
                if args.invert_transform {
                    t = t.invert().map_err(|e| e.to_string())?;
                }
                Some(t)
            } else {
                None
            };

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
                    transform,
                    projection,
                })
            } else {
                return Err("Error: You need to provide --width and --height when using --obj".to_string());
            }
        } else {
            None
        };

        Ok(VesuviusConfig {
            data_dir: args.data_directory,
            obj_file,
            overlay_dir: args.overlay,
            volume,
        })
    }
}

// When compiling natively:
#[cfg(not(target_arch = "wasm32"))]
#[tokio::main]
async fn main() -> eframe::Result<()> {
    let args = Args::parse();
    let catalog = load_catalog();

    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).

    let native_options = Default::default();

    let config = args.try_into();
    match config {
        Ok(config) => eframe::run_native(
            "vesuvius-gui",
            native_options,
            Box::new(|cc| Ok(Box::new(TemplateApp::new(cc, catalog, config)))),
        ),
        Err(e) => {
            eprintln!("{}", e);
            std::process::exit(1);
        }
    }
}

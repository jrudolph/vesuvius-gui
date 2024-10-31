use vesuvius_gui::catalog::load_catalog;
use vesuvius_gui::gui::{TemplateApp, VesuviusConfig};

use clap::Parser;
use vesuvius_gui::model::VolumeReference;

/// Vesuvius GUI, an app to visualize and explore 3D data of the Vesuvius Challenge (https://scrollprize.org)
#[derive(Parser, Debug)]
#[command(about, long_about = None)]
pub struct Args {
    /// Override the data directory. By default, a directory in the user's cache is used
    #[clap(short, long)]
    data_directory: Option<String>,

    /// A directory that contains data to overlay. Only zarr arrays are currently supported
    #[clap(short, long)]
    overlay: Option<String>,

    /// The id of a volume to open
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
        let volume = v
            .clone()
            .map(|vol| <dyn VolumeReference>::VOLUMES.iter().find(|v| v.id() == vol).cloned());

        if let Some(None) = volume {
            return Err(format!(
                "Error: Volume {} not found. Use one of\n{}",
                v.unwrap_or_default(),
                <dyn VolumeReference>::VOLUMES
                    .iter()
                    .map(|v| format!("{} -> {}", v.id(), v.label()))
                    .collect::<Vec<_>>()
                    .join("\n")
            ));
        }
        Ok(VesuviusConfig {
            data_dir: args.data_directory,
            overlay_dir: args.overlay,
            volume: volume.map(|x| x.unwrap()),
        })
    }
}

// When compiling natively:
#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result<()> {
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

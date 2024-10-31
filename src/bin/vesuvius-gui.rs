use vesuvius_gui::catalog::load_catalog;
use vesuvius_gui::gui::{TemplateApp, VesuviusConfig};

use clap::Parser;

/// Vesuvius GUI, an app to visualize and explore 3D data of the Vesuvius Challenge (https://scrollprize.org)
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// Override the data directory. By default, a directory in the user's cache is used
    #[clap(short, long)]
    data_directory: Option<String>,

    /// A directory that contains data to overlay. Only zarr arrays are currently supported
    #[clap(short, long)]
    overlay: Option<String>,
}

impl From<Args> for VesuviusConfig {
    fn from(args: Args) -> Self {
        VesuviusConfig {
            data_dir: args.data_directory,
            overlay_dir: args.overlay,
        }
    }
}

// When compiling natively:
#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result<()> {
    let args = Args::parse();
    let catalog = load_catalog();

    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).

    let native_options = Default::default();
    eframe::run_native(
        "vesuvius-gui",
        native_options,
        Box::new(|cc| Ok(Box::new(TemplateApp::new(cc, catalog, args.into())))),
    )
}

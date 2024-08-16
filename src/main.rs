#![warn(clippy::all, rust_2018_idioms)]
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

// When compiling natively:
#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result<()> {
    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).

    let native_options = Default::default();
    eframe::run_native(
        "vesuvius-gui",
        native_options,
        Box::new(|cc| {
            Ok(Box::new(vesuvius_gui::TemplateApp::new(
                cc,
                std::env::args().nth(1),
                std::env::args().nth(2),
            )))
        }),
    )
}

use vesuvius_gui::catalog::load_catalog;
use vesuvius_gui::gui::TemplateApp;

// When compiling natively:
#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result<()> {
    let catalog = load_catalog();

    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).

    let native_options = Default::default();
    eframe::run_native(
        "vesuvius-gui",
        native_options,
        Box::new(|cc| {
            Ok(Box::new(TemplateApp::new(
                cc,
                catalog,
                std::env::args().nth(2),
                std::env::args().nth(1),
            )))
        }),
    )
}

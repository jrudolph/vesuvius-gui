use vesuvius_gui::gui::TemplateApp;

// When compiling natively:
#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result<()> {
    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).

    let native_options = eframe::NativeOptions {
        initial_window_size: Some([400.0, 300.0].into()),
        min_window_size: Some([300.0, 220.0].into()),
        ..Default::default()
    };
    eframe::run_native(
        "vesuvius-gui",
        native_options,
        Box::new(|cc| Box::new(TemplateApp::new(cc, std::env::args().nth(1), std::env::args().nth(2)))),
    )
}

//! Network-backed full-stack checks via the pane harness. Ignored by default;
//! run with e.g.
//!
//! ```sh
//! PANE_HARNESS_VOLUME=https://…/20250728140407-9.362um-1.2m-113keV-masked.zarr/ \
//! PANE_HARNESS_TIFXYZ=…/w040_1_tps_match \
//! cargo test --release -p vesuvius-gui --test pane_harness -- --ignored --nocapture
//! ```

use std::time::Duration;
use vesuvius_gui::gui::PaneType;
use vesuvius_gui::harness::{HarnessOptions, HarnessWorld, PaneHarness};

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

/// A cold UV view must finish loading: every dispatched chunk lands (or is
/// given up on) and the downloader goes idle, within the timeout.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs network; set PANE_HARNESS_VOLUME (+ PANE_HARNESS_TIFXYZ)"]
async fn cold_view_settles() {
    let Some(volume) = env("PANE_HARNESS_VOLUME") else {
        eprintln!("PANE_HARNESS_VOLUME not set, skipping");
        return;
    };
    let cache = tempfile::tempdir().unwrap();
    let mut world = HarnessWorld::open_volume(&volume, Some(cache.path())).unwrap();
    let pane = match env("PANE_HARNESS_TIFXYZ") {
        Some(dir) => {
            world = world.with_tifxyz(dir).unwrap();
            PaneType::UV
        }
        None => PaneType::XY,
    };
    let zoom = env("PANE_HARNESS_ZOOM").map_or(0.9, |z| z.parse().unwrap());
    let mut harness = PaneHarness::new(
        world,
        HarnessOptions {
            pane,
            zoom,
            ..Default::default()
        },
    );

    let settled = harness.run_until_settled(Duration::from_secs(180), Duration::from_secs(2));
    let report = harness.phase_report(settled);
    println!("{}", report);

    assert!(settled.is_some(), "view never settled");
    assert_eq!(report.pending_at_end, Some(0), "chunks stranded in Pending");
}

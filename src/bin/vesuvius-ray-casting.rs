use image::{ImageBuffer, Rgb};
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use rayon::prelude::*;
use serde::de::value;
use std::sync::Mutex;
use std::{path::Path, sync::Arc};
use vesuvius_gui::{
    model::{NewVolumeReference, VolumeCreationParams, VolumeReference},
    volume::{PaintVolume, VoxelPaintVolume, VoxelVolume},
    zarr::ZarrArray,
};

fn alpha_composite_raycast(
    volume: &dyn VoxelVolume,
    x: f64,
    z: f64,
    y_start: f64,
    y_end: f64,
    step_size: f64,
    max_alpha: f64,
    value_region_min: f64,
    value_region_max: f64,
) -> f64 {
    let mut accumulated_intensity = 0.0;
    let mut accumulated_alpha = 0.0;

    let mut y = y_start;
    while y < y_end && accumulated_alpha < max_alpha {
        let voxel_value = volume.get_interpolated([x, y, z], 1) as f64 / 255.0;

        // lerp and clamp voxel value to the specified range
        let voxel_value = (voxel_value - value_region_min) / (value_region_max - value_region_min);
        let voxel_value = voxel_value.clamp(0.0, 1.0);

        // Simple alpha from voxel intensity
        let alpha = voxel_value * step_size;

        // Alpha compositing: new_color = old_color + (1 - old_alpha) * new_alpha * new_color
        let weight = (1.0 - accumulated_alpha) * alpha;

        accumulated_intensity += weight * voxel_value;
        accumulated_alpha += weight;

        /* println!("Raycasting at ({}, {}, {}): voxel_value = {}, weight = {}, alpha = {}, accumulated_intensity = {}, accumulated_alpha = {}",
        x, y, z, voxel_value, weight, alpha, accumulated_intensity, accumulated_alpha); */

        y += step_size;
    }

    accumulated_intensity
}

fn main() {
    //let array = ZarrArray::<3,u8>::from_url_to_default_cache_dir_blocking("https://d1q9tbl6hor5wj.cloudfront.net/esrf/20250506/SCROLLS_TA_HEL_4.320um_1.0m_116keV_binmean_2_PHerc0343P_TA_0001_masked.zarr/0", client);
    // can be cloned / shared across threads
    //let volume_base = array.into_ctx();

    let value_region_min = 0.3; // Minimum value for the region of interest
    let value_region_max = 0.7; // Maximum value for the region of interest

    // 4um 500P2
    let y_range = 3623..6450;

    // full
    let x_range = 1346..8342;
    let z_range = 1753..14556;
    // middle
    //let x_range = 4800..4900;
    //let z_range = 5000..5100;

    // Ray casting parameters
    let step_size = 1.0;
    let max_alpha = 0.95;

    // Image dimensions
    let width = (x_range.end - x_range.start) as u32;
    let height = (z_range.end - z_range.start) as u32;

    println!("Rendering {}x{} image...", width, height);

    // Create image buffer with mutex for thread-safe access
    let img = ImageBuffer::new(width, height);
    let img_mutex = Arc::new(Mutex::new(img));
    let img_mutex_clone = img_mutex.clone();

    let client = reqwest::blocking::Client::new();
    //let array = ZarrArray::<3,u8>::from_url_to_default_cache_dir_blocking("https://d1q9tbl6hor5wj.cloudfront.net/esrf/20250506/SCROLLS_TA_HEL_4.320um_1.0m_116keV_binmean_2_PHerc0343P_TA_0001_masked.zarr/0", client);
    let array = ZarrArray::<3, u8>::from_url_to_default_cache_dir_blocking(
        "http://serve-volumes/esrf/20250506/4.317um_HA2200_HEL_111keV_1.2m_scroll-fragment-0500P2_D_0001_masked.zarr/0",
        client,
    );

    let base = array.into_ctx().into_ctx();

    // Tile-based processing for better memory locality
    let tile_size = 64usize; // Power-of-2 tile size

    // Calculate tile grid
    let tiles_x = (width as usize + tile_size - 1) / tile_size;
    let tiles_z = (height as usize + tile_size - 1) / tile_size;

    let total_tiles = tiles_x * tiles_z;
    println!(
        "Processing {}x{} tiles of size {}x{} (total: {} tiles)",
        tiles_x, tiles_z, tile_size, tile_size, total_tiles
    );

    // Create progress bar
    let progress_bar = ProgressBar::new(total_tiles as u64);
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} tiles ({percent}%) ETA: {eta}")
            .unwrap()
            .progress_chars("#>-")
    );

    // Process tiles in parallel with progress tracking
    (0..total_tiles)
        .map(|tile_idx| (tile_idx, base.shared(), img_mutex.clone(), progress_bar.clone()))
        .collect::<Vec<_>>()
        .into_par_iter()
        .progress_with(progress_bar)
        .for_each(|(tile_idx, base_shared, img_mutex, pb)| {
            let tile_x = tile_idx % tiles_x;
            let tile_z = tile_idx / tiles_x;

            let x_start = tile_x * tile_size;
            let x_end = (x_start + tile_size).min(width as usize);
            let z_start = tile_z * tile_size;
            let z_end = (z_start + tile_size).min(height as usize);

            if pb.position() % 100 == 0 {
                let img = img_mutex.lock().unwrap();
                img.save("tmp/raycast_output_progress-4um-500p2.png")
                    .expect("Failed to save image");
            }

            /* println!(
                "Processing tile {}/{} ({}x{} at {},{}) ",
                tile_idx + 1,
                tiles_x * tiles_z,
                x_end - x_start,
                z_end - z_start,
                tile_x,
                tile_z
            ); */

            // Create volume for this thread
            let volume = base_shared();

            // Create tile buffer
            let tile_width = x_end - x_start;
            let tile_height = z_end - z_start;
            let mut tile_buffer = vec![Rgb([0u8, 0u8, 0u8]); tile_width * tile_height];

            // Process all pixels in this tile
            for local_x in 0..tile_width {
                for local_z in 0..tile_height {
                    let world_x = x_range.start + x_start + local_x;
                    let world_z = z_range.start + z_start + local_z;

                    let intensity = alpha_composite_raycast(
                        &volume,
                        world_x as f64,
                        world_z as f64,
                        y_range.start as f64, // Start from top (2823)
                        y_range.end as f64,   // End at bottom (1584)
                        step_size,
                        max_alpha,
                        value_region_min,
                        value_region_max,
                    );

                    // Convert intensity to grayscale
                    let pixel_val = (intensity * 255.0).min(255.0) as u8;
                    tile_buffer[local_x * tile_height + local_z] = Rgb([pixel_val, pixel_val, pixel_val]);
                }
            }

            // Acquire lock and blit tile into image
            {
                let mut img = img_mutex.lock().unwrap();
                for local_x in 0..tile_width {
                    for local_z in 0..tile_height {
                        let img_x = x_start + local_x;
                        let img_z = z_start + local_z;
                        let pixel = tile_buffer[local_x * tile_height + local_z];
                        img.put_pixel(img_x as u32, img_z as u32, pixel);
                    }
                }
            }
        });

    let img = img_mutex_clone.lock().unwrap().clone();

    // Save image
    println!("Saving image...");
    img.save("tmp/raycast_output-4um-500p2.png")
        .expect("Failed to save image");

    println!("Ray casting complete! Saved raycast_output.png");
}

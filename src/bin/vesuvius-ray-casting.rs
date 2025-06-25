use image::{ImageBuffer, Rgb};
use serde::de::value;
use std::path::Path;
use vesuvius_gui::{
    model::{NewVolumeReference, VolumeCreationParams, VolumeReference},
    volume::{VoxelPaintVolume, VoxelVolume},
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
    while y > y_end && accumulated_alpha < max_alpha {
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

        y -= step_size;
    }

    accumulated_intensity
}

fn main() {
    let client = reqwest::blocking::Client::new();
    let array = ZarrArray::<3,u8>::from_url_to_default_cache_dir_blocking("https://d1q9tbl6hor5wj.cloudfront.net/esrf/20250506/SCROLLS_TA_HEL_4.320um_1.0m_116keV_binmean_2_PHerc0343P_TA_0001_masked.zarr/0", client);
    let volume = array.into_ctx().into_ctx().into_volume();

    let value_region_min = 0.3; // Minimum value for the region of interest
    let value_region_max = 0.7; // Maximum value for the region of interest

    // from visual inspection
    let y_range = 1584..2823;

    // full
    let x_range = 1466..3739;
    let z_range = 684..4700;

    // center region
    /* let x_range = 2000..3000;
    let z_range = 2000..3000; */

    /* let x_range = 2500..2501;
    let z_range = 2500..2501; */

    // Ray casting parameters
    let step_size = 1.0;
    let max_alpha = 0.95;

    // Image dimensions
    let width = (x_range.end - x_range.start) as u32;
    let height = (z_range.end - z_range.start) as u32;

    println!("Rendering {}x{} image...", width, height);

    // Create image buffer
    let mut img = ImageBuffer::new(width, height);

    // Cast rays for each pixel
    for (img_x, x) in (x_range.start..x_range.end).enumerate() {
        if img_x % 100 == 0 {
            println!("Processing column {}/{}", img_x, width);
            img.save("tmp/raycast_output_tmp.png").expect("Failed to save image");
        }

        for (img_z, z) in (z_range.start..z_range.end).enumerate() {
            let intensity = alpha_composite_raycast(
                &volume,
                x as f64,
                z as f64,
                y_range.end as f64,   // Start from top (2823)
                y_range.start as f64, // End at bottom (1584)
                step_size,
                max_alpha,
                value_region_min,
                value_region_max,
            );

            // Convert intensity to grayscale
            let pixel_val = (intensity * 255.0).min(255.0) as u8;
            img.put_pixel(img_x as u32, img_z as u32, Rgb([pixel_val, pixel_val, pixel_val]));
        }
    }

    // Save image
    println!("Saving image...");
    img.save("tmp/raycast_output.png").expect("Failed to save image");

    println!("Ray casting complete! Saved raycast_output.png");
}

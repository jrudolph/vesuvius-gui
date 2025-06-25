use vesuvius_gui::{
    model::{NewVolumeReference, VolumeCreationParams, VolumeReference},
    zarr::ZarrArray,
};

fn main() {
    let client = reqwest::blocking::Client::new();
    let array = ZarrArray::<3,u8>::from_url_to_default_cache_dir_blocking("https://d1q9tbl6hor5wj.cloudfront.net/esrf/20250506/SCROLLS_TA_HEL_4.320um_1.0m_116keV_binmean_2_PHerc0343P_TA_0001_masked.zarr/0", client);
    let volume = array.into_ctx().into_ctx();

    // from visual inspection
    let x_range = 1466..3739;
    let y_range = 1584..2823;
    let z_range = 684..4700;
}

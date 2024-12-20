use super::{ZarrArray, ZarrContext};
use crate::volume::PaintVolume;
use crate::volume::VoxelPaintVolume;
use crate::volume::VoxelVolume;
use egui::Color32;
use ehttp::Request;
use serde::Deserialize;
use sha2::Digest;
use sha2::Sha256;

#[derive(Debug, Clone, Deserialize)]
pub struct OmeMultiScale {
    pub axes: Vec<OmeAxis>,
    pub datasets: Vec<OmeDataset>,
    pub name: String,
    pub version: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OmeAxis {
    pub name: String,
    pub r#type: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OmeDataset {
    pub coordinate_transformations: Vec<OmeCoordinateTransformation>,
    pub path: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum OmeCoordinateTransformation {
    scale(OmeScale),
}

#[derive(Debug, Clone, Deserialize)]
pub struct OmeScale {
    pub scale: Vec<f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OmeZarrAttrs {
    pub multiscales: Vec<OmeMultiScale>,
}

impl OmeZarrAttrs {
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        let multiscales: Vec<OmeMultiScale> = serde_json::from_str(json)?;
        Ok(OmeZarrAttrs { multiscales })
    }
}

/* pub trait OmeZarrAccess {
    fn load_attrs(&self) -> OmeZarrAttrs;
    fn load_array(&self, path: &str) -> Vec<u8>;
} */

pub struct OmeZarr {
    url: String,
    cache_dir: String,
    attrs: OmeZarrAttrs,
}

pub struct OmeZarrContext<C: ColorScheme> {
    ome_zarr: OmeZarr,
    zarr_contexts: Vec<ZarrContext<3>>, // TODO: make generic
    cache_missing: bool,
    phantom: std::marker::PhantomData<C>,
}

pub trait ColorScheme {
    fn get_color(value: u8) -> Color32;
}
pub struct FourColors {}
impl ColorScheme for FourColors {
    fn get_color(value: u8) -> Color32 {
        match value {
            1 => Color32::RED,
            2 => Color32::GREEN,
            3 => Color32::YELLOW,
            _ => Color32::BLUE,
        }
    }
}
pub struct GrayScale {}
impl ColorScheme for GrayScale {
    fn get_color(value: u8) -> Color32 {
        Color32::from_gray(value)
    }
}

impl<C: ColorScheme> OmeZarrContext<C> {
    pub fn from_url(url: &str, local_cache_dir: &str) -> Self {
        let attrs = Self::load_attrs(url, local_cache_dir);

        let ome_zarr = OmeZarr {
            url: url.to_string(),
            cache_dir: local_cache_dir.to_string(),
            attrs,
        };
        let zarr_contexts = ome_zarr.attrs.multiscales[0]
            .datasets
            .iter()
            .map(|dataset| {
                let url_path = format!("{}/{}", url, dataset.path);
                let cache_path = format!("{}/{}", local_cache_dir, dataset.path);
                ZarrArray::from_url(&url_path, &cache_path).into_ctx().into_ctx()
            })
            .take(4) // FIXME
            .collect();

        Self {
            ome_zarr,
            zarr_contexts,
            cache_missing: false,
            phantom: std::marker::PhantomData,
        }
    }
    pub fn from_url_to_default_cache_dir(url: &str) -> Self {
        let canonical_url = if url.ends_with("/") { &url[..url.len() - 1] } else { url };
        let sha256 = format!("{:x}", Sha256::digest(canonical_url.as_bytes()));
        let local_cache_dir = std::env::temp_dir().join("vesuvius-gui").join(sha256);
        Self::from_url(url, local_cache_dir.to_str().unwrap())
    }

    fn load_attrs(url: &str, local_cache_dir: &str) -> OmeZarrAttrs {
        let target_file = format!("{}/.zattrs", local_cache_dir);
        if !std::path::Path::new(&target_file).exists() {
            let data = ehttp::fetch_blocking(&Request::get(&format!("{}/.zattrs", url)))
                .unwrap()
                .bytes
                .to_vec();
            std::fs::create_dir_all(std::path::Path::new(&target_file).parent().unwrap()).unwrap();
            std::fs::write(&target_file, &data).unwrap();
        }

        let zarray = std::fs::read_to_string(&target_file).unwrap();
        println!("zarray: {}", zarray);
        serde_json::from_str::<OmeZarrAttrs>(&zarray).unwrap()
    }

    fn get(&mut self, xyz: [usize; 3], scale: u8) -> u8 {
        // from max scale to target scale, try to find the value
        //for s in (scale..=scale.min(self.zarr_contexts.len() as u8 - 1)/* self.zarr_contexts.len() as u8 */).rev() {
        let max = self.zarr_contexts.len() as u8 - 1;
        for s in scale.min(max)..=max {
            let scaled_xyz = xyz.iter().map(|&x| x >> s).collect::<Vec<usize>>();
            //println!("xyz: {:?} scaled_xyz: {:?}", xyz, scaled_xyz);
            let v = self.zarr_contexts[s as usize].get(scaled_xyz.try_into().unwrap());
            if let Some(v) = v {
                //if (xyz[0] % 100 == 0) && (xyz[1] % 100 == 0) && (xyz[2] % 100 == 0) {
                //println!("found value {} at scale {} when scale was {}", v, s, scale);
                //}
                return v;
            }
        }
        0
    }
}

impl<C: ColorScheme> PaintVolume for OmeZarrContext<C> {
    fn paint(
        &mut self,
        xyz: [i32; 3],
        u_coord: usize,
        v_coord: usize,
        plane_coord: usize,
        width: usize,
        height: usize,
        sfactor: u8,
        paint_zoom: u8,
        _config: &crate::volume::DrawingConfig,
        buffer: &mut crate::volume::Image,
    ) {
        if !self.cache_missing {
            self.zarr_contexts.iter_mut().for_each(|ctx| {
                // clean missing entries from cache
                let mut access = ctx.cache.lock().unwrap();
                access.purge_missing();
            });
        }

        let scale = sfactor.trailing_zeros() as u8;

        for im_u in 0..width {
            for im_v in 0..height {
                let im_rel_u = (im_u as i32 - width as i32 / 2) * paint_zoom as i32;
                let im_rel_v = (im_v as i32 - height as i32 / 2) * paint_zoom as i32;

                let mut uvw: [f64; 3] = [0.; 3];
                uvw[u_coord] = (xyz[u_coord] + im_rel_u) as f64;
                uvw[v_coord] = (xyz[v_coord] + im_rel_v) as f64;
                uvw[plane_coord] = (xyz[plane_coord]) as f64;

                let [x, y, z] = uvw;

                if x < 0.0 || y < 0.0 || z < 0.0 {
                    continue;
                }

                let v = self.get([z as usize, y as usize, x as usize], scale);
                if v != 0 {
                    buffer.set(im_u, im_v, C::get_color(v));
                }
            }
        }
    }
}

//impl<C: ColorScheme> VoxelPaintVolume for OmeZarrContext<C> {}
impl<C: ColorScheme> VoxelVolume for OmeZarrContext<C> {
    fn get(&mut self, xyz: [f64; 3], downsampling: i32) -> u8 {
        let scale = downsampling.trailing_zeros() as u8;
        self.get(
            [
                (xyz[2] * downsampling as f64) as usize,
                (xyz[1] * downsampling as f64) as usize,
                (xyz[0] * downsampling as f64) as usize,
            ],
            scale,
        )
    }
}

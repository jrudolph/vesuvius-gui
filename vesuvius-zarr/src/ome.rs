#![allow(dead_code)]
use super::{default_cache_dir_for_url, parse_json, v3, ZarrArray, ZarrContext};
use ehttp::Request;
use serde::Deserialize;

fn remote_dataset_present(url: &str, cache_dir: &str, client: &reqwest::blocking::Client) -> bool {
    if std::path::Path::new(&format!("{}/zarr.json", cache_dir)).exists()
        || std::path::Path::new(&format!("{}/.zarray", cache_dir)).exists()
    {
        return true;
    }
    let url = url.trim_end_matches('/');
    for probe in &["zarr.json", ".zarray"] {
        let probe_url = format!("{}/{}", url, probe);
        if let Ok(resp) = client.get(&probe_url).send() {
            if resp.status() == reqwest::StatusCode::OK {
                return true;
            }
        }
    }
    false
}

fn build_blocking_client() -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder()
        .pool_max_idle_per_host(32)
        .pool_idle_timeout(Some(std::time::Duration::from_secs(60)))
        .http2_adaptive_window(true)
        .tcp_keepalive(Some(std::time::Duration::from_secs(30)))
        .timeout(Some(std::time::Duration::from_secs(60)))
        .build()
        .expect("failed to build reqwest client for v3 OmeZarrContext")
}

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
    #[allow(non_camel_case_types)]
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

#[derive(Debug, Clone)]
pub struct OmeZarr {
    attrs: OmeZarrAttrs,
}

pub struct OmeZarrContext {
    pub ome_zarr: OmeZarr,
    pub cache_missing: bool,
    pub zarr_contexts: Vec<ZarrContext<3>>, // TODO: make generic
}

impl OmeZarrContext {
    pub fn from_url(url: &str, local_cache_dir: &str) -> Self {
        // v3 first: if the group exposes a v3 `zarr.json` (with multiscales)
        // we route the per-dataset opens through the v3 sharded c3d path.
        // Use a blocking client so range requests work; the v2 async-download
        // pipeline does not support HTTP Range.
        let client = build_blocking_client();
        if let Some(attrs_value) = super::v3::read_v3_group_attributes_remote(url, local_cache_dir, &client) {
            let attrs: OmeZarrAttrs =
                serde_json::from_value(attrs_value).expect("v3 group attributes missing valid `multiscales`");
            let ome_zarr = OmeZarr { attrs };
            // Probe each dataset for presence (zarr.json or .zarray) and skip
            // any that are missing on the server. Matches the local v3 path's
            // filter and is the right call when only a subset of multiscale
            // levels has been published.
            let zarr_contexts = ome_zarr.attrs.multiscales[0]
                .datasets
                .iter()
                .filter_map(|dataset| {
                    let url_path = format!("{}/{}", url, dataset.path);
                    let cache_path = format!("{}/{}", local_cache_dir, dataset.path);
                    if !remote_dataset_present(&url_path, &cache_path, &client) {
                        log::warn!("OmeZarrContext::from_url: skipping missing dataset `{}`", dataset.path);
                        return None;
                    }
                    Some(
                        ZarrArray::<3, u8>::from_url_auto(&url_path, &cache_path, client.clone())
                            .into_ctx()
                            .into_ctx(),
                    )
                })
                .collect();
            return Self {
                ome_zarr,
                zarr_contexts,
                cache_missing: false,
            };
        }

        // v2 path (legacy async downloader).
        let attrs = Self::load_attrs(url, local_cache_dir);

        let ome_zarr = OmeZarr { attrs };
        let zarr_contexts = ome_zarr.attrs.multiscales[0]
            .datasets
            .iter()
            .map(|dataset| {
                let url_path = format!("{}/{}", url, dataset.path);
                let cache_path = format!("{}/{}", local_cache_dir, dataset.path);
                ZarrArray::from_url(&url_path, &cache_path).into_ctx().into_ctx()
            })
            .collect();

        Self {
            ome_zarr,
            zarr_contexts,
            cache_missing: false,
        }
    }
    pub fn from_url_to_default_cache_dir(url: &str) -> Self {
        let url = if url.ends_with("/") { &url[..url.len() - 1] } else { url };
        Self::from_url(url, &default_cache_dir_for_url(url))
    }
    /// Blocking variant of `from_url`: every per-dataset `ZarrArray` is built
    /// with `from_url_blocking`, so `load_chunk` always blocks until the
    /// chunk is on disk (or returns `None` definitively for 404s). Designed
    /// for the unified-cache backfillers, where the cache layer drives its
    /// own dedup + scheduling and the async download path in v2 would
    /// otherwise hide in-flight state behind `None`.
    pub fn from_url_blocking(url: &str, local_cache_dir: &str) -> Self {
        let client = build_blocking_client();
        if let Some(attrs_value) = super::v3::read_v3_group_attributes_remote(url, local_cache_dir, &client) {
            let attrs: OmeZarrAttrs =
                serde_json::from_value(attrs_value).expect("v3 group attributes missing valid `multiscales`");
            let ome_zarr = OmeZarr { attrs };
            let zarr_contexts = ome_zarr.attrs.multiscales[0]
                .datasets
                .iter()
                .filter_map(|dataset| {
                    let url_path = format!("{}/{}", url, dataset.path);
                    let cache_path = format!("{}/{}", local_cache_dir, dataset.path);
                    if !remote_dataset_present(&url_path, &cache_path, &client) {
                        log::warn!(
                            "OmeZarrContext::from_url_blocking: skipping missing dataset `{}`",
                            dataset.path
                        );
                        return None;
                    }
                    Some(
                        ZarrArray::<3, u8>::from_url_auto(&url_path, &cache_path, client.clone())
                            .into_ctx()
                            .into_ctx(),
                    )
                })
                .collect();
            return Self { ome_zarr, zarr_contexts, cache_missing: true };
        }

        // v2 path, but via the blocking accessor so all `load_chunk` calls
        // synchronously resolve to bytes-or-None.
        let attrs = Self::load_attrs(url, local_cache_dir);
        let ome_zarr = OmeZarr { attrs };
        let zarr_contexts = ome_zarr.attrs.multiscales[0]
            .datasets
            .iter()
            .map(|dataset| {
                let url_path = format!("{}/{}", url, dataset.path);
                let cache_path = format!("{}/{}", local_cache_dir, dataset.path);
                ZarrArray::from_url_blocking(&url_path, &cache_path, client.clone())
                    .into_ctx()
                    .into_ctx()
            })
            .collect();
        Self { ome_zarr, zarr_contexts, cache_missing: true }
    }
    pub fn from_url_blocking_to_default_cache_dir(url: &str) -> Self {
        let url = if url.ends_with("/") { &url[..url.len() - 1] } else { url };
        Self::from_url_blocking(url, &default_cache_dir_for_url(url))
    }
    pub fn from_path(path: &str) -> Self {
        // Prefer v3 group `zarr.json` (the c3d-compressed volumes) over the
        // legacy v2 `.zattrs`. Both have the same `multiscales` shape inside.
        let attrs = if let Some(attrs_value) = v3::read_v3_group_attributes(path) {
            serde_json::from_value::<OmeZarrAttrs>(attrs_value)
                .expect("v3 group attributes missing valid `multiscales`")
        } else {
            let target_file = format!("{}/.zattrs", path);
            let zarray = std::fs::read_to_string(&target_file).unwrap();
            parse_json::<OmeZarrAttrs>(&zarray, &target_file)
        };

        let ome_zarr = OmeZarr { attrs };
        // Skip multiscale levels that aren't materialised locally — a partial
        // download (e.g. only `0/` extracted) is a common shape and shouldn't
        // panic the loader. Missing higher-LOD arrays just mean the multiscale
        // fallback in `get` has fewer levels to try.
        let zarr_contexts = ome_zarr.attrs.multiscales[0]
            .datasets
            .iter()
            .filter_map(|dataset| {
                let dataset_path = format!("{}/{}", path, dataset.path);
                let has_v3 = std::path::Path::new(&format!("{}/zarr.json", dataset_path)).exists();
                let has_v2 = std::path::Path::new(&format!("{}/.zarray", dataset_path)).exists();
                if !has_v3 && !has_v2 {
                    log::warn!("OmeZarrContext::from_path: skipping missing dataset `{}`", dataset.path);
                    return None;
                }
                Some(ZarrArray::from_path_auto(&dataset_path).into_ctx().into_ctx())
            })
            .collect();

        Self {
            ome_zarr,
            zarr_contexts,
            cache_missing: false,
        }
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
        parse_json::<OmeZarrAttrs>(&zarray, &target_file)
    }

    pub fn get(&self, xyz: [usize; 3], scale: u8) -> u8 {
        let max = self.zarr_contexts.len() as u8 - 1;
        for s in scale.min(max)..=max {
            let scaled_xyz = xyz.iter().map(|&x| x >> s).collect::<Vec<usize>>();
            let v = self.zarr_contexts[s as usize].get(scaled_xyz.try_into().unwrap());
            if let Some(v) = v {
                return v;
            }
        }
        0
    }
    pub fn get_interpolated(&self, xyz: [f64; 3], scale: u8) -> u8 {
        let max = self.zarr_contexts.len() as u8 - 1;
        for s in scale.min(max)..=max {
            let scaled_xyz = xyz.iter().map(|&x| x / (1 << s) as f64).collect::<Vec<f64>>();
            let v = self.zarr_contexts[s as usize].get_interpolated(scaled_xyz.try_into().unwrap());
            if let Some(v) = v {
                return v;
            }
        }
        0
    }
}

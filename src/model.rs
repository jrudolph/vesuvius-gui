use crate::{
    downloader::SimpleDownloader,
    volume::{LayersMappedVolume, Volume, VolumeGrid500Mapped, VolumeGrid64x4Mapped, VoxelPaintVolume},
    zarr::{GrayScale, OmeZarrContext, ZarrArray},
};
use std::sync::Arc;

#[derive(Copy, Clone, Debug)]
pub struct Quality {
    pub bit_mask: u8,
    pub downsampling_factor: u8,
}
impl Quality {
    pub const FULL: Quality = Quality {
        bit_mask: 0xff,
        downsampling_factor: 1,
    };
}

pub trait VolumeReference: Send + Sync {
    fn id(&self) -> String;
    fn sub_dir(&self, data_dir: &str) -> String;
    fn label(&self) -> String;
    fn url_path_base(&self) -> String;
}
impl dyn VolumeReference {
    pub const VOLUMES: [&'static dyn VolumeReference; 25] = [
        &FullVolumeReference::SCROLL1,
        &FullVolumeReference::SCROLL1B,
        &FullVolumeReference::SCROLL2,
        &FullVolumeReference::SCROLL2B,
        &FullVolumeReference::SCROLL2_88keV,
        &FullVolumeReference::SCROLL332_3_24_UM,
        &FullVolumeReference::SCROLL332_7_91_UM,
        &FullVolumeReference::SCROLL1667,
        &FullVolumeReference::SCROLL1667_7_91_UM,
        &FullVolumeReference::SCROLL172,
        &FullVolumeReference::FRAGMENT_PHerc0051Cr04Fr08_3_24_UM_53keV,
        &FullVolumeReference::FRAGMENT_PHerc0051Cr04Fr08_3_24_UM_70keV,
        &FullVolumeReference::FRAGMENT_PHerc0051Cr04Fr08_3_24_UM_88keV,
        &FullVolumeReference::FRAGMENT_PHerc0051Cr04Fr08_7_91_UM_53keV,
        &FullVolumeReference::FRAGMENT_PHerc1667Cr01Fr03,
        &FullVolumeReference::FRAGMENT_1_54keV,
        &FullVolumeReference::FRAGMENT_1_88keV,
        &FullVolumeReference::FRAGMENT_2_54keV,
        &FullVolumeReference::FRAGMENT_2_88keV,
        &FullVolumeReference::FRAGMENT_3_54keV,
        &FullVolumeReference::FRAGMENT_3_88keV,
        &FullVolumeReference::FRAGMENT_4_54keV,
        &FullVolumeReference::FRAGMENT_4_88keV,
        &SurfaceVolumeReference::SEGMENT_20230827161847,
        &SurfaceVolumeReference::SEGMENT_20231005123335,
    ];
}
impl TryFrom<String> for &'static dyn VolumeReference {
    type Error = String;
    fn try_from(value: String) -> Result<Self, Self::Error> {
        let vol = <dyn VolumeReference>::VOLUMES
            .iter()
            .find(|x| x.id() == value)
            .ok_or(format!("Volume {} not found", value))?;
        Ok(*vol)
    }
}

pub struct DynamicFullVolumeReference {
    pub scroll_id: String,
    pub volume: String,
}
impl DynamicFullVolumeReference {
    pub fn new(scroll_id: String, volume: String) -> DynamicFullVolumeReference {
        DynamicFullVolumeReference { scroll_id, volume }
    }
}
impl VolumeReference for DynamicFullVolumeReference {
    fn id(&self) -> String {
        self.volume.to_string()
    }
    fn sub_dir(&self, data_dir: &str) -> String {
        format!("{}/scroll{}/{}/", data_dir, self.scroll_id, self.volume)
    }
    fn label(&self) -> String {
        format!("Scroll {} / {}", self.scroll_id, self.volume)
    }
    fn url_path_base(&self) -> String {
        format!("scroll/{}/volume/{}/", self.scroll_id, self.volume)
    }
}

pub struct FullVolumeReference {
    pub scroll_id: &'static str,
    pub volume: &'static str,
}
#[allow(non_upper_case_globals)]
impl FullVolumeReference {
    pub const SCROLL1: FullVolumeReference = FullVolumeReference {
        scroll_id: "1",
        volume: "20230205180739",
    };
    pub const SCROLL1B: FullVolumeReference = FullVolumeReference {
        scroll_id: "1",
        volume: "20230206171837",
    };
    pub const SCROLL2: FullVolumeReference = FullVolumeReference {
        scroll_id: "2",
        volume: "20230210143520",
    };
    pub const SCROLL2B: FullVolumeReference = FullVolumeReference {
        scroll_id: "2",
        volume: "20230206082907",
    };
    pub const SCROLL2_88keV: FullVolumeReference = FullVolumeReference {
        scroll_id: "2",
        volume: "20230212125146",
    };
    pub const SCROLL332_3_24_UM: FullVolumeReference = FullVolumeReference {
        scroll_id: "0332",
        volume: "20231027191953",
    };
    pub const SCROLL332_7_91_UM: FullVolumeReference = FullVolumeReference {
        scroll_id: "0332",
        volume: "20231117143551",
    };
    pub const SCROLL1667: FullVolumeReference = FullVolumeReference {
        scroll_id: "1667",
        volume: "20231107190228",
    };
    pub const SCROLL1667_7_91_UM: FullVolumeReference = FullVolumeReference {
        scroll_id: "1667",
        volume: "20231117161658",
    };
    pub const SCROLL172: FullVolumeReference = FullVolumeReference {
        scroll_id: "172",
        volume: "20241024131838",
    };
    pub const FRAGMENT_PHerc0051Cr04Fr08_3_24_UM_53keV: FullVolumeReference = FullVolumeReference {
        scroll_id: "PHerc0051Cr04Fr08",
        volume: "20231121152933",
    };
    pub const FRAGMENT_PHerc0051Cr04Fr08_3_24_UM_70keV: FullVolumeReference = FullVolumeReference {
        scroll_id: "PHerc0051Cr04Fr08",
        volume: "20231201120546",
    };
    pub const FRAGMENT_PHerc0051Cr04Fr08_3_24_UM_88keV: FullVolumeReference = FullVolumeReference {
        scroll_id: "PHerc0051Cr04Fr08",
        volume: "20231201112849",
    };
    pub const FRAGMENT_PHerc0051Cr04Fr08_7_91_UM_53keV: FullVolumeReference = FullVolumeReference {
        scroll_id: "PHerc0051Cr04Fr08",
        volume: "20231130112027",
    };
    pub const FRAGMENT_PHerc1667Cr01Fr03: FullVolumeReference = FullVolumeReference {
        scroll_id: "PHerc1667Cr01Fr03",
        volume: "20231121133215",
    };
    pub const FRAGMENT_1_54keV: FullVolumeReference = FullVolumeReference {
        scroll_id: "Frag1",
        volume: "20230205142449",
    };
    pub const FRAGMENT_1_88keV: FullVolumeReference = FullVolumeReference {
        scroll_id: "Frag1",
        volume: "20230213100222",
    };
    pub const FRAGMENT_2_54keV: FullVolumeReference = FullVolumeReference {
        scroll_id: "Frag2",
        volume: "20230216174557",
    };
    pub const FRAGMENT_2_88keV: FullVolumeReference = FullVolumeReference {
        scroll_id: "Frag2",
        volume: "20230226143835",
    };
    pub const FRAGMENT_3_54keV: FullVolumeReference = FullVolumeReference {
        scroll_id: "Frag3",
        volume: "20230212182547",
    };
    pub const FRAGMENT_3_88keV: FullVolumeReference = FullVolumeReference {
        scroll_id: "Frag3",
        volume: "20230215142309",
    };
    pub const FRAGMENT_4_54keV: FullVolumeReference = FullVolumeReference {
        scroll_id: "Frag4",
        volume: "20230215185642",
    };
    pub const FRAGMENT_4_88keV: FullVolumeReference = FullVolumeReference {
        scroll_id: "Frag4",
        volume: "20230222173037",
    };
}
impl VolumeReference for FullVolumeReference {
    fn id(&self) -> String {
        self.volume.to_string()
    }
    fn sub_dir(&self, data_dir: &str) -> String {
        format!("{}/scroll{}/{}/", data_dir, self.scroll_id, self.volume)
    }
    fn label(&self) -> String {
        format!("Scroll {} / {}", self.scroll_id, self.volume)
    }

    fn url_path_base(&self) -> String {
        format!("scroll/{}/volume/{}/", self.scroll_id, self.volume)
    }
}

pub struct SurfaceVolumeReference {
    pub scroll_id: u16,
    pub segment_id: &'static str,
}
impl SurfaceVolumeReference {
    pub const SEGMENT_20230827161847: SurfaceVolumeReference = SurfaceVolumeReference {
        scroll_id: 1,
        segment_id: "20230827161847",
    };
    pub const SEGMENT_20231005123335: SurfaceVolumeReference = SurfaceVolumeReference {
        scroll_id: 1,
        segment_id: "20231005123335",
    };
}
impl VolumeReference for SurfaceVolumeReference {
    fn id(&self) -> String {
        self.segment_id.to_string()
    }
    fn sub_dir(&self, data_dir: &str) -> String {
        format!("{}/scroll{}/segment/{}/", data_dir, self.scroll_id, self.segment_id)
    }
    fn label(&self) -> String {
        format!("Scroll {} / Segment {}", self.scroll_id, self.segment_id)
    }

    fn url_path_base(&self) -> String {
        format!("scroll/{}/segment/{}/", self.scroll_id, self.segment_id)
    }
}

pub struct VolumeCreationParams {
    pub cache_dir: String,
}

pub enum VolumeLocation {
    LocalPath(String),
    RemoteUrl(String),
}

pub enum NewVolumeReference {
    Volume64x4(Arc<dyn VolumeReference>),
    OmeZarr { id: String, location: VolumeLocation },
    Zarr { id: String, location: VolumeLocation },
    Cells { id: String, path: String },
    Layers { id: String, path: String },
}
impl NewVolumeReference {
    const TILE_SERVER: &'static str = "https://vesuvius.virtual-void.net";

    pub fn id(&self) -> String {
        match self {
            NewVolumeReference::Volume64x4(v) => v.id(),
            NewVolumeReference::OmeZarr { id, .. } => id.clone(),
            NewVolumeReference::Zarr { id, .. } => id.clone(),
            NewVolumeReference::Cells { id, .. } => id.clone(),
            NewVolumeReference::Layers { id, .. } => id.clone(),
        }
    }
    pub fn label(&self) -> String {
        match self {
            NewVolumeReference::Volume64x4(v) => v.label(),
            NewVolumeReference::OmeZarr { id, .. } => id.clone(),
            NewVolumeReference::Zarr { id, .. } => id.clone(),
            NewVolumeReference::Cells { id, .. } => id.clone(),
            NewVolumeReference::Layers { id, .. } => id.clone(),
        }
    }
    pub fn volume(&self, params: &VolumeCreationParams) -> Volume {
        match self {
            NewVolumeReference::Volume64x4(v) => {
                let (sender, _) = std::sync::mpsc::channel::<(usize, usize, usize, Quality)>();

                let volume_dir = v.sub_dir(&params.cache_dir);

                let downloader = Box::new(SimpleDownloader::new(
                    &volume_dir,
                    Self::TILE_SERVER,
                    &v.url_path_base(),
                    None,
                    sender,
                    false,
                ));
                let v = VolumeGrid64x4Mapped::from_data_dir(&volume_dir, downloader);
                v.into_volume()
            }
            NewVolumeReference::OmeZarr { location, .. } => {
                match location {
                    VolumeLocation::RemoteUrl(url) => {
                        OmeZarrContext::<GrayScale>::from_url_to_default_cache_dir(url).into_volume()
                    }
                    VolumeLocation::LocalPath(path) => {
                        OmeZarrContext::<GrayScale>::from_path(path).into_volume()
                    }
                }
            }
            NewVolumeReference::Zarr { location, .. } => {
                match location {
                    VolumeLocation::RemoteUrl(url) => ZarrArray::from_url_to_default_cache_dir(url)
                        .into_ctx()
                        .into_ctx()
                        .into_volume(),
                    VolumeLocation::LocalPath(path) => ZarrArray::from_path(path)
                        .into_ctx()
                        .into_ctx()
                        .into_volume(),
                }
            }

            NewVolumeReference::Cells { path, .. } => VolumeGrid500Mapped::from_data_dir(path).into_volume(),
            NewVolumeReference::Layers { path, .. } => LayersMappedVolume::from_data_dir(path).into_volume(),
        }
    }

    pub fn from_url(url: impl Into<String>) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let url = url.into();
        let normalized_url = if url.ends_with('/') {
            url[..url.len() - 1].to_string()
        } else {
            url
        };
        Self::from_location(VolumeLocation::RemoteUrl(normalized_url))
    }

    pub fn from_path(path: impl Into<String>) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let path = path.into();
        let normalized_path = if path.ends_with('/') {
            path[..path.len() - 1].to_string()
        } else {
            path
        };
        Self::from_location(VolumeLocation::LocalPath(normalized_path))
    }

    fn from_location(location: VolumeLocation) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        fn check_file_content(location: &VolumeLocation, file: &str, content_check: &str) -> bool {
            match location {
                VolumeLocation::RemoteUrl(url) => {
                    let file_url = format!("{}/{}", url, file);
                    ehttp::fetch_blocking(&ehttp::Request::get(file_url))
                        .ok()
                        .filter(|response| response.status == 200)
                        .map(|response| {
                            let content = String::from_utf8_lossy(&response.bytes);
                            content.contains(content_check)
                        })
                        .unwrap_or(false)
                }
                VolumeLocation::LocalPath(path) => {
                    let file_path = format!("{}/{}", path, file);
                    std::fs::read_to_string(file_path)
                        .map(|content| content.contains(content_check))
                        .unwrap_or(false)
                }
            }
        }

        let (id, location_str) = match &location {
            VolumeLocation::RemoteUrl(url) => (
                url.split('/').last().unwrap_or("unknown").to_string(),
                url.as_str()
            ),
            VolumeLocation::LocalPath(path) => (
                path.split('/').last().unwrap_or("unknown").to_string(),
                path.as_str()
            ),
        };

        // Try OME-Zarr first
        if check_file_content(&location, ".zattrs", "multiscales") {
            return Ok(NewVolumeReference::OmeZarr { id, location });
        }

        // Try regular Zarr
        if check_file_content(&location, ".zarray", "zarr_format")
            || check_file_content(&location, ".zarray", "chunks")
        {
            return Ok(NewVolumeReference::Zarr { id, location });
        }

        let location_type = match location {
            VolumeLocation::RemoteUrl(_) => "URL",
            VolumeLocation::LocalPath(_) => "Path",
        };

        Err(format!(
            "{} {} is neither a valid OME-Zarr nor Zarr array (no .zattrs or .zarray found)",
            location_type, location_str
        )
        .into())
    }
}

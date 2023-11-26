#[derive(Copy, Clone, Debug)]
pub struct Quality {
    pub bit_mask: u8,
    pub downsampling_factor: u8,
}

pub trait VolumeReference: Send + Sync {
    fn sub_dir(&self, data_dir: &str) -> String;
    fn label(&self) -> String;
    fn url_path_base(&self) -> String;
}
impl dyn VolumeReference {
    pub const VOLUMES: [&'static dyn VolumeReference; 7] = [
        &FullVolumeReference::SCROLL1,
        &FullVolumeReference::SCROLL2,
        &FullVolumeReference::SCROLL332,
        &FullVolumeReference::SCROLL1667,
        &FullVolumeReference::FRAGMENT_PHerc1667Cr01Fr03,
        &SurfaceVolumeReference::SEGMENT_20230827161847,
        &SurfaceVolumeReference::SEGMENT_20231005123335,
    ];
}

pub struct FullVolumeReference {
    pub scroll_id: &'static str,
    pub volume: &'static str,
}
impl FullVolumeReference {
    pub const SCROLL1: FullVolumeReference = FullVolumeReference {
        scroll_id: "1",
        volume: "20230205180739",
    };
    pub const SCROLL2: FullVolumeReference = FullVolumeReference {
        scroll_id: "2",
        volume: "20230210143520",
    };
    pub const SCROLL332: FullVolumeReference = FullVolumeReference {
        scroll_id: "332",
        volume: "20231027191953",
    };
    pub const SCROLL1667: FullVolumeReference = FullVolumeReference {
        scroll_id: "1667",
        volume: "20231107190228",
    };
    pub const FRAGMENT_PHerc1667Cr01Fr03: FullVolumeReference = FullVolumeReference {
        scroll_id: "PHerc1667Cr01Fr03",
        volume: "20231121133215",
    };
}
impl VolumeReference for FullVolumeReference {
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

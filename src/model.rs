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
    fn sub_dir(&self, data_dir: &str) -> String;
    fn label(&self) -> String;
    fn url_path_base(&self) -> String;
}
impl dyn VolumeReference {
    pub const VOLUMES: [&'static dyn VolumeReference; 21] = [
        &FullVolumeReference::SCROLL1,
        &FullVolumeReference::SCROLL1B,
        &FullVolumeReference::SCROLL2,
        &FullVolumeReference::SCROLL2B,
        &FullVolumeReference::SCROLL2_88keV,
        &FullVolumeReference::SCROLL332_3_24_UM,
        &FullVolumeReference::SCROLL332_7_91_UM,
        &FullVolumeReference::SCROLL1667,
        &FullVolumeReference::SCROLL1667_7_91_UM,
        &FullVolumeReference::FRAGMENT_PHerc0051Cr04Fr08,
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
    fn sub_dir(&self, data_dir: &str) -> String { format!("{}/scroll{}/{}/", data_dir, self.scroll_id, self.volume) }
    fn label(&self) -> String { format!("Scroll {} / {}", self.scroll_id, self.volume) }

    fn url_path_base(&self) -> String { format!("scroll/{}/volume/{}/", self.scroll_id, self.volume) }
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
    pub const FRAGMENT_PHerc0051Cr04Fr08: FullVolumeReference = FullVolumeReference {
        scroll_id: "PHerc0051Cr04Fr08",
        volume: "20231121152933",
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
    fn sub_dir(&self, data_dir: &str) -> String { format!("{}/scroll{}/{}/", data_dir, self.scroll_id, self.volume) }
    fn label(&self) -> String { format!("Scroll {} / {}", self.scroll_id, self.volume) }

    fn url_path_base(&self) -> String { format!("scroll/{}/volume/{}/", self.scroll_id, self.volume) }
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
    fn label(&self) -> String { format!("Scroll {} / Segment {}", self.scroll_id, self.segment_id) }

    fn url_path_base(&self) -> String { format!("scroll/{}/segment/{}/", self.scroll_id, self.segment_id) }
}

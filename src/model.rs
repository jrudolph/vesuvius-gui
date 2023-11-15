#[derive(Copy, Clone, Debug)]
pub struct Quality {
    pub bit_mask: u8,
    pub downsampling_factor: u8,
}

pub struct VolumeReference {
    pub scroll_id: u16,
    pub volume: &'static str,
}
impl VolumeReference {
    pub const SCROLL1: VolumeReference = VolumeReference {
        scroll_id: 1,
        volume: "20230205180739",
    };
    pub const SCROLL2: VolumeReference = VolumeReference {
        scroll_id: 1,
        volume: "20230210143520",
    };
    pub const SCROLL332: VolumeReference = VolumeReference {
        scroll_id: 332,
        volume: "20230210143520",
    };
    pub const SCROLL1667: VolumeReference = VolumeReference {
        scroll_id: 1667,
        volume: "20231027191953",
    };

    pub const VOLUMES: [VolumeReference; 4] = [Self::SCROLL1, Self::SCROLL2, Self::SCROLL332, Self::SCROLL1667];

    pub fn sub_dir(&self, data_dir: &str) -> String {
        format!("{}/scroll{}/{}/", data_dir, self.scroll_id, self.volume)
    }
}

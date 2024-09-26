use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    io::Cursor,
    io::Read,
};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Hash)]
#[serde(rename_all = "camelCase")]
pub struct Scroll {
    pub id: String,
    pub num: String,
    pub old_id: String,
    #[serde(default)]
    pub is_fragment: bool,
}
impl Scroll {
    pub fn label(&self) -> String {
        let what = if self.is_fragment { "Fragment" } else { "Scroll" };
        format!("{} {} - {}", what, self.num, self.id)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Urls {
    pub composite_url: String,
    pub mask_url: String,
    pub meta_url: String,
    pub obj_url: String,
    pub ppm_url: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Segment {
    pub id: String,
    pub scroll: Scroll,
    pub volume: Option<String>,
    pub height: usize,
    pub width: usize,
    pub urls: Urls,
    pub area_cm2: Option<f64>,
}
impl PartialEq for Segment {
    fn eq(&self, other: &Self) -> bool { self.id == other.id && self.scroll == other.scroll }
}

#[derive(Default)]
pub struct Catalog {
    segments_by_scroll: HashMap<Scroll, Vec<Segment>>,
    scrolls: Vec<Scroll>,
}
impl Catalog {
    pub fn from_segments(segments: Vec<Segment>) -> Self {
        let mut scrolls = HashSet::new();
        let mut segments_by_scroll: HashMap<Scroll, Vec<Segment>> = segments
            .into_iter()
            .chunk_by(|segment| {
                scrolls.insert(segment.scroll.clone());
                segment.scroll.clone()
            })
            .into_iter()
            .map(|(scroll, group)| (scroll, group.collect()))
            .collect();

        segments_by_scroll.iter_mut().for_each(|(_, segments)| {
            segments.sort_by(|a, b| a.id.cmp(&b.id));
        });
        let mut scrolls: Vec<Scroll> = scrolls.into_iter().collect();
        scrolls.sort_by_key(|s| (s.is_fragment, s.num.clone()));

        Catalog {
            segments_by_scroll,
            scrolls,
        }
    }
    pub fn scrolls(&self) -> Vec<Scroll> { self.scrolls.clone() }
    /// Returns an iterator over the segments for the given scroll
    pub fn segments(&self, scroll: &Scroll) -> impl Iterator<Item = &Segment> {
        self.segments_by_scroll.get(scroll).into_iter().flat_map(|v| v.iter())
    }
}

pub fn load_segments() -> Vec<Segment> {
    let zst_compressed = include_bytes!("../../vesuvius-segments-2024-09-26.json.zst");
    let mut uncompressed = Vec::new();
    ruzstd::StreamingDecoder::new(Cursor::new(zst_compressed))
        .unwrap()
        .read_to_end(&mut uncompressed)
        .unwrap();

    // interpret as utf8
    let json = String::from_utf8(uncompressed).unwrap();
    serde_json::from_str(&json).unwrap()
}

pub fn load_catalog() -> Catalog { Catalog::from_segments(load_segments()) }

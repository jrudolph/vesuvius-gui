use crate::catalog::{Catalog, Segment};
use directories::BaseDirs;
use std::{
    collections::HashSet,
    io::Cursor,
    path::PathBuf,
    sync::mpsc::{Receiver, Sender},
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SegmentId {
    pub scroll_id: String,
    pub segment_id: String,
}
impl From<&Segment> for SegmentId {
    fn from(segment: &Segment) -> Self {
        Self {
            scroll_id: segment.scroll.id.clone(),
            segment_id: segment.id.clone(),
        }
    }
}

pub struct ObjRepository {
    cached_objs: HashSet<SegmentId>, // by segment id
    download_notifier: Receiver<SegmentId>,
    download_notify_sender: Sender<SegmentId>,
}
impl ObjRepository {
    pub fn new(catalog: &Catalog) -> Self {
        let mut cached_objs: HashSet<SegmentId> = HashSet::new();

        for s in catalog.scrolls.iter() {
            for seg in catalog.segments_by_scroll.get(s).unwrap() {
                let file = Self::file_for(seg);
                if file.exists() {
                    cached_objs.insert(seg.into());
                }
            }
        }

        let (download_notify_sender, download_notifier) = std::sync::mpsc::channel();
        Self {
            cached_objs,
            download_notifier,
            download_notify_sender,
        }
    }

    pub fn is_cached(&mut self, segment: &Segment) -> bool {
        self.handle_notifications();
        self.cached_objs.contains(&segment.into())
    }
    pub fn get(&mut self, segment: &Segment) -> Option<PathBuf> {
        if self.is_cached(segment) {
            Some(Self::file_for(segment))
        } else {
            None
        }
    }

    pub fn download(&mut self, segment: &Segment, on_done: impl 'static + Send + FnOnce(Segment) -> ()) -> () {
        let s = segment.clone();
        let obj_file = Self::file_for(segment);
        // use existing or download
        println!(
            "Downloading obj file from {} to {}",
            segment.urls.obj_url,
            &obj_file.to_str().unwrap()
        );
        let sender = self.download_notify_sender.clone();
        ehttp::fetch(ehttp::Request::get(&segment.urls.obj_url), move |response| {
            if let Ok(response) = response {
                std::fs::create_dir_all(&obj_file.parent().unwrap()).unwrap();
                let mut file = std::fs::File::create(&obj_file).unwrap();
                let bytes = response.bytes;
                println!("Downloaded {} bytes", bytes.len());
                std::io::copy(&mut Cursor::new(bytes), &mut file).unwrap();
                let _ = sender.send((&s).into()); // ignore result
                on_done(s);
            }
        });
    }

    fn handle_notifications(&mut self) {
        for id in self.download_notifier.try_iter() {
            self.cached_objs.insert(id);
        }
    }

    fn file_for(segment: &Segment) -> PathBuf {
        let dir = BaseDirs::new().unwrap().cache_dir().join("vesuvius-gui");
        dir.join(format!("segments/{}/{}.obj", &segment.scroll.old_id, segment.id))
    }
}

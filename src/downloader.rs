use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc::Sender,
        Arc, Mutex,
    },
    thread,
    time::Duration,
};

use crate::model::*;

#[derive(Copy, Clone, Debug)]
pub enum DownloadState {
    Queuing,
    Downloading,
    Done,
    Failed,
    Delayed,
    Pruned, // was not in view any more and therefore pruned
}
impl DownloadState {
    pub fn needs_reload(&self) -> bool {
        match self {
            DownloadState::Queuing => false,
            DownloadState::Downloading => false,
            DownloadState::Done => true,
            DownloadState::Failed => true,
            DownloadState::Pruned => true,
            DownloadState::Delayed => true,
        }
    }
}

type DownloadTask = (Arc<Mutex<DownloadState>>, usize, usize, usize, Quality);

enum DownloadMessage {
    Download(DownloadTask),
    Position(i32, i32, i32, usize, usize),
}

pub struct Downloader {
    download_queue: Sender<DownloadMessage>,
}
impl Downloader {
    pub fn new(
        dir: &str,
        tile_server_base: &'static str,
        volume: &'static dyn VolumeReference,
        authorization: Option<String>,
        download_notifier: Sender<()>,
    ) -> Downloader {
        let (sender, receiver) = std::sync::mpsc::channel::<DownloadMessage>();

        let count = Arc::new(AtomicUsize::new(0));

        std::fs::create_dir_all(dir.to_string()).unwrap();
        let dir = dir.to_string();
        let volume = volume.clone();
        thread::spawn(move || {
            let mut queue = Vec::new();
            let mut pos = (0, 0, 0, 0 as usize, 0 as usize);
            loop {
                while let Ok(msg) = receiver.try_recv() {
                    match msg {
                        DownloadMessage::Download(task) => {
                            queue.push(task);
                        }
                        DownloadMessage::Position(x, y, z, width, height) => pos = (x, y, z, width, height),
                    }
                }

                /*let larger_edge = pos.3.max(pos.4) as i32;
                queue.retain(|(state, x, y, z, q)| {
                    let f = q.downsampling_factor as i32;
                    let dx = *x as i32 * 64 * f + 32 * f - pos.0;
                    let dy = *y as i32 * 64 * f + 32 * f - pos.1;
                    let dz = *z as i32 * 64 * f + 32 * f - pos.2;

                    if (dx.abs() > larger_edge / 2 && dy.abs() > larger_edge / 2) && dz.abs() > larger_edge / 2 {
                        /* println!(
                            "Pruning {} {} {} {} {:?}",
                            x,
                            y,
                            z,
                            q.downsampling_factor,
                            *state.lock().unwrap()
                        ); */
                        *state.lock().unwrap() = DownloadState::Pruned;
                        false
                    } else {
                        true
                    }
                }); */

                let cur = count.load(Ordering::Acquire);
                if cur >= 32 || queue.is_empty() {
                    thread::sleep(Duration::from_millis(50));
                    continue;
                }

                if count
                    .compare_exchange(cur, cur + 1, Ordering::Acquire, Ordering::Acquire)
                    .is_ok()
                {
                    queue.sort_by_key(|(_, x, y, z, q)| {
                        let f = q.downsampling_factor as i32;
                        let dx = *x as i32 * 64 * f + 32 * f - pos.0;
                        let dy = *y as i32 * 64 * f + 32 * f - pos.1;
                        let dz = *z as i32 * 64 * f + 32 * f - pos.2;
                        let score = (q.downsampling_factor, -(dx * dx + dy * dy + dz * dz));
                        //println!("{} {} {} {}", x, y, z, score);
                        score
                    });

                    let (state, x, y, z, quality) = queue.pop().unwrap();
                    {
                        *state.lock().unwrap() = DownloadState::Downloading;
                        //let url = format!("https://vesuvius.virtual-void.net/tiles/scroll/332/volume/20231027191953/download/128-16?x={}&y={}&z={}", x, y, z);
                        //let url = format!("http://localhost:8095/tiles/scroll/332/volume/20231027191953/download/128-16?x={}&y={}&z={}", x, y, z);
                        //let url = format!("http://5.161.229.51:8095/tiles/scroll/332/volume/20231027191953/download/128-16?x={}&y={}&z={}", x, y, z);
                        let url = format!(
                            "{}/tiles/{}download/64-4?x={}&y={}&z={}&bitmask={}&downsampling={}",
                            tile_server_base,
                            volume.url_path_base(),
                            x,
                            y,
                            z,
                            quality.bit_mask,
                            quality.downsampling_factor
                        );
                        println!("Downloading {} {} {} {}", x, y, z, quality.downsampling_factor);
                        //let url = format!("https://vesuvius.virtual-void.net/tiles/scroll/1667/volume/20231107190228/download/64-4?x={}&y={}&z={}&bitmask={}&downsampling={}", x, y, z, quality.bit_mask, quality.downsampling_factor);
                        //let url = format!("http://localhost:8095/tiles/scroll/1/volume/20230205180739/download/64-4?x={}&y={}&z={}&bitmask={}&downsampling={}", x, y, z, quality.bit_mask, quality.downsampling_factor);
                        let mut request = ehttp::Request::get(url.clone());
                        if let Some(authorization) = authorization.clone() {
                            request.headers.insert(
                                "Authorization".to_string(),
                                format!("Basic {}", base64::encode(authorization)),
                            );
                        }

                        let notifier = download_notifier.clone();
                        let dir = dir.clone();
                        //println!("downloading tile {}/{}/{} q{}", x, y, z, quality.downsampling_factor);
                        let c2 = count.clone();
                        ehttp::fetch(request, move |response| {
                            if let Ok(res) = response {
                                if res.status == 200 {
                                    //println!("got tile {}/{}/{} q{}", x, y, z, quality.downsampling_factor);
                                    let bytes = res.bytes;
                                    // save bytes to file
                                    let file_name = format!(
                                        "{}/64-4/d{:02}/z{:03}/xyz-{:03}-{:03}-{:03}-b{:03}-d{:02}.bin",
                                        dir,
                                        quality.downsampling_factor,
                                        z,
                                        x,
                                        y,
                                        z,
                                        quality.bit_mask,
                                        quality.downsampling_factor
                                    );
                                    std::fs::create_dir_all(format!(
                                        "{}/64-4/d{:02}/z{:03}",
                                        dir, quality.downsampling_factor, z
                                    ))
                                    .unwrap();
                                    std::fs::write(file_name, bytes).unwrap();
                                    *state.lock().unwrap() = DownloadState::Done;
                                    notifier.send(());
                                } else if res.status == 420 {
                                    println!("delayed tile {}/{}/{} q{}", x, y, z, quality.downsampling_factor);
                                    *state.lock().unwrap() = DownloadState::Delayed;
                                } else {
                                    println!(
                                        "failed to download tile {}/{}/{} q{}: {}",
                                        x, y, z, quality.downsampling_factor, res.status
                                    );
                                    *state.lock().unwrap() = DownloadState::Failed;
                                }
                            } else {
                                println!(
                                    "failed to download tile {}/{}/{} q{}: {}",
                                    x,
                                    y,
                                    z,
                                    quality.downsampling_factor,
                                    response.err().unwrap()
                                );
                                *state.lock().unwrap() = DownloadState::Failed;
                            }

                            c2.fetch_sub(1, Ordering::Acquire);
                        });
                    }
                }
            }
        });

        Downloader { download_queue: sender }
    }

    pub fn check_authorization(tile_server_base: &'static str, authorization: Option<String>) -> bool {
        // check if request to tile server is authorized
        let vol1 = FullVolumeReference::SCROLL1;
        let url = format!(
            "{}/tiles/scroll/{}/volume/{}/",
            tile_server_base, vol1.scroll_id, vol1.volume
        );
        let mut request = ehttp::Request::get(url.clone());
        if let Some(authorization) = authorization {
            request.headers.insert(
                "Authorization".to_string(),
                format!("Basic {}", base64::encode(authorization)),
            );
        }
        match ehttp::fetch_blocking(&request) {
            Ok(res) => {
                if res.status == 200 {
                    return true;
                } else if res.status == 401 {
                    return false;
                } else {
                    println!("failed to check authorization: {}", res.status);
                    false
                }
            }
            Err(e) => {
                println!("Request failed: {}", e);
                false
            }
        }
    }

    pub fn queue(&self, task: DownloadTask) { self.download_queue.send(DownloadMessage::Download(task)).unwrap(); }
    pub fn position(&self, x: i32, y: i32, z: i32, width: usize, height: usize) {
        self.download_queue
            .send(DownloadMessage::Position(x, y, z, width, height))
            .unwrap();
    }
}

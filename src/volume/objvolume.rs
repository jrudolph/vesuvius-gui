use super::{PaintVolume, VoxelPaintVolume, VoxelVolume};
use std::{cell::RefCell, sync::Arc};
use wavefront_obj::obj::{self, Object, Primitive};

struct ObjFile {
    object: Object,
}

pub struct ObjVolume {
    volume: Arc<RefCell<dyn VoxelPaintVolume>>,
    obj: ObjFile,
}
impl ObjVolume {
    pub fn new(obj_file_path: &str, base_volume: Arc<RefCell<dyn VoxelPaintVolume>>) -> Self {
        let mut objects = Self::load_obj(obj_file_path).objects;
        println!("Loaded obj file with {} objects", objects.len());
        for o in objects.iter() {
            println!(
                "Object: {}, geometries: {} vertices: {}",
                o.name,
                o.geometry.len(),
                o.vertices.len()
            );
        }

        let object = objects.remove(1);
        let obj = ObjFile { object };

        Self {
            volume: base_volume,
            obj,
        }
    }

    fn load_obj(file_path: &str) -> obj::ObjSet {
        let obj_file = std::fs::read_to_string(file_path).unwrap();
        obj::parse(obj_file).unwrap()
    }

    pub fn width(&self) -> usize {
        5048 // FIXME, hardcoded for 1847 segment
    }
    pub fn height(&self) -> usize {
        9163 // FIXME
    }
    pub fn convert_to_volume_coords(&self, coord: [i32; 3]) -> [i32; 3] {
        let u = coord[0];
        let v = coord[1];
        let w = coord[2] as f64;

        let obj = &self.obj.object;
        for s in obj.geometry[0].shapes.iter() {
            match s.primitive {
                Primitive::Triangle(i1, i2, i3) => {
                    let v1 = &obj.tex_vertices[i1.1.unwrap()];
                    let v2 = &obj.tex_vertices[i2.1.unwrap()];
                    let v3 = &obj.tex_vertices[i3.1.unwrap()];

                    let u1 = (v1.u * self.width() as f64) as i32;
                    let v1 = ((1.0 - v1.v) * self.height() as f64) as i32;

                    let u2 = (v2.u * self.width() as f64) as i32;
                    let v2 = ((1.0 - v2.v) * self.height() as f64) as i32;

                    let u3 = (v3.u * self.width() as f64) as i32;
                    let v3 = ((1.0 - v3.v) * self.height() as f64) as i32;

                    let min_u_t = u1.min(u2).min(u3);
                    let max_u_t = u1.max(u2).max(u3);

                    let min_v_t = v1.min(v2).min(v3);
                    let max_v_t = v1.max(v2).max(v3);

                    // broad pre-check against triangle bounding box
                    if min_u_t <= u && u <= max_u_t && min_v_t <= v && v <= max_v_t {
                        let w0 = orient2d(u2, v2, u3, v3, u, v);
                        let w1 = orient2d(u3, v3, u1, v1, u, v);
                        let w2 = orient2d(u1, v1, u2, v2, u, v);

                        if w0 >= 0 && w1 >= 0 && w2 >= 0 {
                            let xyz1 = &obj.vertices[i1.0];
                            let xyz2 = &obj.vertices[i2.0];
                            let xyz3 = &obj.vertices[i3.0];

                            // barymetric interpolation
                            let invwsum = 1. / (w0 + w1 + w2) as f64;
                            let x = (w0 as f64 * xyz1.x + w1 as f64 * xyz2.x + w2 as f64 * xyz3.x) * invwsum;
                            let y = (w0 as f64 * xyz1.y + w1 as f64 * xyz2.y + w2 as f64 * xyz3.y) * invwsum;
                            let z = (w0 as f64 * xyz1.z + w1 as f64 * xyz2.z + w2 as f64 * xyz3.z) * invwsum;

                            let (nx, ny, nz) = if coord[2] == 0 {
                                (0.0, 0.0, 0.0)
                            } else {
                                let nxyz1 = &obj.normals[i1.2.unwrap()];
                                let nxyz2 = &obj.normals[i2.2.unwrap()];
                                let nxyz3 = &obj.normals[i3.2.unwrap()];

                                let nx = (w0 as f64 * nxyz1.x + w1 as f64 * nxyz2.x + w2 as f64 * nxyz3.x) * invwsum;
                                let ny = (w0 as f64 * nxyz1.y + w1 as f64 * nxyz2.y + w2 as f64 * nxyz3.y) * invwsum;
                                let nz = (w0 as f64 * nxyz1.z + w1 as f64 * nxyz2.z + w2 as f64 * nxyz3.z) * invwsum;

                                (nx, ny, nz)
                            };

                            return [(x + w * nx) as i32, (y + w * ny) as i32, (z + w * nz) as i32];
                        }
                    }
                }
                _ => todo!(),
            }
        }

        [-1, -1, -1]
    }
}

fn orient2d(u1: i32, v1: i32, u2: i32, v2: i32, u3: i32, v3: i32) -> i32 {
    (u2 - u1) * (v3 - v1) - (v2 - v1) * (u3 - u1)
}

impl PaintVolume for ObjVolume {
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
        _config: &super::DrawingConfig,
        buffer: &mut [u8],
    ) {
        assert!(u_coord == 0);
        assert!(v_coord == 1);
        assert!(plane_coord == 2);

        let mut volume = self.volume.borrow_mut();

        let ffactor = sfactor as f64;

        let w_factor = xyz[2] as f64;

        let min_u = xyz[0] - width as i32 / 2 * paint_zoom as i32;
        let max_u = xyz[0] + width as i32 / 2 * paint_zoom as i32;
        let min_v = xyz[1] - height as i32 / 2 * paint_zoom as i32;
        let max_v = xyz[1] + height as i32 / 2 * paint_zoom as i32;

        /* println!(
            "xyz: {:?}, width: {}, height: {}, sfactor: {}, paint_zoom: {}",
            xyz, width, height, sfactor, paint_zoom
        );
        println!("min_u: {}, max_u: {}, min_v: {}, max_v: {}", min_u, max_u, min_v, max_v); */

        let obj = &self.obj.object;
        for s in obj.geometry[0].shapes.iter() {
            match s.primitive {
                Primitive::Triangle(i1, i2, i3) => {
                    let v1 = &obj.tex_vertices[i1.1.unwrap()];
                    let v2 = &obj.tex_vertices[i2.1.unwrap()];
                    let v3 = &obj.tex_vertices[i3.1.unwrap()];

                    let u1 = (v1.u * self.width() as f64) as i32;
                    let v1 = ((1.0 - v1.v) * self.height() as f64) as i32;

                    let u2 = (v2.u * self.width() as f64) as i32;
                    let v2 = ((1.0 - v2.v) * self.height() as f64) as i32;

                    let u3 = (v3.u * self.width() as f64) as i32;
                    let v3 = ((1.0 - v3.v) * self.height() as f64) as i32;

                    let min_u_t = u1.min(u2).min(u3);
                    let max_u_t = u1.max(u2).max(u3);

                    let min_v_t = v1.min(v2).min(v3);
                    let max_v_t = v1.max(v2).max(v3);

                    // clip to paint area
                    if !(min_u_t > max_u || max_u_t < min_u || min_v_t > max_v || max_v_t < min_v) {
                        /*
                        Implement the simple tri rasterization algorithm from https://fgiesen.wordpress.com/2013/02/08/triangle-rasterization-in-practice/

                        // Compute triangle bounding box
                        int minX = min3(v0.x, v1.x, v2.x);
                        int minY = min3(v0.y, v1.y, v2.y);
                        int maxX = max3(v0.x, v1.x, v2.x);
                        int maxY = max3(v0.y, v1.y, v2.y);

                        // Clip against screen bounds
                        minX = max(minX, 0);
                        minY = max(minY, 0);
                        maxX = min(maxX, screenWidth - 1);
                        maxY = min(maxY, screenHeight - 1);

                        // Rasterize
                        Point2D p;
                        for (p.y = minY; p.y <= maxY; p.y++) {
                            for (p.x = minX; p.x <= maxX; p.x++) {
                                // Determine barycentric coordinates
                                int w0 = orient2d(v1, v2, p);
                                int w1 = orient2d(v2, v0, p);
                                int w2 = orient2d(v0, v1, p);

                                // If p is on or inside all edges, render pixel.
                                if (w0 >= 0 && w1 >= 0 && w2 >= 0)
                                    renderPixel(p, w0, w1, w2);
                            }
                        } */

                        //println!("u1: {}, v1: {}, u2: {}, v2: {}, u3: {}, v3: {}", u1, v1, u2, v2, u3, v3);

                        let u1i = u1 - min_u;
                        let v1i = v1 - min_v;

                        let u2i = u2 - min_u;
                        let v2i = v2 - min_v;

                        let u3i = u3 - min_u;
                        let v3i = v3 - min_v;

                        /* println!(
                            "u1i: {}, v1i: {}, u2i: {}, v2i: {}, u3i: {}, v3i: {}",
                            u1i, v1i, u2i, v2i, u3i, v3i
                        ); */

                        // align values to paint_zoom to avoid gaps because of integer processing
                        fn paint_zoom_align(v: i32, paint_zoom: u8) -> i32 {
                            let v = v / paint_zoom as i32;
                            v * paint_zoom as i32
                        }
                        fn paint_zoom_align_up(v: i32, paint_zoom: u8) -> i32 {
                            paint_zoom_align(v + paint_zoom as i32 - 1, paint_zoom)
                        }

                        // calculate triangle bounding box in paint coordinates, make sure to slightly widen to align
                        // to paint_zoom to avoid gaps at the edges because of integer processing
                        let tmin_u = paint_zoom_align(u1i.min(u2i).min(u3i), paint_zoom).max(0);
                        let tmax_u = paint_zoom_align_up(u1i.max(u2i).max(u3i), paint_zoom)
                            .min(width as i32 * paint_zoom as i32 - 1);

                        let tmin_v = paint_zoom_align(v1i.min(v2i).min(v3i), paint_zoom).max(0);
                        let tmax_v = paint_zoom_align_up(v1i.max(v2i).max(v3i), paint_zoom)
                            .min(height as i32 * paint_zoom as i32 - 1);

                        /* println!(
                            "tmin_u: {}, tmax_u: {}, tmin_v: {}, tmax_v: {}",
                            tmin_u, tmax_u, tmin_v, tmax_v
                        ); */

                        for v in (tmin_v..=tmax_v).step_by(paint_zoom as usize) {
                            for u in (tmin_u..=tmax_u).step_by(paint_zoom as usize) {
                                let w0 = orient2d(u2i, v2i, u3i, v3i, u, v);
                                let w1 = orient2d(u3i, v3i, u1i, v1i, u, v);
                                let w2 = orient2d(u1i, v1i, u2i, v2i, u, v);

                                //println!("At u:{} v:{} w0: {}, w1: {}, w2: {}", u, v, w0, w1, w2);

                                if w0 >= 0 && w1 >= 0 && w2 >= 0 {
                                    let idx = (v / paint_zoom as i32) * width as i32 + (u / paint_zoom as i32);
                                    if idx >= 0 && idx < buffer.len() as i32 {
                                        let xyz1 = &obj.vertices[i1.0];
                                        let xyz2 = &obj.vertices[i2.0];
                                        let xyz3 = &obj.vertices[i3.0];

                                        // barymetric interpolation
                                        let invwsum = 1. / (w0 + w1 + w2) as f64;
                                        let x =
                                            (w0 as f64 * xyz1.x + w1 as f64 * xyz2.x + w2 as f64 * xyz3.x) * invwsum;
                                        let y =
                                            (w0 as f64 * xyz1.y + w1 as f64 * xyz2.y + w2 as f64 * xyz3.y) * invwsum;
                                        let z =
                                            (w0 as f64 * xyz1.z + w1 as f64 * xyz2.z + w2 as f64 * xyz3.z) * invwsum;

                                        let (nx, ny, nz) = if xyz[2] == 0 {
                                            (0.0, 0.0, 0.0)
                                        } else {
                                            let nxyz1 = &obj.normals[i1.2.unwrap()];
                                            let nxyz2 = &obj.normals[i2.2.unwrap()];
                                            let nxyz3 = &obj.normals[i3.2.unwrap()];

                                            let nx = (w0 as f64 * nxyz1.x + w1 as f64 * nxyz2.x + w2 as f64 * nxyz3.x)
                                                * invwsum;
                                            let ny = (w0 as f64 * nxyz1.y + w1 as f64 * nxyz2.y + w2 as f64 * nxyz3.y)
                                                * invwsum;
                                            let nz = (w0 as f64 * nxyz1.z + w1 as f64 * nxyz2.z + w2 as f64 * nxyz3.z)
                                                * invwsum;

                                            (nx, ny, nz)
                                        };

                                        let v = volume.get(
                                            [
                                                (x + w_factor * nx) / ffactor,
                                                (y + w_factor * ny) / ffactor,
                                                (z + w_factor * nz) / ffactor,
                                            ],
                                            sfactor as i32,
                                        );

                                        buffer[idx as usize] = v;
                                    }
                                }
                            }
                        }
                    }
                }
                _ => todo!(),
            }
        }
    }
}

#[allow(dead_code)] // useful for debugging triangle shapes
fn line(x0: i32, y0: i32, x1: i32, y1: i32, buffer: &mut [u8], width: usize) {
    // simple bresenham algorithm from https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm

    let dx = (x1 as i32 - x0 as i32).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 as i32 - y0 as i32).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut error = dx + dy;

    let mut x = x0 as i32;
    let mut y = y0 as i32;

    while x != x1 || y != y1 {
        //println!("x: {}, y: {}", x, y);
        let idx = y * width as i32 + x;
        if idx >= 0 && idx < buffer.len() as i32 {
            buffer[idx as usize] = 255;
        }

        let e2 = 2 * error;
        if e2 >= dy {
            error += dy;
            x += sx;
        }
        if e2 <= dx {
            error += dx;
            y += sy;
        }
    }
}

impl VoxelVolume for ObjVolume {
    fn get(&mut self, _xyz: [f64; 3], _downsampling: i32) -> u8 { todo!() }
}

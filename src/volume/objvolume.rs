#![allow(dead_code, unused_variables)] // FIXME
use super::{PaintVolume, VoxelPaintVolume, VoxelVolume};
use std::u128;
use wavefront_obj::obj::{self, ObjSet, Object, Primitive, Shape};

struct ObjFile {
    object: Object,
}

pub struct ObjVolume {
    volume: Box<dyn VoxelPaintVolume>,
    obj: ObjFile,
}
impl ObjVolume {
    pub fn new(obj_file_path: &str, base_volume: Box<dyn VoxelPaintVolume>) -> Self {
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
        config: &super::DrawingConfig,
        buffer: &mut [u8],
    ) {
        assert!(u_coord == 0);
        assert!(v_coord == 1);
        assert!(plane_coord == 2);

        let w_factor = xyz[2] as f64 * sfactor as f64;

        /*
        let im_rel_u = (im_u as i32 - width as i32 / 2) * paint_zoom as i32;
                let im_rel_v = (im_v as i32 - height as i32 / 2) * paint_zoom as i32;

                let mut uvw: [f64; 3] = [0.; 3];
                uvw[u_coord] = (xyz[u_coord] + im_rel_u) as f64 / fi32;
                uvw[v_coord] = (xyz[v_coord] + im_rel_v) as f64 / fi32;
                uvw[plane_coord] = (xyz[plane_coord]) as f64 / fi32;

                let v = self.get(uvw, _sfactor as i32);
                buffer[im_v * width + im_u] = v; */

        let min_u = (xyz[0] - width as i32 / 2 * paint_zoom as i32) / sfactor as i32;
        let max_u = (xyz[0] + width as i32 / 2 * paint_zoom as i32) / sfactor as i32;
        let min_v = (xyz[1] - height as i32 / 2 * paint_zoom as i32) / sfactor as i32;
        let max_v = (xyz[1] + height as i32 / 2 * paint_zoom as i32) / sfactor as i32;

        /* println!(
            "xyz: {:?}, width: {}, height: {}, sfactor: {}, paint_zoom: {}",
            xyz, width, height, sfactor, paint_zoom
        ); */
        //println!("min_u: {}, max_u: {}, min_v: {}, max_v: {}", min_u, max_u, min_v, max_v);

        /*
                int orient2d(const Point2D& a, const Point2D& b, const Point2D& c)
        {
            return (b.x-a.x)*(c.y-a.y) - (b.y-a.y)*(c.x-a.x);
        }

                 */
        fn orient2d(u1: i32, v1: i32, u2: i32, v2: i32, u3: i32, v3: i32) -> i32 {
            (u2 - u1) * (v3 - v1) - (v2 - v1) * (u3 - u1)
        }

        let mut done = false;
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

                    if !done
                        && (u1 >= min_u && u1 < max_u && v1 >= min_v && v1 < max_v)
                        && (u2 >= min_u && u2 < max_u && v2 >= min_v && v2 < max_v)
                        && (u3 >= min_u && u3 < max_u && v3 >= min_v && v3 < max_v)
                    {
                        /*
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

                        let tmin_u = u1i.min(u2i).min(u3i);
                        let tmax_u = u1i.max(u2i).max(u3i);

                        let tmin_v = v1i.min(v2i).min(v3i);
                        let tmax_v = v1i.max(v2i).max(v3i);

                        /* println!(
                            "tmin_u: {}, tmax_u: {}, tmin_v: {}, tmax_v: {}",
                            tmin_u, tmax_u, tmin_v, tmax_v
                        ); */

                        for v in tmin_v..=tmax_v {
                            for u in tmin_u..=tmax_u {
                                let w0 = orient2d(u2i, v2i, u3i, v3i, u, v);
                                let w1 = orient2d(u3i, v3i, u1i, v1i, u, v);
                                let w2 = orient2d(u1i, v1i, u2i, v2i, u, v);

                                //println!("At u:{} v:{} w0: {}, w1: {}, w2: {}", u, v, w0, w1, w2);

                                if w0 >= 0 && w1 >= 0 && w2 >= 0 {
                                    let idx = v * width as i32 + u;
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

                                        let v = self
                                            .volume
                                            .get([x + w_factor * nx, y + w_factor * ny, z + w_factor * nz], 1);

                                        buffer[idx as usize] = v;
                                    }
                                }
                            }
                        }

                        //done = true;

                        //line(u1i, v1i, u2i, v2i, buffer, width);
                        //line(u2i, v2i, u3i, v3i, buffer, width);
                        //line(u3i, v3i, u1i, v1i, buffer, width);
                    }
                }
                _ => todo!(),
            }
        }

        //line(0, 0, width - 1, height - 1, buffer, width);

        //todo!()
    }
}

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

    //buffer[y1 * width + x1] = 255;
}

impl VoxelVolume for ObjVolume {
    fn get(&mut self, xyz: [f64; 3], downsampling: i32) -> u8 { todo!() }
}

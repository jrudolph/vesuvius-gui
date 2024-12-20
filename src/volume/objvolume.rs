use super::{Image, PaintVolume, SurfaceVolume, VoxelPaintVolume, VoxelVolume};
use std::{cell::RefCell, sync::Arc};
use wavefront_obj::obj::{self, Object, Primitive, Vertex};

struct ObjFile {
    object: Object,
}
impl ObjFile {
    fn has_inverted_uv_tris(&self) -> bool {
        for s in self.object.geometry[0].shapes.iter().skip(1) {
            match s.primitive {
                Primitive::Triangle(i1, i2, i3) => {
                    let v1 = &self.object.tex_vertices[i1.1.unwrap()];
                    let v2 = &self.object.tex_vertices[i2.1.unwrap()];
                    let v3 = &self.object.tex_vertices[i3.1.unwrap()];

                    let u1 = (v1.u * 100000.) as i32;
                    let v1 = (v1.v * 100000.) as i32;

                    let u2 = (v2.u * 100000.) as i32;
                    let v2 = (v2.v * 100000.) as i32;

                    let u3 = (v3.u * 100000.) as i32;
                    let v3 = (v3.v * 100000.) as i32;

                    let u = (u1 + u2 + u3) / 3;
                    let v = (v1 + v2 + v3) / 3;

                    let w0 = orient2d(u2, v2, u3, v3, u, v);
                    let w1 = orient2d(u3, v3, u1, v1, u, v);
                    let w2 = orient2d(u1, v1, u2, v2, u, v);

                    return w0 >= 0 && w1 >= 0 && w2 >= 0;
                }
                _ => (),
            }
        }
        false
    }
}

pub struct ObjVolume {
    volume: Arc<RefCell<dyn VoxelPaintVolume>>,
    obj: ObjFile,
    width: usize,
    height: usize,
    invert_tris: bool,
}
impl ObjVolume {
    pub fn new(
        obj_file_path: &str,
        base_volume: Arc<RefCell<dyn VoxelPaintVolume>>,
        width: usize,
        height: usize,
    ) -> Self {
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

        let object = objects.remove(0);
        let obj = ObjFile { object };
        let invert_tris = obj.has_inverted_uv_tris();

        Self {
            volume: base_volume,
            obj,
            width,
            height,
            invert_tris,
        }
    }

    fn load_obj(file_path: &str) -> obj::ObjSet {
        let obj_file = std::fs::read_to_string(file_path).unwrap();
        // filter out material definitions that wavefront_obj does not cope well with
        let obj_file = obj_file
            .lines()
            .filter(|line| !line.starts_with("mtllib"))
            .filter(|line| !line.starts_with("usemtl"))
            .collect::<Vec<_>>()
            .join("\n");
        obj::parse(obj_file).unwrap()
    }

    pub fn width(&self) -> usize { self.width }
    pub fn height(&self) -> usize { self.height }
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
                    let v1 = (self.v(v1.v) * self.height() as f64) as i32;

                    let u2 = (v2.u * self.width() as f64) as i32;
                    let v2 = (self.v(v2.v) * self.height() as f64) as i32;

                    let u3 = (v3.u * self.width() as f64) as i32;
                    let v3 = (self.v(v3.v) * self.height() as f64) as i32;

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

    fn v(&self, v: f64) -> f64 {
        if self.invert_tris {
            v
        } else {
            1.0 - v
        }
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
        config: &super::DrawingConfig,
        buffer: &mut Image,
    ) {
        assert!(u_coord == 0);
        assert!(v_coord == 1);
        assert!(plane_coord == 2);

        let draw_outlines = config.draw_xyz_outlines;

        let real_xyz = self.convert_to_volume_coords(xyz);

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
                    let v1 = (self.v(v1.v) * self.height() as f64) as i32;

                    let u2 = (v2.u * self.width() as f64) as i32;
                    let v2 = (self.v(v2.v) * self.height() as f64) as i32;

                    let u3 = (v3.u * self.width() as f64) as i32;
                    let v3 = (self.v(v3.v) * self.height() as f64) as i32;

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

                        // TODO: would probably better to operate in paint coordinates instead
                        for v in (tmin_v..=tmax_v).step_by(paint_zoom as usize) {
                            for u in (tmin_u..=tmax_u).step_by(paint_zoom as usize) {
                                let w0 = orient2d(u2i, v2i, u3i, v3i, u, v);
                                let w1 = orient2d(u3i, v3i, u1i, v1i, u, v);
                                let w2 = orient2d(u1i, v1i, u2i, v2i, u, v);

                                //println!("At u:{} v:{} w0: {}, w1: {}, w2: {}", u, v, w0, w1, w2);

                                if w0 >= 0 && w1 >= 0 && w2 >= 0 {
                                    if u >= 0
                                        && u < width as i32 * paint_zoom as i32
                                        && v >= 0
                                        && v < height as i32 * paint_zoom as i32
                                    {
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

                                        let x = x + w_factor * nx;
                                        let y = y + w_factor * ny;
                                        let z = z + w_factor * nz;

                                        let value = volume.get([x / ffactor, y / ffactor, z / ffactor], sfactor as i32);

                                        buffer.set_gray(
                                            u as usize / paint_zoom as usize,
                                            v as usize / paint_zoom as usize,
                                            value,
                                        );

                                        if draw_outlines {
                                            if (x - real_xyz[0] as f64).abs() < 2.0 {
                                                buffer.set_rgb(
                                                    u as usize / paint_zoom as usize,
                                                    v as usize / paint_zoom as usize,
                                                    0,
                                                    0,
                                                    255,
                                                );
                                            } else if (y - real_xyz[1] as f64).abs() < 2.0 {
                                                buffer.set_rgb(
                                                    u as usize / paint_zoom as usize,
                                                    v as usize / paint_zoom as usize,
                                                    255,
                                                    0,
                                                    0,
                                                );
                                            } else if (z - real_xyz[2] as f64).abs() < 2.0 {
                                                buffer.set_rgb(
                                                    u as usize / paint_zoom as usize,
                                                    v as usize / paint_zoom as usize,
                                                    0,
                                                    255,
                                                    0,
                                                );
                                            }
                                        }
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

impl SurfaceVolume for ObjVolume {
    fn paint_plane_intersection(
        &self,
        xyz: [i32; 3],
        u_coord: usize,
        v_coord: usize,
        plane_coord: usize,
        width: usize,
        height: usize,
        _sfactor: u8,
        paint_zoom: u8,
        _config: &super::DrawingConfig,
        image: &mut Image,
    ) {
        let u = xyz[u_coord];
        let v = xyz[v_coord];

        let min_u = u - width as i32 / 2 * paint_zoom as i32;
        let max_u = u + width as i32 / 2 * paint_zoom as i32;
        let min_v = v - height as i32 / 2 * paint_zoom as i32;
        let max_v = v + height as i32 / 2 * paint_zoom as i32;

        let w = xyz[plane_coord];

        let mut mins = [0.0, 0.0, 0.0];
        let mut maxs = [0.0, 0.0, 0.0];

        mins[u_coord] = min_u as f64;
        maxs[u_coord] = max_u as f64;

        mins[v_coord] = min_v as f64;
        maxs[v_coord] = max_v as f64;

        mins[plane_coord] = w as f64;
        maxs[plane_coord] = w as f64;

        let obj = &self.obj.object;
        for s in obj.geometry[0].shapes.iter() {
            match s.primitive {
                Primitive::Triangle(i1, i2, i3) => {
                    let v1 = &obj.vertices[i1.0];
                    let v2 = &obj.vertices[i2.0];
                    let v3 = &obj.vertices[i3.0];

                    let minx = v1.x.min(v2.x).min(v3.x);
                    let maxx = v1.x.max(v2.x).max(v3.x);

                    let miny = v1.y.min(v2.y).min(v3.y);
                    let maxy = v1.y.max(v2.y).max(v3.y);

                    let minz = v1.z.min(v2.z).min(v3.z);
                    let maxz = v1.z.max(v2.z).max(v3.z);

                    // clip to paint area
                    if !(minx > maxs[0]
                        || maxx < mins[0]
                        || miny > maxs[1]
                        || maxy < mins[1]
                        || minz > maxs[2]
                        || maxz < mins[2])
                    {
                        fn coord(v: &Vertex) -> [f64; 3] { [v.x, v.y, v.z] }
                        fn intersects(v1: &Vertex, v2: &Vertex, w: i32, plane_coord: usize) -> bool {
                            (coord(v1)[plane_coord] - w as f64).signum() != (coord(v2)[plane_coord] - w as f64).signum()
                        }

                        let mut points: Vec<[f64; 3]> = vec![];

                        let mut add_intersection_points = |v1: &Vertex, v2: &Vertex| {
                            if intersects(v1, v2, w, plane_coord) {
                                let x1 = coord(v1)[u_coord];
                                let y1 = coord(v1)[v_coord];

                                let x2 = coord(v2)[u_coord];
                                let y2 = coord(v2)[v_coord];

                                let d1 = w as f64 - coord(v1)[plane_coord];
                                let d2 = w as f64 - coord(v2)[plane_coord];

                                let t = d1 / (d1 - d2);

                                let mut coords = [0.0, 0.0, 0.0];
                                coords[u_coord] = x1 + t * (x2 - x1);
                                coords[v_coord] = y1 + t * (y2 - y1);
                                coords[plane_coord] = w as f64;

                                points.push(coords);
                            }
                        };

                        add_intersection_points(v1, v2);
                        add_intersection_points(v2, v3);
                        add_intersection_points(v3, v1);

                        if points.len() == 2 {
                            let p1 = points[0];
                            let p2 = points[1];

                            // convert to image coordinates
                            let x0 = ((p1[u_coord] - min_u as f64) / paint_zoom as f64) as i32;
                            let y0 = ((p1[v_coord] - min_v as f64) / paint_zoom as f64) as i32;

                            let x1 = ((p2[u_coord] - min_u as f64) / paint_zoom as f64) as i32;
                            let y1 = ((p2[v_coord] - min_v as f64) / paint_zoom as f64) as i32;

                            line(x0, y0, x1, y1, image, width, height, 0xff, 0, 0xff);
                        }
                    }
                }
                _ => todo!(),
            }
        }
    }
}

// align values to paint_zoom to avoid gaps because of integer processing
fn paint_zoom_align(v: i32, paint_zoom: u8) -> i32 {
    let v = v / paint_zoom as i32;
    v * paint_zoom as i32
}
fn paint_zoom_align_up(v: i32, paint_zoom: u8) -> i32 { paint_zoom_align(v + paint_zoom as i32 - 1, paint_zoom) }

#[allow(dead_code)] // useful for debugging triangle shapes
fn line(x0: i32, y0: i32, x1: i32, y1: i32, buffer: &mut Image, width: usize, height: usize, r: u8, g: u8, b: u8) {
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
        if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
            buffer.set_rgb(x as usize, y as usize, r, g, b);
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

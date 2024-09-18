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

        println!(
            "xyz: {:?}, width: {}, height: {}, sfactor: {}, paint_zoom: {}",
            xyz, width, height, sfactor, paint_zoom
        );
        println!("min_u: {}, max_u: {}, min_v: {}, max_v: {}", min_u, max_u, min_v, max_v);

        let obj = &self.obj.object;
        for s in obj.geometry[0].shapes.iter() {
            match s.primitive {
                Primitive::Triangle(i1, i2, i3) => {
                    let v1 = &obj.tex_vertices[i1.1.unwrap()];
                    let v2 = &obj.tex_vertices[i2.1.unwrap()];
                    let v3 = &obj.tex_vertices[i3.1.unwrap()];

                    let u1 = (v1.u * self.width() as f64) as i32;
                    let v1 = (v1.v * self.height() as f64) as i32;

                    let u2 = (v2.u * self.width() as f64) as i32;
                    let v2 = (v2.v * self.height() as f64) as i32;

                    let u3 = (v3.u * self.width() as f64) as i32;
                    let v3 = (v3.v * self.height() as f64) as i32;

                    //println!("u1: {}, v1: {}, u2: {}, v2: {}, u3: {}, v3: {}", u1, v1, u2, v2, u3, v3);

                    if (u1 >= min_u && u1 < max_u && v1 >= min_v && v1 < max_v)
                        && (u2 >= min_u && u2 < max_u && v2 >= min_v && v2 < max_v)
                        && (u3 >= min_u && u3 < max_u && v3 >= min_v && v3 < max_v)
                    {
                        line(
                            (u1 - min_u) as usize,
                            (v1 - min_v) as usize,
                            (u2 - min_u) as usize,
                            (v2 - min_v) as usize,
                            buffer,
                            width,
                        );
                        line(
                            (u2 - min_u) as usize,
                            (v2 - min_v) as usize,
                            (u3 - min_u) as usize,
                            (v3 - min_v) as usize,
                            buffer,
                            width,
                        );
                        line(
                            (u3 - min_u) as usize,
                            (v3 - min_v) as usize,
                            (u1 - min_u) as usize,
                            (v1 - min_v) as usize,
                            buffer,
                            width,
                        );
                    }
                }
                _ => todo!(),
            }
        }

        //line(0, 0, width - 1, height - 1, buffer, width);

        //todo!()
    }
}

fn line(x0: usize, y0: usize, x1: usize, y1: usize, buffer: &mut [u8], width: usize) {
    // simple bresenham algorithm from https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm

    let dx = (x1 as i32 - x0 as i32).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 as i32 - y0 as i32).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut error = dx + dy;

    let mut x = x0 as i32;
    let mut y = y0 as i32;

    while x as usize != x1 || y as usize != y1 {
        //println!("x: {}, y: {}", x, y);
        let idx = y as usize * width + x as usize;
        if idx >= 0 && idx < buffer.len() {
            buffer[idx] = 255;
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

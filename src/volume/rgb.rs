use super::{AutoPaintVolume, PaintVolume, Volume, VolumeGrid64x4Mapped, VoxelPaintVolume, VoxelVolume};
use crate::model::FullVolumeReference;
use egui::Color32;
use libm::modf;

pub struct RGBVolume {
    base_volumes: [Volume; 3],
}

type Transform = [[f64; 4]; 4];

/*
{
    "type": "Transform3D",
    "source": "20231121152933",
    "target": "20231201120546",
    "transform-type": "AffineTransform",
    "params": [
        [
            1.0006160646190843,
            -2.5422595962086987e-05,
            -0.000134181941768234,
            2.1239017560166307
        ],
        [
            -0.000663892422442385,
            1.0038428739804488,
            0.00019622720911857127,
            1.9961987437992526
        ],
        [
            -0.0008938271087464538,
            0.006836708326308265,
            0.9997624437294631,
            -2.7442182541238207
        ],
        [
            0.0,
            0.0,
            0.0,
            1.0
        ]
    ]
}
    33 - 46, i.e. index 0 -> 1
     */

const TRANSFORM_0_1: Transform = [
    [
        1.000302727922068,
        0.0004507424491822119,
        -0.0002537566416235583,
        2.693713874845783,
    ],
    [
        -0.0007774542370093613,
        1.0007174537512895,
        0.0008593689178191039,
        3.626107150975347,
    ],
    [
        0.00021757047910274662,
        -0.0003276378656418654,
        1.0002703890943743,
        -0.7785313524877353,
    ],
    [0.0, 0.0, 0.0, 1.0],
];

/* [
    [
        1.0006160646190843,
        -2.5422595962086987e-05,
        -0.000134181941768234,
        2.1239017560166307,
    ],
    [
        -0.000663892422442385,
        1.0038428739804488,
        0.00019622720911857127,
        1.9961987437992526,
    ],
    [
        -0.0008938271087464538,
        0.006836708326308265,
        0.9997624437294631,
        3.7442182541238207,
    ],
    [0.0, 0.0, 0.0, 1.0],
]; */

/*

 "params": [
        [
            1.0007215501024045,
            0.0013773873122267283,
            -0.000246981145704375,
            -1.6923463396947025
        ],
        [
            -0.0011517996910433237,
            1.0049720453430941,
            0.00033707652112412867,
            1.539705878372157
        ],
        [
            -0.0004210555420767604,
            0.005751308306241877,
            1.00008807708756,
            -6.120943102326436
        ],
        [
            0.0,
            0.0,
            0.0,
            1.0
        ]
    ]

    33 -> 49, i.e. index 0 -> 2

*/

const TRANSFORM_0_2: Transform = [
    [
        1.000302727922068,
        0.00045074244918093514,
        -0.0002537566416235756,
        0.6937138748464312,
    ],
    [
        -0.0013829100811429459,
        0.999815968852928,
        0.0013668822010657995,
        5.238679401274581,
    ],
    [
        -8.515744296359678e-5,
        -0.0007783803148242519,
        1.000524145735997,
        -0.472245227335804,
    ],
    [0.0, 0.0, 0.0, 1.0],
];

/* [
    [
        1.0007215501024045,
        0.0013773873122267283,
        -0.000246981145704375,
        -1.6923463396947025,
    ],
    [
        -0.0011517996910433237,
        1.0049720453430941,
        0.00033707652112412867,
        1.539705878372157,
    ],
    [
        -0.0004210555420767604,
        0.005751308306241877,
        1.00008807708756,
        -6.120943102326436,
    ],
    [0.0, 0.0, 0.0, 1.0],
]; */

fn transform(xyz: [f64; 3], transform: &Transform) -> [f64; 3] {
    let mut result = [0.; 3];
    for i in 0..3 {
        result[i] = transform[i][3];
        for j in 0..3 {
            result[i] += transform[i][j] * xyz[j];
        }
    }
    result
}

impl RGBVolume {
    pub fn new_for_frag6() -> Self {
        RGBVolume {
            base_volumes: [
                &FullVolumeReference::FRAGMENT_PHerc0051Cr04Fr08_3_24_UM_53keV,
                &FullVolumeReference::FRAGMENT_PHerc0051Cr04Fr08_3_24_UM_70keV,
                &FullVolumeReference::FRAGMENT_PHerc0051Cr04Fr08_3_24_UM_88keV,
            ]
            .map(|vol| VolumeGrid64x4Mapped::for_volume(vol).into_volume()),
        }
    }
}

impl VoxelVolume for RGBVolume {
    fn get(&self, xyz: [f64; 3], downsampling: i32) -> u8 {
        self.base_volumes[0].get(xyz, downsampling)
    }
    fn get_color(&self, xyz: [f64; 3], downsampling: i32) -> Color32 {
        let v0 = filter(self.base_volumes[0].get(xyz, downsampling));
        let v1 = filter(self.base_volumes[1].get(transform(xyz, &TRANSFORM_0_1), downsampling));
        let v2 = filter(self.base_volumes[2].get(transform(xyz, &TRANSFORM_0_2), downsampling));

        Color32::from_rgb(v0, v1, v2)
    }
    fn get_color_interpolated(&self, xyz: [f64; 3], downsampling: i32) -> Color32 {
        let v0 = filter(self.base_volumes[0].get_interpolated(xyz, downsampling));
        let v1 = filter(self.base_volumes[1].get_interpolated(transform(xyz, &TRANSFORM_0_1), downsampling));
        let v2 = filter(self.base_volumes[2].get_interpolated(transform(xyz, &TRANSFORM_0_2), downsampling));

        /* let diff = (v0 as i32 - v2 as i32 + 128).abs().clamp(0, 255) as u8;
        Color32::from_gray(diff) */
        Color32::from_rgb(v0, v1, v2)
    }
}

impl PaintVolume for RGBVolume {
    fn paint(
        &self,
        xyz: [i32; 3],
        u_coord: usize,
        v_coord: usize,
        plane_coord: usize,
        width: usize,
        height: usize,
        sfactor: u8,
        paint_zoom: u8,
        config: &super::DrawingConfig,
        buffer: &mut super::Image,
    ) {
        let fi32 = sfactor as f64;

        for im_v in 0..height {
            for im_u in 0..width {
                let im_rel_u = (im_u as i32 - width as i32 / 2) * paint_zoom as i32;
                let im_rel_v = (im_v as i32 - height as i32 / 2) * paint_zoom as i32;

                let mut uvw: [f64; 3] = [0.; 3];
                uvw[u_coord] = (xyz[u_coord] + im_rel_u) as f64 / fi32;
                uvw[v_coord] = (xyz[v_coord] + im_rel_v) as f64 / fi32;
                uvw[plane_coord] = (xyz[plane_coord]) as f64 / fi32;

                let v0 = filter(self.base_volumes[0].get(uvw, sfactor as i32));
                let v1 = filter(self.base_volumes[1].get(transform(uvw, &TRANSFORM_0_1), sfactor as i32));
                let v2 = filter(self.base_volumes[2].get(transform(uvw, &TRANSFORM_0_2), sfactor as i32));

                buffer.set_rgb(im_u, im_v, v0, v1, v2);
                /* let diff = (v0 as i32 - v2 as i32 + 128).abs().clamp(0, 255) as u8;
                buffer.set_gray(im_u, im_v, diff); */
            }
        }
    }
}

fn filter(v: u8) -> u8 {
    if v < 0 {
        0
    } else {
        v
    }
}

/*
fn calculate_transform() {
    let p1_33 = [5490, 914, 6966];
    let p1_46 = [5493, 920, 6968];
    let p1_49 = [5491, 921, 6968];

    let p2_33 = [5360, 1374, 7628];
    let p2_46 = [5363, 1381, 7630];
    let p2_49 = [5361, 1382, 7630];

    let p3_33 = [3784, 829, 839];
    let p3_46 = [3788, 831, 839];
    let p3_49 = [3786, 830, 838];

    let p4_33 = [3709, 665, 4399];
    let p4_46 = [3712, 670, 4400];
    let p4_49 = [3710, 671, 4400];

    // recalculate transform 33 -> 46 by solving Api_33 = pi_46 for each point

    println!("const TRANSFORM_33_46: Transform = {:?};", transform_33_46);
}

*/

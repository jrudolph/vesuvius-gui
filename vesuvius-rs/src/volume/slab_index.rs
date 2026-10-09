//! Per-axis spatial index over mesh primitives (triangles, quads) for slicing
//! a surface with an axis-aligned plane: for every axis, the primitives whose
//! bounding box overlaps each `SLAB`-voxel-thick slab along that axis. A plane
//! query then only visits primitives near the plane instead of the whole mesh.

use rayon::prelude::*;

const SLAB: i32 = 32;

pub(crate) struct SlabIndex {
    axes: [AxisSlabs; 3],
}

/// CSR layout: the primitives of slab `first_slab + i` are
/// `items[starts[i]..starts[i + 1]]`.
struct AxisSlabs {
    first_slab: i32,
    starts: Vec<u32>,
    items: Vec<u32>,
}

impl SlabIndex {
    /// Index primitives by their bounding boxes `[min_x, min_y, min_z, max_x, max_y, max_z]`.
    /// Primitives with a non-finite box are left out.
    pub(crate) fn build(aabbs: &[[f32; 6]]) -> Self {
        assert!(aabbs.len() <= u32::MAX as usize, "too many primitives for SlabIndex");
        let axes: Vec<AxisSlabs> = (0..3).into_par_iter().map(|axis| AxisSlabs::build(aabbs, axis)).collect();
        let [x, y, z]: [AxisSlabs; 3] = axes.try_into().ok().unwrap();
        Self { axes: [x, y, z] }
    }

    /// Indices of the primitives whose bounding box may contain `[axis] == w`
    /// (a superset: the whole slab around `w`).
    pub(crate) fn near_plane(&self, axis: usize, w: i32) -> &[u32] {
        self.axes[axis].slab(slab_of(w as f32))
    }
}

fn slab_of(v: f32) -> i32 {
    (v / SLAB as f32).floor() as i32
}

fn is_valid(bb: &[f32; 6]) -> bool {
    bb.iter().all(|v| v.is_finite())
}

impl AxisSlabs {
    fn build(aabbs: &[[f32; 6]], axis: usize) -> Self {
        let (min, max) = aabbs
            .iter()
            .filter(|bb| is_valid(bb))
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), bb| {
                (lo.min(bb[axis]), hi.max(bb[3 + axis]))
            });
        if min > max {
            return Self {
                first_slab: 0,
                starts: vec![0],
                items: vec![],
            };
        }
        let first_slab = slab_of(min);
        let num_slabs = (slab_of(max) - first_slab + 1) as usize;
        let slab_range = |bb: &[f32; 6]| {
            (slab_of(bb[axis]) - first_slab) as usize..=(slab_of(bb[3 + axis]) - first_slab) as usize
        };

        let mut starts = vec![0u32; num_slabs + 1];
        for bb in aabbs.iter().filter(|bb| is_valid(bb)) {
            for s in slab_range(bb) {
                starts[s + 1] += 1;
            }
        }
        for s in 0..num_slabs {
            starts[s + 1] += starts[s];
        }

        let mut fill = starts.clone();
        let mut items = vec![0u32; starts[num_slabs] as usize];
        for (i, bb) in aabbs.iter().enumerate().filter(|(_, bb)| is_valid(bb)) {
            for s in slab_range(bb) {
                items[fill[s] as usize] = i as u32;
                fill[s] += 1;
            }
        }

        Self {
            first_slab,
            starts,
            items,
        }
    }

    fn slab(&self, slab: i32) -> &[u32] {
        let i = slab - self.first_slab;
        if i < 0 || i as usize + 1 >= self.starts.len() {
            return &[];
        }
        let i = i as usize;
        &self.items[self.starts[i] as usize..self.starts[i + 1] as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_primitives_crossing_plane() {
        let inf = f32::INFINITY;
        let aabbs = [
            [0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
            [0.0, 0.0, 100.0, 10.0, 10.0, 200.0],
            [inf, inf, inf, -inf, -inf, -inf],
            [0.0, 0.0, -50.0, 10.0, 10.0, -40.0],
        ];
        let index = SlabIndex::build(&aabbs);
        assert!(index.near_plane(2, 5).contains(&0));
        assert!(!index.near_plane(2, 5).contains(&1));
        assert!(index.near_plane(2, 150).contains(&1));
        assert!(index.near_plane(2, 200).contains(&1));
        assert!(index.near_plane(2, -45).contains(&3));
        assert!(index.near_plane(2, 1000).is_empty());
        assert!(index.near_plane(2, -1000).is_empty());
        assert!(!index.near_plane(0, 5).contains(&2));
    }
}

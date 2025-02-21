use nalgebra::{DMatrix, Matrix, Matrix3x4, Matrix4, MatrixXx4, Vector3, Vector4};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Transform3D {
    #[serde(rename = "type")]
    transform_type: String,
    source: String,
    target: String,
    #[serde(rename = "transform-type")]
    affine_type: String,
    params: Vec<Vec<f64>>,
}

impl Transform3D {
    pub fn new(source: String, target: String, matrix: Matrix4<f64>) -> Self {
        let params: Vec<Vec<f64>> = (0..4).map(|i| (0..4).map(|j| matrix[(i, j)]).collect()).collect();

        Self {
            transform_type: "Transform3D".to_string(),
            source,
            target,
            affine_type: "AffineTransform".to_string(),
            params,
        }
    }
}

/// Compute 3D affine transformation matrix from control points
pub fn compute_affine_transform(
    source_points: &[Vector3<f64>],
    target_points: &[Vector3<f64>],
) -> Result<Matrix4<f64>, String> {
    if source_points.len() != target_points.len() {
        return Err("Source and target point counts must match".to_string());
    }
    if source_points.len() < 4 {
        return Err("Need at least 4 non-coplanar points".to_string());
    }

    let n_points = source_points.len();
    let mut a = DMatrix::zeros(n_points * 3, 12);
    let mut b = DMatrix::zeros(n_points * 3, 1);

    // Build the system of equations
    for (i, (src, tgt)) in source_points.iter().zip(target_points.iter()).enumerate() {
        // Equation for x coordinate
        a[(i * 3, 0)] = src.x;
        a[(i * 3, 1)] = src.y;
        a[(i * 3, 2)] = src.z;
        a[(i * 3, 3)] = 1.0;
        b[(i * 3, 0)] = tgt.x;

        // Equation for y coordinate
        a[(i * 3 + 1, 4)] = src.x;
        a[(i * 3 + 1, 5)] = src.y;
        a[(i * 3 + 1, 6)] = src.z;
        a[(i * 3 + 1, 7)] = 1.0;
        b[(i * 3 + 1, 0)] = tgt.y;

        // Equation for z coordinate
        a[(i * 3 + 2, 8)] = src.x;
        a[(i * 3 + 2, 9)] = src.y;
        a[(i * 3 + 2, 10)] = src.z;
        a[(i * 3 + 2, 11)] = 1.0;
        b[(i * 3 + 2, 0)] = tgt.z;
    }

    // Solve using pseudo-inverse (least squares)
    let x = match a.svd(true, true).solve(&b, 1e-10) {
        Ok(solution) => solution,
        _ => return Err("Failed to solve the system of equations".to_string()),
    };

    // Construct the 4x4 transformation matrix
    let mut transform = Matrix4::identity();
    for i in 0..3 {
        for j in 0..4 {
            transform[(i, j)] = x[(i * 4 + j, 0)];
        }
    }

    Ok(transform)
}

/// Transform a single point using the affine transformation matrix
pub fn transform_point(matrix: &Matrix4<f64>, point: &Vector3<f64>) -> Vector3<f64> {
    let homogeneous = Vector4::new(point.x, point.y, point.z, 1.0);
    let transformed = matrix * homogeneous;
    Vector3::new(transformed.x, transformed.y, transformed.z)
}

// Example usage
fn main() {
    let p1_33 = [5490.0, 914.0, 6966.0];
    let p1_46 = [5493.0, 920.0, 6968.0];
    let p1_49 = [5491.0, 921.0, 6968.0];

    let p2_33 = [5360.0, 1374.0, 7628.0];
    let p2_46 = [5363.0, 1381.0, 7630.0];
    let p2_49 = [5361.0, 1382.0, 7630.0];

    let p3_33 = [3784.0, 829.0, 839.0];
    let p3_46 = [3788.0, 831.0, 839.0];
    let p3_49 = [3786.0, 830.0, 838.0];

    let p4_33 = [3709.0, 665.0, 4399.0];
    let p4_46 = [3712.0, 670.0, 4400.0];
    let p4_49 = [3710.0, 671.0, 4400.0];

    // p33 points
    let source_points = vec![p1_33.into(), p2_33.into(), p3_33.into(), p4_33.into()];

    // p46 points
    let target_points = vec![p1_49.into(), p2_49.into(), p3_49.into(), p4_49.into()];

    match compute_affine_transform(&source_points, &target_points) {
        Ok(transform) => {
            let transform3d = Transform3D::new("source_frame".to_string(), "target_frame".to_string(), transform);
            println!("Transform: {:#?}", transform3d);

            // Verify a point
            let test_point = Vector3::new(0.5, 0.5, 0.5);
            let transformed = transform_point(&transform, &test_point);
            println!("Test point: {:?} -> Transformed: {:?}", test_point, transformed);
        }
        Err(e) => eprintln!("Error computing transform: {}", e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_affine_transform() {
        // Test points
        let source_points = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(1.0, 1.0, 1.0),
        ];

        let target_points = vec![
            Vector3::new(1.0, 2.0, 1.0),
            Vector3::new(2.0, 2.0, 1.0),
            Vector3::new(1.0, 3.0, 1.0),
            Vector3::new(1.0, 2.0, 2.0),
            Vector3::new(2.0, 3.0, 2.0),
        ];

        let transform = compute_affine_transform(&source_points, &target_points).expect("Failed to compute transform");

        // Verify transformation
        for (src, tgt) in source_points.iter().zip(target_points.iter()) {
            let transformed = transform_point(&transform, src);
            assert_relative_eq!(transformed, tgt, epsilon = 1e-10);
        }
    }
}

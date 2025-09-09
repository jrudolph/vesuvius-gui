use anyhow::*;
use nalgebra::Matrix4;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct AffineTransform {
    /// 3x4 affine transformation matrix in xyz coordinate order
    pub matrix: [[f64; 4]; 3],
}
impl AffineTransform {
    pub fn from_villa_transform_json_file(path: &std::path::Path) -> Result<Self, Error> {
        let json = std::fs::read_to_string(path).context("Failed to read transform file")?;
        Self::from_villa_transform_json(&json)
    }
    pub fn from_villa_transform_json(json: &str) -> Result<Self, Error> {
        let tf: VillaTransformFile = serde_json::from_str(json)?;
        Ok(AffineTransform {
            matrix: tf.transformation_matrix,
        })
    }
    pub fn from_simple_json_array(json: &str) -> Result<Self, Error> {
        let matrix: [[f64; 4]; 3] = serde_json::from_str(json)?;
        Ok(AffineTransform { matrix })
    }
    /// Convenience function to parse either a JSON array or a path to a JSON file, depending on whether the input starts with "[" or not
    pub fn from_json_array_or_path(json_or_path: &str) -> Result<Self, Error> {
        let json_or_path = json_or_path.trim();
        if json_or_path.starts_with("[") {
            Self::from_simple_json_array(json_or_path)
        } else {
            Self::from_villa_transform_json_file(std::path::Path::new(json_or_path))
        }
    }

    /// Invert this affine transformation matrix
    pub fn invert(&self) -> Result<Self, Error> {
        let m = &self.matrix;
        let homogeneous = Matrix4::from([
            [m[0][0], m[0][1], m[0][2], m[0][3]],
            [m[1][0], m[1][1], m[1][2], m[1][3]],
            [m[2][0], m[2][1], m[2][2], m[2][3]],
            [0.0, 0.0, 0.0, 1.0],
        ]);

        let inv = homogeneous
            .try_inverse()
            .ok_or_else(|| anyhow!("Matrix is not invertible"))?;

        let inv_data = inv.data.0;
        Ok(AffineTransform {
            matrix: [
                [inv_data[0][0], inv_data[0][1], inv_data[0][2], inv_data[0][3]],
                [inv_data[1][0], inv_data[1][1], inv_data[1][2], inv_data[1][3]],
                [inv_data[2][0], inv_data[2][1], inv_data[2][2], inv_data[2][3]],
            ],
        })
    }
}

/// Definition of the JSON schema for transform files as used in villa and defined in
/// https://github.com/ScrollPrize/villa/blob/2a0bf2afdc1e16640ec8f4ce3c7f67f87d41fb06/foundation/volume-registration/transform_schema.json
#[derive(Debug, Clone, Deserialize, Serialize)]
struct VillaTransformFile {
    schema_version: String,
    fixed_volume: String,
    transformation_matrix: [[f64; 4]; 3],
    fixed_landmarks: Vec<[f64; 3]>,
    moving_landmarks: Vec<[f64; 3]>,
}

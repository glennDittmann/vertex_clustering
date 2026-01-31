//! WASM bindings for vertex_clustering
//!
//! This module provides JavaScript/TypeScript bindings for the vertex clustering functionality.
//! You can either use the one-shot `cluster2d` / `cluster3d` functions, or instantiate a
//! clusterer and call `simplify()` later (e.g. to maintain the clusterer at runtime).

use crate::utils::{Vertex2, Vertex3};
use crate::{VertexClusterer2, VertexClusterer3};
use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Clusterer classes (instantiate once, call simplify() when needed)
// ---------------------------------------------------------------------------

/// 2D vertex clusterer. Instantiate with vertices and grid size (clustering happens in the
/// constructor). Call `simplify()` for simplified vertices and `compute_clusters()` for cluster data.
#[wasm_bindgen]
pub struct VertexClusterer2D {
    inner: VertexClusterer2,
    min: [f64; 2],
}

#[wasm_bindgen]
impl VertexClusterer2D {
    /// Create a 2D clusterer. Clustering is performed immediately; the clusterer is kept in
    /// memory. Call `simplify()` and/or `compute_clusters()` when needed.
    #[wasm_bindgen(constructor)]
    pub fn new(vertices: &[f64], grid_size: f64) -> Result<VertexClusterer2D, JsValue> {
        validate_2d(vertices, grid_size)?;
        let vertices_2d: Vec<Vertex2> = vertices
            .chunks_exact(2)
            .map(|chunk| [chunk[0], chunk[1]])
            .collect();
        let min = vertices_2d
            .iter()
            .fold([f64::INFINITY, f64::INFINITY], |acc, v| {
                [acc[0].min(v[0]), acc[1].min(v[1])]
            });
        let inner = VertexClusterer2::new(vertices_2d, None, grid_size);
        Ok(VertexClusterer2D { inner, min })
    }

    /// Return an array of cluster objects (id, bounds, vertices). Can be called multiple times.
    #[wasm_bindgen]
    pub fn compute_clusters(&self) -> Result<JsValue, JsValue> {
        build_clusters_2d(&self.inner, &self.min)
    }

    /// Return simplified vertices as a flat array [x1, y1, x2, y2, ...]. Can be called multiple times.
    #[wasm_bindgen]
    pub fn simplify(&self) -> Result<JsValue, JsValue> {
        build_simplified_vertices_2d(&self.inner)
    }
}

/// 3D vertex clusterer. Instantiate with vertices and grid size (clustering happens in the
/// constructor). Call `simplify()` when you need the simplified result.
#[wasm_bindgen]
pub struct VertexClusterer3D {
    inner: VertexClusterer3,
}

#[wasm_bindgen]
impl VertexClusterer3D {
    /// Create a 3D clusterer. Clustering is performed immediately; the clusterer is kept in
    /// memory. Call `simplify()` later to get the simplified vertices.
    #[wasm_bindgen(constructor)]
    pub fn new(vertices: &[f64], grid_size: f64) -> Result<VertexClusterer3D, JsValue> {
        validate_3d(vertices, grid_size)?;
        let vertices_3d: Vec<Vertex3> = vertices
            .chunks_exact(3)
            .map(|chunk| [chunk[0], chunk[1], chunk[2]])
            .collect();
        let inner = VertexClusterer3::new(vertices_3d, None, grid_size);
        Ok(VertexClusterer3D { inner })
    }

    /// Return simplified vertices as a flat array [x1,y1,z1, x2,y2,z2, ...]. Can be called multiple times.
    #[wasm_bindgen]
    pub fn simplify(&self) -> Result<JsValue, JsValue> {
        let (simplified, _weights) = self.inner.simplify();
        let result = js_sys::Object::new();
        let simplified_flat: Vec<f64> =
            simplified.iter().flat_map(|v| [v[0], v[1], v[2]]).collect();
        let arr = js_sys::Array::new();
        for &v in &simplified_flat {
            arr.push(&v.into());
        }
        js_sys::Reflect::set(&result, &"simplified_vertices".into(), &arr)?;
        Ok(result.into())
    }
}

// ---------------------------------------------------------------------------
// One-shot functions (cluster + simplify in a single call)
// ---------------------------------------------------------------------------

/// Cluster vertices in 2D space.
///
/// # Arguments
/// * `vertices` - Flat array of vertex coordinates: [x1, y1, x2, y2, ...]
/// * `grid_size` - Size of each grid cell for clustering
///
/// # Returns
/// A JavaScript object with:
/// - `simplified_vertices`: Array of simplified vertex coordinates [x1, y1, x2, y2, ...]
/// - `clusters`: Array of cluster objects with bounds and vertices
#[wasm_bindgen]
pub fn cluster2d(vertices: &[f64], grid_size: f64) -> Result<JsValue, JsValue> {
    validate_2d(vertices, grid_size)?;
    let vertices_2d: Vec<Vertex2> = vertices
        .chunks_exact(2)
        .map(|chunk| [chunk[0], chunk[1]])
        .collect();
    let min = vertices_2d
        .iter()
        .fold([f64::INFINITY, f64::INFINITY], |acc, v| {
            [acc[0].min(v[0]), acc[1].min(v[1])]
        });
    let clusterer = VertexClusterer2::new(vertices_2d, None, grid_size);
    build_cluster2d_result(&clusterer, &min)
}

/// Cluster vertices in 3D space.
///
/// # Arguments
/// * `vertices` - Flat array of vertex coordinates: [x1, y1, z1, x2, y2, z2, ...]
/// * `grid_size` - Size of each grid cell for clustering
///
/// # Returns
/// A JavaScript object with simplified vertices
#[wasm_bindgen]
pub fn cluster3d(vertices: &[f64], grid_size: f64) -> Result<JsValue, JsValue> {
    validate_3d(vertices, grid_size)?;
    let vertices_3d: Vec<Vertex3> = vertices
        .chunks_exact(3)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect();
    let clusterer = VertexClusterer3::new(vertices_3d, None, grid_size);
    let (simplified, _weights) = clusterer.simplify();
    let result = js_sys::Object::new();
    let simplified_flat: Vec<f64> = simplified.iter().flat_map(|v| [v[0], v[1], v[2]]).collect();
    let arr = js_sys::Array::new();
    for &v in &simplified_flat {
        arr.push(&v.into());
    }
    js_sys::Reflect::set(&result, &"simplified_vertices".into(), &arr)?;
    Ok(result.into())
}

fn validate_2d(vertices: &[f64], grid_size: f64) -> Result<(), JsValue> {
    if vertices.len() % 2 != 0 {
        return Err(JsValue::from_str(
            "Vertices array must have even length (pairs of x, y coordinates)",
        ));
    }
    if grid_size <= 0.0 {
        return Err(JsValue::from_str("Grid size must be positive"));
    }
    if vertices.is_empty() {
        return Err(JsValue::from_str("Vertices array cannot be empty"));
    }
    Ok(())
}

fn validate_3d(vertices: &[f64], grid_size: f64) -> Result<(), JsValue> {
    if vertices.len() % 3 != 0 {
        return Err(JsValue::from_str(
            "Vertices array must have length divisible by 3 (triplets of x, y, z coordinates)",
        ));
    }
    if grid_size <= 0.0 {
        return Err(JsValue::from_str("Grid size must be positive"));
    }
    if vertices.is_empty() {
        return Err(JsValue::from_str("Vertices array cannot be empty"));
    }
    Ok(())
}

/// Returns only the simplified vertices as a flat JS array [x1, y1, x2, y2, ...].
fn build_simplified_vertices_2d(clusterer: &VertexClusterer2) -> Result<JsValue, JsValue> {
    let (simplified, _weights) = clusterer.simplify();
    let flat: Vec<f64> = simplified.iter().flat_map(|v| [v[0], v[1]]).collect();
    let arr = js_sys::Array::new();
    for &v in &flat {
        arr.push(&v.into());
    }
    Ok(arr.into())
}

/// Returns an array of cluster objects (id, bounds, vertices). Includes empty bins for full grid rendering.
fn build_clusters_2d(clusterer: &VertexClusterer2, min: &[f64; 2]) -> Result<JsValue, JsValue> {
    let [min_x, min_y] = *min;
    let grid_size = clusterer.bin_size();
    let clusters_array = js_sys::Array::new();

    for x_idx in 0..clusterer.num_bins_x() {
        for y_idx in 0..clusterer.num_bins_y() {
            let bin_vertices: Vec<f64> = clusterer
                .get_bin(x_idx, y_idx)
                .map(|bin| bin.iter().flat_map(|(v, _)| [v[0], v[1]]).collect())
                .unwrap_or_default();

            let bottom_left = [
                min_x + x_idx as f64 * grid_size,
                min_y + y_idx as f64 * grid_size,
            ];
            let bottom_right = [
                min_x + (x_idx as f64 + 1.0) * grid_size,
                min_y + y_idx as f64 * grid_size,
            ];
            let top_right = [
                min_x + (x_idx as f64 + 1.0) * grid_size,
                min_y + (y_idx as f64 + 1.0) * grid_size,
            ];
            let top_left = [
                min_x + x_idx as f64 * grid_size,
                min_y + (y_idx as f64 + 1.0) * grid_size,
            ];

            let cluster_obj = js_sys::Object::new();
            js_sys::Reflect::set(
                &cluster_obj,
                &"id".into(),
                &format!("cluster_{}_{}", x_idx, y_idx).into(),
            )?;
            js_sys::Reflect::set(
                &cluster_obj,
                &"bounds".into(),
                &create_bounds_object(bottom_left, bottom_right, top_right, top_left)?,
            )?;

            let vertex_array = js_sys::Array::new();
            for chunk in bin_vertices.chunks_exact(2) {
                let v = js_sys::Object::new();
                js_sys::Reflect::set(&v, &"x".into(), &chunk[0].into()).unwrap();
                js_sys::Reflect::set(&v, &"y".into(), &chunk[1].into()).unwrap();
                vertex_array.push(&v);
            }
            js_sys::Reflect::set(&cluster_obj, &"vertices".into(), &vertex_array)?;
            clusters_array.push(&cluster_obj);
        }
    }
    Ok(clusters_array.into())
}

/// One-shot result: both simplified_vertices and clusters (for backward compatibility).
fn build_cluster2d_result(
    clusterer: &VertexClusterer2,
    min: &[f64; 2],
) -> Result<JsValue, JsValue> {
    let result = js_sys::Object::new();
    js_sys::Reflect::set(
        &result,
        &"simplified_vertices".into(),
        &build_simplified_vertices_2d(clusterer)?,
    )?;
    js_sys::Reflect::set(
        &result,
        &"clusters".into(),
        &build_clusters_2d(clusterer, min)?,
    )?;
    Ok(result.into())
}

/// Helper function to create a bounds object for 2D clusters
fn create_bounds_object(
    bottom_left: [f64; 2],
    bottom_right: [f64; 2],
    top_right: [f64; 2],
    top_left: [f64; 2],
) -> Result<JsValue, JsValue> {
    let bounds = js_sys::Object::new();

    let create_vertex = |coords: [f64; 2]| -> JsValue {
        let v = js_sys::Object::new();
        js_sys::Reflect::set(&v, &"x".into(), &coords[0].into()).unwrap();
        js_sys::Reflect::set(&v, &"y".into(), &coords[1].into()).unwrap();
        v.into()
    };

    js_sys::Reflect::set(&bounds, &"bottom_left".into(), &create_vertex(bottom_left))?;
    js_sys::Reflect::set(
        &bounds,
        &"bottom_right".into(),
        &create_vertex(bottom_right),
    )?;
    js_sys::Reflect::set(&bounds, &"top_right".into(), &create_vertex(top_right))?;
    js_sys::Reflect::set(&bounds, &"top_left".into(), &create_vertex(top_left))?;

    Ok(bounds.into())
}

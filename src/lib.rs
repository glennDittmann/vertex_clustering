//! # Vertex Clustering
//!
//! `vertex_clustering` allows to apply two-, three- and four-dimensional grids to sets of vertices for point cloud simplification.

pub use vertex_clusterer_2d::VertexClusterer2;
pub use vertex_clusterer_3d::VertexClusterer3;
pub use vertex_clusterer_4d::VertexClusterer4;

mod utils;
pub mod vertex_clusterer_2d;
pub mod vertex_clusterer_3d;
pub mod vertex_clusterer_4d;

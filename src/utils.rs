use rand::{distributions::Uniform, prelude::Distribution};
use std::ops::RangeInclusive;

pub type Vertex2 = [f64; 2];
pub type Vertex3 = [f64; 3];
pub type Vertex4 = [f64; 4];

pub const EPS: f64 = 1e-6;

/// Returns the next value down that is a multiple of `bin_size` away from `min`.
pub fn next_value_down_relative(value: f64, min: f64, bin_size: f64) -> f64 {
    let relative_value = value - min;
    let floored_relative_value = (relative_value / bin_size).floor() * bin_size;
    min + floored_relative_value
}

/// Returns the next value up that is a multiple of `bin_size` away from `min`.
pub fn next_value_up_relative(value: f64, min: f64, bin_size: f64) -> f64 {
    let relative_value = value - min;
    let ceiled_relative_value = ((relative_value + 0.00001) / bin_size).ceil() * bin_size;
    min + ceiled_relative_value
}

/// Sets values that are really close to zero (within EPSILON) to zero.
pub fn round_to_closest_integer(value: f64) -> f64 {
    let nearest_integer = value.round();
    if (value - nearest_integer).abs() < EPS {
        nearest_integer
    } else {
        value
    }
}

/// Get the min and max coordinates for all `Vertex2` in `vertices`.
pub fn bbox_2d(vertices: &[Vertex2]) -> (Vertex2, Vertex2) {
    let mut min = [f64::INFINITY, f64::INFINITY];
    let mut max = [f64::NEG_INFINITY, f64::NEG_INFINITY];

    for v in vertices.iter() {
        min[0] = min[0].min(v[0]);
        min[1] = min[1].min(v[1]);
        max[0] = max[0].max(v[0]);
        max[1] = max[1].max(v[1]);
    }

    (min, max)
}

/// Get the min and max coordinates for all `Vertex3` in `vertices`.
pub fn bbox_3d(vertices: &[Vertex3]) -> (Vertex3, Vertex3) {
    let mut min = [f64::INFINITY, f64::INFINITY, f64::INFINITY];
    let mut max = [f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];

    for v in vertices.iter() {
        min[0] = min[0].min(v[0]);
        min[1] = min[1].min(v[1]);
        min[2] = min[2].min(v[2]);

        max[0] = max[0].max(v[0]);
        max[1] = max[1].max(v[1]);
        max[2] = max[2].max(v[2]);
    }

    (min, max)
}

/// Get the min and max coordinates for all `Vertex4` in `vertices`.
pub fn bbox_4d(vertices: &[Vertex4]) -> (Vertex4, Vertex4) {
    let mut min = [f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::INFINITY];
    let mut max = [
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];

    for v in vertices.iter() {
        min[0] = min[0].min(v[0]);
        min[1] = min[1].min(v[1]);
        min[2] = min[2].min(v[2]);
        min[3] = min[3].min(v[3]);

        max[0] = max[0].max(v[0]);
        max[1] = max[1].max(v[1]);
        max[2] = max[2].max(v[2]);
        max[3] = max[3].max(v[3]);
    }

    (min, max)
}

/// Samples `n` vertices in 2D space from the [standard] distribution.
///
/// If no range is specified, the unit-square centered around the origin is used, i.e. `[-0.5, 0.5]`.
pub fn sample_vertices_2d(n: usize, range: Option<RangeInclusive<f64>>) -> Vec<Vertex2> {
    let mut rng = rand::thread_rng();
    let range = range.unwrap_or(-0.5..=0.5);
    let uniform = Uniform::from(range);

    let mut vertices: Vec<[f64; 2]> = Vec::with_capacity(n);
    for _ in 0..n {
        let x = uniform.sample(&mut rng);
        let y = uniform.sample(&mut rng);
        vertices.push([x, y]);
    }

    vertices
}

/// Samples `n` vertices in 3D space from the [standard] distribution.
pub fn sample_vertices_3d(n: usize, range: Option<RangeInclusive<f64>>) -> Vec<Vertex3> {
    let mut rng = rand::thread_rng();
    let range = range.unwrap_or(-0.5..=0.5);
    let uniform = Uniform::from(range);

    let mut vertices: Vec<[f64; 3]> = Vec::with_capacity(n);
    for _ in 0..n {
        let x = uniform.sample(&mut rng);
        let y = uniform.sample(&mut rng);
        let z = uniform.sample(&mut rng);

        vertices.push([x, y, z]);
    }

    vertices
}

/// Samples `n` vertices in 3D space from the [standard] distribution.
pub fn sample_vertices_4d(n: usize, range: Option<RangeInclusive<f64>>) -> Vec<Vertex4> {
    let mut rng = rand::thread_rng();
    let range = range.unwrap_or(-0.5..=0.5);
    let uniform = Uniform::from(range);

    let mut vertices: Vec<Vertex4> = Vec::with_capacity(n);
    for _ in 0..n {
        let x = uniform.sample(&mut rng);
        let y = uniform.sample(&mut rng);
        let z = uniform.sample(&mut rng);
        let s = uniform.sample(&mut rng);

        vertices.push([x, y, z, s]);
    }

    vertices
}

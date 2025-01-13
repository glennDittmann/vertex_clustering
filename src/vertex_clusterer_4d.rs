use std::fmt;

use crate::utils::{
    bbox_4d, next_value_down_relative, next_value_up_relative, round_to_closest_integer, Vertex4,
    EPS,
};

type Bin = Vec<(Vertex4, f64)>;

/// A simple `VertexClusterer` that samples a range of values into a number of bins.
#[derive(PartialEq)]
pub struct VertexClusterer4 {
    bin_size: f64,
    bins: Vec<Vec<Vec<Vec<Bin>>>>,
    // the interval starts of the "first" bin, i.e. bottom-left. So this goes from min[0] to min[0] + bin_size (for the y-value repsectively)
    first_bin_interval_start: Vertex4,
    // the interval starts of the "last" bin, i.e. top-right. So this goes from max[0] to max[0] + bin_size (for the y-value repsectively)
    last_bin_interval_start: Vertex4,
    num_bins_x: usize,
    num_bins_y: usize,
    num_bins_z: usize,
    num_bins_w: usize,
    weighted: bool,
}

impl VertexClusterer4 {
    /// Create a `VertexClusterer4` from a list of vertices.
    pub fn new(vertices: Vec<Vertex4>, weights: Option<Vec<f64>>, bin_size: f64) -> Self {
        let ([min_x, min_y, min_z, min_w], [max_x, max_y, max_z, max_w]) = bbox_4d(&vertices);

        let last_bin_x_start = next_value_down_relative(max_x, min_x, bin_size);
        let last_bin_x_end = next_value_up_relative(max_x, min_x, bin_size);

        let last_bin_y_start = next_value_down_relative(max_y, min_y, bin_size);
        let last_bin_y_end = next_value_up_relative(max_y, min_y, bin_size);

        let last_bin_z_start = next_value_down_relative(max_z, min_z, bin_size);
        let last_bin_z_end = next_value_up_relative(max_z, min_z, bin_size);

        let last_bin_w_start = next_value_down_relative(max_w, min_w, bin_size);
        let last_bin_w_end = next_value_up_relative(max_w, min_w, bin_size);

        // There is always at least one bin (in x & y-direction). Compute the number of extra bins
        let mut extra_bins_x = (last_bin_x_end - bin_size - min_x) / bin_size;
        let mut extra_bins_y = (last_bin_y_end - bin_size - min_y) / bin_size;
        let mut extra_bins_z = (last_bin_z_end - bin_size - min_z) / bin_size;
        let mut extra_bins_w = (last_bin_w_end - bin_size - min_w) / bin_size;
        extra_bins_x = round_to_closest_integer(extra_bins_x);
        extra_bins_y = round_to_closest_integer(extra_bins_y);
        extra_bins_z = round_to_closest_integer(extra_bins_z);
        extra_bins_w = round_to_closest_integer(extra_bins_w);

        let num_bins_x = extra_bins_x + 1.0;
        let num_bins_y = extra_bins_y + 1.0;
        let num_bins_z = extra_bins_z + 1.0;
        let num_bins_s = extra_bins_w + 1.0;

        let mut sampler = Self {
            bin_size,
            bins: vec![
                vec![
                    vec![
                        vec![vec![]; num_bins_s.floor() as usize];
                        num_bins_z.floor() as usize
                    ];
                    num_bins_y.floor() as usize
                ];
                num_bins_x.floor() as usize
            ],
            first_bin_interval_start: [min_x, min_y, min_z, min_w],
            last_bin_interval_start: [
                last_bin_x_start,
                last_bin_y_start,
                last_bin_z_start,
                last_bin_w_start,
            ],
            num_bins_x: num_bins_x.floor() as usize,
            num_bins_y: num_bins_y.floor() as usize,
            num_bins_z: num_bins_z.floor() as usize,
            num_bins_w: num_bins_s.floor() as usize,
            weighted: weights.is_some(),
        };

        let weights = weights.unwrap_or_else(|| vec![0.0; vertices.len()]);

        sampler.put_points_in_bins(vertices, weights);

        sampler
    }

    /// Returns the range of the bin at the given indices, as `(start_x, end_x, start_y, end_y, start_z, end_z)`.
    fn bin_range(
        &self,
        x_idx: usize,
        y_idx: usize,
        z_idx: usize,
        s_idx: usize,
    ) -> (f64, f64, f64, f64, f64, f64, f64, f64) {
        let start_x = self.first_bin_interval_start[0] + x_idx as f64 * self.bin_size;
        let start_y = self.first_bin_interval_start[1] + y_idx as f64 * self.bin_size;
        let start_z = self.first_bin_interval_start[2] + z_idx as f64 * self.bin_size;
        let start_s = self.first_bin_interval_start[3] + s_idx as f64 * self.bin_size;

        let end_x = start_x + self.bin_size - f64::EPSILON;
        let end_y = start_y + self.bin_size - f64::EPSILON;
        let end_z = start_z + self.bin_size - f64::EPSILON;
        let end_s = start_s + self.bin_size - f64::EPSILON;

        (
            start_x, end_x, start_y, end_y, start_z, end_z, start_s, end_s,
        )
    }

    /// Get the bin size.
    pub fn bin_size(&self) -> f64 {
        self.bin_size
    }

    /// Get the bin at the given position.
    ///
    /// (0, 0, 0, 0) is the bottom left bin.
    pub fn get_bin(&self, x_idx: usize, y_idx: usize, z_idx: usize, s_idx: usize) -> Option<&Bin> {
        if x_idx < self.num_bins_x && y_idx < self.num_bins_y && z_idx < self.num_bins_z {
            let bin = &self.bins[x_idx][y_idx][z_idx][s_idx];
            if bin.is_empty() {
                return None;
            }
            Some(&self.bins[x_idx][y_idx][z_idx][s_idx])
        } else {
            None
        }
    }

    /// Get the mean of the bin at the given position.
    pub fn get_bin_mean(
        &self,
        x_idx: usize,
        y_idx: usize,
        z_idx: usize,
        s_idx: usize,
    ) -> Option<(Vertex4, f64)> {
        if let Some(bin) = self.get_bin(x_idx, y_idx, z_idx, s_idx) {
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            let mut sum_z = 0.0;
            let mut sum_s = 0.0;
            let mut sum_w = 0.0;

            for ([x, y, z, s], w) in bin {
                sum_x += x;
                sum_y += y;
                sum_z += z;
                sum_s += s;
                sum_w += w;
            }

            return Some((
                [
                    sum_x / bin.len() as f64,
                    sum_y / bin.len() as f64,
                    sum_z / bin.len() as f64,
                    sum_s / bin.len() as f64,
                ],
                sum_w / bin.len() as f64,
            ));
        }
        None
    }

    /// Map a point to its corresponding bin.
    fn put_point_in_bin(&mut self, vertex: Vertex4, weight: f64) {
        // 1) Compute the correct indices
        let next_down_x =
            next_value_down_relative(vertex[0], self.first_bin_interval_start[0], self.bin_size);
        let next_down_y =
            next_value_down_relative(vertex[1], self.first_bin_interval_start[1], self.bin_size);
        let next_down_z =
            next_value_down_relative(vertex[2], self.first_bin_interval_start[2], self.bin_size);
        let next_down_s =
            next_value_down_relative(vertex[3], self.first_bin_interval_start[3], self.bin_size);

        let mut x_idx = (next_down_x - self.first_bin_interval_start[0]) / self.bin_size;
        let mut y_idx = (next_down_y - self.first_bin_interval_start[1]) / self.bin_size;
        let mut z_idx = (next_down_z - self.first_bin_interval_start[2]) / self.bin_size;
        let mut s_idx = (next_down_s - self.first_bin_interval_start[3]) / self.bin_size;

        // 2) Sanitize x_idx and y_idx:
        // with small numbers for value and bin size, we can get floating point errors
        // e.g. the x_idx might be slightly smaller, such that x_idx = 2.9999999999999998, where it should be 3.0
        // hence we need to correct it
        // When the value is slightly to big, e.g. x_idx = 3.0000000000000002, the "as usize" later takes care of that
        if (x_idx.ceil() - x_idx).abs() < 1e-6 {
            x_idx = x_idx.ceil();
        }
        if (y_idx.ceil() - y_idx).abs() < 1e-6 {
            y_idx = y_idx.ceil();
        }
        if (z_idx.ceil() - z_idx).abs() < 1e-6 {
            z_idx = z_idx.ceil();
        }
        if (s_idx.ceil() - s_idx).abs() < 1e-6 {
            s_idx = s_idx.ceil();
        }

        // 3) Assert the computed indices are for the correct bin
        let (start_x, end_x, start_y, end_y, start_z, end_z, start_s, end_s) = self.bin_range(
            x_idx as usize,
            y_idx as usize,
            z_idx as usize,
            s_idx as usize,
        );

        assert!((vertex[0] - start_x).abs() < EPS || vertex[0] >= start_x);
        assert!((end_x - vertex[0]).abs() < EPS || vertex[0] < end_x);

        assert!((vertex[1] - start_y).abs() < EPS || vertex[1] >= start_y);
        assert!((end_y - vertex[1]).abs() < EPS || vertex[1] < end_y);

        assert!((vertex[2] - start_z).abs() < EPS || vertex[2] >= start_z);
        assert!((end_z - vertex[2]).abs() < EPS || vertex[2] < end_z);

        assert!((vertex[3] - start_s).abs() < EPS || vertex[3] >= start_s);
        assert!((end_s - vertex[3]).abs() < EPS || vertex[3] < end_s);

        self.bins[x_idx as usize][y_idx as usize][z_idx as usize][s_idx as usize]
            .push((vertex, weight));
    }

    /// Map points to their corresponding bins.
    fn put_points_in_bins(&mut self, vertices: Vec<Vertex4>, weights: Vec<f64>) {
        for (idx, vertex) in vertices.iter().enumerate() {
            self.put_point_in_bin(*vertex, weights[idx]);
        }
    }

    /// Get the number of bins.
    pub fn num_bins(&self) -> usize {
        self.num_bins_x * self.num_bins_y * self.num_bins_z * self.num_bins_w
    }

    /// Get the number of bins in the x-direction.
    pub fn num_bins_x(&self) -> usize {
        self.num_bins_x
    }

    /// Get the number of bins in the y-direction.
    pub fn num_bins_y(&self) -> usize {
        self.num_bins_y
    }

    /// Get the number of bins in the z-direction.
    pub fn num_bins_z(&self) -> usize {
        self.num_bins_z
    }

    /// Get the number of bins in the w-direction.
    pub fn num_bins_w(&self) -> usize {
        self.num_bins_w
    }

    /// Simplifies the clustered point cloud, i.e. returns the mean of each bin.
    pub fn simplify(&self) -> (Vec<Vertex4>, Vec<f64>) {
        let mut simplified_vertices = Vec::new();
        let mut simplified_weights = Vec::new();

        for x_idx in 0..self.num_bins_x {
            for y_idx in 0..self.num_bins_y {
                for z_idx in 0..self.num_bins_z {
                    for s_idx in 0..self.num_bins_w {
                        if let Some((mean_vertex, mean_weight)) =
                            self.get_bin_mean(x_idx, y_idx, z_idx, s_idx)
                        {
                            simplified_vertices.push(mean_vertex);
                            simplified_weights.push(mean_weight);
                        }
                    }
                }
            }
        }
        (simplified_vertices, simplified_weights)
    }

    /// Get all vertices in the bins.
    pub fn vertices(&self) -> Vec<&(Vertex4, f64)> {
        self.bins
            .iter()
            .flat_map(|bin| bin.iter())
            .flat_map(|bin| bin.iter())
            .flat_map(|bin| bin.iter())
            .flat_map(|bin| bin.iter())
            .collect()
    }
}

impl fmt::Display for VertexClusterer4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for x_idx in 0..self.num_bins_x {
            for y_idx in 0..self.num_bins_y {
                for z_idx in 0..self.num_bins_z {
                    for s_idx in 0..self.num_bins_w {
                        writeln!(
                            f,
                            "Bin ({}, {}, {}, {}): {:?}",
                            x_idx,
                            y_idx,
                            z_idx,
                            s_idx,
                            self.get_bin(x_idx, y_idx, z_idx, s_idx).unwrap()
                        )?;
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use rand::{distributions::Uniform, prelude::Distribution};

    use super::*;

    fn validate_sampler(sampler: &VertexClusterer4) {
        // env_logger::init();
        let mut num_validated = 0;
        for x_idx in 0..sampler.num_bins_x {
            for y_idx in 0..sampler.num_bins_y {
                for z_idx in 0..sampler.num_bins_z {
                    for s_idx in 0..sampler.num_bins_w {
                        if let Some(bin) = sampler.get_bin(x_idx, y_idx, z_idx, s_idx) {
                            let (start_x, end_x, start_y, end_y, start_z, end_z, start_s, end_s) =
                                sampler.bin_range(x_idx, y_idx, z_idx, s_idx);

                            for ([x, y, z, s], _) in bin {
                                // x-component is in between the start and end of the bin
                                assert!(*x >= start_x);
                                assert!(*x < end_x);
                                // y-component..
                                assert!(*y >= start_y);
                                assert!(*y < end_y);
                                // z-component..
                                assert!(*z >= start_z);
                                assert!(*z < end_z);
                                // s-component..
                                assert!(*s >= start_s);
                                assert!(*s < end_s);
                            }
                        }
                        num_validated += 1;
                    }
                }
            }
        }
        assert_eq!(num_validated, sampler.num_bins());
    }

    #[test]
    fn test_vertex_clusterer_4d() {
        let mut rng = rand::thread_rng();
        let uniform = Uniform::from(-0.5..=0.5);

        let mut vertices: Vec<Vertex4> = Vec::with_capacity(1000);
        for _ in 0..1000 {
            let x = uniform.sample(&mut rng);
            let y = uniform.sample(&mut rng);
            let z = uniform.sample(&mut rng);
            let s = uniform.sample(&mut rng);

            vertices.push([x, y, z, s]);
        }

        let sampler = VertexClusterer4::new(vertices, None, 0.1);
        validate_sampler(&sampler);
    }
}

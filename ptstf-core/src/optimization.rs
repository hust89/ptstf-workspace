use crate::flow::TriadicFlowCurve;
use nalgebra::Vector2;
use rayon::prelude::*;

#[inline]
fn sigmoid(z: f64) -> f64 {
    if z >= 40.0 {
        1.0
    } else if z <= -40.0 {
        0.0
    } else {
        1.0 / (1.0 + (-z).exp())
    }
}

/// Evaluates the Smooth Crossing Probability Field P_cross(gamma_r) using continuous parameter steps [2].
pub fn smooth_crossing_probability(curve: &TriadicFlowCurve, steps: &[f64], tau: f64) -> f64 {
    let mut non_crossing_prod = 1.0;
    let sign_p0 = curve.p0.x.signum();
    for &s in steps {
        let pt_x = curve.evaluate(s).x;
        let val = -tau * pt_x * sign_p0;
        non_crossing_prod *= 1.0 - sigmoid(val);
    }
    1.0 - non_crossing_prod
}

/// Pre-indexed inverted index mapping semantic nodes to incident triadic curves.
/// Enables O(deg(i)) local gradient evaluations instead of O(M) global scans per node.
#[derive(Clone, Debug, Default)]
pub struct IncidentCurveIndex {
    /// For each node i: a list of (curve_index, alpha)
    pub node_to_curves: Vec<Vec<(usize, f64)>>,
}

impl IncidentCurveIndex {
    /// Builds an inverted index mapping each node to the unique curves in which it participates.
    pub fn build(
        num_nodes: usize,
        curve_indices: &[(usize, usize, usize)],
        alphas: &[f64],
    ) -> Self {
        let mut node_to_curves = vec![Vec::new(); num_nodes];
        for (curve_idx, &(i, j, k)) in curve_indices.iter().enumerate() {
            let alpha = alphas.get(curve_idx).copied().unwrap_or(1.0);
            if i < num_nodes {
                node_to_curves[i].push((curve_idx, alpha));
            }
            if j < num_nodes && j != i {
                node_to_curves[j].push((curve_idx, alpha));
            }
            if k < num_nodes && k != i && k != j {
                node_to_curves[k].push((curve_idx, alpha));
            }
        }
        Self { node_to_curves }
    }
}

pub struct GradientFlowOptimizer {
    pub learning_rate: f64,
    pub mu: f64,
    pub tau: f64,
}

impl GradientFlowOptimizer {
    pub fn new(learning_rate: f64, mu: f64, tau: f64) -> Self {
        Self {
            learning_rate,
            mu,
            tau,
        }
    }

    /// Step updates coordinates y_i via representation gradient ascent (Theorem 9.1) [2].
    /// Automatically builds an incident-curve inverted index for O(deg(i)) local gradient evaluation.
    pub fn step_representation_flow(
        &self,
        y: &mut [Vector2<f64>],
        curve_indices: &[(usize, usize, usize)],
        alphas: &[f64],
        a_param: f64,
        b_param: f64,
    ) {
        let index = IncidentCurveIndex::build(y.len(), curve_indices, alphas);
        self.step_representation_flow_with_index(y, curve_indices, &index, a_param, b_param);
    }

    /// High-performance step update reusing a precomputed IncidentCurveIndex across multiple epochs.
    /// Completely eliminates O(N) heap reallocations and scales linearly with node incidence using Rayon.
    pub fn step_representation_flow_with_index(
        &self,
        y: &mut [Vector2<f64>],
        curve_indices: &[(usize, usize, usize)],
        index: &IncidentCurveIndex,
        a_param: f64,
        b_param: f64,
    ) {
        let steps = [0.0, 0.25, 0.5, 0.75, 1.0];
        let num_nodes = y.len();
        const EPSILON: f64 = 1e-6;

        let gradients: Vec<Vector2<f64>> = (0..num_nodes)
            .into_par_iter()
            .map(|i| {
                let mut grad = Vector2::new(0.0, 0.0);
                let incident = if i < index.node_to_curves.len() {
                    &index.node_to_curves[i]
                } else {
                    return grad;
                };

                for dim in 0..2 {
                    let mut grad_dim = 0.0;
                    for &sign in &[-1.0, 1.0] {
                        let perturbed_coord = Vector2::new(
                            if dim == 0 { y[i].x + sign * EPSILON } else { y[i].x },
                            if dim == 1 { y[i].y + sign * EPSILON } else { y[i].y },
                        );

                        let mut loss = 0.0;
                        for &(curve_idx, alpha) in incident {
                            let (idx0, idx1, idx2) = curve_indices[curve_idx];
                            let p0 = if idx0 == i { perturbed_coord } else { y[idx0] };
                            let p1 = if idx1 == i { perturbed_coord } else { y[idx1] };
                            let p2 = if idx2 == i { perturbed_coord } else { y[idx2] };

                            let perturbed_curve = TriadicFlowCurve::new(p0, p1, p2);
                            let energy = perturbed_curve.bending_energy(a_param, b_param);
                            let p_cross =
                                smooth_crossing_probability(&perturbed_curve, &steps, self.tau);
                            loss -= self.mu * alpha * energy * p_cross;
                        }
                        grad_dim += sign * loss;
                    }
                    grad[dim] = grad_dim / (2.0 * EPSILON);
                }
                grad
            })
            .collect();

        // Apply update step
        for (coord, grad) in y.iter_mut().zip(gradients) {
            *coord += self.learning_rate * grad;
        }
    }
}

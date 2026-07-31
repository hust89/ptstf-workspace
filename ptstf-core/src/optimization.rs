use nalgebra::Vector2;
use crate::flow::TriadicFlowCurve;

#[inline]
fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
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

pub struct GradientFlowOptimizer {
    pub learning_rate: f64,
    pub mu: f64,
    pub tau: f64,
}

impl GradientFlowOptimizer {
    pub fn new(learning_rate: f64, mu: f64, tau: f64) -> Self {
        Self { learning_rate, mu, tau }
    }

    /// Step updates coordinates y_i via representation gradient ascent (Theorem 9.1) [2].
    /// Dynamically reconstructs perturbed curves using coordinate index mapping to ensure correct gradients.
    pub fn step_representation_flow(
        &self,
        y: &mut [Vector2<f64>],
        curve_indices: &[(usize, usize, usize)], // Replaced static curves with index linkage!
        alphas: &[f64],
        a_param: f64,
        b_param: f64,
    ) {
        let steps = [0.0, 0.25, 0.5, 0.75, 1.0];
        let num_nodes = y.len();
        let mut gradients = vec![Vector2::new(0.0, 0.0); num_nodes];
        const EPSILON: f64 = 1e-6;

        for i in 0..num_nodes {
            for dim in 0..2 {
                let mut grad_dim = 0.0;
                for &sign in &[-1.0, 1.0] {
                    let mut temp_y = y.to_vec();
                    temp_y[i][dim] += sign * EPSILON;

                    // Dynamically reconstruct each curve with the perturbed coordinates
                    let mut total_loss = 0.0;
                    for (idx, &indices) in curve_indices.iter().enumerate() {
                        let perturbed_curve = TriadicFlowCurve::new(
                            temp_y[indices.0],
                            temp_y[indices.1],
                            temp_y[indices.2],
                        );
                        let energy = perturbed_curve.bending_energy(a_param, b_param);
                        let p_cross = smooth_crossing_probability(&perturbed_curve, &steps, self.tau);
                        total_loss -= self.mu * alphas[idx] * energy * p_cross;
                    }
                    grad_dim += sign * total_loss;
                }
                gradients[i][dim] = grad_dim / (2.0 * EPSILON);
            }
        }

        // Apply update step
        for i in 0..num_nodes {
            y[i] += self.learning_rate * gradients[i];
        }
    }
}

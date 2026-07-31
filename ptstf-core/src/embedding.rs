use nalgebra::Vector2;
use crate::PtstfError;

#[derive(Clone)]
pub struct ConductanceField {
    pub embeddings: Vec<Vector2<f64>>,
    pub discrete_conductances: Vec<f64>,
    pub eta: f64,
}

impl ConductanceField {
    pub fn new(embeddings: Vec<Vector2<f64>>, discrete_conductances: Vec<f64>, eta: f64) -> Result<Self, PtstfError> {
        if embeddings.is_empty() {
            return Err(PtstfError::EmptyDataset);
        }
        if embeddings.len() != discrete_conductances.len() {
            return Err(PtstfError::DimensionMismatch { expected: embeddings.len(), found: discrete_conductances.len() });
        }
        if eta <= 0.0 {
            return Err(PtstfError::InvalidParameter("eta must be > 0".to_string()));
        }
        Ok(Self { embeddings, discrete_conductances, eta })
    }

    /// Evaluates the continuous 2D spatial field h(x) via Gaussian kernel density estimation [2].
    pub fn evaluate(&self, x: Vector2<f64>) -> f64 {
        let eta_factor = -2.0 * self.eta * self.eta;
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for (phi, &h_discrete) in self.embeddings.iter().zip(&self.discrete_conductances) {
            let dist_sq = (x - phi).norm_squared();
            let weight = (eta_factor * dist_sq).exp();
            numerator += weight * h_discrete;
            denominator += weight;
        }
        if denominator.abs() < 1e-15 { 0.0 } else { numerator / denominator }
    }
}

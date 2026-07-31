use nalgebra::{DMatrix, Vector2};
use crate::PtstfError;

pub struct TransitionTensor {
    pub data: Vec<f64>,
    pub m: usize,
}

pub struct NcpFactors {
    pub u1: DMatrix<f64>, // M x R
    pub u2: DMatrix<f64>, // M x R
    pub u3: DMatrix<f64>, // M x R
    pub alpha: Vec<f64>,  // R
    pub rank: usize,
}

impl TransitionTensor {
    pub fn new(data: Vec<f64>, m: usize) -> Result<Self, PtstfError> {
        if data.len() != m * m * m {
            return Err(PtstfError::DimensionMismatch { expected: m * m * m, found: data.len() });
        }
        Ok(Self { data, m })
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize, k: usize) -> f64 {
        self.data[i + j * self.m + k * self.m * self.m]
    }

    /// Performs Non-negative Canonical Polyadic (NCP) Decomposition via multiplicative updates.
    pub fn decompose(&self, rank: usize, max_iter: usize) -> Result<NcpFactors, PtstfError> {
        let m = self.m;
        let mut u1 = DMatrix::from_element(m, rank, 0.5);
        let mut u2 = DMatrix::from_element(m, rank, 0.5);
        let mut u3 = DMatrix::from_element(m, rank, 0.5);
        let mut alpha = vec![1.0; rank];
        const EPSILON: f64 = 1e-9;

        for _iter in 0..max_iter {
            // Update Factor U1
            for r in 0..rank {
                for i in 0..m {
                    let mut num = 0.0;
                    let mut den = 0.0;
                    for j in 0..m {
                        for k in 0..m {
                            let x_val = self.get(i, j, k);
                            let recon = (0..rank)
                                .map(|rr| alpha[rr] * u1[(i, rr)] * u2[(j, rr)] * u3[(k, rr)])
                                .sum::<f64>() + EPSILON;
                            let factor_product = u2[(j, r)] * u3[(k, r)];
                            num += x_val * factor_product;
                            den += recon * factor_product;
                        }
                    }
                    u1[(i, r)] *= (num / (den + EPSILON)).sqrt();
                }
            }

            // Update Factor U2
            for r in 0..rank {
                for j in 0..m {
                    let mut num = 0.0;
                    let mut den = 0.0;
                    for i in 0..m {
                        for k in 0..m {
                            let x_val = self.get(i, j, k);
                            let recon = (0..rank)
                                .map(|rr| alpha[rr] * u1[(i, rr)] * u2[(j, rr)] * u3[(k, rr)])
                                .sum::<f64>() + EPSILON;
                            let factor_product = u1[(i, r)] * u3[(k, r)];
                            num += x_val * factor_product;
                            den += recon * factor_product;
                        }
                    }
                    u2[(j, r)] *= (num / (den + EPSILON)).sqrt();
                }
            }

            // Update Factor U3
            for r in 0..rank {
                for k in 0..m {
                    let mut num = 0.0;
                    let mut den = 0.0;
                    for i in 0..m {
                        for j in 0..m {
                            let x_val = self.get(i, j, k);
                            let recon = (0..rank)
                                .map(|rr| alpha[rr] * u1[(i, rr)] * u2[(j, rr)] * u3[(k, rr)])
                                .sum::<f64>() + EPSILON;
                            let factor_product = u1[(i, r)] * u2[(j, r)];
                            num += x_val * factor_product;
                            den += recon * factor_product;
                        }
                    }
                    u3[(k, r)] *= (num / (den + EPSILON)).sqrt();
                }
            }

            // Factor Normalization and Alpha Update
            for r in 0..rank {
                let sum1: f64 = u1.column(r).sum();
                let sum2: f64 = u2.column(r).sum();
                let sum3: f64 = u3.column(r).sum();
                
                if sum1 > 0.0 && sum2 > 0.0 && sum3 > 0.0 {
                    alpha[r] = sum1 * sum2 * sum3;
                    for i in 0..m { u1[(i, r)] /= sum1; }
                    for j in 0..m { u2[(j, r)] /= sum2; }
                    for k in 0..m { u3[(k, r)] /= sum3; }
                }
            }
        }

        Ok(NcpFactors { u1, u2, u3, alpha, rank })
    }
}

impl NcpFactors {
    /// Evaluates the stable control centroids. Denominators are guaranteed strictly positive (> 0) [2].
    pub fn compute_stable_centroid(&self, mode: usize, factor_idx: usize, embeddings: &[Vector2<f64>]) -> Result<Vector2<f64>, PtstfError> {
        let factor_matrix = match mode {
            1 => &self.u1,
            2 => &self.u2,
            3 => &self.u3,
            _ => return Err(PtstfError::InvalidParameter("Mode must be 1, 2, or 3".to_string())),
        };
        let mut numerator = Vector2::new(0.0, 0.0);
        let mut denominator = 0.0;
        for i in 0..factor_matrix.nrows() {
            let weight = factor_matrix[(i, factor_idx)];
            numerator += weight * embeddings[i];
            denominator += weight;
        }
        if denominator <= 0.0 {
            return Err(PtstfError::InvalidParameter("NCP factor column sum is zero".to_string()));
        }
        Ok(numerator / denominator)
    }
}

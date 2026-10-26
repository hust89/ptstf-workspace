use crate::PtstfError;
use nalgebra::DMatrix;

pub struct TrigramGraph {
    pub adjacency: DMatrix<f64>,
    pub num_nodes: usize,
}

impl TrigramGraph {
    pub fn new(adjacency: DMatrix<f64>) -> Result<Self, PtstfError> {
        if !adjacency.is_square() {
            return Err(PtstfError::InvalidParameter(
                "Adjacency matrix must be square".to_string(),
            ));
        }
        let num_nodes = adjacency.nrows();
        Ok(Self {
            adjacency,
            num_nodes,
        })
    }

    /// Computes the Symmetric Normalized Graph Laplacian: L_sym = I - D^(-1/2) * A * D^(-1/2)
    pub fn compute_laplacian(&self) -> DMatrix<f64> {
        let n = self.num_nodes;
        let mut d_inv_sqrt = DMatrix::zeros(n, n);
        for i in 0..n {
            let row_sum: f64 = self.adjacency.row(i).sum();
            if row_sum > 0.0 {
                d_inv_sqrt[(i, i)] = 1.0 / row_sum.sqrt();
            }
        }
        let identity = DMatrix::identity(n, n);
        let normalized_adj = &d_inv_sqrt * &self.adjacency * &d_inv_sqrt;
        identity - normalized_adj
    }
}

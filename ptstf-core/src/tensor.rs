use crate::PtstfError;
use std::collections::HashMap;

/// Sparse Triplet representation for 3D transition probability entries
#[derive(Debug, Clone, PartialEq)]
pub struct SparseEntry {
    pub i: usize,
    pub j: usize,
    pub k: usize,
    pub val: f64,
}

/// Sparse Coordinate (COO) / HashMap 3D Transition Tensor P(w_k | w_i, w_j)
///
/// Backed by a HashMap keyed on (i, j, k). Memory is O(nnz) rather than V^3,
/// so this scales past 10k words without dense allocation.
#[derive(Debug, Clone, Default)]
pub struct SparseTensor {
    entries: HashMap<(usize, usize, usize), f64>,
    /// Row sums over k, keyed on packed (i * dim + j). Kept for fast P(k|i,j).
    row_sums: HashMap<usize, f64>,
    pub dim: usize,
}

impl SparseTensor {
    pub fn new(dim: usize) -> Self {
        Self {
            entries: HashMap::new(),
            row_sums: HashMap::new(),
            dim,
        }
    }

    #[inline]
    fn ctx_key(&self, i: usize, j: usize) -> usize {
        debug_assert!(i < self.dim && j < self.dim);
        i.wrapping_mul(self.dim).wrapping_add(j)
    }

    /// Insert (or overwrite) a raw count for trigram (i, j, k).
    pub fn insert_count(&mut self, i: usize, j: usize, k: usize, count: f64) {
        debug_assert!(i < self.dim && j < self.dim && k < self.dim);
        let prev = self.entries.insert((i, j, k), count).unwrap_or(0.0);
        *self.row_sums.entry(self.ctx_key(i, j)).or_insert(0.0) += count - prev;
    }

    /// Accumulate into a raw count (streaming ingest).
    pub fn add_count(&mut self, i: usize, j: usize, k: usize, delta: f64) {
        debug_assert!(i < self.dim && j < self.dim && k < self.dim);
        *self.entries.entry((i, j, k)).or_insert(0.0) += delta;
        *self.row_sums.entry(self.ctx_key(i, j)).or_insert(0.0) += delta;
    }

    #[inline]
    pub fn count(&self, i: usize, j: usize, k: usize) -> f64 {
        *self.entries.get(&(i, j, k)).unwrap_or(&0.0)
    }

    #[inline]
    pub fn row_sum(&self, i: usize, j: usize) -> f64 {
        *self.row_sums.get(&self.ctx_key(i, j)).unwrap_or(&0.0)
    }

    /// Normalized transition probability P(k | i, j) with uniform fallback.
    #[inline]
    pub fn prob(&self, i: usize, j: usize, k: usize) -> f64 {
        let s = self.row_sum(i, j);
        if s <= 0.0 {
            1.0 / self.dim as f64
        } else {
            self.count(i, j, k) / s
        }
    }

    /// Iterate over all stored nonzero entries as (&(i,j,k), &count).
    pub fn iter(&self) -> impl Iterator<Item = (&(usize, usize, usize), &f64)> {
        self.entries.iter()
    }

    pub fn nnz(&self) -> usize {
        self.entries.len()
    }

    /// Recompute row sums from scratch. Prefer this after bulk ingestion
    /// instead of maintaining row_sums incrementally on every add_count.
    pub fn rebuild_row_sums(&mut self) {
        self.row_sums.clear();
        for (&(i, j, _k), &v) in self.entries.iter() {
            *self.row_sums.entry(self.ctx_key(i, j)).or_insert(0.0) += v;
        }
    }

    /// Materialize as a Vec of SparseEntry, sorted by (i, j, k).
    pub fn to_triplets(&self) -> Vec<SparseEntry> {
        let mut v: Vec<SparseEntry> = self
            .entries
            .iter()
            .map(|(&(i, j, k), &val)| SparseEntry { i, j, k, val })
            .collect();
        v.sort_by(|a, b| (a.i, a.j, a.k).cmp(&(b.i, b.j, b.k)));
        v
    }

    /// Build from triplets (counts). Overwrites existing entries.
    pub fn from_triplets(dim: usize, trips: &[SparseEntry]) -> Self {
        let mut t = Self::new(dim);
        for e in trips {
            t.insert_count(e.i, e.j, e.k, e.val);
        }
        t
    }

    /// Iterate nonzeros grouped by context (i, j). Yields runs of k-values
    /// sharing the same (i, j), so callers can compute per-context sums
    /// exactly once instead of per-nonzero.
    ///
    /// The key list is sorted once into a scratch Vec (O(nnz log nnz)), then
    /// walked in order. No HashMap operations inside the consumer's hot loop
    /// beyond the O(1) get() used to resolve references.
    pub fn iter_grouped_by_context(&self) -> GroupedIter<'_> {
        let mut keys: Vec<(usize, usize, usize)> = self.entries.keys().copied().collect();
        keys.sort_unstable_by(|a, b| (a.0, a.1, a.2).cmp(&(b.0, b.1, b.2)));
        GroupedIter {
            tensor: self,
            keys,
            pos: 0,
        }
    }

    /// Convert to CSF (Compressed Sparse Fiber) ordering: i -> j -> k.
    ///
    /// Returns a proper CSF with prefix-sum fiber offsets so that `fiber(i, j)`
    /// is O(1). Correctness-only export format; hot path stays on the HashMap.
    pub fn to_csf(&self) -> CsfTensor {
        let dim = self.dim;

        // Group by (i, j) with sorted k within each fiber.
        let mut by_i: Vec<Vec<(usize, usize, f64)>> = vec![Vec::new(); dim];
        for (&(i, j, k), &v) in self.entries.iter() {
            by_i[i].push((j, k, v));
        }
        for row in by_i.iter_mut() {
            row.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
        }

        let mut ptr_i: Vec<usize> = Vec::with_capacity(dim + 1);
        let mut fiber_j: Vec<usize> = Vec::new();
        let mut fiber_off: Vec<usize> = Vec::new(); // start offset of each fiber in idx_k/vals
        let mut idx_k: Vec<usize> = Vec::new();
        let mut vals: Vec<f64> = Vec::new();

        ptr_i.push(0usize); // fiber count before i = 0
        for i in 0..dim {
            let row = &by_i[i];
            let mut start = 0usize;
            while start < row.len() {
                let j = row[start].0;
                let mut end = start;
                while end < row.len() && row[end].0 == j {
                    end += 1;
                }
                fiber_j.push(j);
                fiber_off.push(idx_k.len());
                for t in start..end {
                    idx_k.push(row[t].1);
                    vals.push(row[t].2);
                }
                start = end;
            }
            ptr_i.push(fiber_j.len());
        }
        // Sentinel so fiber(i, last_j) can compute len without special-casing.
        fiber_off.push(idx_k.len());

        CsfTensor {
            ptr_i,
            fiber_j,
            fiber_off,
            idx_k,
            vals,
            dim,
        }
    }
}

pub struct GroupedIter<'a> {
    tensor: &'a SparseTensor,
    keys: Vec<(usize, usize, usize)>,
    pos: usize,
}

impl<'a> Iterator for GroupedIter<'a> {
    type Item = (&'a (usize, usize, usize), &'a f64);
    fn next(&mut self) -> Option<Self::Item> {
        let key = *self.keys.get(self.pos)?;
        self.pos += 1;
        // O(1) lookup returning references with the map's lifetime.
        self.tensor.entries.get_key_value(&key)
    }
}

/// CSF view produced by `SparseTensor::to_csf`. Safe, O(1) fiber access.
#[derive(Debug, Clone)]
pub struct CsfTensor {
    /// Fiber boundaries by i: ptr_i[i]..ptr_i[i+1] indexes fibers for row i.
    pub ptr_i: Vec<usize>,
    /// j-index of each (i, j) fiber. len = #fibers.
    pub fiber_j: Vec<usize>,
    /// Start offset of each fiber in idx_k/vals. len = #fibers + 1.
    pub fiber_off: Vec<usize>,
    pub idx_k: Vec<usize>,
    pub vals: Vec<f64>,
    pub dim: usize,
}

impl CsfTensor {
    /// Nonzeros in fiber (i, j) as (k, value). O(1) locate + slice iterate.
    pub fn fiber(&self, i: usize, j: usize) -> Box<dyn Iterator<Item = (usize, f64)> + '_> {
        let mut found: Option<(usize, usize)> = None;
        for f in self.ptr_i[i]..self.ptr_i[i + 1] {
            if self.fiber_j[f] == j {
                let lo = self.fiber_off[f];
                let hi = self.fiber_off[f + 1];
                found = Some((lo, hi));
                break;
            }
        }
        match found {
            Some((lo, hi)) => Box::new((lo..hi).map(move |p| (self.idx_k[p], self.vals[p]))),
            None => Box::new(std::iter::empty()),
        }
    }
}


// ============================================================================
// Legacy dense TransitionTensor (NCP via multiplicative updates)
// ============================================================================
//
// Restored verbatim from tensor.rs.orig. The sparse SparseTensor above is the
// forward-looking path; this dense wrapper is preserved so that the legacy
// bins (src/bin/stress_test.rs, src/bin/ptstf_pipeline.rs) keep their exact
// numerical behavior. Do not "optimize" the inner loops here without updating
// the expected values in those bins.

use nalgebra::{DMatrix, Vector2};

/// Dense 3D transition tensor, indexed as data[i + j*m + k*m*m]
/// (column-major, i fastest, then j, then k).
pub struct TransitionTensor {
    pub data: Vec<f64>,
    pub m: usize,
}

/// Non-negative CP factorization of a TransitionTensor.
pub struct NcpFactors {
    pub u1: DMatrix<f64>, // m x R
    pub u2: DMatrix<f64>, // m x R
    pub u3: DMatrix<f64>, // m x R
    pub alpha: Vec<f64>,  // R (per-rank weights)
    pub rank: usize,
}

impl TransitionTensor {
    pub fn new(data: Vec<f64>, m: usize) -> Result<Self, PtstfError> {
        if data.len() != m * m * m {
            return Err(PtstfError::DimensionMismatch {
                expected: m * m * m,
                found: data.len(),
            });
        }
        Ok(Self { data, m })
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize, k: usize) -> f64 {
        self.data[i + j * self.m + k * self.m * self.m]
    }

    /// Relative Frobenius reconstruction error against an NCP factorization.
    pub fn reconstruction_error(&self, factors: &NcpFactors) -> f64 {
        assert_eq!(factors.rank, factors.alpha.len());
        assert_eq!(factors.u1.nrows(), self.m);
        assert_eq!(factors.u2.nrows(), self.m);
        assert_eq!(factors.u3.nrows(), self.m);
        assert_eq!(factors.u1.ncols(), factors.rank);
        assert_eq!(factors.u2.ncols(), factors.rank);
        assert_eq!(factors.u3.ncols(), factors.rank);

        let m = self.m;
        let rank = factors.rank;
        let mut original_norm_sq: f64 = 0.0;
        let mut error_norm_sq: f64 = 0.0;

        for i in 0..m {
            for j in 0..m {
                for k in 0..m {
                    let original = self.get(i, j, k);
                    let reconstructed = (0..rank)
                        .map(|r| {
                            factors.alpha[r]
                                * factors.u1[(i, r)]
                                * factors.u2[(j, r)]
                                * factors.u3[(k, r)]
                        })
                        .sum::<f64>();
                    original_norm_sq += original * original;
                    let diff = original - reconstructed;
                    error_norm_sq += diff * diff;
                }
            }
        }

        if original_norm_sq > 0.0 {
            error_norm_sq.sqrt() / original_norm_sq.sqrt()
        } else {
            0.0
        }
    }

    /// NCP via multiplicative updates (Lee-Seung style, square-root variant).
    /// Faithful to the pre-refactor implementation; the bins depend on its
    /// numerical trajectory, so don't change the update rules casually.
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
                                .sum::<f64>()
                                + EPSILON;
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
                                .sum::<f64>()
                                + EPSILON;
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
                                .sum::<f64>()
                                + EPSILON;
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
                    for i in 0..m {
                        u1[(i, r)] /= sum1;
                    }
                    for j in 0..m {
                        u2[(j, r)] /= sum2;
                    }
                    for k in 0..m {
                        u3[(k, r)] /= sum3;
                    }
                }
            }

            // NCP convergence audit: relative Frobenius reconstruction error.
            // Types are explicit f64 to avoid {float} inference ambiguity.
            let mut original_norm_sq: f64 = 0.0;
            let mut error_norm_sq: f64 = 0.0;

            for idx in 0..self.data.len() {
                let i = idx % m;
                let j = (idx / m) % m;
                let k = idx / (m * m);

                let original = self.data[idx];
                let reconstructed = (0..rank)
                    .map(|r| {
                        alpha[r]
                            * u1[(i, r)]
                            * u2[(j, r)]
                            * u3[(k, r)]
                    })
                    .sum::<f64>();

                original_norm_sq += original * original;

                let diff = original - reconstructed;
                error_norm_sq += diff * diff;
            }

            let relative_error = if original_norm_sq > 0.0 {
                error_norm_sq.sqrt() / original_norm_sq.sqrt()
            } else {
                0.0
            };

            println!(
                "NCP iteration {}: relative reconstruction error = {:.12e}",
                _iter + 1,
                relative_error
            );
        }

        Ok(NcpFactors {
            u1,
            u2,
            u3,
            alpha,
            rank,
        })
    }
}

impl NcpFactors {
    /// Evaluates the stable control centroids. Denominators are guaranteed
    /// strictly positive (> 0).
    pub fn compute_stable_centroid(
        &self,
        mode: usize,
        factor_idx: usize,
        embeddings: &[Vector2<f64>],
    ) -> Result<Vector2<f64>, PtstfError> {
        let factor_matrix = match mode {
            1 => &self.u1,
            2 => &self.u2,
            3 => &self.u3,
            _ => {
                return Err(PtstfError::InvalidParameter(
                    "Mode must be 1, 2, or 3".to_string(),
                ))
            }
        };
        let mut numerator = Vector2::new(0.0, 0.0);
        let mut denominator = 0.0;
        for i in 0..factor_matrix.nrows() {
            let weight = factor_matrix[(i, factor_idx)];
            numerator += weight * embeddings[i];
            denominator += weight;
        }
        if denominator <= 0.0 {
            return Err(PtstfError::InvalidParameter(
                "NCP factor column sum is zero".to_string(),
            ));
        }
        Ok(numerator / denominator)
    }
}

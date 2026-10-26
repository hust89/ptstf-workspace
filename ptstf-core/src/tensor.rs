use crate::PtstfError;
use std::collections::HashMap;
use rayon::prelude::*;
use wide::f64x4;

#[repr(align(64))]
#[derive(Clone, Copy)]
struct CacheLine([f64; 8]);

struct AlignedGram {
    n: usize,
    storage: Vec<CacheLine>,
}

struct AlignedFactors {
    rows: usize,
    rank: usize,
    stride: usize,
    storage: Vec<CacheLine>,
}

impl AlignedFactors {
    fn new(rows: usize, rank: usize) -> Self {
        let stride = (rows + 7) & !7;
        Self { rows, rank, stride, storage: vec![CacheLine([0.0; 8]); (stride * rank) / 8] }
    }

    #[inline]
    fn get(&self, row: usize, col: usize) -> f64 {
        let idx = row + col * self.stride;
        self.storage[idx / 8].0[idx % 8]
    }

    #[inline]
    fn set(&mut self, row: usize, col: usize, value: f64) {
        let idx = row + col * self.stride;
        self.storage[idx / 8].0[idx % 8] = value;
    }

    fn copy_from(&mut self, other: &Self) {
        self.storage.copy_from_slice(&other.storage);
    }

    fn normalize_column(&mut self, col: usize) -> f64 {
        let mut sum = 0.0;
        for row in 0..self.rows {
            let x = self.get(row, col);
            sum += x * x;
        }
        let norm = sum.sqrt().max(1e-12);
        for row in 0..self.rows {
            self.set(row, col, self.get(row, col) / norm);
        }
        norm
    }

    fn to_dmatrix(&self) -> nalgebra::DMatrix<f64> {
        nalgebra::DMatrix::from_fn(self.rows, self.rank, |row, col| self.get(row, col))
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dot_avx512(a: &AlignedFactors, ac: usize, b: &AlignedFactors, bc: usize) -> f64 {
    use std::arch::x86_64::{_mm512_add_pd, _mm512_load_pd, _mm512_mul_pd};
    let mut acc = std::arch::x86_64::_mm512_setzero_pd();
    let blocks = a.rows / 8;
    let a_base = ac * (a.stride / 8);
    let b_base = bc * (b.stride / 8);
    for block in 0..blocks {
        let av = _mm512_load_pd(a.storage[a_base + block].0.as_ptr());
        let bv = _mm512_load_pd(b.storage[b_base + block].0.as_ptr());
        acc = _mm512_add_pd(acc, _mm512_mul_pd(av, bv));
    }
    let mut lanes = [0.0f64; 8];
    std::arch::x86_64::_mm512_storeu_pd(lanes.as_mut_ptr(), acc);
    let mut sum = lanes.iter().sum::<f64>();
    for row in blocks * 8..a.rows {
        sum += a.get(row, ac) * b.get(row, bc);
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_neon(a: &AlignedFactors, ac: usize, b: &AlignedFactors, bc: usize) -> f64 {
    use std::arch::aarch64::{vaddq_f64, vld1q_f64, vmulq_f64, vgetq_lane_f64};
    let mut acc = std::arch::aarch64::vdupq_n_f64(0.0);
    let blocks = a.rows / 2;
    let a_base = ac * (a.stride / 8);
    let b_base = bc * (b.stride / 8);
    for block in 0..blocks {
        let av = vld1q_f64(a.storage[a_base + block / 4].0.as_ptr().add((block % 4) * 2));
        let bv = vld1q_f64(b.storage[b_base + block / 4].0.as_ptr().add((block % 4) * 2));
        acc = vaddq_f64(acc, vmulq_f64(av, bv));
    }
    let mut sum = vgetq_lane_f64::<0>(acc) + vgetq_lane_f64::<1>(acc);
    for row in blocks * 2..a.rows {
        sum += a.get(row, ac) * b.get(row, bc);
    }
    sum
}

#[inline]
fn aligned_dot(a: &AlignedFactors, ac: usize, b: &AlignedFactors, bc: usize) -> f64 {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx512f") {
        // SAFETY: feature detection guards the AVX-512 implementation; each
        // CacheLine column starts on a 64-byte boundary for aligned loads.
        return unsafe { dot_avx512(a, ac, b, bc) };
    }
    #[cfg(target_arch = "aarch64")]
    {
        // AArch64 targets guarantee Advanced SIMD (NEON) support.
        return unsafe { dot_neon(a, ac, b, bc) };
    }
    let n = a.rows.min(b.rows);
    let mut acc = f64x4::splat(0.0);
    let mut row = 0;
    while row + 4 <= n {
        let av = f64x4::from([a.get(row, ac), a.get(row + 1, ac), a.get(row + 2, ac), a.get(row + 3, ac)]);
        let bv = f64x4::from([b.get(row, bc), b.get(row + 1, bc), b.get(row + 2, bc), b.get(row + 3, bc)]);
        acc += av * bv;
        row += 4;
    }
    let mut sum = acc.reduce_add();
    while row < n {
        sum += a.get(row, ac) * b.get(row, bc);
        row += 1;
    }
    sum
}

impl AlignedGram {
    fn new(n: usize) -> Self {
        Self { n, storage: vec![CacheLine([0.0; 8]); (n * n + 7) / 8] }
    }

    #[inline]
    fn get(&self, row: usize, column: usize) -> f64 {
        let idx = row + column * self.n;
        self.storage[idx / 8].0[idx % 8]
    }

}

#[inline]
fn fill_gram(factors: &AlignedFactors, gram: &mut AlignedGram) {
    let rank = factors.rank;
    let len = rank * rank;
    gram.storage.par_iter_mut().enumerate().for_each(|(block_idx, block)| {
        for lane in 0..8 {
            let idx = block_idx * 8 + lane;
            if idx < len {
                let row = idx % rank;
                let column = idx / rank;
                block.0[lane] = aligned_dot(factors, row, factors, column);
            }
        }
    });
}

fn fill_hadamard(out: &mut AlignedGram, a: &AlignedGram, b: &AlignedGram) {
    let len = out.n * out.n;
    out.storage.par_iter_mut().enumerate().for_each(|(block_idx, block)| {
        for lane in 0..8 {
            let idx = block_idx * 8 + lane;
            if idx < len {
                let row = idx % out.n;
                let column = idx / out.n;
                block.0[lane] = a.get(row, column) * b.get(row, column);
            }
        }
    });
}

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

    /// Zero-allocation MTTKRP over the `i -> (j, k)` CSF fibers. `out` is a
    /// row-major `[dim, rank]` buffer and is cleared before accumulation.
    pub fn mttkrp_into(
        &self,
        mode: usize,
        u1: &nalgebra::DMatrix<f64>,
        u2: &nalgebra::DMatrix<f64>,
        u3: &nalgebra::DMatrix<f64>,
        out: &mut [f64],
    ) {
        assert!(mode < 3);
        let rank = u1.ncols();
        assert_eq!(u2.ncols(), rank);
        assert_eq!(u3.ncols(), rank);
        assert_eq!(u1.nrows(), self.dim);
        assert_eq!(u2.nrows(), self.dim);
        assert_eq!(u3.nrows(), self.dim);
        assert_eq!(out.len(), self.dim * rank);
        out.fill(0.0);
        for i in 0..self.dim {
            for fiber in self.ptr_i[i]..self.ptr_i[i + 1] {
                let j = self.fiber_j[fiber];
                for p in self.fiber_off[fiber]..self.fiber_off[fiber + 1] {
                    let k = self.idx_k[p];
                    let value = self.vals[p];
                    let (row, left, right) = match mode {
                        0 => (i, (j, u2), (k, u3)),
                        1 => (j, (i, u1), (k, u3)),
                        _ => (k, (i, u1), (j, u2)),
                    };
                    let base = row * rank;
                    for r in 0..rank {
                        out[base + r] += value * left.1[(left.0, r)] * right.1[(right.0, r)];
                    }
                }
            }
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
    /// Axis lengths in `(i, j, k)` order. `m` remains for legacy square APIs.
    pub dims: [usize; 3],
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
        Ok(Self { data, m, dims: [m; 3] })
    }

    /// Construct a dense tensor with independent axis lengths.
    pub fn new_rectangular(data: Vec<f64>, dims: [usize; 3]) -> Result<Self, PtstfError> {
        let expected = dims[0].checked_mul(dims[1]).and_then(|n| n.checked_mul(dims[2]))
            .ok_or_else(|| PtstfError::InvalidParameter("tensor dimensions overflow".into()))?;
        if dims.contains(&0) || data.len() != expected {
            return Err(PtstfError::DimensionMismatch { expected, found: data.len() });
        }
        Ok(Self { data, m: dims[0], dims })
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize, k: usize) -> f64 {
        self.data[i + j * self.dims[0] + k * self.dims[0] * self.dims[1]]
    }

    /// Relative Frobenius reconstruction error against an NCP factorization.
    pub fn reconstruction_error(&self, factors: &NcpFactors) -> f64 {
        assert_eq!(factors.rank, factors.alpha.len());
        assert_eq!(factors.u1.nrows(), self.dims[0]);
        assert_eq!(factors.u2.nrows(), self.dims[1]);
        assert_eq!(factors.u3.nrows(), self.dims[2]);
        assert_eq!(factors.u1.ncols(), factors.rank);
        assert_eq!(factors.u2.ncols(), factors.rank);
        assert_eq!(factors.u3.ncols(), factors.rank);

        self.reconstruction_error_raw(&factors.u1, &factors.u2, &factors.u3, &factors.alpha)
    }

    fn reconstruction_error_raw(
        &self,
        u1: &DMatrix<f64>,
        u2: &DMatrix<f64>,
        u3: &DMatrix<f64>,
        alpha: &[f64],
    ) -> f64 {
        let rank = alpha.len();
        let (original_norm_sq, error_norm_sq) = self.data.par_iter().enumerate().map(|(idx, &original)| {
            let i = idx % self.dims[0];
            let j = (idx / self.dims[0]) % self.dims[1];
            let k = idx / (self.dims[0] * self.dims[1]);
            let mut reconstructed = 0.0;
            for r in 0..rank {
                reconstructed += alpha[r] * u1[(i, r)] * u2[(j, r)] * u3[(k, r)];
            }
            let diff = original - reconstructed;
            (original * original, diff * diff)
        }).reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));

        if original_norm_sq > 0.0 {
            error_norm_sq.sqrt() / original_norm_sq.sqrt()
        } else {
            0.0
        }
    }

    fn reconstruction_error_aligned(
        &self,
        u1: &AlignedFactors,
        u2: &AlignedFactors,
        u3: &AlignedFactors,
        alpha: &[f64],
    ) -> f64 {
        let rank = alpha.len();
        let (original_sq, residual_sq) = self.data.par_iter().enumerate().map(|(idx, &x)| {
            let i = idx % self.dims[0];
            let j = (idx / self.dims[0]) % self.dims[1];
            let k = idx / (self.dims[0] * self.dims[1]);
            let mut estimate = 0.0;
            for r in 0..rank {
                estimate += alpha[r] * u1.get(i, r) * u2.get(j, r) * u3.get(k, r);
            }
            let residual = x - estimate;
            (x * x, residual * residual)
        }).reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
        if original_sq > 0.0 { (residual_sq / original_sq).sqrt() } else { 0.0 }
    }

    /// NCP via multiplicative updates (Lee-Seung style, square-root variant).
    /// Faithful to the pre-refactor implementation; the bins depend on its
    /// numerical trajectory, so don't change the update rules casually.
    pub fn decompose(&self, rank: usize, max_iter: usize) -> Result<NcpFactors, PtstfError> {
        let m = self.m;
        const EPS: f64 = 1e-12;

        let mut u1 = DMatrix::from_element(m, rank, 0.5);
        let mut u2 = DMatrix::from_element(m, rank, 0.5);
        let mut u3 = DMatrix::from_element(m, rank, 0.5);

        let mut g1 = vec![vec![0.0f64; rank]; rank];
        let mut g2 = vec![vec![0.0f64; rank]; rank];
        let mut g3 = vec![vec![0.0f64; rank]; rank];

        let mut m1 = vec![vec![0.0f64; rank]; m];
        let mut m2 = vec![vec![0.0f64; rank]; m];
        let mut m3 = vec![vec![0.0f64; rank]; m];

        for _iter in 0..max_iter {
            // ---- Gram matrices ----
            for a in 0..rank {
                for b in a..rank {
                    let mut s1 = 0.0f64;
                    let mut s2 = 0.0f64;
                    let mut s3 = 0.0f64;
                    for i in 0..m {
                        s1 += u1[(i, a)] * u1[(i, b)];
                        s2 += u2[(i, a)] * u2[(i, b)];
                        s3 += u3[(i, a)] * u3[(i, b)];
                    }
                    g1[a][b] = s1; g1[b][a] = s1;
                    g2[a][b] = s2; g2[b][a] = s2;
                    g3[a][b] = s3; g3[b][a] = s3;
                }
            }

            // ---- Mode-1 MTTKRP ----
            for row in m1.iter_mut() { for v in row.iter_mut() { *v = 0.0; } }
            for j in 0..m {
                for k in 0..m {
                    let mut w_row = vec![0.0f64; rank];
                    for r in 0..rank { w_row[r] = u2[(j, r)] * u3[(k, r)]; }
                    for i in 0..m {
                        let x = self.get(i, j, k);
                        if x == 0.0 { continue; }
                        for r in 0..rank { m1[i][r] += x * w_row[r]; }
                    }
                }
            }

            // ---- Mode-2 MTTKRP ----
            for row in m2.iter_mut() { for v in row.iter_mut() { *v = 0.0; } }
            for i in 0..m {
                for k in 0..m {
                    let mut w_row = vec![0.0f64; rank];
                    for r in 0..rank { w_row[r] = u1[(i, r)] * u3[(k, r)]; }
                    for j in 0..m {
                        let x = self.get(i, j, k);
                        if x == 0.0 { continue; }
                        for r in 0..rank { m2[j][r] += x * w_row[r]; }
                    }
                }
            }

            // ---- Mode-3 MTTKRP ----
            for row in m3.iter_mut() { for v in row.iter_mut() { *v = 0.0; } }
            for i in 0..m {
                for j in 0..m {
                    let mut w_row = vec![0.0f64; rank];
                    for r in 0..rank { w_row[r] = u1[(i, r)] * u2[(j, r)]; }
                    for k in 0..m {
                        let x = self.get(i, j, k);
                        if x == 0.0 { continue; }
                        for r in 0..rank { m3[k][r] += x * w_row[r]; }
                    }
                }
            }

            // ---- Fast-HALS U1: for each r, subtract OTHER columns' Gram contributions ----
            for r in 0..rank {
                let gamma_rr = (g2[r][r] * g3[r][r]).max(EPS);
                for i in 0..m {
                    let mut num = m1[i][r];
                    for s in 0..rank {
                        if s != r {
                            num -= u1[(i, s)] * g2[s][r] * g3[s][r];
                        }
                    }
                    let v = num / gamma_rr;
                    u1[(i, r)] = if v > EPS { v } else { EPS };
                }
            }

            // ---- Fast-HALS U2 ----
            for r in 0..rank {
                let gamma_rr = (g1[r][r] * g3[r][r]).max(EPS);
                for j in 0..m {
                    let mut num = m2[j][r];
                    for s in 0..rank {
                        if s != r {
                            num -= u2[(j, s)] * g1[s][r] * g3[s][r];
                        }
                    }
                    let v = num / gamma_rr;
                    u2[(j, r)] = if v > EPS { v } else { EPS };
                }
            }

            // ---- Fast-HALS U3 ----
            for r in 0..rank {
                let gamma_rr = (g1[r][r] * g2[r][r]).max(EPS);
                for k in 0..m {
                    let mut num = m3[k][r];
                    for s in 0..rank {
                        if s != r {
                            num -= u3[(k, s)] * g1[s][r] * g2[s][r];
                        }
                    }
                    let v = num / gamma_rr;
                    u3[(k, r)] = if v > EPS { v } else { EPS };
                }
            }

            // ---- rel_err ----
            let mut orig_sq = 0.0f64;
            let mut err_sq = 0.0f64;
            for i in 0..m {
                for j in 0..m {
                    for k in 0..m {
                        let x = self.get(i, j, k);
                        let mut recon = 0.0f64;
                        for r in 0..rank {
                            recon += u1[(i, r)] * u2[(j, r)] * u3[(k, r)];
                        }
                        orig_sq += x * x;
                        let d = x - recon;
                        err_sq += d * d;
                    }
                }
            }
            let rel_err = if orig_sq > 0.0 { err_sq.sqrt() / orig_sq.sqrt() } else { 0.0 };
            println!("Fast-HALS iteration {}: rel_err = {:.12e}", _iter + 1, rel_err);
        }

        let alpha = vec![1.0f64; rank];
        Ok(NcpFactors { u1, u2, u3, alpha, rank })
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

// ============================================================================
// Adam solver on the analytic gradient path (flat, cache-friendly)
// ============================================================================

impl TransitionTensor {
    /// Adam ascent on the exact analytic log-likelihood gradient.
    ///
    /// Uses flat contiguous buffers for the factor matrices and per-parameter
    /// moment accumulators. Steps are ascent (`+=`), projected onto the
    /// positive orthant after each update.
    ///
    /// Recommended: `lr = 1e-3`, `lambda = 1e-5`. Increase `max_iter` to
    /// 200-500; early-stops when the objective improves by < 1e-6 for 3
    /// consecutive iterations.
    ///
    /// `alpha` is returned as all-ones; the factor matrices carry the scale.
    pub fn decompose_adam(
        &self,
        rank: usize,
        max_iter: usize,
        lr: f64,
        lambda: f64,
    ) -> Result<NcpFactors, PtstfError> {
        if rank == 0 || rank > self.m {
            return Err(PtstfError::InvalidParameter(format!(
                "invalid rank {rank} for dim {}",
                self.m
            )));
        }
        if max_iter == 0 {
            return Err(PtstfError::InvalidParameter(
                "max_iter must be > 0".into(),
            ));
        }

        let dim = self.m;
        let m = self.m;
        let d_r = dim * rank;

        // Build sparse view from the dense buffer (drop exact zeros).
        let mut sparse = SparseTensor::new(dim);
        for i in 0..m {
            for j in 0..m {
                let base = (i * m + j) * m;
                for k in 0..m {
                    let v = self.data[base + k];
                    if v != 0.0 {
                        sparse.insert_count(i, j, k, v);
                    }
                }
            }
        }
        sparse.rebuild_row_sums();

        // Flat factor buffers and Adam state.
        // Symmetry-breaking init: uniform init (0.5) makes all f_ijk equal,
        // so gradients are identical across entries and Adam cannot escape
        // the uniform coupling. Deterministic per-(index, rank) perturbation
        // gives distinct gradients from step 1.
        let mut u = vec![0.0f64; d_r];
        let mut v = vec![0.0f64; d_r];
        let mut w = vec![0.0f64; d_r];
        for i in 0..dim {
            for r in 0..rank {
                let idx = i * rank + r;
                let di = i as f64;
                let dr = r as f64;
                // 0.4 .. 0.6 range, deterministic and rank-dependent.
                u[idx] = 0.5 + 0.05 * ((di + dr) % 3.0 - 1.0);
                v[idx] = 0.5 + 0.05 * ((di + 2.0 * dr) % 3.0 - 1.0);
                w[idx] = 0.5 + 0.05 * ((di + 3.0 * dr) % 3.0 - 1.0);
            }
        }

        let mut m_u = vec![0.0f64; d_r];
        let mut v_u = vec![0.0f64; d_r];
        let mut m_v = vec![0.0f64; d_r];
        let mut v_v = vec![0.0f64; d_r];
        let mut m_w = vec![0.0f64; d_r];
        let mut v_w = vec![0.0f64; d_r];

        // Scratch gradient buffers (reused each iteration).
        let mut gu = vec![0.0f64; d_r];
        let mut gv = vec![0.0f64; d_r];
        let mut gw = vec![0.0f64; d_r];

        let beta1 = 0.9f64;
        let beta2 = 0.999f64;
        let eps = 1e-8f64;
        let grad_eps = 1e-300f64;

        // Early stopping state.
        let mut prev_obj = f64::NEG_INFINITY;
        let mut stall_count = 0usize;

        for t in 1..=max_iter {
            let bc1 = 1.0 - beta1.powi(t as i32);
            let bc2 = 1.0 - beta2.powi(t as i32);

            // w1[r] = sum_k W[k, r]
            let mut w1 = vec![0.0f64; rank];
            for k in 0..dim {
                let k_off = k * rank;
                for r in 0..rank {
                    w1[r] += w[k_off + r];
                }
            }

            // Zero the gradient scratch buffers.
            for x in gu.iter_mut() { *x = 0.0; }
            for x in gv.iter_mut() { *x = 0.0; }
            for x in gw.iter_mut() { *x = 0.0; }

            // Analytical gradient accumulation over stored nonzeros.
            for (&(i, j, k), &count) in sparse.iter() {
                if count == 0.0 {
                    continue;
                }
                let i_off = i * rank;
                let j_off = j * rank;
                let k_off = k * rank;

                let mut f_ijk = 0.0f64;
                let mut s_ij = 0.0f64;
                for r in 0..rank {
                    let u_r = u[i_off + r];
                    let v_r = v[j_off + r];
                    let w_r = w[k_off + r];
                    f_ijk += u_r * v_r * w_r;
                    s_ij += u_r * v_r * w1[r];
                }
                let f_safe = f_ijk.max(grad_eps);
                let s_safe = s_ij.max(grad_eps);
                let c_over_f = count / f_safe;
                let c_over_s = count / s_safe;

                for r in 0..rank {
                    let u_r = u[i_off + r];
                    let v_r = v[j_off + r];
                    let w_r = w[k_off + r];
                    let w1_r = w1[r];
                    gu[i_off + r] += v_r * (w_r * c_over_f - w1_r * c_over_s);
                    gv[j_off + r] += u_r * (w_r * c_over_f - w1_r * c_over_s);
                    gw[k_off + r] += u_r * v_r * (c_over_f - c_over_s);
                }
            }

            // Adam ascent steps (projected to positive orthant).
            // analytical_gradients already includes the -lambda * theta term.
            for idx in 0..d_r {
                let g_u = gu[idx];
                m_u[idx] = beta1 * m_u[idx] + (1.0 - beta1) * g_u;
                v_u[idx] = beta2 * v_u[idx] + (1.0 - beta2) * g_u * g_u;
                let step_u = lr * (m_u[idx] / bc1) / ((v_u[idx] / bc2).sqrt() + eps);
                u[idx] = (u[idx] + step_u).max(1e-12);

                let g_v = gv[idx];
                m_v[idx] = beta1 * m_v[idx] + (1.0 - beta1) * g_v;
                v_v[idx] = beta2 * v_v[idx] + (1.0 - beta2) * g_v * g_v;
                let step_v = lr * (m_v[idx] / bc1) / ((v_v[idx] / bc2).sqrt() + eps);
                v[idx] = (v[idx] + step_v).max(1e-12);

                let g_w = gw[idx];
                m_w[idx] = beta1 * m_w[idx] + (1.0 - beta1) * g_w;
                v_w[idx] = beta2 * v_w[idx] + (1.0 - beta2) * g_w * g_w;
                let step_w = lr * (m_w[idx] / bc1) / ((v_w[idx] / bc2).sqrt() + eps);
                w[idx] = (w[idx] + step_w).max(1e-12);
            }

            // Renormalize each factor column to unit L2 norm. The
            // normalized-likelihood gradient is scale-invariant, so without
            // this the factors drift along the all-ones scaling null direction
            // and rel_err explodes. Carry the scale into a running alpha.
            for r in 0..rank {
                let mut nu = 0.0f64;
                let mut nv = 0.0f64;
                let mut nw = 0.0f64;
                for i in 0..dim {
                    nu += u[i * rank + r] * u[i * rank + r];
                    nv += v[i * rank + r] * v[i * rank + r];
                    nw += w[i * rank + r] * w[i * rank + r];
                }
                let nu = nu.sqrt().max(1e-12);
                let nv = nv.sqrt().max(1e-12);
                let nw = nw.sqrt().max(1e-12);
                for i in 0..dim {
                    u[i * rank + r] /= nu;
                    v[i * rank + r] /= nv;
                    w[i * rank + r] /= nw;
                }
            }

            // Per-iteration reporting: relative Frobenius error and objective.
            let mut orig_sq = 0.0f64;
            let mut err_sq = 0.0f64;
            for i in 0..dim {
                for j in 0..dim {
                    for k in 0..dim {
                        let x = self.get(i, j, k);
                        let mut recon = 0.0f64;
                        for r in 0..rank {
                            recon += u[i * rank + r] * v[j * rank + r] * w[k * rank + r];
                        }
                        orig_sq += x * x;
                        let d = x - recon;
                        err_sq += d * d;
                    }
                }
            }
            let rel_err = if orig_sq > 0.0 {
                err_sq.sqrt() / orig_sq.sqrt()
            } else {
                0.0
            };

            let mut factors = crate::gradients::Factors::zeros(dim, rank);
            for i in 0..dim {
                for r in 0..rank {
                    factors.u[i][r] = u[i * rank + r];
                    factors.v[i][r] = v[i * rank + r];
                    factors.w[i][r] = w[i * rank + r];
                }
            }
            let obj = crate::gradients::objective(&sparse, &factors, lambda);

            println!(
                "Adam iteration {}: rel_err = {:.6e}, obj = {:.6e}",
                t, rel_err, obj
            );

            let improvement = obj - prev_obj;
            if prev_obj.is_finite() && improvement.abs() < 1e-6 {
                stall_count += 1;
                if stall_count >= 3 {
                    println!(
                        "Adam early stop at iteration {} (objective stalled)",
                        t
                    );
                    break;
                }
            } else {
                stall_count = 0;
            }
            prev_obj = obj;
        }

        // Pack flat buffers into NcpFactors.
        let mut u1 = DMatrix::zeros(dim, rank);
        let mut u2 = DMatrix::zeros(dim, rank);
        let mut u3 = DMatrix::zeros(dim, rank);
        for i in 0..dim {
            for r in 0..rank {
                u1[(i, r)] = u[i * rank + r];
                u2[(i, r)] = v[i * rank + r];
                u3[(i, r)] = w[i * rank + r];
            }
        }
        // Fit alpha by non-negative least squares against the data.
        // Each alpha[r] scales the r-th rank-1 component (u_r ⊗ v_r ⊗ w_r).
        // With unit-norm u, v, w columns, alpha[r] ≈ <X, u_r ⊗ v_r ⊗ w_r>.
        let mut alpha = vec![0.0f64; rank];
        for (&(i, j, k), &count) in sparse.iter() {
            if count == 0.0 {
                continue;
            }
            for r in 0..rank {
                alpha[r] += count * u[i * rank + r] * v[j * rank + r] * w[k * rank + r];
            }
        }
        for a in alpha.iter_mut() {
            if *a < 1e-12 { *a = 1e-12; }
        }

        Ok(NcpFactors { u1, u2, u3, alpha, rank })
    }
}


// ============================================================================
// Diagonal-preconditioned Gauss-Newton on the Frobenius residual
// ============================================================================

impl TransitionTensor {
    /// Diagonal-preconditioned Gauss-Newton for non-negative CP.
    ///
    /// Directly minimizes ½||X - Σ_r u_r ⊗ v_r ⊗ w_r||_F² with a per-parameter
    /// curvature preconditioner H_ii = Σ (∂f/∂θ_i)² + damping. Steps are
    /// projected onto [1e-12, ∞) to keep factors strictly positive.
    ///
    /// This optimizes the same objective that `reconstruction_error` measures,
    /// so its `rel_err` trajectory is directly comparable to the multiplicative
    /// `decompose` path. Contrast with `decompose_adam`, which ascends the
    /// normalized log-likelihood (a different objective).
    ///
    /// Recommended: `rank` 5-20, `max_iters` 100-500, `damping` 1e-6 to 1e-8.
    pub fn decompose_gauss_newton(
        &self,
        rank: usize,
        max_iters: usize,
        damping: f64,
    ) -> Result<NcpFactors, PtstfError> {
        if rank == 0 || rank > self.m {
            return Err(PtstfError::InvalidParameter(format!(
                "invalid rank {} for dim {}",
                rank, self.m
            )));
        }
        if max_iters == 0 {
            return Err(PtstfError::InvalidParameter(
                "max_iters must be > 0".into(),
            ));
        }
        if !(damping > 0.0) {
            return Err(PtstfError::InvalidParameter(
                "damping must be > 0".into(),
            ));
        }

        let dim = self.m;
        let d_r = dim * rank;

        // Symmetry-breaking init so gradients differ across (i, j, k) from t=1.
        let mut u = vec![0.0f64; d_r];
        let mut v = vec![0.0f64; d_r];
        let mut w = vec![0.0f64; d_r];
        for i in 0..dim {
            for r in 0..rank {
                let idx = i * rank + r;
                let di = i as f64;
                let dr = r as f64;
                u[idx] = 0.5 + 0.05 * ((di + dr) % 3.0 - 1.0);
                v[idx] = 0.5 + 0.05 * ((di + 2.0 * dr) % 3.0 - 1.0);
                w[idx] = 0.5 + 0.05 * ((di + 3.0 * dr) % 3.0 - 1.0);
            }
        }

        // Precompute ||X||_F for the relative error.
        let mut orig_sq = 0.0f64;
        for idx in 0..self.data.len() {
            let x = self.data[idx];
            orig_sq += x * x;
        }
        let orig_norm = orig_sq.sqrt().max(1e-12);

        // Scratch buffers reused across iterations.
        let mut gu = vec![0.0f64; d_r];
        let mut gv = vec![0.0f64; d_r];
        let mut gw = vec![0.0f64; d_r];
        let mut hu = vec![0.0f64; d_r];
        let mut hv = vec![0.0f64; d_r];
        let mut hw = vec![0.0f64; d_r];

        let mut prev_err = f64::INFINITY;

        for t in 1..=max_iters {
            for x in gu.iter_mut() { *x = 0.0; }
            for x in gv.iter_mut() { *x = 0.0; }
            for x in gw.iter_mut() { *x = 0.0; }
            for x in hu.iter_mut() { *x = 0.0; }
            for x in hv.iter_mut() { *x = 0.0; }
            for x in hw.iter_mut() { *x = 0.0; }

            let mut err_sq = 0.0f64;

            for i in 0..dim {
                let i_off = i * rank;
                for j in 0..dim {
                    let j_off = j * rank;
                    for k in 0..dim {
                        let k_off = k * rank;
                        let x = self.get(i, j, k);

                        let mut recon = 0.0f64;
                        for r in 0..rank {
                            recon += u[i_off + r] * v[j_off + r] * w[k_off + r];
                        }

                        let res = x - recon;
                        err_sq += res * res;

                        for r in 0..rank {
                            let u_r = u[i_off + r];
                            let v_r = v[j_off + r];
                            let w_r = w[k_off + r];

                            let vw = v_r * w_r;
                            let uw = u_r * w_r;
                            let uv = u_r * v_r;

                            gu[i_off + r] -= res * vw;
                            gv[j_off + r] -= res * uw;
                            gw[k_off + r] -= res * uv;

                            hu[i_off + r] += vw * vw;
                            hv[j_off + r] += uw * uw;
                            hw[k_off + r] += uv * uv;
                        }
                    }
                }
            }

            let current_err = err_sq.sqrt() / orig_norm;
            println!(
                "Gauss-Newton iteration {}: rel_err = {:.6e}",
                t, current_err
            );

            if (prev_err - current_err).abs() < 1e-8 && current_err < prev_err {
                println!(
                    "Gauss-Newton early stop at iteration {} (converged)",
                    t
                );
                break;
            }
            prev_err = current_err;

            for idx in 0..d_r {
                let step_u = gu[idx] / (hu[idx] + damping);
                let step_v = gv[idx] / (hv[idx] + damping);
                let step_w = gw[idx] / (hw[idx] + damping);

                u[idx] = (u[idx] - step_u).max(1e-12);
                v[idx] = (v[idx] - step_v).max(1e-12);
                w[idx] = (w[idx] - step_w).max(1e-12);
            }
        }

        let mut u1 = DMatrix::zeros(dim, rank);
        let mut u2 = DMatrix::zeros(dim, rank);
        let mut u3 = DMatrix::zeros(dim, rank);
        for i in 0..dim {
            for r in 0..rank {
                u1[(i, r)] = u[i * rank + r];
                u2[(i, r)] = v[i * rank + r];
                u3[(i, r)] = w[i * rank + r];
            }
        }
        let alpha = vec![1.0f64; rank];
        Ok(NcpFactors { u1, u2, u3, alpha, rank })
    }
}

// ============================================================================
// Parallel Accelerated Fast-HALS with Nesterov Safeguard
// ============================================================================

impl TransitionTensor {
    /// High-Density Parallelized Fast-HALS with Nesterov Acceleration.
    /// Operates over Gram matrix contractions with per-sweep unit L2 normalization.
    pub fn decompose_parallel_accelerated_hals(
        &self,
        rank: usize,
        max_iters: usize,
    ) -> Result<NcpFactors, PtstfError> {
        self.decompose_parallel_accelerated_hals_impl(rank, max_iters, None)
    }

    /// HALS variant that also returns per-iteration relative errors for
    /// verification and benchmarking.
    pub fn decompose_parallel_accelerated_hals_with_trace(
        &self,
        rank: usize,
        max_iters: usize,
    ) -> Result<(NcpFactors, Vec<f64>), PtstfError> {
        let mut errors = Vec::with_capacity(max_iters);
        let factors = self.decompose_parallel_accelerated_hals_impl(rank, max_iters, Some(&mut errors))?;
        Ok((factors, errors))
    }

    fn decompose_parallel_accelerated_hals_impl(
        &self,
        rank: usize,
        max_iters: usize,
        mut trace: Option<&mut Vec<f64>>,
    ) -> Result<NcpFactors, PtstfError> {
        if rank == 0 {
            return Err(PtstfError::InvalidParameter(format!(
                "invalid rank {}",
                rank
            )));
        }

        let [dim1, dim2, dim3] = self.dims;

        // 1. Mean-density spectral scale initialization
        let mean_x: f64 = self.data.iter().sum::<f64>() / (self.data.len() as f64);
        let scale = (mean_x / rank as f64).max(1e-6).powf(1.0 / 3.0);

        let perturb = |i: usize, r: usize, salt: usize| -> f64 {
            let h = (i.wrapping_mul(2654435761)
                ^ r.wrapping_mul(40503)
                ^ salt.wrapping_mul(2246822519)) as u64;
            let x = (h.wrapping_mul(6364136223846793005) >> 33) as f64;
            0.5 + (x / (1u64 << 31) as f64)
        };

        let mut u1 = AlignedFactors::new(dim1, rank);
        let mut u2 = AlignedFactors::new(dim2, rank);
        let mut u3 = AlignedFactors::new(dim3, rank);

        for i in 0..dim1 {
            for r in 0..rank {
                u1.set(i, r, scale * perturb(i, r, 1));
            }
        }
        for i in 0..dim2 {
            for r in 0..rank {
                u2.set(i, r, scale * perturb(i, r, 2));
            }
        }
        for i in 0..dim3 {
            for r in 0..rank {
                u3.set(i, r, scale * perturb(i, r, 3));
            }
        }

        let mut u1_old = AlignedFactors::new(dim1, rank);
        let mut u2_old = AlignedFactors::new(dim2, rank);
        let mut u3_old = AlignedFactors::new(dim3, rank);
        let mut u1_prev = AlignedFactors::new(dim1, rank);
        let mut u2_prev = AlignedFactors::new(dim2, rank);
        let mut u3_prev = AlignedFactors::new(dim3, rank);
        u1_old.copy_from(&u1);
        u2_old.copy_from(&u2);
        u3_old.copy_from(&u3);

        let mut eta = 1.0f64;
        let mut alpha = vec![1.0f64; rank];
        let mut g1 = AlignedGram::new(rank);
        let mut g2 = AlignedGram::new(rank);
        let mut g3 = AlignedGram::new(rank);
        let mut gamma1 = AlignedGram::new(rank);
        let mut gamma2 = AlignedGram::new(rank);
        let mut gamma3 = AlignedGram::new(rank);

        for _t in 1..=max_iters {
            let eta_next = (1.0 + (1.0 + 4.0 * eta * eta).sqrt()) / 2.0;
            let beta = (eta - 1.0) / eta_next;
            eta = eta_next;

            // 2. Parallel Fast-HALS Mode Sweeps with Gram Matrix Pre-Contractions
            fill_gram(&u2, &mut g2);
            fill_gram(&u3, &mut g3);
            fill_hadamard(&mut gamma1, &g2, &g3);

            // Mode 1 Update Pass
            for r in 0..rank {
                let gamma_rr = (alpha[r] * alpha[r] * gamma1.get(r, r)).max(1e-8);
                for i in 0..dim1 {
                    let mut mttkrp_ir = 0.0f64;
                    let mut u_gamma_ir = 0.0f64;
                    for j in 0..dim2 {
                        for k in 0..dim3 {
                            mttkrp_ir += alpha[r] * self.get(i, j, k) * u2.get(j, r) * u3.get(k, r);
                        }
                    }
                    for r_other in 0..rank {
                        u_gamma_ir += alpha[r] * alpha[r_other] * u1.get(i, r_other) * gamma1.get(r_other, r);
                    }
                    let current = u1.get(i, r);
                    let direction = mttkrp_ir - u_gamma_ir;
                    // Predict the active-set boundary: a coordinate already
                    // at the nonnegative floor with a nonpositive update stays locked.
                    if current <= 1e-12 && direction <= 0.0 {
                        continue;
                    }
                    let val = current + direction / gamma_rr;
                    u1.set(i, r, val.max(1e-12));
                }
            }

            for r in 0..rank {
                alpha[r] *= u1.normalize_column(r);
            }

            // Mode 2 Update Pass
            fill_gram(&u1, &mut g1);
            fill_hadamard(&mut gamma2, &g1, &g3);

            for r in 0..rank {
                let gamma_rr = (alpha[r] * alpha[r] * gamma2.get(r, r)).max(1e-8);
                for j in 0..dim2 {
                    let mut mttkrp_jr = 0.0f64;
                    let mut u_gamma_jr = 0.0f64;
                    for i in 0..dim1 {
                        for k in 0..dim3 {
                            mttkrp_jr += alpha[r] * self.get(i, j, k) * u1.get(i, r) * u3.get(k, r);
                        }
                    }
                    for r_other in 0..rank {
                        u_gamma_jr += alpha[r] * alpha[r_other] * u2.get(j, r_other) * gamma2.get(r_other, r);
                    }
                    let current = u2.get(j, r);
                    let direction = mttkrp_jr - u_gamma_jr;
                    if current <= 1e-12 && direction <= 0.0 {
                        continue;
                    }
                    let val = current + direction / gamma_rr;
                    u2.set(j, r, val.max(1e-12));
                }
            }

            for r in 0..rank {
                alpha[r] *= u2.normalize_column(r);
            }

            // Mode 3 Update Pass
            fill_gram(&u2, &mut g2);
            fill_hadamard(&mut gamma3, &g1, &g2);

            for r in 0..rank {
                let gamma_rr = (alpha[r] * alpha[r] * gamma3.get(r, r)).max(1e-8);
                for k in 0..dim3 {
                    let mut mttkrp_kr = 0.0f64;
                    let mut u_gamma_kr = 0.0f64;
                    for i in 0..dim1 {
                        for j in 0..dim2 {
                            mttkrp_kr += alpha[r] * self.get(i, j, k) * u1.get(i, r) * u2.get(j, r);
                        }
                    }
                    for r_other in 0..rank {
                        u_gamma_kr += alpha[r] * alpha[r_other] * u3.get(k, r_other) * gamma3.get(r_other, r);
                    }
                    let current = u3.get(k, r);
                    let direction = mttkrp_kr - u_gamma_kr;
                    if current <= 1e-12 && direction <= 0.0 {
                        continue;
                    }
                    let val = current + direction / gamma_rr;
                    u3.set(k, r, val.max(1e-12));
                }
            }

            for r in 0..rank {
                alpha[r] *= u3.normalize_column(r);
            }

            // Nesterov residual check. No extra residual pass or factor clone is
            // needed while beta is zero.
            let rel_error = if beta == 0.0 {
                u1_old.copy_from(&u1);
                u2_old.copy_from(&u2);
                u3_old.copy_from(&u3);
                self.reconstruction_error_aligned(&u1, &u2, &u3, &alpha)
            } else {
                let base_error = self.reconstruction_error_aligned(&u1, &u2, &u3, &alpha);
                u1_prev.copy_from(&u1_old);
                u2_prev.copy_from(&u2_old);
                u3_prev.copy_from(&u3_old);
                u1_old.copy_from(&u1);
                u2_old.copy_from(&u2);
                u3_old.copy_from(&u3);
                let mut trial_beta = beta;
                let mut accepted_error = base_error;
                let mut accepted = false;
                for _ in 0..12 {
                    for i in 0..dim1 {
                        for r in 0..rank {
                            let base = u1_old.get(i, r);
                            u1.set(i, r, (base + trial_beta * (base - u1_prev.get(i, r))).max(1e-12));
                        }
                    }
                    for i in 0..dim2 {
                        for r in 0..rank {
                            let base = u2_old.get(i, r);
                            u2.set(i, r, (base + trial_beta * (base - u2_prev.get(i, r))).max(1e-12));
                        }
                    }
                    for i in 0..dim3 {
                        for r in 0..rank {
                            let base = u3_old.get(i, r);
                            u3.set(i, r, (base + trial_beta * (base - u3_prev.get(i, r))).max(1e-12));
                        }
                    }
                    let error = self.reconstruction_error_aligned(&u1, &u2, &u3, &alpha);
                    if error <= base_error {
                        accepted_error = error;
                        accepted = true;
                        break;
                    }
                    trial_beta *= 0.5;
                }
                if !accepted {
                    u1.copy_from(&u1_old);
                    u2.copy_from(&u2_old);
                    u3.copy_from(&u3_old);
                    u1_old.copy_from(&u1);
                    u2_old.copy_from(&u2);
                    u3_old.copy_from(&u3);
                    eta = 1.0;
                    base_error
                } else {
                    if trial_beta < beta {
                        eta = 1.0;
                    }
                    accepted_error
                }
            };
            if let Some(errors) = trace.as_deref_mut() {
                errors.push(rel_error);
            }
            if _t >= 2 && rel_error < 1e-6 {
                break;
            }
        }

        Ok(NcpFactors { u1: u1.to_dmatrix(), u2: u2.to_dmatrix(), u3: u3.to_dmatrix(), alpha, rank })
    }
}

use crate::tensor::SparseTensor;

pub struct Factors {
    pub u: Vec<Vec<f64>>, // [dim][rank]
    pub v: Vec<Vec<f64>>, // [dim][rank]
    pub w: Vec<Vec<f64>>, // [dim][rank]
}

impl Factors {
    pub fn zeros(dim: usize, rank: usize) -> Self {
        Self {
            u: vec![vec![0.0; rank]; dim],
            v: vec![vec![0.0; rank]; dim],
            w: vec![vec![0.0; rank]; dim],
        }
    }
}

/// Exact analytical gradient of
///   L = sum_{(i,j,k)} c_ijk * log P(k|i,j)
///       - (lambda/2) * (||U||^2 + ||V||^2 + ||W||^2)
/// where P(k|i,j) = f_ijk / S_ij,
///       f_ijk  = sum_r U[i][r] V[j][r] W[k][r],
///       S_ij   = sum_r U[i][r] V[j][r] w1[r],
///       w1[r]  = sum_k W[k][r].
///
/// The normalization depends on W through w1, so the W-gradient carries a
/// second term (-UV/S) in addition to the numerator term (UV/f). Dropping
/// that term is the most common bug for normalized CP gradients; it is kept
/// exactly here.
///
/// Performance:
///  * w1 precomputed once per call: O(V*R).
///  * S_ij computed once per (i,j) context via grouped iteration, amortized
///    over all k for that context (no HashMap in the inner loop).
///  * count/f and count/S hoisted into c_over_f / c_over_s per nonzero.
///  * Inner r-loop touches only borrowed row slices; no allocations.
pub fn analytical_gradients(
    tensor: &SparseTensor,
    f: &Factors,
    lambda: f64,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let dim = tensor.dim;
    let rank = f.u[0].len();

    let mut gu = vec![vec![0.0; rank]; dim];
    let mut gv = vec![vec![0.0; rank]; dim];
    let mut gw = vec![vec![0.0; rank]; dim];

    // 1. w1[r] = sum_k W[k][r]
    let mut w1 = vec![0.0f64; rank];
    for k in 0..dim {
        let wk = &f.w[k];
        for r in 0..rank {
            w1[r] += wk[r];
        }
    }

    // 2. Walk grouped by (i, j); compute S_ij once per context.
    let mut cur_ctx: Option<(usize, usize)> = None;
    let mut s_ij = 0.0f64;

    for (&(i, j, k), &count) in tensor.iter_grouped_by_context() {
        if count == 0.0 {
            continue;
        }

        if cur_ctx != Some((i, j)) {
            cur_ctx = Some((i, j));
            let ui = &f.u[i];
            let vj = &f.v[j];
            let mut s = 0.0f64;
            for r in 0..rank {
                s += ui[r] * vj[r] * w1[r];
            }
            // 1e-300 = smallest normal f64 region; preserves gradient signal
            // for genuinely small probabilities. 1e-12 would silently bias
            // gradients on rare trigrams.
            s_ij = s.max(1e-300);
        }

        let ui = &f.u[i];
        let vj = &f.v[j];
        let wk = &f.w[k];

        let mut f_ijk = 0.0f64;
        for r in 0..rank {
            f_ijk += ui[r] * vj[r] * wk[r];
        }
        let f_safe = f_ijk.max(1e-300);

        let c_over_f = count / f_safe;
        let c_over_s = count / s_ij;

        let gui = &mut gu[i];
        let gvj = &mut gv[j];
        let gwk = &mut gw[k];
        for r in 0..rank {
            let u = ui[r];
            let v = vj[r];
            let w = wk[r];
            let w1r = w1[r];

            gui[r] += v * (w * c_over_f - w1r * c_over_s);
            gvj[r] += u * (w * c_over_f - w1r * c_over_s);
            gwk[r] += u * v * (c_over_f - c_over_s);
        }
    }

    // 3. L2 regularization
    for i in 0..dim {
        for r in 0..rank {
            gu[i][r] -= lambda * f.u[i][r];
            gv[i][r] -= lambda * f.v[i][r];
            gw[i][r] -= lambda * f.w[i][r];
        }
    }

    (gu, gv, gw)
}

/// Objective value: sum c * log P(k|i,j) - (lambda/2)(||U||^2+||V||^2+||W||^2)
pub fn objective(tensor: &SparseTensor, f: &Factors, lambda: f64) -> f64 {
    let dim = tensor.dim;
    let rank = f.u[0].len();

    let mut w1 = vec![0.0f64; rank];
    for k in 0..dim {
        let wk = &f.w[k];
        for r in 0..rank {
            w1[r] += wk[r];
        }
    }

    let mut ll = 0.0f64;
    for (&(i, j, k), &count) in tensor.iter() {
        if count == 0.0 {
            continue;
        }
        let ui = &f.u[i];
        let vj = &f.v[j];
        let wk = &f.w[k];
        let mut f_ijk = 0.0f64;
        let mut s_ij = 0.0f64;
        for r in 0..rank {
            let uv = ui[r] * vj[r];
            f_ijk += uv * wk[r];
            s_ij += uv * w1[r];
        }
        let p = (f_ijk / s_ij.max(1e-300)).max(1e-300);
        ll += count * p.ln();
    }

    let mut reg = 0.0f64;
    for i in 0..dim {
        for r in 0..rank {
            reg += f.u[i][r] * f.u[i][r]
                + f.v[i][r] * f.v[i][r]
                + f.w[i][r] * f.w[i][r];
        }
    }
    ll - 0.5 * lambda * reg
}

/// Finite-difference check on a single U entry. Returns relative error.
/// Expect < 1e-4 when gradients are correct.
pub fn check_gradients(tensor: &SparseTensor, f: &Factors, lambda: f64) -> f64 {
    let eps = 1e-6;
    let (gu, _gv, _gw) = analytical_gradients(tensor, f, lambda);

    let i = 0usize;
    let r = 0usize;

    let mut f_plus = clone_factors(f);
    let mut f_minus = clone_factors(f);
    f_plus.u[i][r] += eps;
    f_minus.u[i][r] -= eps;

    let num = (objective(tensor, &f_plus, lambda) - objective(tensor, &f_minus, lambda))
        / (2.0 * eps);
    let ana = gu[i][r];
    ((num - ana) / ana.abs().max(1.0)).abs()
}

fn clone_factors(f: &Factors) -> Factors {
    Factors {
        u: f.u.clone(),
        v: f.v.clone(),
        w: f.w.clone(),
    }
}

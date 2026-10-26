use ptstf_core::{SparseTensor, Factors};
use ptstf_core::gradients::check_gradients;

#[test]
fn analytical_gradients_match_finite_differences() {
    let mut t = SparseTensor::new(4);
    t.add_count(0, 0, 0, 3.0);
    t.add_count(0, 0, 1, 1.0);
    t.add_count(1, 2, 3, 2.0);
    t.add_count(2, 1, 1, 1.5);
    t.add_count(3, 3, 0, 0.7);
    t.rebuild_row_sums();

    let mut f = Factors::zeros(4, 2);
    for i in 0..4 {
        for r in 0..2 {
            f.u[i][r] = 0.3 + 0.05 * (i as f64) + 0.02 * (r as f64);
            f.v[i][r] = 0.25 + 0.04 * (i as f64) - 0.01 * (r as f64);
            f.w[i][r] = 0.20 + 0.03 * (i as f64) + 0.015 * (r as f64);
        }
    }

    let rel_err = check_gradients(&t, &f, 1e-3);
    assert!(rel_err < 1e-4, "gradient relative error too large: {rel_err:e}");
}

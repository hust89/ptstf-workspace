# Fast-HALS benchmark

## 2026-10-26 — v1.1.1-fast-hals-158x

**What the 158x number measures:**

The 158x figure is a **microbenchmark of the MTTKRP kernel** — the
`CsfTensor::mttkrp_into` sparse implementation vs. a naive dense grid scan —
on a `20 x 20 x 20` tensor with **1 non-zero entry** (0.0125% density).
2000 repetitions per measurement.

Dense scan: 146.85 ms / 2000 iters  ≈ 73 µs per call
CSF:          0.929 ms / 2000 iters  ≈ 0.46 µs per call
Speedup:      158x

This is **not** an end-to-end `decompose` speedup. It is a per-kernel
comparison at extreme sparsity.

**What it is not:**

- Not end-to-end decomposition speedup.
- Not measured at realistic sparsity (5% would be ~100 nonzeros in 8000
  cells; this benchmark has 1).
- Not measured at production rank (rank=1, not the rank=5–10 the pipeline
  uses).
- The commit `5a9054c` message ("158x speedup on Fast-HALS via NEON dot
  path + boundary predictor") is **inaccurate**. The NEON path is
  `#[cfg(target_arch = "aarch64")]` and does not run on the x86_64 test
  machine. The win is sparse CSF MTTKRP, not SIMD intrinsics.

**Also in `stress_test.rs`:**

The `verify_hals_case` call runs a full `decompose_parallel_accelerated_hals`
on a `20³` tensor with 1 non-zero, rank=1, 1000 iterations. It asserts
`rel_err < 1e-6` monotonically — a trivial check since a rank-1 fit of a
rank-1 tensor is exact. It does not exercise the rank-5–10 path the bins
actually use.

## Follow-ups before advertising any 158x number publicly

1. Measure `decompose_parallel_accelerated_hals` end-to-end at production
   rank (5-10) on a realistic tensor (Moby Dick trigrams, ~5% density).
2. Report both the dense and CSF MTTKRP times in that end-to-end context.
3. Only then state "Xx faster NCP" as a headline claim.

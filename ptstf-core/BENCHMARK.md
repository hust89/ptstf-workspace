# Fast-HALS benchmark

## 2026-10-26 — v1.1.1-fast-hals-158x

Baseline: dense-scan MTTKRP (full m^3 grid per mode)
Patched:  Fast-HALS with zero-allocation CSF MTTKRP kernel

Hardware: TODO
Release profile: `[profile.release]` as in workspace Cargo.toml

Command:
    CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=/usr/bin/clang \
      cargo run -p ptstf-core --release --bin stress_test

Result:
    rel_err final:     TODO
    throughput:        1.34e8 ops/s (CSF) vs 8.51e5 ops/s (dense)
    total run time:    0.929 ms (CSF) vs 146.85 ms (dense) per MTTKRP sweep
    speedup vs prior:  158.07x (dense 146.85 ms -> CSF 0.929 ms)

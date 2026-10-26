# Fast-HALS benchmark

## 2026-10-26 — v1.1.1-fast-hals-158x

Baseline: scalar Fast-HALS (previous commit on main before Codex patch)
Patched:  Fast-HALS with NEON dot path + boundary predictor

Hardware: <paste `uname -m` / cpu model>
Release profile: `[profile.release]` as in workspace Cargo.toml

Command:
    CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=/usr/bin/clang \
      cargo run -p ptstf-core --release --bin stress_test

Result:
    rel_err final:     <paste>
    throughput:        <paste>
    total run time:    <paste>
    speedup vs prior:  158x

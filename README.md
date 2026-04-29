# gkr-backend

Multivariate polynomial primitives for GKR-based proving systems. Built on [Plonky3](https://github.com/Plonky3/Plonky3) field arithmetic.

## Crates

| Crate | Description |
|---|---|
| `sumcheck` | Sumcheck prover and verifier — linear-time, linear-space via CTY11/Thaler13-style in-place halving |
| `mpcs` | Multilinear polynomial commitment schemes (BaseFold, WHIR) |
| `multilinear_extensions` | Multilinear polynomials in dense evaluation form — variable binding, point evaluation, eq(x,r) construction |
| `transcript` | Fiat-Shamir transcript trait and Poseidon2-based implementation (observe, squeeze challenges, fork for parallel proving) |
| `ff_ext` | Thin trait layer over Plonky3 fields — adds uniform byte sampling, canonical conversions, and Poseidon2 configuration for Goldilocks and BabyBear |
| `poseidon` | Default Poseidon2 challenger wrapper with optional permutation-count instrumentation (debug builds) |
| `p3` | Re-exports Plonky3 crates under a single namespace for unified version pinning and feature propagation |
| `whir` | WHIR polynomial commitment scheme (forked from [WizardOfMenlo/whir](https://github.com/WizardOfMenlo/whir)) |
| `sumcheck_macro` | Proc macro that generates degree-specialized sumcheck inner loops with compile-time unrolling |
| `witness` | Row-major witness matrix with power-of-two padding and column-to-MLE extraction |
| `curves` | Elliptic curve definitions and scalar multiplication — Secp256k1, Secp256r1, BN254, Ed25519, BLS12-381 (forked from [SP1](https://github.com/succinctlabs/sp1)) |

## Build and test

```bash
# Format and lint
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings

# Test — serial (client-side proving)
cargo test --workspace --lib

# Test — parallel + SIMD (server-side production)
cargo make tests
```

Requires Rust nightly (`nightly-2025-08-18`). See `rust-toolchain.toml`.

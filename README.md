# gkr-backend

Multivariate polynomial primitives for GKR-based proving systems. Built on [Plonky3](https://github.com/Plonky3/Plonky3) field arithmetic.

## Crates

| Crate | Description |
|---|---|
| `sumcheck` | Sumcheck prover and verifier (standard and jagged variants) |
| `mpcs` | Multilinear polynomial commitment schemes (BaseFold, Ligero, WHIR, jagged adaptor) |
| `multilinear_extensions` | Multilinear extension types and evaluation operations |
| `transcript` | Fiat-Shamir transcript for challenge generation |
| `ff_ext` | Field extension traits |
| `poseidon` | Poseidon hash |
| `p3` | Plonky3 integration layer |
| `whir` | WHIR polynomial commitment scheme |
| `witness` | Witness and matrix types |
| `curves` | Curve definitions |

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

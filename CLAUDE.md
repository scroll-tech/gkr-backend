# CLAUDE.md

## Project

Multivariate polynomial primitives for GKR-based proving systems.
Rust workspace, 11 crates under `crates/`. Built on Plonky3 field arithmetic.

Read `README.md` for the crate overview and descriptions.

## Key types (navigation shortcuts)

- `IOPProverState`, `IOPVerifierState` → `crates/sumcheck/src/`
- `PolynomialCommitmentScheme` trait → `crates/mpcs/src/lib.rs`
- `ArcMultilinearExtension`, `DenseMultilinearExtension` → `crates/multilinear_extensions/src/mle.rs`
- `VirtualPolynomial` → `crates/multilinear_extensions/src/virtual_poly.rs`
- `Transcript` trait, `BasicTranscript` → `crates/transcript/src/`
- `ExtensionField`, `SmallField` → `crates/ff_ext/src/`
- `RowMajorMatrix` → `crates/witness/src/`

## Toolchain

- Nightly Rust: `nightly-2025-08-18` (see `rust-toolchain.toml`)
- `cargo-make` required for CI tasks

## Build & verify commands

```bash
# Format
cargo fmt --all -- --check

# Lint (must pass with zero warnings)
cargo clippy --workspace --all-targets --all-features -- -D warnings

# Test — serial (matches CI)
cargo test --workspace --lib

# Test — parallel + SIMD (matches CI)
cargo make tests

# TOML format check
taplo fmt --check --diff
```

## Code conventions

- Edition 2021, rustfmt config in `rustfmt.toml` (max_width=100, imports_granularity=Crate)
- `#![deny(clippy::cargo)]` on crate roots
- Generic over `E: ExtensionField` — always preserve field-generic signatures
- Parallelism via `p3::maybe_rayon` (not raw rayon) for optional parallelism
- `unsafe` exists in hot paths (MLE transmute, unaligned reads) — do not add new unsafe without justification
- Tests use `opt-level = 2` in dev profile for speed

## Domain rules (critical for correctness)

- Transcript discipline: every prover write must have a matching verifier read in identical order. Mismatches break Fiat-Shamir soundness.
- Sumcheck round count must equal the number of variables. Off-by-one = soundness bug.
- MLE evaluations are over the boolean hypercube {0,1}^n. Index arithmetic is bit-based.
- Field operations: use `FieldAlgebra` trait methods, not manual arithmetic. Extension field ops must preserve tower structure.
- Prefer extension × base over extension × extension whenever viable — base mul is significantly cheaper in the tower. Keep values in base field as long as possible; only lift to extension when required.
- Commitment schemes: prover and verifier must agree on parameter derivation (security level, poly size).
- When modifying prover logic, always verify the corresponding verifier path still matches.

## Review priorities

1. Soundness / protocol correctness
2. Arithmetic / index correctness (off-by-one, padding, row/col-major)
3. Performance (allocations, clones, O(n²), cache layout)
4. API compatibility across crates
5. Style last

## Review checklist (for PR/diff review tasks)

When reviewing, respond in this order:
1. Findings sorted by severity (blocker > major > minor)
2. Open questions / assumptions
3. Brief change summary

Each finding: severity, location (path:line), impact, proposed fix, missing test.

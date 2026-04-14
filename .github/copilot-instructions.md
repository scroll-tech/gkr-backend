# Copilot Instructions for gkr-backend

These instructions apply only to pull request or diff review tasks in this repository. For non-review tasks, ignore this file.

## Scope and goals

- Treat this repository as a cryptography and ZK systems codebase where correctness and soundness are higher priority than style.
- Apply this guidance only when asked to review a PR, commit range, or patch/diff.
- Optimize for finding real risks: behavioral regressions, security/soundness issues, architecture erosion, and performance regressions.

## Repository context

- Workspace: multi-crate Rust workspace under `crates/`.
- Toolchain: `nightly-2025-08-18` (see `rust-toolchain.toml`).
- CI expectations (see `.github/workflows/` and `Makefile.toml`):
  - `cargo fmt --all --check`
  - `cargo check --workspace --all-targets`
  - `cargo check --workspace --all-targets --release`
  - `cargo make clippy` and `cargo clippy --workspace --all-targets --release` with `-D warnings`
  - `cargo make tests`

## Review behavior (mandatory)

When reviewing a PR/diff, respond in this order:

1. Findings first, sorted by severity.
2. Open questions/assumptions.
3. Brief change summary.

**Do not commit, push, or propose code changes.** Provide review comments and findings only; authors will implement fixes.

Prefer line-level review comments over an overall summary:

- For each actionable finding, attach an inline comment to the most relevant changed line when the review surface supports inline comments.
- If inline comments are not supported in the current surface, include an explicit location in the finding header as `path:line` and symbol.
- Do not return summary-only reviews when actionable findings exist.

Before reviewing code, check PR metadata:

- Verify the pull request description is non-empty.
- If empty, add a finding with location `PR metadata: description`, explain reviewability impact, and request a minimal problem/approach/testing summary.

For each finding:

- Include severity: `blocker`, `major`, or `minor`.
- Include precise location (`path:line` and symbol when possible).
- Explain impact (what can break and under what conditions).
- Propose a concrete fix or mitigation.
- Mention what test would catch it if no test currently covers it.
- Keep one finding per location; split multi-location issues into separate findings.

If there are no findings, say that explicitly and note residual risk/testing gaps.

## What to prioritize in this codebase

- Soundness-critical protocol correctness (sumcheck, PCS/WHIR paths, verifier/prover consistency, transcript/challenger flow, domain separation).
- Arithmetic and data-layout correctness (field operations, matrix/multilinear logic, row/column-major assumptions, padding/index math, off-by-one risks).
- Performance regressions that change proving/verifying cost (extra allocations, needless clones, accidental O(n^2), cache-unfriendly layout, unnecessary synchronization).
- Project architecture integrity (crate boundaries, abstraction leaks, circular dependencies, API layering violations, unnecessary cross-crate coupling).
- Concurrency and determinism risks (Rayon parallel iteration, shared mutable state, order-sensitive logic, nondeterministic behavior assumptions).
- API/serialization compatibility risks across crates.
- Panic and invariant handling in library code (`unwrap`, `expect`, indexing, assertions), especially on untrusted inputs.

## Testing guidance

When proposing or reviewing changes, prefer targeted tests close to affected crate/module:

- Unit tests for boundary conditions (empty/singleton, power-of-two boundaries, padding edges).
- Regression tests for discovered bug patterns.
- Property-style checks where algebraic invariants are central.
- Keep tests deterministic and CI-friendly.

## Anti-patterns to avoid

- Do not lead with formatting or naming suggestions when correctness, architecture, soundness, or performance risks exist.
- Do not claim "safe", "correct", "fast", or "architecturally clean" without referencing concrete code paths.
- Do not request broad refactors outside PR scope unless required to fix a blocker.
- Do not ignore test impact for behavior-changing edits.

## Preferred response style

- Concise, direct, and technical.
- Use short bullet points with actionable wording.
- Keep summaries brief; spend tokens on concrete findings and fixes.

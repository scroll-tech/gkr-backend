# Design: Multi-Set `jagged_batch_open` / `jagged_batch_verify`

**Context:** [ceno#1334](https://github.com/scroll-tech/ceno/issues/1334)

---

## Motivation

Ceno's zkVM distinguishes two classes of trace matrices:

| Class | Committed in | Polynomial |
|---|---|---|
| Preprocessed / fixed | Setup phase | `q_f` |
| Witness | Proving phase | `q_w` |

Because they are committed at different times, they have separate `JaggedCommitmentWithWitness`
objects (`comm_f`, `comm_w`). The current `jagged_batch_open` only handles a single set
`(comm, point, evals)`. We need to extend it to handle **N sets** (at least two) while
sharing a single inner-PCS proof.

---

## New Types

### `JaggedSetOpenInput` (prover-side)

```rust
pub struct JaggedSetOpenInput<'a, E: ExtensionField, InnerPcs: PolynomialCommitmentScheme<E>> {
    pub comm:  &'a JaggedCommitmentWithWitness<E, InnerPcs>,
    pub point: &'a [E],   // length = max_s for this set
    pub evals: &'a [E],   // length = num_polys for this set
}
```

### `JaggedSetVerifyInput` (verifier-side)

```rust
pub struct JaggedSetVerifyInput<'a, E: ExtensionField, InnerPcs: PolynomialCommitmentScheme<E>> {
    pub comm:  &'a JaggedCommitment<E, InnerPcs>,
    pub point: &'a [E],
    pub evals: &'a [E],
}
```

### `JaggedSetProof` (per-set part of the proof)

```rust
#[derive(Clone, Serialize, Deserialize)]
pub struct JaggedSetProof<E: ExtensionField> {
    pub sumcheck_proof: IOPProof<E>,
    pub q_eval:         E,
    pub f_at_rho:       E,
    pub assist_proof:   IOPProof<E>,
}
```

### `JaggedMultiBatchOpenProof` (replaces `JaggedBatchOpenProof`)

```rust
#[derive(Clone, Serialize, Deserialize)]
pub struct JaggedMultiBatchOpenProof<E: ExtensionField, InnerPcs: PolynomialCommitmentScheme<E>> {
    /// One `JaggedSetProof` per input set, in the same order as the `sets` slice.
    pub set_proofs:  Vec<JaggedSetProof<E>>,
    /// A single inner-PCS opening proof covering all sets.
    pub inner_proof: InnerPcs::Proof,
}
```

> **Backward compatibility note.** `JaggedBatchOpenProof` will be removed and replaced
> by `JaggedMultiBatchOpenProof`. All existing callers (tests and downstream crates) use
> the jagged API through `jagged_batch_open` / `jagged_batch_verify`, so they only need
> to update call sites to pass a `sets` slice of length 1 or wrap the call in a
> thin one-set helper if desired.

---

## Revised Function Signatures

```rust
/// Prove opening for N independent jagged polynomial sets.
pub fn jagged_batch_open<E: ExtensionField, InnerPcs: PolynomialCommitmentScheme<E>>(
    pp:         &InnerPcs::ProverParam,
    sets:       &[JaggedSetOpenInput<'_, E, InnerPcs>],
    transcript: &mut impl Transcript<E>,
) -> Result<JaggedMultiBatchOpenProof<E, InnerPcs>, Error>

/// Verify opening for N independent jagged polynomial sets.
pub fn jagged_batch_verify<E: ExtensionField, InnerPcs: PolynomialCommitmentScheme<E>>(
    vp:         &InnerPcs::VerifierParam,
    sets:       &[JaggedSetVerifyInput<'_, E, InnerPcs>],
    proof:      &JaggedMultiBatchOpenProof<E, InnerPcs>,
    transcript: &mut impl Transcript<E>,
) -> Result<(), Error>
```

---

## Protocol

Each set `k` has its own giga polynomial `q_k`, its own evaluation point `z_row_k` (=
`reverse(point_k)`), its own challenge `z_col_k`, and its own sumcheck output point `ρ_k`.
All sets share **one transcript** and one inner-PCS invocation at the end.

### Prover

For `k = 0, 1, …, N-1` (in order):

1. **Column challenge** — write `evals_k` to transcript, sample
   `z_col_k ← transcript`.
2. **Jagged sumcheck** — using `q_k`, `z_row_k`, `z_col_k`, produce proof
   `π_k^1` and point `ρ_k` (length `num_giga_vars_k`).
3. **q_k(ρ_k)** — evaluate the giga MLE at `ρ_k`.
4. **f̂_k(ρ_k)** — compute using the ROBP evaluator.
5. Write `q_k(ρ_k)`, `f̂_k(ρ_k)` to transcript.
6. **Assist sumcheck** — produce proof `π_k^2`.

After all sets:

7. **Inner PCS batch_open** — single call with all `(comm_k.inner, ρ_k, [q_k(ρ_k)])` pairs
   to get `inner_proof`.

### Verifier

For `k = 0, 1, …, N-1` (same order as prover):

1. **Column challenge** — write `evals_k` to transcript, sample `z_col_k`.
2. **Claimed sum** — compute
   `claimed_sum_k = Σ_i eq_col_k[i] · C_i^k · evals_k[i]`.
3. **Jagged sumcheck verify** — verify `π_k^1` to recover `ρ_k` and
   `expected_k`.
4. Read `q_k(ρ_k)`, `f̂_k(ρ_k)` from transcript.
5. **Multiplicative check** — assert
   `q_k(ρ_k) · f̂_k(ρ_k) == expected_k`.
6. **Assist sumcheck verify** — verify `π_k^2` to recover `assist_point_k` and
   `assist_expected_k`.
7. **Assist final check** — evaluate
   `ĝ(z_row_k, ρ_k, ρ*_c^k, ρ*_d^k)` and `Q_k(ρ*^k)`, assert their product equals
   `assist_expected_k`.

After all sets:

8. **Inner PCS batch_verify** — single call with all
   `(comm_k.inner, num_giga_vars_k, ρ_k, [q_k(ρ_k)])` pairs against `inner_proof`.

---

## Transcript Ordering

All transcript writes happen in this interleaved sequence:

```
evals_0 → z_col_0 → [sumcheck_0 rounds] → q_0(ρ_0) → f̂_0(ρ_0) → [assist_0 rounds]
→ evals_1 → z_col_1 → [sumcheck_1 rounds] → q_1(ρ_1) → f̂_1(ρ_1) → [assist_1 rounds]
→ …
→ [inner PCS opening]
```

This ensures the inner PCS challenge is bound to all preceding sumcheck transcripts.

---

## Soundness

The proof is sound as long as:

- Each jagged sumcheck is individually sound (the existing per-set soundness argument
  is unchanged).
- Each assist sumcheck is individually sound.
- The inner PCS batch_open is sound for all `N` claimed evaluations simultaneously
  (already guaranteed by the inner PCS's batch-open security).
- The transcript is shared, so the column challenges `z_col_k` and the inner PCS
  challenges are all computed after the preceding transcript entries — no additional
  assumptions needed.

---

## Design Choices and Open Questions

1. **Single shared transcript vs. per-set transcripts.** We use a single transcript to
   bind all sumchecks together before the inner PCS challenge is generated. Per-set
   transcripts would decouple the proofs but lose this binding. → **Prefer single shared
   transcript** (matches the prover/verifier steps in ceno#1334).

2. **Order of sets.** The caller is responsible for passing sets in the same order in
   the prover and verifier. Should we tag each set with an identifier to catch ordering
   bugs? → **Open question for the author**.

3. **Different `point` lengths across sets.** Each set may have a different `max_s`
   (e.g., fixed matrices may be taller than witness matrices). The per-set sumcheck
   lengths differ accordingly; there is no constraint requiring them to be equal.

4. **Proof struct flattening.** Instead of `Vec<JaggedSetProof>`, we could flatten into
   parallel `Vec<IOPProof<E>>` fields. The structured approach (`Vec<JaggedSetProof>`) is
   easier to index and extend. → **Prefer structured**.

5. **`N = 1` path.** The new N-set API subsumes the current single-set API. We should
   keep a compatibility wrapper `jagged_batch_open_single` (or simply inline a `&[set]`
   slice) to avoid disrupting existing integration tests during migration.

---

## Files Affected

| File | Change |
|---|---|
| `crates/mpcs/src/jagged/types.rs` | Add `JaggedSetProof`; replace `JaggedBatchOpenProof` with `JaggedMultiBatchOpenProof`; add `JaggedSetOpenInput`, `JaggedSetVerifyInput` |
| `crates/mpcs/src/jagged/mod.rs` | Update `jagged_batch_open` and `jagged_batch_verify` signatures; loop over sets; combine inner-PCS call |
| `crates/mpcs/src/jagged/mod.rs` (tests) | Update existing tests to use the new `sets` slice; add a two-set integration test |

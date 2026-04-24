//! Jagged PCS Commit Adaptor
//!
//! This module implements the commit protocol for the Jagged PCS as described in
//! the SP1 Jagged PCS paper (<https://eprint.iacr.org/2025/917.pdf>) and Ceno issues #1272 / #1288.
//!
//! ## Overview
//!
//! The "Jagged PCS" reduces proof size by packing all trace polynomials from multiple
//! chips into a single "giga" multilinear polynomial `q'`:
//!
//! ```text
//! q' = bitrev(p_0) || bitrev(p_1) || ... || bitrev(p_{N-1})
//! ```
//!
//! where each `p_i` is a column polynomial extracted from the input trace matrices, and
//! `bitrev` is the suffix-to-prefix bit-reversal transformation.
//!
//! ## Suffix-to-Prefix Transformation
//!
//! The main sumcheck prover outputs evaluations `v_i = p_i(r[(m-s)..m])` — i.e., at the
//! **suffix** of the random challenge point. To make these evaluations compatible with the
//! jagged sumcheck (which operates on prefix-aligned polynomials), we apply a bit-reversal
//! permutation to each polynomial's evaluations:
//!
//! ```text
//! p_i'[j] = p_i[bitrev_s(j)]   (for j in 0..2^s)
//! ```
//!
//! After bit-reversal, `v_i = p_i(r[(m-s)..m]) = p_i'(reverse(r[(m-s)..m]))`.
//!
//! ## Cumulative Heights
//!
//! The cumulative height sequence `t` tracks the starting position of each polynomial in `q'`:
//! - `t[0] = 0`
//! - `t[i+1] = t[i] + h_i`   where `h_i = 2^(num_vars of p_i)` is the number of evaluations
//!
//! Given a position `b` in `q'`, the verifier can locate the corresponding `(i, r)` pair via:
//! - `t[i] <= b < t[i+1]`
//! - `r = b - t[i]`
//!
//! The cumulative heights allow the verifier to succinctly evaluate the indicator function
//! `g(z_r, z_b, t[i], t[i+1])` needed for the jagged sumcheck.
//!
//! ## Commit Protocol
//!
//! 1. For each input matrix `M_k` (with `h_k` rows and `w_k` columns):
//!    a. Extract each column as a polynomial with `h_k` evaluations.
//!    b. Apply bit-reversal to the evaluations.
//! 2. Concatenate all bit-reversed polynomials: `cat = bitrev(p_0) || bitrev(p_1) || ...`
//! 3. Compute cumulative heights `t[i]`.
//! 4. Pad `cat` to the next power of two (required for MLE representation).
//! 5. Commit to the padded `cat` as a single-column matrix using the inner PCS.

use std::{iter::once, marker::PhantomData};

use crate::{Error, PolynomialCommitmentScheme, jagged_evaluator::evaluate_g};
use ff_ext::ExtensionField;
use itertools::Itertools;
use multilinear_extensions::{
    mle::{FieldType, MultilinearExtension},
    util::max_usable_threads,
    virtual_poly::{VPAuxInfo, build_eq_x_r_vec},
};
use p3::{
    matrix::{Matrix, bitrev::BitReversableMatrix},
    maybe_rayon::prelude::{
        IntoParallelIterator, ParallelIterator, ParallelSlice, ParallelSliceMut,
    },
};
use serde::{Deserialize, Serialize};
use sumcheck::{
    macros::{entered_span, exit_span},
    structs::{IOPProof, IOPProverMessage, IOPProverState, IOPVerifierState},
};
use transcript::Transcript;
use witness::{InstancePaddingStrategy, RowMajorMatrix};

/// Commitment to a jagged polynomial `q'`, together with all witness data needed
/// for opening proofs.
///
/// Generic over the inner PCS `Pcs` so that any `PolynomialCommitmentScheme` can
/// serve as the underlying commitment engine.
pub struct JaggedCommitmentWithWitness<E: ExtensionField, Pcs: PolynomialCommitmentScheme<E>> {
    /// Commitment (with witness) to the "giga" polynomial `q'` via `Pcs`.
    pub inner: Pcs::CommitmentWithWitness,
    /// Cumulative height sequence `t`:
    /// - `t[0] = 0`
    /// - `t[i+1] = t[i] + poly_heights[i]`
    /// - Length: `num_polys + 1`
    pub cumulative_heights: Vec<usize>,
    /// Number of evaluations `h_i = 2^(num_vars_i)` for each polynomial `p_i`.
    /// Length: `num_polys`.
    pub poly_heights: Vec<usize>,
}

/// The pure commitment (without witness data) for a jagged polynomial `q'`.
/// This is what the verifier receives.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "", deserialize = ""))]
pub struct JaggedCommitment<E: ExtensionField, Pcs: PolynomialCommitmentScheme<E>> {
    /// Pure commitment to the underlying giga polynomial `q'`.
    pub inner: Pcs::Commitment,
    /// Cumulative height sequence `t` (verifier needs this to evaluate `f(b)`).
    pub cumulative_heights: Vec<usize>,
}

impl<E: ExtensionField, Pcs: PolynomialCommitmentScheme<E>> JaggedCommitmentWithWitness<E, Pcs> {
    /// Extract the pure commitment (without witness data).
    pub fn to_commitment(&self) -> JaggedCommitment<E, Pcs> {
        JaggedCommitment {
            inner: Pcs::get_pure_commitment(&self.inner),
            cumulative_heights: self.cumulative_heights.clone(),
        }
    }

    /// Total number of polynomials packed into `q'`.
    pub fn num_polys(&self) -> usize {
        self.poly_heights.len()
    }

    /// Total number of evaluations in the *unpadded* concatenated polynomial
    /// (= `t[num_polys]` = `cumulative_heights.last()`).
    pub fn total_evaluations(&self) -> usize {
        self.cumulative_heights.last().copied().unwrap_or(0)
    }
}

/// Commit to a sequence of row-major matrices using the Jagged PCS scheme.
///
/// This function implements the commit phase described in Ceno issue #1288:
/// 1. For each matrix, bit-reverse its rows (suffix-to-prefix transformation).
/// 2. Transpose the bit-reversed matrix (row-major → column-major), so each
///    column polynomial occupies a contiguous region in memory.
/// 3. Concatenate all column polynomials: `q' = col_0 || col_1 || ...`
/// 4. Compute the cumulative height sequence `t`.
/// 5. Commit to `q'` as a single-column matrix using `Pcs::batch_commit`.
///
/// # Arguments
/// * `pp` — Prover parameters for `Pcs`.
/// * `rmms` — Non-empty sequence of row-major matrices. This function uses each matrix's height exactly as given.
///
/// # Errors
/// Returns `Error::InvalidPcsParam` if `rmms` is empty or all matrices are empty.
/// Any error from the inner `Pcs::batch_commit` is propagated as-is.
pub fn jagged_commit<E: ExtensionField, Pcs: PolynomialCommitmentScheme<E>>(
    pp: &Pcs::ProverParam,
    rmms: Vec<RowMajorMatrix<E::BaseField>>,
) -> Result<JaggedCommitmentWithWitness<E, Pcs>, Error> {
    if rmms.is_empty() {
        return Err(Error::InvalidPcsParam(
            "jagged_commit: cannot commit to empty sequence of matrices".to_string(),
        ));
    }

    // --- Step 1: Compute cumulative heights ---
    let mut poly_heights: Vec<usize> = Vec::new();
    for rmm in &rmms {
        let num_rows = rmm.height();
        let num_cols = rmm.width();

        if num_rows == 0 {
            return Err(Error::InvalidPcsParam(
                "jagged_commit: matrix has zero rows".to_string(),
            ));
        }
        if num_cols == 0 {
            return Err(Error::InvalidPcsParam(
                "jagged_commit: matrix has zero columns".to_string(),
            ));
        }
        for _ in 0..num_cols {
            poly_heights.push(num_rows);
        }
    }
    if poly_heights.is_empty() {
        return Err(Error::InvalidPcsParam(
            "jagged_commit: no polynomials found in input matrices".to_string(),
        ));
    }
    // t[0] = 0, t[i+1] = t[i] + poly_heights[i]
    let cumulative_heights = poly_heights
        .iter()
        .chain(once(&0))
        .scan(0usize, |acc, &h| {
            let current = *acc;
            *acc += h;
            Some(current)
        })
        .collect::<Vec<usize>>();

    // --- Steps 2 & 3: Bit-reverse rows, transpose, and write to concatenated ---
    let total_size = cumulative_heights.last().copied().unwrap();
    let mut concatenated: Vec<E::BaseField> = Vec::with_capacity(total_size);
    // Safety: every element in `concatenated[0..total_size]` is fully written
    // by the transpose loop below before it is read.
    #[allow(clippy::uninit_vec)]
    unsafe {
        concatenated.set_len(total_size)
    };

    // `poly_idx` tracks which poly (column index in cumulative_heights) is the
    // first polynomial of the current matrix.
    let mut poly_idx = 0;
    for rmm in &rmms {
        // Step 2: Bit-reverse the rows (suffix-to-prefix transformation).
        // br.values[i * n_cols + j] = original[bitrev(i)][j]
        let br = rmm.as_view().bit_reverse_rows().to_row_major_matrix();

        let n_cols = br.width();
        let n_rows = br.height();
        let n_cells = n_cols * n_rows;

        // The start position in `concatenated` for this matrix's block of polynomials.
        let start = cumulative_heights[poly_idx];

        // Step 3: Transpose — write each column j of `br` (= one polynomial)
        // into its corresponding contiguous slice in `concatenated`.
        (0..n_cols)
            .into_par_iter()
            .zip(concatenated[start..start + n_cells].par_chunks_mut(n_rows))
            .for_each(|(j, chunk)| {
                br.values
                    .iter()
                    .skip(j)
                    .step_by(n_cols)
                    .zip_eq(chunk.iter_mut())
                    .for_each(|(v, out)| *out = *v);
            });

        poly_idx += n_cols;
    }

    // --- Step 4: Commit via the inner PCS ---
    // q' is committed as a single-column matrix with height = total_size.
    let giga_rmm = RowMajorMatrix::<E::BaseField>::new_by_values(
        concatenated,
        1, // width = 1 (single polynomial q')
        InstancePaddingStrategy::Default,
    );

    let inner = Pcs::batch_commit(pp, vec![giga_rmm])?;

    Ok(JaggedCommitmentWithWitness {
        inner,
        cumulative_heights,
        poly_heights,
    })
}

// ---------------------------------------------------------------------------
// Jagged Sumcheck Prover (M-table streaming algorithm)
// ---------------------------------------------------------------------------

// Streaming sumcheck prover using the M-table algorithm from
// "Time-Space Trade-Offs for Sumcheck" (eprint 2025/1473), Section 4.

/// Default log2 of the maximum epoch size for the streaming phase.
/// Epoch sizes are `[2^0, 2^1, ..., 2^LOG2_MAX_EPOCH]` = `[1, 2, 4, 8]`,
/// covering 1 + 2 + 4 + 8 = 15 streaming rounds before switching to standard sumcheck.
const LOG2_MAX_EPOCH: u32 = 3;

/// All inputs needed for the jagged sumcheck.
pub struct JaggedSumcheckInput<'a, E: ExtensionField> {
    /// Giga polynomial evaluations (concatenated, bit-reversed).
    pub q_evals: &'a [E::BaseField],
    /// n = log2(padded_total_size).
    pub num_giga_vars: usize,
    /// Cumulative height sequence t[j], length num_polys + 1.
    pub cumulative_heights: &'a [usize],
    /// Precomputed eq table for the row evaluation point: `build_eq_x_r_vec(z_row)`.
    pub eq_row: Vec<E>,
    /// Precomputed eq table for the column challenge point: `build_eq_x_r_vec(z_col)`.
    pub eq_col: Vec<E>,
}

/// Iterator that yields `(col, row)` pairs for consecutive giga indices.
/// Uses one binary search at construction, then O(1) per step.
struct ColRowIter<'a> {
    cumulative_heights: &'a [usize],
    col: usize,
    row: usize,
    num_polys: usize,
}

impl<'a> Iterator for ColRowIter<'a> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<(usize, usize)> {
        if self.col >= self.num_polys {
            return None;
        }
        let result = (self.col, self.row);
        self.row += 1;
        let poly_height = self.cumulative_heights[self.col + 1] - self.cumulative_heights[self.col];
        if self.row >= poly_height {
            self.row = 0;
            self.col += 1;
        }
        Some(result)
    }
}

impl<'a, E: ExtensionField> JaggedSumcheckInput<'a, E> {
    fn total_evaluations(&self) -> usize {
        *self.cumulative_heights.last().unwrap_or(&0)
    }

    /// Return an iterator yielding `(col, row)` for consecutive giga indices
    /// starting from `start`. One binary search at construction, O(1) per step.
    fn col_row_iter(&self, start: usize) -> ColRowIter<'_> {
        let num_polys = self.cumulative_heights.len() - 1;
        if start >= self.total_evaluations() {
            return ColRowIter {
                cumulative_heights: self.cumulative_heights,
                col: num_polys,
                row: 0,
                num_polys,
            };
        }
        let j = self.cumulative_heights.partition_point(|&t| t <= start) - 1;
        ColRowIter {
            cumulative_heights: self.cumulative_heights,
            col: j,
            row: start - self.cumulative_heights[j],
            num_polys,
        }
    }

    /// Brute-force computation of sum_b q'(b) * f(b).
    /// O(2^n) time — only for debugging and tests.
    #[cfg(test)]
    fn compute_claimed_sum(&self) -> E {
        self.q_evals[..self.total_evaluations()]
            .iter()
            .zip(self.col_row_iter(0))
            .fold(E::ZERO, |sum, (&q, (col, row))| {
                sum + (self.eq_row[row] * self.eq_col[col]) * q
            })
    }

    /// Brute-force MLE evaluation of q'(rb) and f(rb) at the given point.
    /// O(2^n) time and memory — only for debugging and tests.
    /// Returns `(q_at_point, f_at_point)`.
    #[cfg(test)]
    fn final_evaluations_slow(&self, point: &[E]) -> (E, E) {
        let n = self.num_giga_vars;
        let total_evals = self.total_evaluations();

        // Build q' MLE (padded with zeros) and evaluate at point.
        let mut q_padded: Vec<E::BaseField> = self.q_evals.to_vec();
        q_padded.resize(1 << n, Default::default());
        let q_mle = MultilinearExtension::from_evaluations_vec(n, q_padded);
        let q_at_point = q_mle.evaluate(point);

        // Build f MLE from eq tables and evaluate at point.
        let mut f_evals = vec![E::ZERO; 1 << n];
        for (f_eval, (col, row)) in f_evals[..total_evals].iter_mut().zip(self.col_row_iter(0)) {
            *f_eval = self.eq_row[row] * self.eq_col[col];
        }
        let f_mle = MultilinearExtension::from_evaluations_ext_vec(n, f_evals);
        let f_at_point = f_mle.evaluate(point);

        (q_at_point, f_at_point)
    }
}

/// Run the full jagged sumcheck: streaming phase (rounds 1..K) + standard phase (rounds K+1..n).
///
/// `log2_max_epoch` controls the streaming phase epoch schedule. Epoch sizes are
/// `[1, 2, 4, ..., 2^k]` where `k = log2_max_epoch`. Pass `None` to use the default
/// `LOG2_MAX_EPOCH = 3`, giving epoch sizes `[1, 2, 4, 8]` and 15 streaming rounds.
///
/// Returns the proof and the full list of challenges (r_1, ..., r_n).
pub fn jagged_sumcheck_prove<E: ExtensionField>(
    input: &JaggedSumcheckInput<E>,
    transcript: &mut impl Transcript<E>,
    log2_max_epoch: Option<u32>,
) -> (IOPProof<E>, Vec<E>) {
    let n = input.num_giga_vars;
    let max_degree: usize = 2;

    let k = log2_max_epoch.unwrap_or(LOG2_MAX_EPOCH);
    let epoch_sizes: Vec<usize> = (0..=k).map(|i| 1usize << i).collect();

    let mut challenges: Vec<E> = Vec::with_capacity(n);
    let mut proof_messages: Vec<IOPProverMessage<E>> = Vec::with_capacity(n);

    // Write transcript header (must match verifier's expectations).
    transcript.append_message(&n.to_le_bytes());
    transcript.append_message(&max_degree.to_le_bytes());

    // --- Streaming phase: epochs j' = 1, 2, 4, ..., 2^k ---
    for &epoch_size in &epoch_sizes {
        // Epoch j' handles rounds j'..2j'-1. Skip if all rounds are done.
        if epoch_size > n {
            break;
        }

        // Build M-table for this epoch.
        let span = entered_span!("build_m_table", epoch = epoch_size);
        let m_table = build_m_table(input, &challenges, epoch_size);
        exit_span!(span);

        // Extract rounds j = epoch_size .. min(2*epoch_size - 1, n)
        let span = entered_span!("compute_rounds_from_m", epoch = epoch_size);
        for j in epoch_size..(2 * epoch_size).min(n + 1) {
            let d = j - epoch_size; // intra-epoch offset
            let intra_challenges = challenges[epoch_size - 1..epoch_size - 1 + d].to_vec();

            let [_h0, h1, h2] = compute_round_from_m(&m_table, epoch_size, &intra_challenges);

            // Append [h(1), h(2)] to transcript and sample challenge.
            transcript.append_field_element_ext(&h1);
            transcript.append_field_element_ext(&h2);
            let challenge = transcript
                .sample_and_append_challenge(b"Internal round")
                .elements;

            proof_messages.push(IOPProverMessage {
                evaluations: vec![h1, h2],
            });
            challenges.push(challenge);
        }
        exit_span!(span);
    }

    // --- Phase 2: Bind and materialize, then standard sumcheck ---
    let k = challenges.len(); // actual number of streaming rounds completed
    if k < n {
        let span = entered_span!("bind_and_materialize");
        let (q_bound, f_bound) = bind_and_materialize(input, &challenges);
        exit_span!(span);

        let remaining_vars = n - k;
        let q_mle = MultilinearExtension::from_evaluations_ext_vec(remaining_vars, q_bound);
        let f_mle = MultilinearExtension::from_evaluations_ext_vec(remaining_vars, f_bound);

        // Use VirtualPolynomial + round-by-round proving (no extra transcript header).
        use multilinear_extensions::virtual_poly::VirtualPolynomial;
        use std::sync::Arc;
        let q_arc = Arc::new(q_mle);
        let f_arc = Arc::new(f_mle);
        let vp = VirtualPolynomial::new_from_product(vec![q_arc, f_arc], E::ONE);

        let span = entered_span!("standard_sumcheck", rounds = remaining_vars);
        let mut prover_state =
            IOPProverState::prover_init_with_extrapolation_aux(true, vp, None, None);
        let mut challenge = None;
        for _ in 0..remaining_vars {
            let prover_msg =
                IOPProverState::prove_round_and_update_state(&mut prover_state, &challenge);
            prover_msg
                .evaluations
                .iter()
                .for_each(|e| transcript.append_field_element_ext(e));
            challenge = Some(transcript.sample_and_append_challenge(b"Internal round"));
            challenges.push(challenge.unwrap().elements);
            proof_messages.push(prover_msg);
        }
        exit_span!(span);
    }

    (
        IOPProof {
            proofs: proof_messages,
        },
        challenges,
    )
}

/// Build M-table for epoch j'.
///
/// M[beta1 * 2^{j'} + beta2] = sum_b Q_bound(beta1, b) * F_bound(beta2, b)
///
/// where:
/// - Q_bound(beta, b) = sum_{a in {0,1}^{j'-1}} eq(R, a) * q'[a || beta || b]
/// - F_bound(beta, b) = sum_{a in {0,1}^{j'-1}} eq(R, a) * f[a || beta || b]
fn build_m_table<E: ExtensionField>(
    input: &JaggedSumcheckInput<E>,
    challenges: &[E],  // R_{j'} = (r_1, ..., r_{j'-1})
    epoch_size: usize, // j'
) -> Vec<E> {
    let n = input.num_giga_vars;
    let bound_vars = epoch_size - 1; // j' - 1

    let eq_r = if bound_vars > 0 {
        build_eq_x_r_vec(&challenges[..bound_vars])
    } else {
        vec![E::ONE]
    };

    let beta_count = 1usize << epoch_size; // 2^{j'}
    let a_count = 1usize << bound_vars; // 2^{j'-1}
    let chunk_size = a_count * beta_count; // 2^{2j'-1}
    // When n < 2j'-1, all variables fit in a single chunk (no b dimension).
    let n_chunks = 1usize << n.saturating_sub(2 * epoch_size - 1); // 2^{max(0, n - 2j' + 1)}

    let m_size = beta_count * beta_count; // 2^{2j'}

    // Step 1: Each thread processes a batch of b-chunks, producing a local M-table.
    let span = entered_span!(
        "streaming_pass",
        n_chunks = n_chunks,
        beta_count = beta_count
    );
    let indices: Vec<usize> = (0..n_chunks).collect();
    let n_threads = max_usable_threads();
    let batch_size = (n_chunks / n_threads).max(1);
    let partial_tables: Vec<Vec<E>> = indices
        .par_chunks(batch_size)
        .map(|batch| {
            let mut local_m = vec![E::ZERO; m_size];
            let mut q_bound = vec![E::ZERO; beta_count];
            let mut f_bound = vec![E::ZERO; beta_count];

            for &b_idx in batch {
                let chunk_start = b_idx * chunk_size;
                q_bound
                    .iter_mut()
                    .zip(f_bound.iter_mut())
                    .enumerate()
                    .for_each(|(beta, (q_b, f_b))| {
                        let base = chunk_start + beta * a_count;
                        let (q_acc, f_acc) = eq_r
                            .iter()
                            .zip(input.q_evals.get(base..).unwrap_or(&[]))
                            .zip(input.col_row_iter(base))
                            .fold(
                                (E::ZERO, E::ZERO),
                                |(q_acc, f_acc), ((&eq_r_a, &q), (col, row))| {
                                    (
                                        q_acc + eq_r_a * q,
                                        f_acc + eq_r_a * (input.eq_row[row] * input.eq_col[col]),
                                    )
                                },
                            );
                        *q_b = q_acc;
                        *f_b = f_acc;
                    });

                // Outer product accumulation into local M-table.
                for b1 in 0..beta_count {
                    if q_bound[b1] == E::ZERO {
                        continue;
                    }
                    for b2 in 0..beta_count {
                        local_m[b1 * beta_count + b2] += q_bound[b1] * f_bound[b2];
                    }
                }
            }
            local_m
        })
        .collect();
    exit_span!(span);

    // Step 2: Sum partial M-tables in parallel, each thread handles a slice of cells.
    let span = entered_span!("reduce_partial_tables", n_partials = partial_tables.len());
    let n_partials = partial_tables.len();
    if n_partials == 0 {
        exit_span!(span);
        return vec![E::ZERO; m_size];
    }
    let mut m_table = partial_tables[0].clone();
    let cell_batch = (m_size / n_threads).max(1);
    m_table
        .par_chunks_mut(cell_batch)
        .enumerate()
        .for_each(|(ci, cells)| {
            let start = ci * cell_batch;
            for partial in &partial_tables[1..] {
                for (j, cell) in cells.iter_mut().enumerate() {
                    *cell += partial[start + j];
                }
            }
        });
    exit_span!(span);
    m_table
}

/// Extract round univariate h_j(x) from M-table.
///
/// For round j in epoch j', d = j - j' intra-epoch challenges have been collected.
/// Returns [h(0), h(1), h(2)].
fn compute_round_from_m<E: ExtensionField>(
    m_table: &[E],
    epoch_size: usize,      // j'
    intra_challenges: &[E], // r_{j'}, ..., r_{j-1} (d elements)
) -> [E; 3] {
    let d = intra_challenges.len();
    let beta_count = 1usize << epoch_size;
    let pad_bits = epoch_size - d - 1; // number of "future" bits to sum over
    let pad_count = 1usize << pad_bits;

    let eq_intra = if d > 0 {
        build_eq_x_r_vec(intra_challenges)
    } else {
        vec![E::ONE]
    };

    let a_count = 1usize << d; // 2^d

    let mut h = [E::ZERO; 3];

    for a in 0..a_count {
        for c in 0..a_count {
            let eq_weight = eq_intra[a] * eq_intra[c];
            if eq_weight == E::ZERO {
                continue;
            }

            // Sum over all pad bit assignments (same pad for both beta1/beta2
            // since they correspond to the same physical "future" variables).
            for p in 0..pad_count {
                // beta = a_bits || x_bit || pad_bits  (little-endian)
                // beta_val = a + x_bit * 2^d + pad * 2^{d+1}
                let base1 = a + (p << (d + 1));
                let base2 = c + (p << (d + 1));
                let b1_0 = base1; // x=0
                let b1_1 = base1 + (1 << d); // x=1
                let b2_0 = base2;
                let b2_1 = base2 + (1 << d);

                let m00 = m_table[b1_0 * beta_count + b2_0];
                let m10 = m_table[b1_1 * beta_count + b2_0];
                let m01 = m_table[b1_0 * beta_count + b2_1];
                let m11 = m_table[b1_1 * beta_count + b2_1];

                h[0] += eq_weight * m00;
                h[1] += eq_weight * m11;
                // h(2) via bilinear: (1-2)^2*M00 + 2(1-2)*M10 + (1-2)*2*M01 + 4*M11
                h[2] += eq_weight * (m00 - m10.double() - m01.double() + m11.double().double());
            }
        }
    }

    h
}

/// Bind first K variables and materialize reduced q' and f as extension-field vectors.
///
/// q_bound[idx] = sum_{a in {0,1}^K} eq(R, a) * q'[a + idx * 2^K]
/// f_bound[idx] = sum_{a in {0,1}^K} eq(R, a) * f[a + idx * 2^K]
fn bind_and_materialize<E: ExtensionField>(
    input: &JaggedSumcheckInput<E>,
    challenges: &[E], // R_K = (r_1, ..., r_K)
) -> (Vec<E>, Vec<E>) {
    let n = input.num_giga_vars;
    let k = challenges.len();
    let remaining_size = 1usize << (n - k);
    let a_count = 1usize << k;

    let eq_r = build_eq_x_r_vec(challenges);

    // Each output index is independent — parallelize over idx.
    let results: Vec<(E, E)> = (0..remaining_size)
        .into_par_iter()
        .map(|idx| {
            let base = idx * a_count;
            eq_r.iter()
                .zip(input.q_evals.get(base..).unwrap_or(&[]))
                .zip(input.col_row_iter(base))
                .fold(
                    (E::ZERO, E::ZERO),
                    |(q_acc, f_acc), ((&eq_r_a, &q), (col, row))| {
                        (
                            q_acc + eq_r_a * q,
                            f_acc + eq_r_a * (input.eq_row[row] * input.eq_col[col]),
                        )
                    },
                )
        })
        .collect();

    results.into_iter().unzip()
}

// ---------------------------------------------------------------------------
// Jagged Batch Open / Verify
// ---------------------------------------------------------------------------

/// Proof for the jagged batch opening protocol.
///
/// Contains a sumcheck proof (reducing K column evaluation claims to a single
/// point on q'), the evaluation q'(ρ), and an inner PCS opening proof.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "", deserialize = ""))]
pub struct JaggedBatchOpenProof<E: ExtensionField, Pcs: PolynomialCommitmentScheme<E>> {
    pub sumcheck_proof: IOPProof<E>,
    pub q_eval: E,
    pub inner_proof: Pcs::Proof,
}

/// Convert a `usize` to its little-endian binary representation as field elements.
fn int_to_field_bits<E: ExtensionField>(val: usize, num_bits: usize) -> Vec<E> {
    (0..num_bits)
        .map(|i| if (val >> i) & 1 == 1 { E::ONE } else { E::ZERO })
        .collect()
}

/// Evaluate f̂(ρ) using the ROBP evaluator.
///
/// f̂(ρ) = Σ_{y} eq_col[y] · ĝ(z_row_padded, ρ, bits(t_y), bits(t_{y+1}))
fn compute_f_at_point<E: ExtensionField>(
    z_row_padded: &[E],
    rho_padded: &[E],
    eq_col: &[E],
    cumulative_heights: &[usize],
    n_robp: usize,
) -> E {
    let num_polys = cumulative_heights.len() - 1;
    let mut f_val = E::ZERO;
    for y in 0..num_polys {
        if eq_col[y] == E::ZERO {
            continue;
        }
        let t_lo = int_to_field_bits::<E>(cumulative_heights[y], n_robp);
        let t_hi = int_to_field_bits::<E>(cumulative_heights[y + 1], n_robp);
        let g_val = evaluate_g(z_row_padded, rho_padded, &t_lo, &t_hi);
        f_val += eq_col[y] * g_val;
    }
    f_val
}

/// Prove that evaluation claims `evals[i] = p_i(point)` are consistent with a
/// jagged commitment.
///
/// All polynomials must have the same height. `point` is the common evaluation
/// point (a suffix `r[s..]` of the GKR challenge, length `s = log2(poly_height)`).
///
/// The protocol:
/// 1. Batch the K column claims via a random column challenge `z_col`.
/// 2. Run the jagged sumcheck to reduce to a single evaluation of q'.
/// 3. Open q' at the sumcheck output point via the inner PCS.
pub fn jagged_batch_open<E: ExtensionField, Pcs: PolynomialCommitmentScheme<E>>(
    pp: &Pcs::ProverParam,
    comm: &JaggedCommitmentWithWitness<E, Pcs>,
    point: &[E],
    evals: &[E],
    transcript: &mut impl Transcript<E>,
) -> Result<JaggedBatchOpenProof<E, Pcs>, Error> {
    let num_polys = comm.num_polys();
    if evals.len() != num_polys {
        return Err(Error::InvalidPcsParam(format!(
            "jagged_batch_open: expected {} evals, got {}",
            num_polys,
            evals.len()
        )));
    }

    let s = point.len();
    let expected_height = 1usize << s;
    for (i, &h) in comm.poly_heights.iter().enumerate() {
        if h != expected_height {
            return Err(Error::InvalidPcsParam(format!(
                "jagged_batch_open: poly {} has height {}, expected {}",
                i, h, expected_height
            )));
        }
    }

    let total_evals = comm.total_evaluations();
    let num_giga_vars = total_evals.next_power_of_two().trailing_zeros() as usize;

    // z_row = reverse(point) to account for bit-reversal in jagged_commit.
    let z_row: Vec<E> = point.iter().rev().cloned().collect();

    // Write evals to transcript, then sample z_col.
    transcript.append_field_element_exts(evals);
    let num_col_vars = (num_polys.next_power_of_two().trailing_zeros() as usize).max(1);
    let z_col: Vec<E> = transcript.sample_and_append_vec(b"jagged_z_col", num_col_vars);

    let eq_col = build_eq_x_r_vec(&z_col);

    // Extract q' base-field evaluations from the inner PCS witness.
    let q_mles = Pcs::get_arc_mle_witness_from_commitment(&comm.inner);
    assert_eq!(
        q_mles.len(),
        1,
        "jagged commit produces exactly one polynomial"
    );
    let q_mle = &q_mles[0];
    let q_evals_base: &[E::BaseField] = match q_mle.evaluations() {
        FieldType::Base(slice) => slice,
        _ => {
            return Err(Error::InvalidPcsParam(
                "jagged_batch_open: expected base-field evaluations for q'".into(),
            ));
        }
    };

    // Run the jagged sumcheck.
    let eq_row = build_eq_x_r_vec(&z_row);
    let input = JaggedSumcheckInput {
        q_evals: q_evals_base,
        num_giga_vars,
        cumulative_heights: &comm.cumulative_heights,
        eq_row,
        eq_col,
    };
    let (sumcheck_proof, challenges) = jagged_sumcheck_prove(&input, transcript, None);

    // Evaluate q'(ρ).
    let q_eval = q_mle.evaluate(&challenges);

    // Write q_eval to transcript before inner PCS open.
    transcript.append_field_element_ext(&q_eval);

    // Open q' at ρ via inner PCS batch_open.
    let inner_proof = Pcs::batch_open(
        pp,
        vec![(&comm.inner, vec![(challenges, vec![q_eval])])],
        transcript,
    )?;

    Ok(JaggedBatchOpenProof {
        sumcheck_proof,
        q_eval,
        inner_proof,
    })
}

/// Verify that evaluation claims `evals[i] = p_i(point)` are consistent with a
/// jagged commitment.
pub fn jagged_batch_verify<E: ExtensionField, Pcs: PolynomialCommitmentScheme<E>>(
    vp: &Pcs::VerifierParam,
    comm: &JaggedCommitment<E, Pcs>,
    point: &[E],
    evals: &[E],
    proof: &JaggedBatchOpenProof<E, Pcs>,
    transcript: &mut impl Transcript<E>,
) -> Result<(), Error> {
    let num_polys = comm.cumulative_heights.len() - 1;
    if evals.len() != num_polys {
        return Err(Error::InvalidPcsOpen(format!(
            "jagged_batch_verify: expected {} evals, got {}",
            num_polys,
            evals.len()
        )));
    }

    let total_evals = *comm.cumulative_heights.last().unwrap();
    let num_giga_vars = total_evals.next_power_of_two().trailing_zeros() as usize;

    // z_row = reverse(point)
    let z_row: Vec<E> = point.iter().rev().cloned().collect();

    // Replay transcript: write evals, sample z_col.
    transcript.append_field_element_exts(evals);
    let num_col_vars = (num_polys.next_power_of_two().trailing_zeros() as usize).max(1);
    let z_col: Vec<E> = transcript.sample_and_append_vec(b"jagged_z_col", num_col_vars);

    // claimed_sum = Σ_j eq_col[j] · evals[j]
    let eq_col = build_eq_x_r_vec(&z_col);
    let claimed_sum: E = eq_col[..num_polys]
        .iter()
        .zip(evals.iter())
        .map(|(&eq, &ev)| eq * ev)
        .sum();

    // Verify the jagged sumcheck.
    let aux_info = VPAuxInfo {
        max_degree: 2,
        max_num_variables: num_giga_vars,
        phantom: PhantomData::<E>,
    };
    let subclaim =
        IOPVerifierState::<E>::verify(claimed_sum, &proof.sumcheck_proof, &aux_info, transcript);
    let rho: Vec<E> = subclaim.point.iter().map(|c| c.elements).collect();

    // The ROBP needs enough bits to represent the max cumulative height (= total_evals).
    // When total_evals is an exact power of 2, num_giga_vars bits can't represent it.
    let n_robp = num_giga_vars + if total_evals.is_power_of_two() { 1 } else { 0 };

    // Compute f̂(ρ) via the ROBP evaluator.
    let mut z_row_padded = z_row;
    z_row_padded.resize(n_robp, E::ZERO);
    let mut rho_padded = rho.clone();
    rho_padded.resize(n_robp, E::ZERO);
    let f_at_rho = compute_f_at_point(
        &z_row_padded,
        &rho_padded,
        &eq_col,
        &comm.cumulative_heights,
        n_robp,
    );

    // Write q_eval to transcript (must match prover).
    transcript.append_field_element_ext(&proof.q_eval);

    // Check subclaim: q'(ρ) · f̂(ρ) == expected_evaluation.
    if proof.q_eval * f_at_rho != subclaim.expected_evaluation {
        return Err(Error::InvalidPcsOpen(
            "jagged_batch_verify: q_eval * f(rho) != subclaim expected evaluation".into(),
        ));
    }

    // Verify the inner PCS opening.
    Pcs::batch_verify(
        vp,
        vec![(
            comm.inner.clone(),
            vec![(num_giga_vars, (rho, vec![proof.q_eval]))],
        )],
        &proof.inner_proof,
        transcript,
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        basefold::{Basefold, BasefoldRSParams},
        test_util::setup_pcs,
    };
    use ff_ext::GoldilocksExt2;
    use p3::{field::FieldAlgebra, goldilocks::Goldilocks};

    type F = Goldilocks;
    type E = GoldilocksExt2;
    type Pcs = Basefold<E, BasefoldRSParams>;

    fn make_rmm(num_rows: usize, num_cols: usize) -> RowMajorMatrix<F> {
        let values: Vec<F> = (0..num_rows * num_cols)
            .map(|i| F::from_canonical_u64(i as u64 + 1))
            .collect();
        RowMajorMatrix::<F>::new_by_values(values, num_cols, InstancePaddingStrategy::Default)
    }

    #[test]
    fn test_cumulative_heights_single_matrix() {
        // 4x2 matrix → 2 polynomials with 4 evaluations each → cumulative = [0, 4, 8]
        let rmm = make_rmm(4, 2);
        let num_rows = rmm.height();
        let num_cols = rmm.width();
        let mut poly_heights = Vec::new();
        for _ in 0..num_cols {
            poly_heights.push(num_rows);
        }
        let mut ch = vec![0usize];
        for &h in &poly_heights {
            ch.push(ch.last().unwrap() + h);
        }
        assert_eq!(poly_heights, vec![4, 4]);
        assert_eq!(ch, vec![0, 4, 8]);
    }

    #[test]
    fn test_cumulative_heights_multiple_matrices() {
        // 4x1 + 8x2 → heights [4, 8, 8] → cumulative [0, 4, 12, 20]
        let m1 = make_rmm(4, 1);
        let m2 = make_rmm(8, 2);
        let mut poly_heights: Vec<usize> = Vec::new();
        for rmm in &[m1, m2] {
            for _ in 0..rmm.width() {
                poly_heights.push(rmm.height());
            }
        }
        let mut ch = vec![0usize];
        for &h in &poly_heights {
            ch.push(ch.last().unwrap() + h);
        }
        assert_eq!(poly_heights, vec![4, 8, 8]);
        assert_eq!(ch, vec![0, 4, 12, 20]);
    }

    #[test]
    fn test_jagged_commit_smoke() {
        // Two matrices: 4x1 and 4x2 → 3 polynomials, total 12 evals, padded to 16
        let (pp, _vp) = setup_pcs::<E, Pcs>(4);
        let m1 = make_rmm(4, 1);
        let m2 = make_rmm(4, 2);

        let comm = jagged_commit::<E, Pcs>(&pp, vec![m1, m2]).expect("commit should succeed");

        assert_eq!(comm.num_polys(), 3);
        assert_eq!(comm.poly_heights, vec![4, 4, 4]);
        assert_eq!(comm.cumulative_heights, vec![0, 4, 8, 12]);
        assert_eq!(comm.total_evaluations(), 12);

        let pure = comm.to_commitment();
        assert_eq!(pure.cumulative_heights, vec![0, 4, 8, 12]);
    }

    #[test]
    fn test_jagged_commit_single_poly() {
        // 8x1 matrix → 1 polynomial, 8 evals, no padding needed
        let (pp, _vp) = setup_pcs::<E, Pcs>(3);
        let m = make_rmm(8, 1);

        let comm = jagged_commit::<E, Pcs>(&pp, vec![m]).expect("commit should succeed");

        assert_eq!(comm.num_polys(), 1);
        assert_eq!(comm.poly_heights, vec![8]);
        assert_eq!(comm.cumulative_heights, vec![0, 8]);
        assert_eq!(comm.total_evaluations(), 8);
    }

    #[test]
    fn test_jagged_commit_empty_error() {
        let (pp, _vp) = setup_pcs::<E, Pcs>(4);
        let result = jagged_commit::<E, Pcs>(&pp, vec![]);
        assert!(matches!(result, Err(Error::InvalidPcsParam(_))));
    }

    // --- Sumcheck tests ---

    use multilinear_extensions::virtual_poly::{VPAuxInfo, build_eq_x_r_vec};
    use rand::thread_rng;
    use std::marker::PhantomData;
    use sumcheck::structs::IOPVerifierState;
    use transcript::basic::BasicTranscript;

    #[test]
    fn test_jagged_sumcheck_small() {
        use ff_ext::FromUniformBytes;

        let mut rng = thread_rng();

        // 3 polynomials of height 4 (s=2), total 12 evals, padded to 16 (n=4).
        let num_polys = 3usize;
        let poly_height = 4usize;
        let s = 2; // log2(poly_height)
        let total_evals = num_polys * poly_height;
        let num_giga_vars = 4; // ceil(log2(12)) = 4, 2^4 = 16

        let q_evals: Vec<F> = (0..total_evals)
            .map(|i| F::from_canonical_u64(i as u64 + 1))
            .collect();

        let cumulative_heights: Vec<usize> = (0..=num_polys).map(|i| i * poly_height).collect();

        let z_row: Vec<E> = (0..s).map(|_| E::random(&mut rng)).collect();
        // z_col needs ceil(log2(num_polys)) = 2 variables
        let z_col: Vec<E> = (0..2).map(|_| E::random(&mut rng)).collect();

        let input = JaggedSumcheckInput {
            q_evals: &q_evals,
            num_giga_vars,
            cumulative_heights: &cumulative_heights,
            eq_row: build_eq_x_r_vec(&z_row),
            eq_col: build_eq_x_r_vec(&z_col),
        };

        let claimed_sum = input.compute_claimed_sum();

        let mut transcript = BasicTranscript::<E>::new(b"jagged_sumcheck_test");
        let (proof, challenges) = jagged_sumcheck_prove(&input, &mut transcript, None);

        assert_eq!(proof.proofs.len(), num_giga_vars);
        assert_eq!(challenges.len(), num_giga_vars);

        // Verify using the standard sumcheck verifier.
        let mut transcript_v = BasicTranscript::<E>::new(b"jagged_sumcheck_test");
        let aux_info = VPAuxInfo {
            max_degree: 2,
            max_num_variables: num_giga_vars,
            phantom: PhantomData::<E>,
        };
        let subclaim =
            IOPVerifierState::<E>::verify(claimed_sum, &proof, &aux_info, &mut transcript_v);

        // The subclaim point should match our challenges.
        for (sc, ch) in subclaim.point.iter().zip(challenges.iter()) {
            assert_eq!(sc.elements, *ch);
        }

        // Verify the final evaluation: q'(point) * f(point) == expected_evaluation
        let (q_at_point, f_at_point) = input.final_evaluations_slow(&challenges);
        assert_eq!(
            q_at_point * f_at_point,
            subclaim.expected_evaluation,
            "final evaluation mismatch"
        );
    }

    #[test]
    fn test_jagged_sumcheck_all_epochs() {
        // n=16: exercises all 4 epochs (j'=1,2,4,8) + 1 round of standard sumcheck.
        use ff_ext::FromUniformBytes;

        let mut rng = thread_rng();

        let num_polys = 8usize;
        let poly_height = 1 << 13; // 8192, s=13
        let s = 13;
        let total_evals = num_polys * poly_height; // 65536
        let num_giga_vars = 16; // 2^16 = 65536

        let q_evals: Vec<F> = (0..total_evals)
            .map(|i| F::from_canonical_u64((i as u64 * 7 + 3) % (1 << 20)))
            .collect();

        let cumulative_heights: Vec<usize> = (0..=num_polys).map(|i| i * poly_height).collect();

        let z_row: Vec<E> = (0..s).map(|_| E::random(&mut rng)).collect();
        let z_col: Vec<E> = (0..3).map(|_| E::random(&mut rng)).collect(); // ceil(log2(8))=3

        let input = JaggedSumcheckInput {
            q_evals: &q_evals,
            num_giga_vars,
            cumulative_heights: &cumulative_heights,
            eq_row: build_eq_x_r_vec(&z_row),
            eq_col: build_eq_x_r_vec(&z_col),
        };

        let claimed_sum = input.compute_claimed_sum();

        let mut transcript = BasicTranscript::<E>::new(b"jagged_test_16");
        let (proof, challenges) = jagged_sumcheck_prove(&input, &mut transcript, None);

        assert_eq!(proof.proofs.len(), num_giga_vars);
        assert_eq!(challenges.len(), num_giga_vars);

        let mut transcript_v = BasicTranscript::<E>::new(b"jagged_test_16");
        let aux_info = VPAuxInfo {
            max_degree: 2,
            max_num_variables: num_giga_vars,
            phantom: PhantomData::<E>,
        };
        let subclaim =
            IOPVerifierState::<E>::verify(claimed_sum, &proof, &aux_info, &mut transcript_v);

        for (sc, ch) in subclaim.point.iter().zip(challenges.iter()) {
            assert_eq!(sc.elements, *ch);
        }
    }

    #[test]
    fn test_jagged_sumcheck_n25() {
        // n=25: 2^25 = 33M evaluations. Exercises all epochs + 10 rounds of standard sumcheck.
        use ff_ext::FromUniformBytes;

        tracing_forest::init();

        let mut rng = thread_rng();

        let num_polys = 1 << 10; // 1024 polynomials
        let poly_height = 1 << 15; // 32768 each, s=15
        let s = 15;
        let total_evals = num_polys * poly_height; // 2^25 = 33554432
        let num_giga_vars = 25;

        let q_evals: Vec<F> = (0..total_evals)
            .map(|i| F::from_canonical_u64((i as u64 * 13 + 7) % (1 << 30)))
            .collect();

        let cumulative_heights: Vec<usize> = (0..=num_polys).map(|i| i * poly_height).collect();

        let z_row: Vec<E> = (0..s).map(|_| E::random(&mut rng)).collect();
        let z_col: Vec<E> = (0..10).map(|_| E::random(&mut rng)).collect(); // ceil(log2(1024))=10

        let input = JaggedSumcheckInput {
            q_evals: &q_evals,
            num_giga_vars,
            cumulative_heights: &cumulative_heights,
            eq_row: build_eq_x_r_vec(&z_row),
            eq_col: build_eq_x_r_vec(&z_col),
        };

        let claimed_sum = input.compute_claimed_sum();

        let mut transcript = BasicTranscript::<E>::new(b"jagged_test_25");
        let (proof, challenges) = jagged_sumcheck_prove(&input, &mut transcript, None);

        assert_eq!(proof.proofs.len(), num_giga_vars);
        assert_eq!(challenges.len(), num_giga_vars);

        let mut transcript_v = BasicTranscript::<E>::new(b"jagged_test_25");
        let aux_info = VPAuxInfo {
            max_degree: 2,
            max_num_variables: num_giga_vars,
            phantom: PhantomData::<E>,
        };
        let subclaim =
            IOPVerifierState::<E>::verify(claimed_sum, &proof, &aux_info, &mut transcript_v);

        for (sc, ch) in subclaim.point.iter().zip(challenges.iter()) {
            assert_eq!(sc.elements, *ch);
        }

        // Verify the final evaluation: q'(point) * f(point) == expected_evaluation
        let (q_at_point, f_at_point) = input.final_evaluations_slow(&challenges);
        assert_eq!(
            q_at_point * f_at_point,
            subclaim.expected_evaluation,
            "final evaluation mismatch"
        );
    }

    // --- Batch open/verify tests ---

    use ff_ext::FromUniformBytes;

    /// Evaluate a single column polynomial at `point` (the original, non-bit-reversed poly).
    /// `col_evals` are the raw evaluations of the column (before bit-reversal).
    fn eval_column_poly_at_point(col_evals: &[F], point: &[E]) -> E {
        let s = point.len();
        assert_eq!(col_evals.len(), 1 << s);
        let mle = MultilinearExtension::from_evaluations_vec(s, col_evals.to_vec());
        mle.evaluate(point)
    }

    #[test]
    fn test_jagged_batch_open_verify_small() {
        let mut rng = thread_rng();

        let num_rows = 1024usize; // s=10
        let num_cols = 3usize;
        let s = 10;
        let total_evals = num_rows * num_cols;
        let num_giga_vars = total_evals.next_power_of_two().trailing_zeros() as usize;

        let (pp, vp) = setup_pcs::<E, Pcs>(num_giga_vars);
        let rmm = make_rmm(num_rows, num_cols);

        // Extract column polynomials (before bit-reversal) for computing true evaluations.
        let col_polys: Vec<Vec<F>> = (0..num_cols)
            .map(|c| {
                (0..num_rows)
                    .map(|r| rmm.values[r * num_cols + c])
                    .collect()
            })
            .collect();

        // Commit.
        let mut transcript_p = BasicTranscript::<E>::new(b"jagged_batch_test");
        let comm = jagged_commit::<E, Pcs>(&pp, vec![rmm]).expect("commit should succeed");
        Pcs::write_commitment(&comm.to_commitment().inner, &mut transcript_p).unwrap();

        // Random evaluation point of length s.
        let point: Vec<E> = (0..s).map(|_| E::random(&mut rng)).collect();

        // Compute true evaluations.
        let evals: Vec<E> = col_polys
            .iter()
            .map(|col| eval_column_poly_at_point(col, &point))
            .collect();

        // Prover: batch open.
        let proof = jagged_batch_open::<E, Pcs>(&pp, &comm, &point, &evals, &mut transcript_p)
            .expect("batch open should succeed");

        // Verifier: batch verify.
        let mut transcript_v = BasicTranscript::<E>::new(b"jagged_batch_test");
        Pcs::write_commitment(&comm.to_commitment().inner, &mut transcript_v).unwrap();

        jagged_batch_verify::<E, Pcs>(
            &vp,
            &comm.to_commitment(),
            &point,
            &evals,
            &proof,
            &mut transcript_v,
        )
        .expect("batch verify should succeed");
    }

    #[test]
    fn test_jagged_batch_open_verify_single_poly() {
        let mut rng = thread_rng();

        let num_rows = 1024usize; // s=10
        let num_cols = 1usize;
        let s = 10;
        let num_giga_vars = (num_rows * num_cols).next_power_of_two().trailing_zeros() as usize;

        let (pp, vp) = setup_pcs::<E, Pcs>(num_giga_vars);
        let rmm = make_rmm(num_rows, num_cols);

        let col_poly: Vec<F> = (0..num_rows).map(|r| rmm.values[r]).collect();

        let mut transcript_p = BasicTranscript::<E>::new(b"jagged_single");
        let comm = jagged_commit::<E, Pcs>(&pp, vec![rmm]).expect("commit");
        Pcs::write_commitment(&comm.to_commitment().inner, &mut transcript_p).unwrap();

        let point: Vec<E> = (0..s).map(|_| E::random(&mut rng)).collect();
        let evals = vec![eval_column_poly_at_point(&col_poly, &point)];

        let proof = jagged_batch_open::<E, Pcs>(&pp, &comm, &point, &evals, &mut transcript_p)
            .expect("batch open");

        let mut transcript_v = BasicTranscript::<E>::new(b"jagged_single");
        Pcs::write_commitment(&comm.to_commitment().inner, &mut transcript_v).unwrap();

        jagged_batch_verify::<E, Pcs>(
            &vp,
            &comm.to_commitment(),
            &point,
            &evals,
            &proof,
            &mut transcript_v,
        )
        .expect("batch verify");
    }

    #[test]
    fn test_jagged_batch_open_verify_soundness() {
        let mut rng = thread_rng();

        let num_rows = 1024usize;
        let num_cols = 3usize;
        let s = 10;
        let num_giga_vars = (num_rows * num_cols).next_power_of_two().trailing_zeros() as usize;

        let (pp, vp) = setup_pcs::<E, Pcs>(num_giga_vars);
        let rmm = make_rmm(num_rows, num_cols);

        let col_polys: Vec<Vec<F>> = (0..num_cols)
            .map(|c| {
                (0..num_rows)
                    .map(|r| rmm.values[r * num_cols + c])
                    .collect()
            })
            .collect();

        let mut transcript_p = BasicTranscript::<E>::new(b"jagged_soundness");
        let comm = jagged_commit::<E, Pcs>(&pp, vec![rmm]).expect("commit");
        Pcs::write_commitment(&comm.to_commitment().inner, &mut transcript_p).unwrap();

        let point: Vec<E> = (0..s).map(|_| E::random(&mut rng)).collect();
        let mut evals: Vec<E> = col_polys
            .iter()
            .map(|col| eval_column_poly_at_point(col, &point))
            .collect();

        // Tamper with one evaluation.
        evals[1] += E::ONE;

        let proof = jagged_batch_open::<E, Pcs>(&pp, &comm, &point, &evals, &mut transcript_p)
            .expect("batch open with wrong evals still produces a proof");

        let mut transcript_v = BasicTranscript::<E>::new(b"jagged_soundness");
        Pcs::write_commitment(&comm.to_commitment().inner, &mut transcript_v).unwrap();

        let result = jagged_batch_verify::<E, Pcs>(
            &vp,
            &comm.to_commitment(),
            &point,
            &evals,
            &proof,
            &mut transcript_v,
        );
        assert!(result.is_err(), "verify should reject tampered evaluations");
    }
}

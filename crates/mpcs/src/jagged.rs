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
//! q' = bitrev(p_0) || bitrev(p_1) || ... || bitrev(p_N)
//! ```
//!
//! where each `p_i` is a column polynomial extracted from the input trace matrices, and
//! `bitrev` is the suffix-to-prefix bit-reversal transformation.
//!
//! ## Suffix-to-Prefix Transformation
//!
//! The main sumcheck prover outputs evaluations `v_i = p_i(r[(n-s)..n])` — i.e., at the
//! **suffix** of the random challenge point. To make these evaluations compatible with the
//! jagged sumcheck (which operates on prefix-aligned polynomials), we apply a bit-reversal
//! permutation to each polynomial's evaluations:
//!
//! ```text
//! p_i'[j] = p_i[bitrev_s(j)]   (for j in 0..2^s)
//! ```
//!
//! After bit-reversal, `v_i = p_i(r[(n-s)..n]) = p_i'(reverse(r[(n-s)..n]))`.
//!
//! ## Cumulative Heights
//!
//! The cumulative height sequence `t` tracks the starting position of each polynomial in `q'`:
//! - `t[0] = 0`
//! - `t[i+1] = t[i] + h_i`   where `h_i = 2^(num_vars of p_i)` is the number of evaluations
//!
//! Given a position `b` in `q'`, the verifier can locate the corresponding `(i, r)` pair via:
//! - `t[i-1] <= b < t[i]`
//! - `r = b - t[i-1]`
//!
//! The cumulative heights allow the verifier to succinctly evaluate the indicator function
//! `g(z_r, z_b, t[i-1], t[i])` needed for the jagged sumcheck.
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

use std::iter::once;

use crate::{Error, PolynomialCommitmentScheme};
use ff_ext::ExtensionField;
use itertools::Itertools;
use multilinear_extensions::{mle::MultilinearExtension, virtual_poly::build_eq_x_r_vec};
use p3::{
    matrix::{Matrix, bitrev::BitReversableMatrix},
    maybe_rayon::prelude::{
        IndexedParallelIterator, IntoParallelIterator, ParallelIterator, ParallelSliceMut,
    },
};
use serde::{Deserialize, Serialize};
use sumcheck::structs::{IOPProof, IOPProverMessage, IOPProverState};
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

/// Number of streaming rounds before switching to standard sumcheck.
/// Determined by the epoch schedule: 1 + 2 + 4 + 8 = 15.
#[allow(dead_code)]
const STREAMING_ROUNDS: usize = 15;
/// Epoch sizes used in the streaming phase: j' = 1, 2, 4, 8.
const EPOCH_SIZES: [usize; 4] = [1, 2, 4, 8];

/// All inputs needed for the jagged sumcheck.
pub struct JaggedSumcheckInput<'a, E: ExtensionField> {
    /// Giga polynomial evaluations (concatenated, bit-reversed).
    pub q_evals: &'a [E::BaseField],
    /// n = log2(padded_total_size).
    pub num_giga_vars: usize,
    /// Cumulative height sequence t[j], length num_polys + 1.
    pub cumulative_heights: &'a [usize],
    /// Shared row evaluation point (s components).
    pub z_row: &'a [E],
    /// Column challenge point.
    pub z_col: &'a [E],
}

impl<'a, E: ExtensionField> JaggedSumcheckInput<'a, E> {
    fn total_evaluations(&self) -> usize {
        *self.cumulative_heights.last().unwrap_or(&0)
    }

    /// Find which polynomial giga_idx belongs to. Returns (poly_index, local_offset).
    fn col_row(&self, giga_idx: usize) -> (usize, usize) {
        // Binary search: find j such that t[j] <= giga_idx < t[j+1]
        let j = self.cumulative_heights.partition_point(|&t| t <= giga_idx) - 1;
        (j, giga_idx - self.cumulative_heights[j])
    }
}

/// Run the full jagged sumcheck: streaming phase (rounds 1..K) + standard phase (rounds K+1..n).
///
/// Returns the proof and the full list of challenges (r_1, ..., r_n).
pub fn jagged_sumcheck_prove<E: ExtensionField>(
    input: &JaggedSumcheckInput<E>,
    transcript: &mut impl Transcript<E>,
) -> (IOPProof<E>, Vec<E>) {
    let n = input.num_giga_vars;
    let max_degree: usize = 2;

    // Precompute eq tables (once).
    let eq_row = build_eq_x_r_vec(input.z_row);
    let eq_col = build_eq_x_r_vec(input.z_col);

    let mut challenges: Vec<E> = Vec::with_capacity(n);
    let mut proof_messages: Vec<IOPProverMessage<E>> = Vec::with_capacity(n);

    // Write transcript header (must match verifier's expectations).
    transcript.append_message(&n.to_le_bytes());
    transcript.append_message(&max_degree.to_le_bytes());

    // --- Streaming phase: epochs j' = 1, 2, 4, 8 ---
    for &epoch_size in &EPOCH_SIZES {
        // Epoch j' handles rounds j'..2j'-1. Skip if all rounds are done.
        if epoch_size > n {
            break;
        }

        // Build M-table for this epoch.
        let m_table = build_m_table(input, &eq_row, &eq_col, &challenges, epoch_size);

        // Extract rounds j = epoch_size .. min(2*epoch_size - 1, n)
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
    }

    // --- Phase 2: Bind and materialize, then standard sumcheck ---
    let k = challenges.len(); // actual number of streaming rounds completed
    if k < n {
        let (q_bound, f_bound) = bind_and_materialize(input, &eq_row, &eq_col, &challenges);

        let remaining_vars = n - k;
        let q_mle = MultilinearExtension::from_evaluations_ext_vec(remaining_vars, q_bound);
        let f_mle = MultilinearExtension::from_evaluations_ext_vec(remaining_vars, f_bound);

        // Use VirtualPolynomial + round-by-round proving (no extra transcript header).
        use multilinear_extensions::virtual_poly::VirtualPolynomial;
        use std::sync::Arc;
        let q_arc = Arc::new(q_mle);
        let f_arc = Arc::new(f_mle);
        let vp = VirtualPolynomial::new_from_product(vec![q_arc, f_arc], E::ONE);

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
    eq_row: &[E],
    eq_col: &[E],
    challenges: &[E],  // R_{j'} = (r_1, ..., r_{j'-1})
    epoch_size: usize, // j'
) -> Vec<E> {
    let n = input.num_giga_vars;
    let total_evals = input.total_evaluations();
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
    let mut m_table = vec![E::ZERO; m_size];

    let mut q_bound = vec![E::ZERO; beta_count];
    let mut f_bound = vec![E::ZERO; beta_count];

    for b_idx in 0..n_chunks {
        let chunk_start = b_idx * chunk_size;

        // Reset per-chunk accumulators.
        q_bound.iter_mut().for_each(|v| *v = E::ZERO);
        f_bound.iter_mut().for_each(|v| *v = E::ZERO);

        for beta in 0..beta_count {
            for (a, &eq_r_a) in eq_r.iter().enumerate().take(a_count) {
                let giga_idx = chunk_start + beta * a_count + a;
                if giga_idx >= total_evals {
                    continue;
                }

                let (col, row) = input.col_row(giga_idx);
                let q_val: E = input.q_evals[giga_idx].into();
                let f_val = eq_row[row] * eq_col[col];

                q_bound[beta] += eq_r_a * q_val;
                f_bound[beta] += eq_r_a * f_val;
            }
        }

        // Outer product accumulation.
        for b1 in 0..beta_count {
            if q_bound[b1] == E::ZERO {
                continue;
            }
            for b2 in 0..beta_count {
                m_table[b1 * beta_count + b2] += q_bound[b1] * f_bound[b2];
            }
        }
    }

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
    eq_row: &[E],
    eq_col: &[E],
    challenges: &[E], // R_K = (r_1, ..., r_K)
) -> (Vec<E>, Vec<E>) {
    let n = input.num_giga_vars;
    let k = challenges.len();
    let total_evals = input.total_evaluations();
    let remaining_size = 1usize << (n - k);
    let a_count = 1usize << k;

    let eq_r = build_eq_x_r_vec(challenges);

    let mut q_bound = vec![E::ZERO; remaining_size];
    let mut f_bound = vec![E::ZERO; remaining_size];

    for idx in 0..remaining_size {
        for (a, &eq_r_a) in eq_r.iter().enumerate().take(a_count) {
            let giga_idx = a + idx * a_count;
            if giga_idx >= total_evals {
                continue;
            }

            let (col, row) = input.col_row(giga_idx);
            let q_val: E = input.q_evals[giga_idx].into();
            let f_val = eq_row[row] * eq_col[col];

            q_bound[idx] += eq_r_a * q_val;
            f_bound[idx] += eq_r_a * f_val;
        }
    }

    (q_bound, f_bound)
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

    /// Compute v = sum_b q'(b) * f(b) directly (brute force).
    fn compute_claimed_sum(
        q_evals: &[F],
        cumulative_heights: &[usize],
        z_row: &[E],
        z_col: &[E],
        num_giga_vars: usize,
    ) -> E {
        let eq_row = build_eq_x_r_vec(z_row);
        let eq_col = build_eq_x_r_vec(z_col);
        let total_evals = *cumulative_heights.last().unwrap();
        let giga_size = 1usize << num_giga_vars;

        let mut sum = E::ZERO;
        for b in 0..giga_size {
            if b >= total_evals {
                break; // padding region: q'(b) = 0
            }
            let j = cumulative_heights.partition_point(|&t| t <= b) - 1;
            let local = b - cumulative_heights[j];
            let q_val: E = q_evals[b].into();
            let f_val = eq_row[local] * eq_col[j];
            sum += q_val * f_val;
        }
        sum
    }

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

        let claimed_sum =
            compute_claimed_sum(&q_evals, &cumulative_heights, &z_row, &z_col, num_giga_vars);

        let input = JaggedSumcheckInput {
            q_evals: &q_evals,
            num_giga_vars,
            cumulative_heights: &cumulative_heights,
            z_row: &z_row,
            z_col: &z_col,
        };

        let mut transcript = BasicTranscript::<E>::new(b"jagged_sumcheck_test");
        let (proof, challenges) = jagged_sumcheck_prove(&input, &mut transcript);

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
        // Build the full MLE of q' and f, evaluate at the subclaim point.
        let mut q_padded = q_evals.clone();
        q_padded.resize(1 << num_giga_vars, F::ZERO);
        let q_mle = MultilinearExtension::<E>::from_evaluations_vec(num_giga_vars, q_padded);
        let q_at_point = q_mle.evaluate(&challenges);

        // f is defined piecewise; compute its MLE evaluations then evaluate at point.
        let eq_row_table = build_eq_x_r_vec(&z_row);
        let eq_col_table = build_eq_x_r_vec(&z_col);
        let mut f_evals = vec![E::ZERO; 1 << num_giga_vars];
        for b in 0..total_evals {
            let j = cumulative_heights.partition_point(|&t| t <= b) - 1;
            let local = b - cumulative_heights[j];
            f_evals[b] = eq_row_table[local] * eq_col_table[j];
        }
        let f_mle = MultilinearExtension::<E>::from_evaluations_ext_vec(num_giga_vars, f_evals);
        let f_at_point = f_mle.evaluate(&challenges);

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

        let claimed_sum =
            compute_claimed_sum(&q_evals, &cumulative_heights, &z_row, &z_col, num_giga_vars);

        let input = JaggedSumcheckInput {
            q_evals: &q_evals,
            num_giga_vars,
            cumulative_heights: &cumulative_heights,
            z_row: &z_row,
            z_col: &z_col,
        };

        let mut transcript = BasicTranscript::<E>::new(b"jagged_test_16");
        let (proof, challenges) = jagged_sumcheck_prove(&input, &mut transcript);

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

        let claimed_sum =
            compute_claimed_sum(&q_evals, &cumulative_heights, &z_row, &z_col, num_giga_vars);

        let input = JaggedSumcheckInput {
            q_evals: &q_evals,
            num_giga_vars,
            cumulative_heights: &cumulative_heights,
            z_row: &z_row,
            z_col: &z_col,
        };

        let mut transcript = BasicTranscript::<E>::new(b"jagged_test_25");
        let (proof, challenges) = jagged_sumcheck_prove(&input, &mut transcript);

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
    }
}

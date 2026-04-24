//! Jagged PCS
//!
//! Implements the Jagged PCS protocol (commit, batch open, batch verify) tailored to
//! Ceno's scenario, based on the SP1 Jagged PCS paper
//! (<https://eprint.iacr.org/2025/917.pdf>) and Ceno issue
//! [#1272](https://github.com/scroll-tech/ceno/issues/1272).
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
//! where each `p_i` is a column polynomial extracted from the input trace matrices.
//! We cannot use the naive `q = p_0 || ... || p_{N-1}` because the jagged sumcheck
//! requires prefix-aligned evaluation points, but Ceno's main sumcheck evaluates
//! polynomials at a **suffix** of the challenge point. The `bitrev` (suffix-to-prefix
//! bit-reversal) bridges the two, making the jagged PCS applicable (see below).
//!
//! ## Suffix-to-Prefix Transformation
//!
//! Each `p_i` has `s` variables (where `s` varies per polynomial; we write `s` instead
//! of `s_i` for brevity). Let `m` be the length of `z_r`. The main sumcheck prover
//! outputs evaluations `v_i = p_i(z_r[(m-s)..m])` — i.e., at the suffix of the
//! random challenge point.
//! To make these evaluations compatible with the jagged sumcheck (which operates on
//! prefix-aligned polynomials), we apply a bit-reversal permutation to each polynomial's
//! evaluations:
//!
//! ```text
//! p_i'[j] = p_i[bitrev_s(j)]   (for j in 0..2^s)
//! ```
//!
//! After bit-reversal,
//! ```text
//! v_i = p_i(z_r[(m-s)..m])
//!     = p_i'(z_r'[..s])
//! ```
//! where `z_r' = reverse(z_r)`.
//!
//! ## Cumulative Heights
//!
//! The cumulative height sequence `t` tracks the starting position of each polynomial in `q'`:
//! - `t[0] = 0`
//! - `t[i+1] = t[i] + h_i`   where `h_i = 2^(num_vars of p_i)` is the number of evaluations
//!
//! Given a position `b` in `q'`, the inverse mapping `inv(b) = (i, r)` is defined by:
//! - `t[i] <= b < t[i+1]`
//! - `r = b - t[i]`
//!
//! Using the indicator `g(r, b, t[i], t[i+1]) = [r + t[i] = b ∧ b < t[i+1]]`, this has
//! the closed form: `inv(b) = Σ_{i,r} (i, r) · g(r, b, t[i], t[i+1])`.
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
//!
//! ## Batch Open Protocol
//!
//! For notational simplicity, we write `p_i` and `z_r` instead of `p_i'` and `z_r'`
//! (the bit-reversed variants) throughout this section.
//!
//! Each column opening `v_i = p_i(z_r[..s])` requires its own sumcheck. We batch all K
//! openings into one using `eq(z_c, ·)` weights (soundness loss: `log2(N) / |E|`):
//!
//! ```text
//! v = Σ_i eq(z_c, i) · p_i(z_r[..s])
//!   = Σ_{i,r} eq(z_c, i) · eq(z_r, r) · p_i(r)
//!   = Σ_{i,r} f(i, r) · p_i(r)     where f(i, r) = eq(z_c, i) · eq(z_r, r)
//! ```
//!
//! We apply `inv(b)` to rewrite `f` in terms of the giga-index `b`:
//!
//! ```text
//! f(inv(b)) = Σ_{i,r} f(i, r) · g(r, b, t[i], t[i+1])
//!           = Σ_i eq(z_c, i) · ĝ(z_r, b, t[i], t[i+1])
//! ```
//!
//! where `ĝ(z_r, b, ·) = Σ_r eq(z_r, r) · g(r, b, ·)` absorbs the row weight.
//!
//! Since summation over (i,r) is equivalent to summation over b, we can rewrite the batch opening claim as
//!
//! ```text
//! v = Σ_b q'(b) · f(inv(b))
//!   = Σ_b q'(b) · Σ_i eq(z_c, i) · ĝ(z_r, b, t[i], t[i+1])
//! ```
//!
//! Defining `h(b) = f(inv(b)) = Σ_i eq(z_c, i) · ĝ(z_r, b, t[i], t[i+1])` (multilinear in `b`):
//!
//! The sumcheck reduces this to a single opening of `q'(ρ)` plus a verifier check of
//! `h(ρ)`. The ROBP makes `ĝ` efficiently evaluable, so the verifier computes
//! `h(ρ) = Σ_i eq(z_c, i) · ĝ(z_r, ρ, t[i], t[i+1])` in `O(K·n)` time.

pub mod evaluator;
pub mod sumcheck;
mod types;

pub use evaluator::{evaluate_g, evaluate_g_backward, evaluate_g_forward};
pub use sumcheck::{JaggedSumcheckInput, jagged_sumcheck_prove};
pub use types::{JaggedBatchOpenProof, JaggedCommitment, JaggedCommitmentWithWitness};

use std::{iter::once, marker::PhantomData};

use crate::{Error, PolynomialCommitmentScheme};
use ::sumcheck::structs::IOPVerifierState;
use ff_ext::ExtensionField;
use itertools::Itertools;
use multilinear_extensions::{
    mle::FieldType,
    virtual_poly::{VPAuxInfo, build_eq_x_r_vec},
};
use p3::{
    matrix::{Matrix, bitrev::BitReversableMatrix},
    maybe_rayon::prelude::{
        IndexedParallelIterator, IntoParallelIterator, ParallelIterator, ParallelSliceMut,
    },
};
use transcript::Transcript;
use types::int_to_field_bits;
use witness::{InstancePaddingStrategy, RowMajorMatrix};

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
// Jagged Batch Open / Verify
// ---------------------------------------------------------------------------

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
    use multilinear_extensions::mle::MultilinearExtension;
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

    // --- Batch open/verify tests ---

    use ff_ext::FromUniformBytes;
    use rand::thread_rng;
    use transcript::basic::BasicTranscript;

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

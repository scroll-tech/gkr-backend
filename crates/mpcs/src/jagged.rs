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
use p3::{
    matrix::{Matrix, bitrev::BitReversableMatrix},
    maybe_rayon::prelude::{
        IndexedParallelIterator, IntoParallelIterator, ParallelIterator, ParallelSliceMut,
    },
};
use serde::{Deserialize, Serialize};
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
    unsafe { concatenated.set_len(total_size) };

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
}

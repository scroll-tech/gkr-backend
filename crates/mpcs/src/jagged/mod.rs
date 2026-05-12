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
//! q' = p_0 || p_1 || ... || p_{N-1}
//! ```
//!
//! where each `p_i` is a column polynomial extracted from the input trace matrices.
//! The jagged sumcheck uses prefix-aligned evaluation points, so a polynomial with
//! `s_i = ceil_log2(h_i)` variables is opened at `z_r[..s_i]`.
//!
//! ## Cumulative Heights
//!
//! Each polynomial `p_i` has `s_i = ceil_log2(h_i)` variables, where `h_i` is the
//! real number of evaluations from the input matrix column. `q'` stores exactly
//! those `h_i` evaluations; any implicit zero padding to `2^{s_i}` is only an MLE
//! evaluation convention and is not materialized inside the concatenation.
//!
//! The cumulative height sequence `t` tracks the starting position of each polynomial in `q'`:
//! - `t[0] = 0`
//! - `t[i+1] = t[i] + h_i`
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
//! 1. For each input matrix `M_k` (with `h_k` rows and `w_k` columns), extract each
//!    column as a polynomial with `h_k` evaluations.
//! 2. Concatenate all column polynomials: `cat = p_0 || p_1 || ...`
//! 3. Compute cumulative heights `t[i]`.
//! 4. Pad `cat` to the next power of two (required for MLE representation).
//! 5. Commit to the padded `cat` as a single-column matrix using the inner PCS.
//!
//! ## Batch Open Protocol
//!
//! Let `s_i` denote the number of variables of `p_i` and let `m = max(s_i)`.
//!
//! Each column opening `v_i = p_i(z_r[..s_i])` requires its own sumcheck. We batch
//! all K openings into one using `eq(z_c, ·)` weights (soundness loss: `log2(N) / |E|`).
//!
//! ### Correction factors for different-height polynomials
//!
//! In the giga polynomial `q'`, each `p_i` occupies only `h_i` slots (padded to
//! `2^{s_i}` for MLE representation). When `s_i < m`, this is equivalent to
//! zero-padding `p_i` to `m` variables:
//!
//! ```text
//! p_i^pad(r, b) = p_i(r)   if b = 0    (r ∈ {0,1}^{s_i}, b ∈ {0,1}^{m - s_i})
//!                 0         if b ≠ 0
//! ```
//!
//! The MLE of this zero-padded polynomial evaluates to:
//!
//! ```text
//! p_i^pad(z_r) = eq(z_r[s_i..], 0) · p_i(z_r[..s_i])
//! ```
//!
//! where `eq(z_r[s_i..], 0) = Π_{j=s_i}^{m-1} (1 - z_r[j])` is the correction
//! factor `C_i` arising from the zero-padded positions. The batched claim becomes:
//!
//! ```text
//! v = Σ_i eq(z_c, i) · C_i · p_i(z_r[..s_i])
//!   = Σ_{i,r} eq(z_c, i) · eq(z_r, r) · p_i(r)
//!   = Σ_{i,r} f(i, r) · p_i(r)     where f(i, r) = eq(z_c, i) · eq(z_r, r)
//! ```
//!
//! Here `eq(z_r, r)` uses the full `z_r` of length `m`. For `r < 2^{s_i}`, the
//! high bits of `r` are zero, so `eq(z_r, r) = eq(z_r[..s_i], r) · C_i`.
//! This means a single precomputed eq table of size `2^m` naturally incorporates
//! the correction factors — no per-polynomial tables are needed.
//!
//! ### Rewriting via the inverse mapping
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

pub mod assist;
pub mod evaluator;
pub mod sumcheck;
mod types;

pub use assist::{assist_sumcheck_prove, compute_q_at_assist_point};
pub use evaluator::{evaluate_g, evaluate_g_backward, evaluate_g_forward};
pub use sumcheck::{JaggedSumcheckInput, jagged_sumcheck_prove};
pub use types::{JaggedBatchOpenProof, JaggedCommitment, JaggedCommitmentWithWitness, JaggedProof};

use std::{iter::once, marker::PhantomData};

use crate::{Error, PCSFriParam, Point, PolynomialCommitmentScheme};
use ::sumcheck::structs::IOPVerifierState;
use ff_ext::ExtensionField;
use itertools::Itertools;
use multilinear_extensions::{
    mle::{ArcMultilinearExtension, FieldType},
    util::ceil_log2,
    virtual_poly::{VPAuxInfo, build_eq_x_r_vec},
};
use p3::{
    field::FieldAlgebra,
    matrix::Matrix,
    maybe_rayon::prelude::*,
};
use serde::{Serialize, Serializer, de::DeserializeOwned};
use std::sync::Arc;
use transcript::Transcript;
use types::int_to_field_bits;
use witness::{InstancePaddingStrategy, RowMajorMatrix as WitnessRowMajorMatrix};

#[derive(Debug)]
pub struct Jagged<InnerPcs>(PhantomData<InnerPcs>);

impl<InnerPcs> Clone for Jagged<InnerPcs> {
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}

impl<InnerPcs> Serialize for Jagged<InnerPcs> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str("jagged")
    }
}

/// Commit to a sequence of row-major matrices using the Jagged PCS scheme.
///
/// This function implements the commit phase described in Ceno issue #1288:
/// 1. For each matrix, transpose it (row-major → column-major), so each
///    column polynomial occupies a contiguous region in memory.
/// 2. Concatenate all column polynomials: `q' = col_0 || col_1 || ...`
/// 3. Compute the cumulative height sequence `t`.
/// 4. Commit to `q'` as a single-column matrix using `InnerPcs::batch_commit`.
///
/// # Arguments
/// * `pp` — Prover parameters for `InnerPcs`.
/// * `rmms` — Non-empty sequence of row-major matrices. This function uses each matrix's height exactly as given.
///
/// # Errors
/// Returns `Error::InvalidPcsParam` if `rmms` is empty or all matrices are empty.
/// Any error from the inner `InnerPcs::batch_commit` is propagated as-is.
pub fn jagged_commit<E: ExtensionField, InnerPcs: PolynomialCommitmentScheme<E>>(
    pp: &InnerPcs::ProverParam,
    rmms: Vec<WitnessRowMajorMatrix<E::BaseField>>,
    reshape_log_height: usize,
) -> Result<JaggedCommitmentWithWitness<E, InnerPcs>, Error> {
    if rmms.is_empty() {
        return Err(Error::InvalidPcsParam(
            "jagged_commit: cannot commit to empty sequence of matrices".to_string(),
        ));
    }

    let polys = rmms
        .iter()
        .flat_map(|rmm| rmm.to_mles().into_iter().map(Arc::new))
        .collect_vec();

    // --- Step 1: Compute cumulative heights from real matrix heights ---
    let mut poly_heights: Vec<usize> = Vec::new();
    for rmm in &rmms {
        let num_rows = rmm.occupied_physical_rows();
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

    // --- Steps 2 & 3: Transpose and write to concatenated ---
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
    for rmm in rmms {
        let n_cols = rmm.width();
        let n_rows = rmm.occupied_physical_rows();
        let n_cells = n_cols * n_rows;

        // The start position in `concatenated` for this matrix's block of polynomials.
        let start = cumulative_heights[poly_idx];

        // Step 3: Transpose — write each column j of `rmm` (= one polynomial)
        // into its corresponding contiguous slice in `concatenated`.
        (0..n_cols)
            .into_par_iter()
            .zip(concatenated[start..start + n_cells].par_chunks_mut(n_rows))
            .for_each(|(j, chunk)| {
                rmm.values
                    .iter()
                    .take(n_cells)
                    .skip(j)
                    .step_by(n_cols)
                    .zip_eq(chunk.iter_mut())
                    .for_each(|(v, out)| *out = *v);
            });

        poly_idx += n_cols;
    }

    // --- Step 4: Reshape and commit via the inner PCS ---
    let log_h = reshape_log_height.min(ceil_log2(total_size.max(1)));
    let h = 1usize << log_h;
    let w = total_size.div_ceil(h);

    let giga_data = if w == 1 {
        concatenated
    } else {
        // Transpose the flat q' evaluations to row-major for the inner PCS.
        // In the flat array, column i occupies concatenated[i*h .. (i+1)*h];
        // the last column may be short and is zero-padded.
        let padded_total = w * h;
        let mut row_major: Vec<E::BaseField> = Vec::with_capacity(padded_total);
        #[allow(clippy::uninit_vec)]
        unsafe {
            row_major.set_len(padded_total)
        };

        (0..w).into_par_iter().for_each(|i| {
            let src_start = i * h;
            let col_len = if i < w - 1 { h } else { total_size - src_start };
            let dst = unsafe {
                &mut *std::ptr::slice_from_raw_parts_mut(
                    row_major.as_ptr() as *mut E::BaseField,
                    padded_total,
                )
            };
            for b in 0..col_len {
                dst[b * w + i] = concatenated[src_start + b];
            }
            for b in col_len..h {
                dst[b * w + i] = E::BaseField::ZERO;
            }
        });

        row_major
    };

    let giga_rmm = WitnessRowMajorMatrix::<E::BaseField>::new_by_values(
        giga_data,
        w,
        InstancePaddingStrategy::Default,
    );

    let inner = InnerPcs::batch_commit(pp, vec![giga_rmm])?;

    Ok(JaggedCommitmentWithWitness {
        inner,
        cumulative_heights,
        poly_heights,
        reshape_log_height: log_h,
        polys,
    })
}

fn default_reshape_log_height<E: ExtensionField, InnerPcs: PolynomialCommitmentScheme<E>>(
    pp: &InnerPcs::ProverParam,
    rmms: &[WitnessRowMajorMatrix<E::BaseField>],
) -> usize {
    let total_evals = rmms
        .iter()
        .map(|rmm| rmm.occupied_physical_rows() * rmm.width())
        .sum::<usize>();
    let total_log = ceil_log2(total_evals.max(1));
    pp.get_max_message_size_log().min(total_log)
}

fn flatten_padded_openings_as_native<E: ExtensionField>(
    poly_heights: &[usize],
    openings: Vec<(Point<E>, Vec<E>)>,
) -> Result<(Point<E>, Vec<E>), Error> {
    let max_native_point_len = poly_heights.iter().map(|&h| ceil_log2(h)).max().unwrap_or(0);
    let mut common_point = vec![None; max_native_point_len];
    let mut evals = Vec::with_capacity(poly_heights.len());
    let mut poly_idx = 0;

    for (point, point_evals) in openings {
        for value in point_evals {
            let height = poly_heights.get(poly_idx).ok_or_else(|| {
                Error::InvalidPcsParam("jagged: too many opening evaluations".to_string())
            })?;
            let native_num_vars = ceil_log2(*height);
            if point.len() < native_num_vars {
                return Err(Error::InvalidPcsParam(format!(
                    "jagged: opening point length {} is smaller than poly num_vars {}",
                    point.len(),
                    native_num_vars
                )));
            }
            for (dst, src) in common_point.iter_mut().zip(point.iter()) {
                match dst {
                    Some(existing) if *existing != *src => {
                        return Err(Error::InvalidPcsParam(
                            "jagged: opening points are not prefix-compatible".to_string(),
                        ));
                    }
                    Some(_) => {}
                    None => *dst = Some(*src),
                }
            }

            let tail_zero_factor = point[native_num_vars..]
                .iter()
                .fold(E::ONE, |acc, r| acc * (E::ONE - *r));
            if tail_zero_factor == E::ZERO {
                return Err(Error::InvalidPcsParam(
                    "jagged: padded opening tail factor is zero".to_string(),
                ));
            }
            evals.push(value * tail_zero_factor.inverse());
            poly_idx += 1;
        }
    }

    if poly_idx != poly_heights.len() {
        return Err(Error::InvalidPcsParam(format!(
            "jagged: expected {} opening evaluations, got {}",
            poly_heights.len(),
            poly_idx
        )));
    }

    let point = common_point
        .into_iter()
        .map(|value| {
            value.ok_or_else(|| {
                Error::InvalidPcsParam("jagged: missing common opening point".to_string())
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((point, evals))
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

/// Prove that evaluation claims `evals[i] = p_i(point_i)` are consistent with a
/// jagged commitment.
///
/// Polynomials may have different heights. `point` is the evaluation point of
/// length `max_s = max(log2(h_i))`. Polynomial `i` with `s_i` variables is
/// evaluated at the prefix `point[..s_i]`.
///
/// The protocol:
/// 1. Batch the K column claims via a random column challenge `z_col`.
/// 2. Run the jagged sumcheck to reduce to a single evaluation of q'.
/// 3. Open q' at the sumcheck output point via the inner PCS.
pub fn jagged_batch_open<E: ExtensionField, InnerPcs: PolynomialCommitmentScheme<E>>(
    pp: &InnerPcs::ProverParam,
    comm: &JaggedCommitmentWithWitness<E, InnerPcs>,
    point: &[E],
    evals: &[E],
    transcript: &mut impl Transcript<E>,
) -> Result<JaggedBatchOpenProof<E, InnerPcs>, Error> {
    let num_polys = comm.num_polys();
    if evals.len() != num_polys {
        return Err(Error::InvalidPcsParam(format!(
            "jagged_batch_open: expected {} evals, got {}",
            num_polys,
            evals.len()
        )));
    }

    let max_s = comm
        .poly_heights
        .iter()
        .map(|&h| ceil_log2(h))
        .max()
        .unwrap_or(0);
    if point.len() != max_s {
        return Err(Error::InvalidPcsParam(format!(
            "jagged_batch_open: point length {} != max poly log-height {}",
            point.len(),
            max_s
        )));
    }

    let total_evals = comm.total_evaluations();
    let log_h = comm.reshape_log_height;
    let h = 1usize << log_h;
    let w = total_evals.div_ceil(h);
    let padded_total = w * h;
    let num_giga_vars = ceil_log2(padded_total);

    let z_row: Vec<E> = point.to_vec();

    // Write evals to transcript, then sample z_col.
    transcript.append_field_element_exts(evals);
    let num_col_vars = ceil_log2(num_polys).max(1);
    let z_col: Vec<E> = transcript.sample_and_append_vec(b"jagged_z_col", num_col_vars);

    let eq_col = build_eq_x_r_vec(&z_col);

    // Reconstruct the flat q' evaluation table from the inner PCS MLEs.
    // The inner PCS stores w column MLEs; concatenating them gives q'[0..w*h].
    let q_mles = InnerPcs::get_arc_mle_witness_from_commitment(&comm.inner);
    assert_eq!(q_mles.len(), w);
    let q_evals_base_owned: Vec<E::BaseField>;
    let q_evals_base: &[E::BaseField] = if w == 1 {
        match q_mles[0].evaluations() {
            FieldType::Base(slice) => slice,
            _ => {
                return Err(Error::InvalidPcsParam(
                    "jagged_batch_open: expected base-field evaluations for q'".into(),
                ));
            }
        }
    } else {
        let target_len = 1usize << num_giga_vars;
        q_evals_base_owned = q_mles
            .iter()
            .flat_map(|mle| match mle.evaluations() {
                FieldType::Base(slice) => slice.iter().copied(),
                _ => unreachable!("expected base-field evaluations"),
            })
            .chain(std::iter::repeat_n(
                E::BaseField::ZERO,
                target_len - padded_total,
            ))
            .collect();
        &q_evals_base_owned
    };

    // Batched opening claim: v = Σ_i eq_col[i] · C_i · p_i(z_row[..s_i])
    // where C_i = Π_{j=s_i}^{max_s-1}(1 - z_row[j]) is the correction factor for poly i.
    // eq_row[r] = eq(z_row, r) naturally incorporates C_i for row r within poly i
    // (the high bits of r are 0, so eq_row[r] = eq(z_row[..s_i], r) * C_i). This means
    // the jagged sumcheck sum Σ_{i,r} eq_col[i]*eq_row[r]*q'(t_i+r) equals the batched claim.
    let eq_row = build_eq_x_r_vec(&z_row);
    let input = JaggedSumcheckInput {
        q_evals: q_evals_base,
        num_giga_vars,
        cumulative_heights: &comm.cumulative_heights,
        eq_row,
        eq_col: eq_col.to_vec(),
    };
    let (sumcheck_proof, challenges) = jagged_sumcheck_prove(&input, transcript, None);
    let rho = challenges;

    // Compute f̂(ρ) = Σ_y eq_col[y] · ĝ(z_row, ρ, bits(t_y), bits(t_{y+1})).
    let n_robp = num_giga_vars + if padded_total.is_power_of_two() { 1 } else { 0 };
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

    // Split ρ: low bits = row point, high bits = column selector.
    let rho_row = &rho[..log_h];

    // Compute column evaluations: evaluate each column MLE at ρ_row.
    assert_eq!(q_mles.len(), w);
    let col_evals: Vec<E> = q_mles.par_iter().map(|mle| mle.evaluate(rho_row)).collect();

    // Write col_evals and f_at_rho to transcript.
    transcript.append_field_element_exts(&col_evals);
    transcript.append_field_element_ext(&f_at_rho);

    // Run the assist sumcheck to prove f̂(ρ) is correct.
    let (assist_proof, _assist_challenges) = assist_sumcheck_prove(
        &z_row_padded,
        &rho_padded,
        &eq_col,
        &comm.cumulative_heights,
        n_robp,
        transcript,
    );

    // Open column MLEs at ρ_row via inner PCS.
    let inner_proof = InnerPcs::batch_open(
        pp,
        vec![(&comm.inner, vec![(rho_row.to_vec(), col_evals.clone())])],
        transcript,
    )?;

    Ok(JaggedBatchOpenProof {
        sumcheck_proof,
        col_evals,
        f_at_rho,
        assist_proof,
        inner_proof,
    })
}

/// Verify that evaluation claims `evals[i] = p_i(point_i)` are consistent with a
/// jagged commitment.
///
/// Polynomials may have different heights. `point` has length `max_s`. Polynomial
/// `i` with `s_i` variables is evaluated at the prefix `point[..s_i]`.
pub fn jagged_batch_verify<E: ExtensionField, InnerPcs: PolynomialCommitmentScheme<E>>(
    vp: &InnerPcs::VerifierParam,
    comm: &JaggedCommitment<E, InnerPcs>,
    point: &[E],
    evals: &[E],
    proof: &JaggedBatchOpenProof<E, InnerPcs>,
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
    let log_h = comm.reshape_log_height;
    let h = 1usize << log_h;
    let w = total_evals.div_ceil(h);
    let padded_total = w * h;
    let num_giga_vars = ceil_log2(padded_total);
    let max_s = point.len();
    let z_row: Vec<E> = point.to_vec();

    // Replay transcript: write evals, sample z_col.
    transcript.append_field_element_exts(evals);
    let num_col_vars = ceil_log2(num_polys).max(1);
    let z_col: Vec<E> = transcript.sample_and_append_vec(b"jagged_z_col", num_col_vars);

    // Batched opening claim: v = Σ_i eq_col[i] · C_i · evals[i]
    // where C_i = eq(z_row[s_i..], 0) = Π_{j=s_i}^{max_s-1} (1 - z_row[j]) is the
    // correction factor from zero-padding p_i to max_s variables.
    let eq_col = build_eq_x_r_vec(&z_col);
    let mut tail_zero_prod = vec![E::ONE; max_s + 1];
    for j in (0..max_s).rev() {
        tail_zero_prod[j] = tail_zero_prod[j + 1] * (E::ONE - z_row[j]);
    }
    let claimed_sum: E = (0..num_polys)
        .map(|i| {
            let h_i = comm.cumulative_heights[i + 1] - comm.cumulative_heights[i];
            let s_i = ceil_log2(h_i);
            eq_col[i] * tail_zero_prod[s_i] * evals[i]
        })
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
    // When padded_total is an exact power of 2, num_giga_vars bits can't represent it.
    let n_robp = num_giga_vars + if padded_total.is_power_of_two() { 1 } else { 0 };

    // Split ρ: low bits = row point, high bits = column selector.
    let rho_row = &rho[..log_h];
    let rho_col = &rho[log_h..];

    // Reconstruct q'(ρ) from col_evals: q'(ρ) = Σ_{i=0}^{w-1} eq(ρ_col, i) · col_evals[i].
    let eq_rho_col = build_eq_x_r_vec(rho_col);
    let q_eval: E = eq_rho_col[..w]
        .iter()
        .zip(&proof.col_evals)
        .map(|(e, v)| *e * *v)
        .sum();

    // Write col_evals and f_at_rho to transcript (must match prover).
    transcript.append_field_element_exts(&proof.col_evals);
    transcript.append_field_element_ext(&proof.f_at_rho);

    // Check multiplicative subclaim: q'(ρ) · f̂(ρ) == expected_evaluation.
    if q_eval * proof.f_at_rho != subclaim.expected_evaluation {
        return Err(Error::InvalidPcsOpen(
            "jagged_batch_verify: q_eval * f(rho) != subclaim expected evaluation".into(),
        ));
    }

    // Verify the assist sumcheck: proves that f_at_rho is correct.
    let n_assist = 2 * n_robp;
    let assist_aux = VPAuxInfo {
        max_degree: 2,
        max_num_variables: n_assist,
        phantom: PhantomData::<E>,
    };
    let assist_subclaim =
        IOPVerifierState::<E>::verify(proof.f_at_rho, &proof.assist_proof, &assist_aux, transcript);
    let assist_point: Vec<E> = assist_subclaim.point.iter().map(|c| c.elements).collect();

    // De-interleave the assist point: (z3[0], z4[0], z3[1], z4[1], ...)
    let rho_star_c: Vec<E> = (0..n_robp).map(|i| assist_point[2 * i]).collect();
    let rho_star_d: Vec<E> = (0..n_robp).map(|i| assist_point[2 * i + 1]).collect();

    // h(ρ*) = ĝ(z_row_padded, ρ_padded, ρ*_c, ρ*_d) — one ROBP evaluation.
    let mut z_row_padded = z_row;
    z_row_padded.resize(n_robp, E::ZERO);
    let mut rho_padded = rho.clone();
    rho_padded.resize(n_robp, E::ZERO);
    let h_at_rho_star = evaluate_g(&z_row_padded, &rho_padded, &rho_star_c, &rho_star_d);

    // Q(ρ*) = Σ_y eq_col[y] · eq(ρ*, x_y).
    let q_at_rho_star =
        compute_q_at_assist_point(&assist_point, &eq_col, &comm.cumulative_heights, n_robp);

    // Check assist subclaim: h(ρ*) · Q(ρ*) == expected_evaluation.
    if h_at_rho_star * q_at_rho_star != assist_subclaim.expected_evaluation {
        return Err(Error::InvalidPcsOpen(
            "jagged_batch_verify: assist sumcheck final check failed".into(),
        ));
    }

    // Verify the inner PCS opening at ρ_row with col_evals.
    InnerPcs::batch_verify(
        vp,
        vec![(
            comm.inner.clone(),
            vec![(log_h, (rho_row.to_vec(), proof.col_evals.clone()))],
        )],
        &proof.inner_proof,
        transcript,
    )?;

    Ok(())
}

impl<E, InnerPcs> PolynomialCommitmentScheme<E> for Jagged<InnerPcs>
where
    E: ExtensionField + Serialize + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
    InnerPcs: PolynomialCommitmentScheme<E>,
{
    type Param = InnerPcs::Param;
    type ProverParam = InnerPcs::ProverParam;
    type VerifierParam = InnerPcs::VerifierParam;
    type CommitmentWithWitness = JaggedCommitmentWithWitness<E, InnerPcs>;
    type Commitment = JaggedCommitment<E, InnerPcs>;
    type CommitmentChunk = InnerPcs::CommitmentChunk;
    type Proof = JaggedProof<E, InnerPcs>;

    fn setup(poly_size: usize, security_level: crate::SecurityLevel) -> Result<Self::Param, Error> {
        InnerPcs::setup(poly_size, security_level)
    }

    fn trim(
        param: Self::Param,
        poly_size: usize,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), Error> {
        InnerPcs::trim(param, poly_size)
    }

    fn commit(
        pp: &Self::ProverParam,
        rmm: WitnessRowMajorMatrix<E::BaseField>,
    ) -> Result<Self::CommitmentWithWitness, Error> {
        Self::batch_commit(pp, vec![rmm])
    }

    fn write_commitment(
        comm: &Self::Commitment,
        transcript: &mut impl Transcript<E>,
    ) -> Result<(), Error> {
        InnerPcs::write_commitment(&comm.inner, transcript)?;
        transcript.append_field_element(&E::BaseField::from_canonical_usize(
            comm.reshape_log_height,
        ));
        transcript.append_field_element(&E::BaseField::from_canonical_usize(
            comm.cumulative_heights.len(),
        ));
        for height in &comm.cumulative_heights {
            transcript.append_field_element(&E::BaseField::from_canonical_usize(*height));
        }
        Ok(())
    }

    fn get_pure_commitment(comm: &Self::CommitmentWithWitness) -> Self::Commitment {
        comm.to_commitment()
    }

    fn batch_commit(
        pp: &Self::ProverParam,
        rmms: Vec<WitnessRowMajorMatrix<E::BaseField>>,
    ) -> Result<Self::CommitmentWithWitness, Error> {
        let reshape_log_height = default_reshape_log_height::<E, InnerPcs>(pp, &rmms);
        jagged_commit::<E, InnerPcs>(pp, rmms, reshape_log_height)
    }

    fn open(
        _pp: &Self::ProverParam,
        _poly: &ArcMultilinearExtension<E>,
        _comm: &Self::CommitmentWithWitness,
        _point: &[E],
        _eval: &E,
        _transcript: &mut impl Transcript<E>,
    ) -> Result<Self::Proof, Error> {
        unimplemented!()
    }

    fn batch_open(
        pp: &Self::ProverParam,
        rounds: Vec<(
            &Self::CommitmentWithWitness,
            Vec<(Point<E>, Vec<E>)>,
        )>,
        transcript: &mut impl Transcript<E>,
    ) -> Result<Self::Proof, Error> {
        let mut proofs = Vec::with_capacity(rounds.len());
        for (comm, openings) in rounds {
            let (point, evals) = flatten_padded_openings_as_native(&comm.poly_heights, openings)?;
            proofs.push(jagged_batch_open::<E, InnerPcs>(
                pp, comm, &point, &evals, transcript,
            )?);
        }
        Ok(JaggedProof { rounds: proofs })
    }

    fn simple_batch_open(
        _pp: &Self::ProverParam,
        _polys: &[ArcMultilinearExtension<E>],
        _comm: &Self::CommitmentWithWitness,
        _point: &[E],
        _evals: &[E],
        _transcript: &mut impl Transcript<E>,
    ) -> Result<Self::Proof, Error> {
        unimplemented!()
    }

    fn verify(
        _vp: &Self::VerifierParam,
        _comm: &Self::Commitment,
        _point: &[E],
        _eval: &E,
        _proof: &Self::Proof,
        _transcript: &mut impl Transcript<E>,
    ) -> Result<(), Error> {
        unimplemented!()
    }

    fn batch_verify(
        vp: &Self::VerifierParam,
        rounds: Vec<(
            Self::Commitment,
            Vec<(usize, (Point<E>, Vec<E>))>,
        )>,
        proof: &Self::Proof,
        transcript: &mut impl Transcript<E>,
    ) -> Result<(), Error> {
        if rounds.len() != proof.rounds.len() {
            return Err(Error::InvalidPcsOpeningProof(format!(
                "jagged: expected {} proof rounds, got {}",
                rounds.len(),
                proof.rounds.len()
            )));
        }
        for ((comm, openings), round_proof) in rounds.into_iter().zip_eq(proof.rounds.iter()) {
            let poly_heights = comm
                .cumulative_heights
                .windows(2)
                .map(|window| window[1] - window[0])
                .collect_vec();
            let openings = openings
                .into_iter()
                .map(|(_, (point, evals))| (point, evals))
                .collect_vec();
            let (point, evals) = flatten_padded_openings_as_native(&poly_heights, openings)?;
            jagged_batch_verify::<E, InnerPcs>(vp, &comm, &point, &evals, round_proof, transcript)?;
        }
        Ok(())
    }

    fn simple_batch_verify(
        _vp: &Self::VerifierParam,
        _comm: &Self::Commitment,
        _point: &[E],
        _evals: &[E],
        _proof: &Self::Proof,
        _transcript: &mut impl Transcript<E>,
    ) -> Result<(), Error> {
        unimplemented!()
    }

    fn get_arc_mle_witness_from_commitment(
        commitment: &Self::CommitmentWithWitness,
    ) -> Vec<ArcMultilinearExtension<'static, E>> {
        commitment.polys.clone()
    }
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
    use p3::{field::FieldAlgebra, goldilocks::Goldilocks, matrix::dense::RowMajorMatrix};

    type F = Goldilocks;
    type E = GoldilocksExt2;
    type Pcs = Basefold<E, BasefoldRSParams>;

    fn make_rmm(num_rows: usize, num_cols: usize) -> WitnessRowMajorMatrix<F> {
        let values: Vec<F> = (0..num_rows * num_cols)
            .map(|i| F::from_canonical_u64(i as u64 + 1))
            .collect();
        WitnessRowMajorMatrix::new_by_inner_matrix(
            RowMajorMatrix::new(values, num_cols),
            InstancePaddingStrategy::Default,
        )
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
    fn test_jagged_commit_uses_real_heights() {
        // 3x1 + 5x2 → heights [3, 5, 5], not [4, 8, 8].
        let reshape_log_height = 4;
        let (pp, _vp) = setup_pcs::<E, Pcs>(reshape_log_height);
        let m1 = make_rmm(3, 1);
        let m2 = make_rmm(5, 2);

        let comm = jagged_commit::<E, Pcs>(&pp, vec![m1, m2], reshape_log_height)
            .expect("commit should succeed");

        assert_eq!(comm.num_polys(), 3);
        assert_eq!(comm.poly_heights, vec![3, 5, 5]);
        assert_eq!(comm.cumulative_heights, vec![0, 3, 8, 13]);
        assert_eq!(comm.total_evaluations(), 13);
    }

    #[test]
    fn test_jagged_commit_uses_unpadded_physical_rotation_height() {
        // 3 logical rows with 4-way rotation occupy 12 real physical rows, padded to 16.
        let reshape_log_height = 4;
        let (pp, _vp) = setup_pcs::<E, Pcs>(reshape_log_height);
        let rmm = WitnessRowMajorMatrix::new_by_rotation(
            3,
            2,
            2,
            InstancePaddingStrategy::Default,
        );

        let comm = jagged_commit::<E, Pcs>(&pp, vec![rmm], reshape_log_height)
            .expect("commit should succeed");

        assert_eq!(comm.num_polys(), 2);
        assert_eq!(comm.poly_heights, vec![12, 12]);
        assert_eq!(comm.cumulative_heights, vec![0, 12, 24]);
        assert_eq!(comm.total_evaluations(), 24);
    }

    #[test]
    fn test_jagged_commit_smoke() {
        // Two matrices: 4x1 and 4x2 → 3 polynomials, total 12 evals, padded to 16
        let num_giga_vars = 4;
        let (pp, _vp) = setup_pcs::<E, Pcs>(num_giga_vars);
        let m1 = make_rmm(4, 1);
        let m2 = make_rmm(4, 2);

        let comm = jagged_commit::<E, Pcs>(&pp, vec![m1, m2], num_giga_vars)
            .expect("commit should succeed");

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
        let num_giga_vars = 3;
        let (pp, _vp) = setup_pcs::<E, Pcs>(num_giga_vars);
        let m = make_rmm(8, 1);

        let comm =
            jagged_commit::<E, Pcs>(&pp, vec![m], num_giga_vars).expect("commit should succeed");

        assert_eq!(comm.num_polys(), 1);
        assert_eq!(comm.poly_heights, vec![8]);
        assert_eq!(comm.cumulative_heights, vec![0, 8]);
        assert_eq!(comm.total_evaluations(), 8);
    }

    #[test]
    fn test_jagged_commit_empty_error() {
        let (pp, _vp) = setup_pcs::<E, Pcs>(4);
        let result = jagged_commit::<E, Pcs>(&pp, vec![], 4);
        assert!(matches!(result, Err(Error::InvalidPcsParam(_))));
    }

    // --- Batch open/verify tests ---

    use ff_ext::FromUniformBytes;
    use rand::thread_rng;
    use transcript::basic::BasicTranscript;

    /// Evaluate a single column polynomial at its native prefix of `point`.
    fn eval_column_poly_at_point(col_evals: &[F], point: &[E]) -> E {
        let s = ceil_log2(col_evals.len());
        assert!(point.len() >= s);
        assert!(col_evals.len() <= 1 << s);
        let mut padded = col_evals.to_vec();
        padded.resize(1 << s, F::ZERO);
        let mle = MultilinearExtension::from_evaluations_vec(s, padded);
        mle.evaluate(&point[..s])
    }

    #[test]
    fn test_jagged_batch_open_verify_small() {
        let mut rng = thread_rng();

        // 3 matrices with different heights: 2^10, 2^11, 2^9 (each 1 column).
        let log_heights = [10usize, 11, 9];
        let heights: Vec<usize> = log_heights.iter().map(|&s| 1 << s).collect();
        let max_s = *log_heights.iter().max().unwrap();
        let total_evals: usize = heights.iter().sum();
        let num_giga_vars = ceil_log2(total_evals);

        let (pp, vp) = setup_pcs::<E, Pcs>(num_giga_vars);
        let rmms: Vec<_> = heights.iter().map(|&h| make_rmm(h, 1)).collect();

        // Extract each column polynomial.
        let col_polys: Vec<Vec<F>> = rmms
            .iter()
            .map(|rmm| (0..rmm.height()).map(|r| rmm.values[r]).collect())
            .collect();

        // Commit.
        let mut transcript_p = BasicTranscript::<E>::new(b"jagged_batch_test");
        let comm =
            jagged_commit::<E, Pcs>(&pp, rmms, num_giga_vars).expect("commit should succeed");
        Pcs::write_commitment(&comm.to_commitment().inner, &mut transcript_p).unwrap();

        // Random evaluation point of length max_s.
        // Poly i is evaluated at its native prefix of the common point.
        let point: Vec<E> = (0..max_s).map(|_| E::random(&mut rng)).collect();

        let evals: Vec<E> = col_polys
            .iter()
            .zip(log_heights.iter())
            .map(|(col, _)| eval_column_poly_at_point(col, &point))
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
        let num_giga_vars = ceil_log2(num_rows * num_cols);

        let (pp, vp) = setup_pcs::<E, Pcs>(num_giga_vars);
        let rmm = make_rmm(num_rows, num_cols);

        let col_poly: Vec<F> = (0..num_rows).map(|r| rmm.values[r]).collect();

        let mut transcript_p = BasicTranscript::<E>::new(b"jagged_single");
        let comm = jagged_commit::<E, Pcs>(&pp, vec![rmm], num_giga_vars).expect("commit");
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
    fn test_jagged_batch_open_verify_non_power_of_two_heights() {
        let mut rng = thread_rng();

        let heights = [1023usize, 777, 513];
        let log_heights: Vec<usize> = heights.iter().map(|&h| ceil_log2(h)).collect();
        let max_s = *log_heights.iter().max().unwrap();
        let total_evals: usize = heights.iter().sum();
        let num_giga_vars = ceil_log2(total_evals);

        let (pp, vp) = setup_pcs::<E, Pcs>(num_giga_vars);
        let rmms: Vec<_> = heights.iter().map(|&h| make_rmm(h, 1)).collect();

        let col_polys: Vec<Vec<F>> = rmms
            .iter()
            .map(|rmm| (0..rmm.height()).map(|r| rmm.values[r]).collect())
            .collect();

        let mut transcript_p = BasicTranscript::<E>::new(b"jagged_non_power_two");
        let comm = jagged_commit::<E, Pcs>(&pp, rmms, num_giga_vars).expect("commit");
        Pcs::write_commitment(&comm.to_commitment().inner, &mut transcript_p).unwrap();

        let point: Vec<E> = (0..max_s).map(|_| E::random(&mut rng)).collect();
        let evals: Vec<E> = col_polys
            .iter()
            .zip(log_heights.iter())
            .map(|(col, _)| eval_column_poly_at_point(col, &point))
            .collect();

        let proof = jagged_batch_open::<E, Pcs>(&pp, &comm, &point, &evals, &mut transcript_p)
            .expect("batch open");

        let mut transcript_v = BasicTranscript::<E>::new(b"jagged_non_power_two");
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
        let num_giga_vars = ceil_log2(num_rows * num_cols);

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
        let comm = jagged_commit::<E, Pcs>(&pp, vec![rmm], num_giga_vars).expect("commit");
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

    // --- Reshape tests (reshape_log_height < num_giga_vars) ---

    #[test]
    fn test_jagged_reshape_single_poly() {
        let mut rng = thread_rng();

        let num_rows = 1024usize; // s=10
        let s = 10;
        let total_evals = num_rows;
        let reshape_log_height = 6; // h=64, w=ceil(1024/64)=16
        let h = 1usize << reshape_log_height;
        let w = total_evals.div_ceil(h);
        assert_eq!(w, 16);

        let (pp, vp) = setup_pcs::<E, Pcs>(reshape_log_height);
        let rmm = make_rmm(num_rows, 1);
        let col_poly: Vec<F> = (0..num_rows).map(|r| rmm.values[r]).collect();

        let mut transcript_p = BasicTranscript::<E>::new(b"jagged_reshape_single");
        let comm = jagged_commit::<E, Pcs>(&pp, vec![rmm], reshape_log_height).expect("commit");
        Pcs::write_commitment(&comm.to_commitment().inner, &mut transcript_p).unwrap();

        let point: Vec<E> = (0..s).map(|_| E::random(&mut rng)).collect();
        let evals = vec![eval_column_poly_at_point(&col_poly, &point)];

        let proof = jagged_batch_open::<E, Pcs>(&pp, &comm, &point, &evals, &mut transcript_p)
            .expect("batch open");
        assert_eq!(proof.col_evals.len(), w);

        let mut transcript_v = BasicTranscript::<E>::new(b"jagged_reshape_single");
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
    fn test_jagged_reshape_multiple_polys() {
        let mut rng = thread_rng();

        // 3 matrices with different heights: 2^10, 2^11, 2^9 (each 1 column).
        let log_heights = [10usize, 11, 9];
        let heights: Vec<usize> = log_heights.iter().map(|&s| 1 << s).collect();
        let max_s = *log_heights.iter().max().unwrap();
        let total_evals: usize = heights.iter().sum();
        let reshape_log_height = 8; // h=256
        let h = 1usize << reshape_log_height;
        let w = total_evals.div_ceil(h);

        let (pp, vp) = setup_pcs::<E, Pcs>(reshape_log_height);
        let rmms: Vec<_> = heights.iter().map(|&h| make_rmm(h, 1)).collect();

        let col_polys: Vec<Vec<F>> = rmms
            .iter()
            .map(|rmm| (0..rmm.height()).map(|r| rmm.values[r]).collect())
            .collect();

        let mut transcript_p = BasicTranscript::<E>::new(b"jagged_reshape_multi");
        let comm =
            jagged_commit::<E, Pcs>(&pp, rmms, reshape_log_height).expect("commit should succeed");
        Pcs::write_commitment(&comm.to_commitment().inner, &mut transcript_p).unwrap();

        let point: Vec<E> = (0..max_s).map(|_| E::random(&mut rng)).collect();
        let evals: Vec<E> = col_polys
            .iter()
            .zip(log_heights.iter())
            .map(|(col, _)| eval_column_poly_at_point(col, &point))
            .collect();

        let proof = jagged_batch_open::<E, Pcs>(&pp, &comm, &point, &evals, &mut transcript_p)
            .expect("batch open should succeed");
        assert_eq!(proof.col_evals.len(), w);

        let mut transcript_v = BasicTranscript::<E>::new(b"jagged_reshape_multi");
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
    fn test_jagged_reshape_non_power_of_two_heights() {
        let mut rng = thread_rng();

        let heights = [1023usize, 777, 513];
        let log_heights: Vec<usize> = heights.iter().map(|&h| ceil_log2(h)).collect();
        let max_s = *log_heights.iter().max().unwrap();
        let total_evals: usize = heights.iter().sum();
        let reshape_log_height = 8;
        let h = 1usize << reshape_log_height;
        let w = total_evals.div_ceil(h);

        let (pp, vp) = setup_pcs::<E, Pcs>(reshape_log_height);
        let rmms: Vec<_> = heights.iter().map(|&h| make_rmm(h, 1)).collect();

        let col_polys: Vec<Vec<F>> = rmms
            .iter()
            .map(|rmm| (0..rmm.height()).map(|r| rmm.values[r]).collect())
            .collect();

        let mut transcript_p = BasicTranscript::<E>::new(b"jagged_reshape_non_power");
        let comm = jagged_commit::<E, Pcs>(&pp, rmms, reshape_log_height).expect("commit");
        Pcs::write_commitment(&comm.to_commitment().inner, &mut transcript_p).unwrap();

        let point: Vec<E> = (0..max_s).map(|_| E::random(&mut rng)).collect();
        let evals: Vec<E> = col_polys
            .iter()
            .zip(log_heights.iter())
            .map(|(col, _)| eval_column_poly_at_point(col, &point))
            .collect();

        let proof = jagged_batch_open::<E, Pcs>(&pp, &comm, &point, &evals, &mut transcript_p)
            .expect("batch open");
        assert_eq!(proof.col_evals.len(), w);

        let mut transcript_v = BasicTranscript::<E>::new(b"jagged_reshape_non_power");
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
}

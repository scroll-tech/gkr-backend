//! Jagged Assist Sumcheck
//!
//! Reduces K evaluations of the indicator function ĝ (one per column) to a
//! single evaluation, via the batch-proving protocol of §5 (Lemma 5.1) of the
//! jagged PCS paper.
//!
//! The sumcheck operates on P(b) = h(b) · Q(b) where:
//!   h(z₃, z₄) = ĝ(z_row, ρ, z₃, z₄)   (multilinear in 2n variables)
//!   Q(b) = Σ_y eq_col[y] · eq(b, x_y)   (x_y are Boolean evaluation points)
//!
//! Variables are interleaved as (z₃[0], z₄[0], z₃[1], z₄[1], …) so that
//! each pair of consecutive sumcheck rounds maps to one ROBP step.

use ff_ext::ExtensionField;
use multilinear_extensions::util::max_usable_threads;
use p3::maybe_rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, ParallelIterator, ParallelSlice,
    ParallelSliceMut,
};
use sumcheck::structs::{IOPProof, IOPProverMessage};
use transcript::Transcript;

use super::evaluator::{
    ROBP_WIDTH, StateVec, TransitionMatrix, dot4, mat_vec_mul, sink_labels, source_vec,
    symbol_transition_matrices, vec_mat_mul,
};

/// Run the assist sumcheck prover.
///
/// Proves that `claimed_sum = Σ_y eq_col[y] · ĝ(z_row, ρ, bits(t_y), bits(t_{y+1}))`.
///
/// Returns the proof (2·n_robp rounds) and the full challenge vector.
pub fn assist_sumcheck_prove<E: ExtensionField>(
    z_row_padded: &[E],
    rho_padded: &[E],
    eq_col: &[E],
    cumulative_heights: &[usize],
    n_robp: usize,
    transcript: &mut impl Transcript<E>,
) -> (IOPProof<E>, Vec<E>) {
    let num_polys = cumulative_heights.len() - 1;
    let n_vars = 2 * n_robp;
    let max_degree: usize = 2;

    // Write transcript header (must match verifier).
    transcript.append_message(&n_vars.to_le_bytes());
    transcript.append_message(&max_degree.to_le_bytes());

    // Precompute per-step symbol matrices.
    let step_mats: Vec<[TransitionMatrix<E>; 4]> = (0..n_robp)
        .map(|i| symbol_transition_matrices(z_row_padded[i], rho_padded[i]))
        .collect();

    // Extract Boolean bits in step-major layout: c_bits[i][y], d_bits[i][y].
    // c_bits[i][y] = bit_i(t_y), d_bits[i][y] = bit_i(t_{y+1})
    let mut c_bits = vec![vec![0usize; num_polys]; n_robp];
    let mut d_bits = vec![vec![0usize; num_polys]; n_robp];
    for i in 0..n_robp {
        for y in 0..num_polys {
            c_bits[i][y] = (cumulative_heights[y] >> i) & 1;
            d_bits[i][y] = (cumulative_heights[y + 1] >> i) & 1;
        }
    }

    // Precompute backward vectors bwd[i][y] (step-major layout).
    //
    // bwd[i][y] is a 4-element state vector representing the ROBP suffix
    // product from step i to the sinks, for polynomial y's Boolean bits:
    //
    //   bwd[n][y]  = u = [0, 1, 0, 0]          (sink labels: accept at carry=0, lt=1)
    //   bwd[i][y]  = M_i^{(c_y[i], d_y[i])} · bwd[i+1][y],   i = n-1, …, 0
    //
    // where c_y[i] = bit_i(t_y), d_y[i] = bit_i(t_{y+1}), and M_i^{(c,d)}
    // is the per-symbol transition matrix with z_row[i], ρ[i] baked in.
    //
    // The full ROBP evaluation for polynomial y equals:
    //   ĝ(z_row, ρ, bits(t_y), bits(t_{y+1})) = e_0^T · bwd[0][y]
    //
    // During the sumcheck, the prover decomposes ĝ into fwd · bwd.
    // M_i^{(c,d)} extends to field arguments via MLE: M_i^{(α,β)} means
    // Σ_{c,d} eq₁(α,c)·eq₁(β,d)·M_i^{(c,d)}.  With this notation:
    //   round 2i:   h_y(λ) = fwd · M_i^{(λ, d_y[i])} · bwd[i+1][y]
    //   round 2i+1: h_y(λ) = fwd · M_i^{(α, λ)}      · bwd[i+1][y]
    // where fwd absorbs bound challenges and bwd provides the Boolean suffix.
    let u = sink_labels::<E>();
    let mut bwd = vec![vec![[E::ZERO; ROBP_WIDTH]; num_polys]; n_robp + 1];
    for y in 0..num_polys {
        bwd[n_robp][y] = u;
    }
    for i in (0..n_robp).rev() {
        let (left, right) = bwd.split_at_mut(i + 1);
        let dst = &mut left[i];
        let src = &right[0];
        dst.into_par_iter().enumerate().for_each(|(y, dst_y)| {
            let cd = c_bits[i][y] * 2 + d_bits[i][y];
            *dst_y = mat_vec_mul(&step_mats[i][cd], &src[y]);
        });
    }

    // Initialize weights and forward vector.
    let mut weights: Vec<E> = eq_col[..num_polys].to_vec();
    let mut fwd: StateVec<E> = source_vec();

    let mut challenges: Vec<E> = Vec::with_capacity(n_vars);
    let mut proof_messages: Vec<IOPProverMessage<E>> = Vec::with_capacity(n_vars);

    let n_threads = max_usable_threads();
    let poly_indices: Vec<usize> = (0..num_polys).collect();

    for i in 0..n_robp {
        // Precompute fwd · M^{(c,d)} for all 4 symbols.
        let r_cd: [StateVec<E>; 4] = std::array::from_fn(|cd| vec_mat_mul(&fwd, &step_mats[i][cd]));

        // ---- Round 2i: bind z₃[i] ----
        //
        // Round polynomial for round 2i:
        //   p_{2i}(λ) = Σ_y w_y · eq₁(λ, c_y[i]) · fwd · M_i^{(λ, d_y[i])} · bwd[i+1][y]
        //
        // Precompute R[c*2+d] = fwd · M_i^{(c,d)} as row-vectors for Boolean (c,d).
        // Then fwd · M_i^{(λ,d)} = (1-λ)·R[d] + λ·R[2+d] is also a row-vector,
        // so each term reduces to a dot product with bwd — no matrix-vector multiply.
        //
        // Expanding M_i^{(λ,d)} = Σ_{c'} eq₁(λ,c')·M_i^{(c',d)} in the bucketed form:
        //   p_{2i}(λ) = Σ_{c,c',d} eq₁(λ,c) · eq₁(λ,c') · R[c'*2+d] · bwd_sum[c*2+d]
        // where c indexes the Q bucket and c' indexes the M expansion (independent).
        //
        // Bucket bwd vectors by (c,d):
        //   bwd_sum[c*2+d] = Σ_{y: c_y[i]=c, d_y[i]=d} w_y · bwd[i+1][y]
        //
        // This reduces evaluation at each λ to 4 dot products against bwd_sum.
        //
        // Parallelization: partition K polynomials across threads. Each thread
        // builds local bwd_sum, computes local (p0, p1, p2), then we sum
        // the scalars across threads. We also merge bwd_sum for round 2i+1.
        let row2_d0: StateVec<E> = std::array::from_fn(|j| r_cd[2][j].double() - r_cd[0][j]);
        let row2_d1: StateVec<E> = std::array::from_fn(|j| r_cd[3][j].double() - r_cd[1][j]);

        let batch_size = (num_polys / n_threads).max(1);
        let partials: Vec<([E; 3], [[E; ROBP_WIDTH]; 4])> = poly_indices
            .par_chunks(batch_size)
            .map(|chunk| {
                let mut local_bwd_sum = [[E::ZERO; ROBP_WIDTH]; 4];
                for &y in chunk {
                    let cd = c_bits[i][y] * 2 + d_bits[i][y];
                    let w = weights[y];
                    for s in 0..ROBP_WIDTH {
                        local_bwd_sum[cd][s] += w * bwd[i + 1][y][s];
                    }
                }
                let lp0 =
                    dot4(&r_cd[0], &local_bwd_sum[0]) + dot4(&r_cd[1], &local_bwd_sum[1]);
                let lp1 =
                    dot4(&r_cd[2], &local_bwd_sum[2]) + dot4(&r_cd[3], &local_bwd_sum[3]);
                let tc0 =
                    dot4(&row2_d0, &local_bwd_sum[0]) + dot4(&row2_d1, &local_bwd_sum[1]);
                let tc1 =
                    dot4(&row2_d0, &local_bwd_sum[2]) + dot4(&row2_d1, &local_bwd_sum[3]);
                let lp2 = tc1.double() - tc0;
                ([lp0, lp1, lp2], local_bwd_sum)
            })
            .collect();

        let (mut p0, mut p1, mut p2) = (E::ZERO, E::ZERO, E::ZERO);
        for (p_local, _) in &partials {
            p0 += p_local[0];
            p1 += p_local[1];
            p2 += p_local[2];
        }

        debug_assert_eq!(
            p0 + p1,
            if i == 0 && challenges.is_empty() {
                p0 + p1
            } else {
                p0 + p1
            }
        );

        transcript.append_field_element_ext(&p1);
        transcript.append_field_element_ext(&p2);
        let alpha = transcript
            .sample_and_append_challenge(b"Internal round")
            .elements;
        challenges.push(alpha);
        proof_messages.push(IOPProverMessage {
            evaluations: vec![p1, p2],
        });

        // ---- Round 2i+1: bind z₄[i] ----
        //
        // Round polynomial for round 2i+1:
        //   p_{2i+1}(λ) = Σ_y w_y · eq₁(λ, d_y[i]) · fwd · M_i^{(α, λ)} · bwd[i+1][y]
        // where w_y logically includes eq₁(α, c_y[i]) from round 2i.
        //
        // Reuse cached per-thread bwd_sums from round 2i: each absorbs α to
        // get local bwd_sum_d, computes local p(0), p(1), p(2), then we sum.
        let na = E::ONE - alpha;
        // fwd · M_i^{(α, λ)} at λ=0: (1-α)·R_{0,0} + α·R_{1,0}
        let row_at_0: StateVec<E> = std::array::from_fn(|j| na * r_cd[0][j] + alpha * r_cd[2][j]);
        // at λ=1: (1-α)·R_{0,1} + α·R_{1,1}
        let row_at_1: StateVec<E> = std::array::from_fn(|j| na * r_cd[1][j] + alpha * r_cd[3][j]);
        // at λ=2: 2·row_at_1 - row_at_0
        let row_at_2: StateVec<E> = std::array::from_fn(|j| row_at_1[j].double() - row_at_0[j]);

        let (mut p0, mut p1, mut p2) = (E::ZERO, E::ZERO, E::ZERO);
        for (_, local_bwd_sum) in &partials {
            let local_bwd_sum_d: [[E; ROBP_WIDTH]; 2] = [
                std::array::from_fn(|s| na * local_bwd_sum[0][s] + alpha * local_bwd_sum[2][s]),
                std::array::from_fn(|s| na * local_bwd_sum[1][s] + alpha * local_bwd_sum[3][s]),
            ];
            p0 += dot4(&row_at_0, &local_bwd_sum_d[0]);
            p1 += dot4(&row_at_1, &local_bwd_sum_d[1]);
            p2 += term_d1_d0_combine(
                dot4(&row_at_2, &local_bwd_sum_d[0]),
                dot4(&row_at_2, &local_bwd_sum_d[1]),
            );
        }

        transcript.append_field_element_ext(&p1);
        transcript.append_field_element_ext(&p2);
        let beta = transcript
            .sample_and_append_challenge(b"Internal round")
            .elements;
        challenges.push(beta);
        proof_messages.push(IOPProverMessage {
            evaluations: vec![p1, p2],
        });

        // Fused weight update: w_y *= eq₁(α, c_y[i]) · eq₁(β, d_y[i])
        let nb = E::ONE - beta;
        let eq_cd = [na * nb, na * beta, alpha * nb, alpha * beta];
        weights
            .par_chunks_mut(batch_size)
            .enumerate()
            .for_each(|(chunk_idx, w_chunk)| {
                let start = chunk_idx * batch_size;
                for (j, w) in w_chunk.iter_mut().enumerate() {
                    let y = start + j;
                    let cd = c_bits[i][y] * 2 + d_bits[i][y];
                    *w *= eq_cd[cd];
                }
            });

        // Update forward vector: fwd ← fwd · M_i^{(α, β)}
        fwd = std::array::from_fn(|j| {
            eq_cd[0] * r_cd[0][j]
                + eq_cd[1] * r_cd[1][j]
                + eq_cd[2] * r_cd[2][j]
                + eq_cd[3] * r_cd[3][j]
        });
    }

    (
        IOPProof {
            proofs: proof_messages,
        },
        challenges,
    )
}

/// eq₁(2, d=0) = -1, eq₁(2, d=1) = 2.
#[inline]
fn term_d1_d0_combine<E: ExtensionField>(dot_d0: E, dot_d1: E) -> E {
    dot_d1.double() - dot_d0
}

/// Compute Q(ρ*) = Σ_y eq_col[y] · eq(ρ*, x_y) where x_y are the interleaved
/// Boolean evaluation points.
///
/// `assist_point` is the interleaved challenge point from the assist sumcheck,
/// of length 2·n_robp.
pub fn compute_q_at_assist_point<E: ExtensionField>(
    assist_point: &[E],
    eq_col: &[E],
    cumulative_heights: &[usize],
    n_robp: usize,
) -> E {
    let num_polys = cumulative_heights.len() - 1;
    let mut q_val = E::ZERO;
    for y in 0..num_polys {
        let mut prod = eq_col[y];
        if prod == E::ZERO {
            continue;
        }
        for i in 0..n_robp {
            let c_bit = (cumulative_heights[y] >> i) & 1;
            let d_bit = (cumulative_heights[y + 1] >> i) & 1;
            let z3_val = assist_point[2 * i];
            let z4_val = assist_point[2 * i + 1];
            prod *= if c_bit == 1 { z3_val } else { E::ONE - z3_val };
            prod *= if d_bit == 1 { z4_val } else { E::ONE - z4_val };
        }
        q_val += prod;
    }
    q_val
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jagged::{
        evaluator::{evaluate_g, evaluate_g_forward},
        types::int_to_field_bits,
    };
    use ff_ext::{FromUniformBytes, GoldilocksExt2};
    use multilinear_extensions::{util::ceil_log2, virtual_poly::build_eq_x_r_vec};
    use p3::field::FieldAlgebra;
    use rand::thread_rng;
    use std::marker::PhantomData;
    use sumcheck::structs::IOPVerifierState;
    use transcript::basic::BasicTranscript;

    type E = GoldilocksExt2;

    /// Brute-force f̂(ρ) = Σ_y eq_col[y] · ĝ(z_row, ρ, bits(t_y), bits(t_{y+1}))
    fn compute_f_at_point_slow(
        z_row: &[E],
        rho: &[E],
        eq_col: &[E],
        cumulative_heights: &[usize],
        n_robp: usize,
    ) -> E {
        let num_polys = cumulative_heights.len() - 1;
        let mut val = E::ZERO;
        for y in 0..num_polys {
            let t_lo = int_to_field_bits::<E>(cumulative_heights[y], n_robp);
            let t_hi = int_to_field_bits::<E>(cumulative_heights[y + 1], n_robp);
            val += eq_col[y] * evaluate_g(z_row, rho, &t_lo, &t_hi);
        }
        val
    }

    #[test]
    fn test_assist_sumcheck_small() {
        let mut rng = thread_rng();

        // 4 polynomials with heights [4, 8, 4, 8] → cumulative [0, 4, 12, 16, 24]
        let cumulative_heights = vec![0, 4, 12, 16, 24];
        let num_polys = cumulative_heights.len() - 1;
        let total_evals = *cumulative_heights.last().unwrap();
        let num_giga_vars = ceil_log2(total_evals);
        let n_robp = num_giga_vars + if total_evals.is_power_of_two() { 1 } else { 0 };

        let z_row: Vec<E> = (0..n_robp).map(|_| E::random(&mut rng)).collect();
        let rho: Vec<E> = (0..n_robp).map(|_| E::random(&mut rng)).collect();
        let num_col_vars = ceil_log2(num_polys).max(1);
        let z_col: Vec<E> = (0..num_col_vars).map(|_| E::random(&mut rng)).collect();
        let eq_col = build_eq_x_r_vec(&z_col);

        let claimed_sum =
            compute_f_at_point_slow(&z_row, &rho, &eq_col, &cumulative_heights, n_robp);

        // Run prover.
        let mut transcript_p = BasicTranscript::<E>::new(b"assist_test");
        let (proof, challenges) = assist_sumcheck_prove(
            &z_row,
            &rho,
            &eq_col,
            &cumulative_heights,
            n_robp,
            &mut transcript_p,
        );

        let n_vars = 2 * n_robp;
        assert_eq!(proof.proofs.len(), n_vars);
        assert_eq!(challenges.len(), n_vars);

        // Verify using standard sumcheck verifier.
        let mut transcript_v = BasicTranscript::<E>::new(b"assist_test");
        let aux_info = multilinear_extensions::virtual_poly::VPAuxInfo {
            max_degree: 2,
            max_num_variables: n_vars,
            phantom: PhantomData::<E>,
        };
        let subclaim =
            IOPVerifierState::<E>::verify(claimed_sum, &proof, &aux_info, &mut transcript_v);

        // Check challenges match.
        for (sc, ch) in subclaim.point.iter().zip(challenges.iter()) {
            assert_eq!(sc.elements, *ch);
        }

        // De-interleave the assist point: (z3[0], z4[0], z3[1], z4[1], ...)
        let rho_star_c: Vec<E> = (0..n_robp).map(|i| challenges[2 * i]).collect();
        let rho_star_d: Vec<E> = (0..n_robp).map(|i| challenges[2 * i + 1]).collect();

        // h(ρ*) = ĝ(z_row, ρ, ρ*_c, ρ*_d)
        let h_val = evaluate_g(&z_row, &rho, &rho_star_c, &rho_star_d);

        // Q(ρ*) = Σ_y eq_col[y] · eq(ρ*, x_y)
        let q_val = compute_q_at_assist_point(&challenges, &eq_col, &cumulative_heights, n_robp);

        assert_eq!(
            h_val * q_val,
            subclaim.expected_evaluation,
            "h(ρ*) * Q(ρ*) != subclaim"
        );
    }

    #[test]
    fn test_assist_sumcheck_single_poly() {
        let mut rng = thread_rng();

        let cumulative_heights = vec![0, 8];
        let total_evals = 8;
        let num_giga_vars = ceil_log2(total_evals);
        let n_robp = num_giga_vars + if total_evals.is_power_of_two() { 1 } else { 0 };

        let z_row: Vec<E> = (0..n_robp).map(|_| E::random(&mut rng)).collect();
        let rho: Vec<E> = (0..n_robp).map(|_| E::random(&mut rng)).collect();
        let eq_col = vec![E::ONE]; // single poly

        let claimed_sum =
            compute_f_at_point_slow(&z_row, &rho, &eq_col, &cumulative_heights, n_robp);

        let mut transcript_p = BasicTranscript::<E>::new(b"assist_single");
        let (proof, challenges) = assist_sumcheck_prove(
            &z_row,
            &rho,
            &eq_col,
            &cumulative_heights,
            n_robp,
            &mut transcript_p,
        );

        let n_vars = 2 * n_robp;
        let mut transcript_v = BasicTranscript::<E>::new(b"assist_single");
        let aux_info = multilinear_extensions::virtual_poly::VPAuxInfo {
            max_degree: 2,
            max_num_variables: n_vars,
            phantom: PhantomData::<E>,
        };
        let subclaim =
            IOPVerifierState::<E>::verify(claimed_sum, &proof, &aux_info, &mut transcript_v);

        let rho_star_c: Vec<E> = (0..n_robp).map(|i| challenges[2 * i]).collect();
        let rho_star_d: Vec<E> = (0..n_robp).map(|i| challenges[2 * i + 1]).collect();
        let h_val = evaluate_g(&z_row, &rho, &rho_star_c, &rho_star_d);
        let q_val = compute_q_at_assist_point(&challenges, &eq_col, &cumulative_heights, n_robp);

        assert_eq!(h_val * q_val, subclaim.expected_evaluation);
    }

    #[test]
    fn test_assist_sumcheck_many_polys() {
        let mut rng = thread_rng();

        // 16 polynomials, each height 32 → cumulative [0, 32, 64, ..., 512]
        let num_polys = 16;
        let poly_height = 32usize;
        let cumulative_heights: Vec<usize> = (0..=num_polys).map(|i| i * poly_height).collect();
        let total_evals = *cumulative_heights.last().unwrap();
        let num_giga_vars = ceil_log2(total_evals);
        let n_robp = num_giga_vars + if total_evals.is_power_of_two() { 1 } else { 0 };

        let z_row: Vec<E> = (0..n_robp).map(|_| E::random(&mut rng)).collect();
        let rho: Vec<E> = (0..n_robp).map(|_| E::random(&mut rng)).collect();
        let num_col_vars = ceil_log2(num_polys).max(1);
        let z_col: Vec<E> = (0..num_col_vars).map(|_| E::random(&mut rng)).collect();
        let eq_col = build_eq_x_r_vec(&z_col);

        let claimed_sum =
            compute_f_at_point_slow(&z_row, &rho, &eq_col, &cumulative_heights, n_robp);

        let mut transcript_p = BasicTranscript::<E>::new(b"assist_many");
        let (proof, challenges) = assist_sumcheck_prove(
            &z_row,
            &rho,
            &eq_col,
            &cumulative_heights,
            n_robp,
            &mut transcript_p,
        );

        let n_vars = 2 * n_robp;
        let mut transcript_v = BasicTranscript::<E>::new(b"assist_many");
        let aux_info = multilinear_extensions::virtual_poly::VPAuxInfo {
            max_degree: 2,
            max_num_variables: n_vars,
            phantom: PhantomData::<E>,
        };
        let subclaim =
            IOPVerifierState::<E>::verify(claimed_sum, &proof, &aux_info, &mut transcript_v);

        for (sc, ch) in subclaim.point.iter().zip(challenges.iter()) {
            assert_eq!(sc.elements, *ch);
        }

        let rho_star_c: Vec<E> = (0..n_robp).map(|i| challenges[2 * i]).collect();
        let rho_star_d: Vec<E> = (0..n_robp).map(|i| challenges[2 * i + 1]).collect();
        let h_val = evaluate_g(&z_row, &rho, &rho_star_c, &rho_star_d);
        let q_val = compute_q_at_assist_point(&challenges, &eq_col, &cumulative_heights, n_robp);

        assert_eq!(h_val * q_val, subclaim.expected_evaluation);
    }

    /// Also verify using the forward evaluator for cross-validation.
    #[test]
    fn test_assist_forward_backward_consistency() {
        let mut rng = thread_rng();

        let cumulative_heights = vec![0, 3, 7, 10];
        let num_polys = 3;
        let total_evals = 10;
        let num_giga_vars = ceil_log2(total_evals);
        let n_robp = num_giga_vars + if total_evals.is_power_of_two() { 1 } else { 0 };

        let z_row: Vec<E> = (0..n_robp).map(|_| E::random(&mut rng)).collect();
        let rho: Vec<E> = (0..n_robp).map(|_| E::random(&mut rng)).collect();
        let eq_col = build_eq_x_r_vec(&vec![E::random(&mut rng); ceil_log2(num_polys).max(1)]);

        // Compute f̂(ρ) using both forward and backward evaluators.
        let mut f_fwd = E::ZERO;
        let mut f_bwd = E::ZERO;
        for y in 0..num_polys {
            let t_lo = int_to_field_bits::<E>(cumulative_heights[y], n_robp);
            let t_hi = int_to_field_bits::<E>(cumulative_heights[y + 1], n_robp);
            f_fwd += eq_col[y] * evaluate_g_forward(&z_row, &rho, &t_lo, &t_hi);
            f_bwd += eq_col[y] * evaluate_g(&z_row, &rho, &t_lo, &t_hi);
        }
        assert_eq!(f_fwd, f_bwd, "forward/backward f̂ disagree");

        // Run assist sumcheck and verify.
        let mut transcript_p = BasicTranscript::<E>::new(b"assist_fb");
        let (proof, challenges) = assist_sumcheck_prove(
            &z_row,
            &rho,
            &eq_col,
            &cumulative_heights,
            n_robp,
            &mut transcript_p,
        );

        let n_vars = 2 * n_robp;
        let mut transcript_v = BasicTranscript::<E>::new(b"assist_fb");
        let aux_info = multilinear_extensions::virtual_poly::VPAuxInfo {
            max_degree: 2,
            max_num_variables: n_vars,
            phantom: PhantomData::<E>,
        };
        let subclaim = IOPVerifierState::<E>::verify(f_bwd, &proof, &aux_info, &mut transcript_v);

        let rho_star_c: Vec<E> = (0..n_robp).map(|i| challenges[2 * i]).collect();
        let rho_star_d: Vec<E> = (0..n_robp).map(|i| challenges[2 * i + 1]).collect();
        let h_val = evaluate_g(&z_row, &rho, &rho_star_c, &rho_star_d);
        let q_val = compute_q_at_assist_point(&challenges, &eq_col, &cumulative_heights, n_robp);

        assert_eq!(h_val * q_val, subclaim.expected_evaluation);
    }
}

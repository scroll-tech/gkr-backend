//! Jagged Sumcheck Prover
//!
//! Streaming sumcheck prover using the M-table algorithm from
//! "Time-Space Trade-Offs for Sumcheck" (eprint 2025/1473), Section 4.

use ff_ext::ExtensionField;
use multilinear_extensions::{
    mle::MultilinearExtension, util::max_usable_threads, virtual_poly::build_eq_x_r_vec,
};
use p3::maybe_rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, ParallelIterator, ParallelSlice,
    ParallelSliceMut,
};
use sumcheck::{
    macros::{entered_span, exit_span},
    structs::{IOPProof, IOPProverMessage, IOPProverState},
};
use transcript::Transcript;

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

#[cfg(test)]
mod tests {
    use super::*;
    use ff_ext::{FromUniformBytes, GoldilocksExt2};
    use multilinear_extensions::virtual_poly::{VPAuxInfo, build_eq_x_r_vec};
    use p3::{field::FieldAlgebra, goldilocks::Goldilocks};
    use rand::thread_rng;
    use std::marker::PhantomData;
    use sumcheck::structs::IOPVerifierState;
    use transcript::basic::BasicTranscript;

    type F = Goldilocks;
    type E = GoldilocksExt2;

    #[test]
    fn test_jagged_sumcheck_small() {
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
}

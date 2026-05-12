use std::sync::Arc;

use ff_ext::ExtensionField;
use itertools::Itertools;
use multilinear_extensions::{
    mle::{FieldType, MultilinearExtension},
    monomial::Term,
    virtual_poly::{MonomialTerms, VPAuxInfo, VirtualPolynomial},
    virtual_polys::{PolyMeta, VirtualPolynomials},
};
use p3::field::FieldAlgebra;
use rayon::prelude::*;
use transcript::{Challenge, Transcript};

use crate::{
    structs::{IOPProof, IOPProverMessage},
    util::extrapolate_from_table,
};

/// State returned by the frontload sumcheck prover.
///
/// Frontload embeds a compact `k`-variable MLE into an `N`-variable sumcheck
/// domain by binding the real MLE to the early variables and representing every
/// missing later variable as a multiplicative tail factor:
///
/// ```text
/// F(x0, ..., x{N-1}) = f(x0, ..., x{k-1}) * product_{i=k}^{N-1} x_i
/// ```
///
/// The compact MLE is not expanded to `2^N`. Once all of its real variables are
/// fixed, it keeps contributing through those tail factors.
///
/// # Two-phase worker example
///
/// Suppose there are four workers, so `log2(num_workers) = 2`, and the global
/// sumcheck domain has `2^10` points:
///
/// ```text
/// global variables:  x0..x9
/// phase-1 variables: x0..x7
/// phase-2 variables: x8,x9
/// polynomial:        a + b + c
/// a size:            2^10
/// b size:            2^1
/// c size:            2^5
/// ```
///
/// MLEs larger than the worker space are `Normal`: they are split across
/// workers and then combined in phase 2. MLEs no larger than the worker space are
/// duplicated to every worker by `VirtualPolynomials`; frontload treats that
/// old `Phase2Only` metadata as `Phase1Only`.
///
/// ```text
/// a: 10 variables > 2 worker bits => Normal
/// b:  1 variable  <= 2 worker bits => Phase1Only
/// c:  5 variables > 2 worker bits => Normal
/// ```
///
/// For `a`, each worker receives a `2^8` chunk. Phase 1 folds each local chunk
/// over `x0..x7`, producing four worker scalars. Phase 2 builds a two-variable
/// MLE from those scalars and folds the worker bits `x8,x9`.
///
/// For `b`, each worker receives the same compact `2^1` MLE. Its frontload
/// embedding is:
///
/// ```text
/// B(x0..x9) = b(x0) * x1 * x2 * ... * x9
/// ```
///
/// Round 0 folds the real `b` variable. Rounds 1..7 in phase 1 only multiply
/// the local tail challenges. Because `b` has no worker-bit data, the missing
/// worker-bit tail requires `x8 = 1` and `x9 = 1`, so only worker `3 = 0b11`
/// contributes this term during phase 1. Phase 2 carries `b` as a compact
/// constant with the remaining worker-bit tail.
///
/// For `c`, the MLE is split across workers because `2^5 > 2^2`. Each worker
/// receives a `2^3` local chunk:
///
/// ```text
/// worker 0: c_00(x0,x1,x2)
/// worker 1: c_01(x0,x1,x2)
/// worker 2: c_10(x0,x1,x2)
/// worker 3: c_11(x0,x1,x2)
/// ```
///
/// After phase-1 round 2, each worker has one scalar
/// `s_w = c_w(r0,r1,r2)`. There is no exchange within phase 1 to squeeze those
/// four scalars into one. During rounds 3..7, each worker independently applies
/// the missing local-variable tail:
///
/// ```text
/// s'_w = s_w * r3 * r4 * r5 * r6 * r7
/// ```
///
/// Phase 2 then builds a two-variable MLE from `[s'_0, s'_1, s'_2, s'_3]` and
/// folds the worker bits `x8,x9`.
///
/// The returned `challenges` are the verifier challenges sampled across both
/// phases. `final_evaluations` contains the compact MLE evaluations after all
/// relevant folding; for frontload this is stored explicitly because the
/// internal working state is not the legacy suffix-loaded `IOPProverState`.
#[derive(Clone, Debug)]
pub struct FrontloadProverState<E: ExtensionField> {
    pub challenges: Vec<Challenge<E>>,
    pub final_evaluations: Vec<Vec<E>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FrontloadPolyMeta {
    Normal,
    Phase1Only,
}

impl From<PolyMeta> for FrontloadPolyMeta {
    fn from(meta: PolyMeta) -> Self {
        match meta {
            PolyMeta::Normal => Self::Normal,
            PolyMeta::Phase2Only => Self::Phase1Only,
        }
    }
}

pub fn prove<'a, E: ExtensionField>(
    poly: VirtualPolynomial<'a, E>,
    transcript: &mut impl Transcript<E>,
) -> (IOPProof<E>, FrontloadProverState<E>) {
    prove_inner(WorkingState::new(poly), transcript, true)
}

pub fn prove_2phase<'a, E: ExtensionField>(
    virtual_poly: VirtualPolynomials<'a, E>,
    transcript: &mut impl Transcript<E>,
) -> (IOPProof<E>, FrontloadProverState<E>) {
    let log_num_workers = p3::util::log2_strict_usize(virtual_poly.num_threads);
    let max_degree = virtual_poly.degree();
    let (polys, poly_meta) = virtual_poly.get_batched_polys();
    let frontload_poly_meta = poly_meta
        .into_iter()
        .map(FrontloadPolyMeta::from)
        .collect_vec();
    let local_num_vars = polys
        .first()
        .map(|poly| poly.aux_info.max_num_variables)
        .unwrap_or(0);
    let global_num_vars = local_num_vars + log_num_workers;

    transcript.append_message(&global_num_vars.to_le_bytes());
    transcript.append_message(&max_degree.to_le_bytes());

    let global_mle_num_vars = polys
        .first()
        .map(|poly| {
            poly.flattened_ml_extensions
                .iter()
                .zip_eq(&frontload_poly_meta)
                .map(|(mle, meta)| match meta {
                    FrontloadPolyMeta::Normal => mle.num_vars() + log_num_workers,
                    FrontloadPolyMeta::Phase1Only => mle.num_vars(),
                })
                .collect_vec()
        })
        .unwrap_or_default();

    let mut workers = polys
        .into_iter()
        .enumerate()
        .map(|(worker_id, poly)| {
            WorkingState::new_with_metadata(
                poly,
                global_mle_num_vars.clone(),
                frontload_poly_meta
                    .iter()
                    .map(|meta| *meta == FrontloadPolyMeta::Normal)
                    .collect_vec(),
                Some((worker_id, log_num_workers)),
            )
        })
        .collect_vec();
    if let Some(worker) = workers.first() {
        worker.assert_uniform_product_num_vars();
    }
    let mut proofs = Vec::with_capacity(global_num_vars);
    let mut challenge: Option<Challenge<E>> = None;

    for round in 0..local_num_vars {
        workers.par_iter_mut().for_each(|worker| {
            if let Some(challenge) = challenge {
                worker.challenges.push(challenge);
                worker.fold_round(challenge.elements);
            }
        });
        let mut evaluations = if workers
            .first()
            .is_some_and(|worker| worker.any_mle_round_needs_canonical_merge(round))
        {
            round_evaluations_across_workers(&workers, round, max_degree)
        } else {
            workers
                .par_iter()
                .map(|worker| worker.round_evaluations(round))
                .reduce(
                    || vec![E::ZERO; max_degree + 1],
                    |mut acc, evals| {
                        acc.iter_mut().zip_eq(evals).for_each(|(acc, eval)| {
                            *acc += eval;
                        });
                        acc
                    },
                )
        };
        evaluations.remove(0);
        transcript.append_field_element_exts(&evaluations);
        proofs.push(IOPProverMessage { evaluations });
        challenge = Some(transcript.sample_and_append_challenge(b"Internal round"));
    }

    if let Some(challenge) = challenge {
        workers.par_iter_mut().for_each(|worker| {
            worker.challenges.push(challenge);
            worker.fold_round(challenge.elements);
        });
    }

    let phase2_poly = build_phase2_poly(
        &workers,
        &frontload_poly_meta,
        local_num_vars,
        log_num_workers,
    );
    let (phase2_proof, phase2_state) =
        prove_inner(WorkingState::new(phase2_poly), transcript, false);
    proofs.extend(phase2_proof.proofs);
    let challenges = workers
        .first()
        .map(|worker| worker.challenges.clone())
        .unwrap_or_default()
        .into_iter()
        .chain(phase2_state.challenges)
        .collect_vec();
    let final_evaluations = normalize_phase2_final_evaluations(
        phase2_state.final_evaluations,
        &global_mle_num_vars,
        local_num_vars,
        &challenges,
    );

    (
        IOPProof { proofs },
        FrontloadProverState {
            challenges,
            final_evaluations,
        },
    )
}

fn prove_inner<'a, E: ExtensionField>(
    mut state: WorkingState<'a, E>,
    transcript: &mut impl Transcript<E>,
    append_header: bool,
) -> (IOPProof<E>, FrontloadProverState<E>) {
    let num_vars = state.poly.aux_info.max_num_variables;
    let max_degree = state.poly.aux_info.max_degree;

    if append_header {
        transcript.append_message(&num_vars.to_le_bytes());
        transcript.append_message(&max_degree.to_le_bytes());
    }

    let mut proof = Vec::with_capacity(num_vars);
    let mut challenge: Option<Challenge<E>> = None;

    for round in 0..num_vars {
        if let Some(challenge) = challenge.take() {
            state.challenges.push(challenge);
            state.fold_round(challenge.elements);
        }

        let mut evaluations = state.round_evaluations(round);
        evaluations.remove(0);
        transcript.append_field_element_exts(&evaluations);
        proof.push(IOPProverMessage { evaluations });
        challenge = Some(transcript.sample_and_append_challenge(b"Internal round"));
    }

    if let Some(challenge) = challenge {
        state.challenges.push(challenge);
        state.fold_round(challenge.elements);
    }

    let final_evaluations = state.final_evaluations();
    (
        IOPProof { proofs: proof },
        FrontloadProverState {
            challenges: state.challenges,
            final_evaluations,
        },
    )
}

pub fn claimed_sum<E: ExtensionField>(poly: &VirtualPolynomial<'_, E>) -> E {
    let state = WorkingState::new(poly.as_view());
    let evaluations = state.round_evaluations(0);
    evaluations[0] + evaluations[1]
}

pub fn evaluate<E: ExtensionField>(
    poly: &VirtualPolynomial<'_, E>,
    point: &[E],
    raw_mle_evals: &[E],
) -> E {
    assert_eq!(poly.aux_info.max_num_variables, point.len());
    assert_eq!(poly.flattened_ml_extensions.len(), raw_mle_evals.len());
    let mle_evals = poly
        .flattened_ml_extensions
        .iter()
        .zip_eq(raw_mle_evals)
        .map(|(mle, &raw_eval)| {
            point[mle.num_vars()..]
                .iter()
                .fold(raw_eval, |acc, point| acc * *point)
        })
        .collect_vec();
    poly.products
        .iter()
        .map(|MonomialTerms { terms }| {
            terms
                .iter()
                .map(|Term { scalar, product }| {
                    either::for_both!(scalar, scalar => {
                        product.iter().map(|&idx| mle_evals[idx]).product::<E>() * *scalar
                    })
                })
                .sum::<E>()
        })
        .sum()
}

pub fn aux_info<E: ExtensionField>(poly: &VirtualPolynomial<'_, E>) -> VPAuxInfo<E> {
    poly.aux_info.clone()
}

struct WorkingState<'a, E: ExtensionField> {
    poly: VirtualPolynomial<'a, E>,
    mles: Vec<MultilinearExtension<'a, E>>,
    term_metadata: Vec<Vec<TermMetadata>>,
    challenges: Vec<Challenge<E>>,
    global_mle_num_vars: Vec<usize>,
    mle_has_worker_bits: Vec<bool>,
    worker: Option<(usize, usize)>,
}

#[derive(Clone, Copy, Debug)]
struct TermMetadata {
    degree: usize,
    local_num_vars: usize,
    global_num_vars: usize,
    all_same_local_num_vars: bool,
    all_same_global_num_vars: bool,
}

impl<'a, E: ExtensionField> WorkingState<'a, E> {
    fn new(poly: VirtualPolynomial<'a, E>) -> Self {
        let global_mle_num_vars = poly
            .flattened_ml_extensions
            .iter()
            .map(|mle| mle.num_vars())
            .collect_vec();
        let mle_has_worker_bits = vec![false; global_mle_num_vars.len()];
        Self::new_with_metadata(poly, global_mle_num_vars, mle_has_worker_bits, None)
    }

    fn new_with_metadata(
        poly: VirtualPolynomial<'a, E>,
        global_mle_num_vars: Vec<usize>,
        mle_has_worker_bits: Vec<bool>,
        worker: Option<(usize, usize)>,
    ) -> Self {
        let mles = poly
            .flattened_ml_extensions
            .iter()
            .map(|mle| mle.as_ref().as_owned())
            .collect_vec();
        assert_eq!(global_mle_num_vars.len(), mles.len());
        assert_eq!(mle_has_worker_bits.len(), mles.len());
        let term_metadata = poly
            .products
            .iter()
            .map(|MonomialTerms { terms }| {
                terms
                    .iter()
                    .map(|term| {
                        let mut local_num_vars = 0usize;
                        let mut first_local_num_vars = None;
                        let mut first_global_num_vars = None;
                        let mut all_same_local_num_vars = true;
                        let mut all_same_global_num_vars = true;
                        for &idx in &term.product {
                            let mle_num_vars = poly.flattened_ml_extensions[idx].num_vars();
                            let global_num_vars = global_mle_num_vars[idx];
                            local_num_vars = local_num_vars.max(mle_num_vars);
                            if first_local_num_vars
                                .replace(mle_num_vars)
                                .is_some_and(|first| first != mle_num_vars)
                            {
                                all_same_local_num_vars = false;
                            }
                            if first_global_num_vars
                                .replace(global_num_vars)
                                .is_some_and(|first| first != global_num_vars)
                            {
                                all_same_global_num_vars = false;
                            }
                        }
                        TermMetadata {
                            degree: term.product.len(),
                            local_num_vars,
                            global_num_vars: first_global_num_vars.unwrap_or(0),
                            all_same_local_num_vars,
                            all_same_global_num_vars,
                        }
                    })
                    .collect_vec()
            })
            .collect_vec();
        Self {
            poly,
            mles,
            term_metadata,
            challenges: vec![],
            global_mle_num_vars,
            mle_has_worker_bits,
            worker,
        }
    }

    fn round_evaluations(&self, round: usize) -> Vec<E> {
        let mut evaluations = vec![E::ZERO; self.poly.aux_info.max_degree + 1];
        for (MonomialTerms { terms }, metadata) in
            self.poly.products.iter().zip_eq(&self.term_metadata)
        {
            for (term, metadata) in terms.iter().zip_eq(metadata) {
                self.add_term_round_evaluations(term, *metadata, round, &mut evaluations);
            }
        }
        evaluations
    }

    fn add_term_round_evaluations(
        &self,
        term: &Term<either::Either<E::BaseField, E>, usize>,
        metadata: TermMetadata,
        round: usize,
        acc: &mut [E],
    ) {
        debug_assert_eq!(metadata.degree, term.product.len());
        if metadata.all_same_local_num_vars
            && metadata.all_same_global_num_vars
            && metadata.local_num_vars == self.poly.aux_info.max_num_variables
            && self.uniform_term_has_no_frontload_tail(metadata)
        {
            match metadata.degree {
                2 if self.add_uniform_degree2(term, metadata, round, acc) => return,
                3 if self.add_uniform_degree3(term, metadata, round, acc) => return,
                4 if self.add_uniform_degree4(term, metadata, round, acc) => return,
                _ => {}
            }
        }

        let evaluations = self.term_round_evaluations(term, round);
        acc.iter_mut().zip_eq(evaluations).for_each(|(acc, eval)| {
            *acc += eval;
        });
    }

    #[inline]
    fn uniform_term_has_no_frontload_tail(&self, metadata: TermMetadata) -> bool {
        match self.worker {
            None => true,
            Some((_worker_id, log_num_workers)) => {
                log_num_workers == 0
                    || metadata.global_num_vars > self.poly.aux_info.max_num_variables
            }
        }
    }

    fn add_evaluations(
        &self,
        acc: &mut [E],
        degree: usize,
        scalar: E,
        evals: impl IntoIterator<Item = E>,
    ) {
        if degree == self.poly.aux_info.max_degree {
            for (dst, eval) in acc.iter_mut().take(degree + 1).zip(evals) {
                *dst += eval * scalar;
            }
            return;
        }

        let mut evaluations = vec![E::ZERO; self.poly.aux_info.max_degree + 1];
        for (dst, eval) in evaluations.iter_mut().take(degree + 1).zip(evals) {
            *dst = eval * scalar;
        }
        if degree < self.poly.aux_info.max_degree {
            extrapolate_from_table(&mut evaluations, degree + 1);
        }
        acc.iter_mut().zip_eq(evaluations).for_each(|(acc, eval)| {
            *acc += eval;
        });
    }

    fn add_base_evaluations<const N: usize>(
        &self,
        acc: &mut [E],
        degree: usize,
        scalar: &either::Either<E::BaseField, E>,
        evals: [E::BaseField; N],
    ) {
        let scale_eval = |eval: E::BaseField| match scalar {
            either::Either::Left(base_scalar) => E::from(eval * *base_scalar),
            either::Either::Right(ext_scalar) => E::from(eval) * *ext_scalar,
        };

        if degree == self.poly.aux_info.max_degree {
            for (dst, eval) in acc.iter_mut().take(degree + 1).zip(evals) {
                *dst += scale_eval(eval);
            }
            return;
        }

        let mut evaluations = vec![E::ZERO; self.poly.aux_info.max_degree + 1];
        for (dst, eval) in evaluations.iter_mut().take(degree + 1).zip(evals) {
            *dst = scale_eval(eval);
        }
        if degree < self.poly.aux_info.max_degree {
            extrapolate_from_table(&mut evaluations, degree + 1);
        }
        acc.iter_mut().zip_eq(evaluations).for_each(|(acc, eval)| {
            *acc += eval;
        });
    }

    fn add_uniform_degree2(
        &self,
        term: &Term<either::Either<E::BaseField, E>, usize>,
        metadata: TermMetadata,
        round: usize,
        acc: &mut [E],
    ) -> bool {
        let live_vars = metadata.local_num_vars.saturating_sub(round);
        let lane_count = if live_vars == 0 {
            1
        } else {
            1usize << (live_vars - 1)
        };
        let suffix_mask = uniform_suffix_mask(metadata.local_num_vars, round);
        sumcheck_macro::frontload_uniform_sumcheck_code_gen!(
            2,
            |i: usize| self.mles[term.product[i]].evaluations(),
            metadata.local_num_vars,
            round,
            lane_count,
            suffix_mask
        );
        true
    }

    fn add_uniform_degree3(
        &self,
        term: &Term<either::Either<E::BaseField, E>, usize>,
        metadata: TermMetadata,
        round: usize,
        acc: &mut [E],
    ) -> bool {
        let live_vars = metadata.local_num_vars.saturating_sub(round);
        let lane_count = if live_vars == 0 {
            1
        } else {
            1usize << (live_vars - 1)
        };
        let suffix_mask = uniform_suffix_mask(metadata.local_num_vars, round);
        sumcheck_macro::frontload_uniform_sumcheck_code_gen!(
            3,
            |i: usize| self.mles[term.product[i]].evaluations(),
            metadata.local_num_vars,
            round,
            lane_count,
            suffix_mask
        );
        true
    }

    fn add_uniform_degree4(
        &self,
        term: &Term<either::Either<E::BaseField, E>, usize>,
        metadata: TermMetadata,
        round: usize,
        acc: &mut [E],
    ) -> bool {
        let live_vars = metadata.local_num_vars.saturating_sub(round);
        let lane_count = if live_vars == 0 {
            1
        } else {
            1usize << (live_vars - 1)
        };
        let suffix_mask = uniform_suffix_mask(metadata.local_num_vars, round);
        sumcheck_macro::frontload_uniform_sumcheck_code_gen!(
            4,
            |i: usize| self.mles[term.product[i]].evaluations(),
            metadata.local_num_vars,
            round,
            lane_count,
            suffix_mask
        );
        true
    }

    fn term_round_evaluations(
        &self,
        term: &Term<either::Either<E::BaseField, E>, usize>,
        round: usize,
    ) -> Vec<E> {
        let degree = term.product.len();
        if !self.worker_matches_frontload_tail(term) {
            return vec![E::ZERO; self.poly.aux_info.max_degree + 1];
        }
        let has_worker_bit_round = term
            .product
            .iter()
            .any(|&idx| self.mle_round_is_worker_bit(idx, round));
        if !has_worker_bit_round {
            match degree {
                1 => {
                    return sumcheck_macro::frontload_mixed_sumcheck_code_gen!(
                        1, self, term, round
                    );
                }
                2 => {
                    return sumcheck_macro::frontload_mixed_sumcheck_code_gen!(
                        2, self, term, round
                    );
                }
                3 => {
                    return sumcheck_macro::frontload_mixed_sumcheck_code_gen!(
                        3, self, term, round
                    );
                }
                4 => {
                    return sumcheck_macro::frontload_mixed_sumcheck_code_gen!(
                        4, self, term, round
                    );
                }
                5 => {
                    return sumcheck_macro::frontload_mixed_sumcheck_code_gen!(
                        5, self, term, round
                    );
                }
                _ => {}
            }
        }

        let live_vars = term
            .product
            .iter()
            .map(|&idx| {
                self.poly.flattened_ml_extensions[idx]
                    .num_vars()
                    .saturating_sub(round)
            })
            .max()
            .unwrap_or(0);
        let lane_count = if live_vars == 0 {
            1
        } else {
            1usize << (live_vars - 1)
        };
        let required_ones_mask = self.required_future_frontload_mask(term, round, live_vars);
        let fixed_frontload = term
            .product
            .iter()
            .map(|&idx| self.fixed_frontload_factor(idx, round))
            .collect_vec();

        let mut evaluations = vec![E::ZERO; degree + 1];
        let scalar = scalar_to_ext(&term.scalar);
        for_each_active_lane(lane_count, required_ones_mask, |lane| {
            for (z_idx, eval) in evaluations.iter_mut().enumerate() {
                let z = E::from_canonical_u64(z_idx as u64);
                let product = term
                    .product
                    .iter()
                    .zip_eq(&fixed_frontload)
                    .map(|(&idx, &frontload_factor)| {
                        self.mle_round_value(idx, round, lane, z) * frontload_factor
                    })
                    .product::<E>();
                *eval += product * scalar;
            }
        });

        if degree == self.poly.aux_info.max_degree {
            evaluations
        } else {
            let mut extrapolated = vec![E::ZERO; self.poly.aux_info.max_degree + 1];
            extrapolated[..=degree].copy_from_slice(&evaluations);
            extrapolate_from_table(&mut extrapolated, degree + 1);
            extrapolated
        }
    }

    fn worker_matches_frontload_tail(
        &self,
        term: &Term<either::Either<E::BaseField, E>, usize>,
    ) -> bool {
        let Some((worker_id, log_num_workers)) = self.worker else {
            return true;
        };
        if term
            .product
            .iter()
            .any(|&idx| self.mle_has_worker_bits[idx])
        {
            return true;
        }
        worker_id == (1usize << log_num_workers) - 1
    }

    fn required_future_frontload_mask(
        &self,
        term: &Term<either::Either<E::BaseField, E>, usize>,
        round: usize,
        live_vars: usize,
    ) -> usize {
        if live_vars <= 1 {
            return 0;
        }
        let enumerated_var_end = (round + live_vars).min(self.poly.aux_info.max_num_variables);
        ((round + 1)..enumerated_var_end).fold(0usize, |mask, var_idx| {
            let needs_frontload = term
                .product
                .iter()
                .any(|&idx| self.frontload_tail_start(idx) <= var_idx);
            if needs_frontload {
                mask | (1usize << (var_idx - round - 1))
            } else {
                mask
            }
        })
    }

    fn mle_round_value(&self, mle_idx: usize, round: usize, lane: usize, z: E) -> E {
        let (e0, e1) = self.mle_round_endpoints(mle_idx, round, lane);
        e0 + z * (e1 - e0)
    }

    fn mle_round_endpoints_scaled(
        &self,
        mle_idx: usize,
        round: usize,
        lane: usize,
        factor: E,
    ) -> (E, E) {
        let (e0, e1) = self.mle_round_endpoints(mle_idx, round, lane);
        (e0 * factor, e1 * factor)
    }

    fn mle_round_endpoints(&self, mle_idx: usize, round: usize, lane: usize) -> (E, E) {
        let local_num_vars = self.poly.flattened_ml_extensions[mle_idx].num_vars();
        let global_num_vars = self.global_mle_num_vars[mle_idx];
        if round >= local_num_vars {
            let value = read_eval_or_zero(&self.mles[mle_idx], 0);
            if round < global_num_vars {
                return if self.worker_round_bit(mle_idx, round) == 0 {
                    (value, E::ZERO)
                } else {
                    (E::ZERO, value)
                };
            }
            return (E::ZERO, value);
        }

        let remaining_vars = local_num_vars - round;
        let suffix_mask = if remaining_vars == 1 {
            0
        } else {
            (1usize << (remaining_vars - 1)) - 1
        };
        let suffix = lane & suffix_mask;
        let evals = self.mles[mle_idx].evaluations();
        let e0 = field_index_or_zero(evals, suffix << 1);
        let e1 = field_index_or_zero(evals, (suffix << 1) + 1);
        (e0, e1)
    }

    fn fixed_frontload_factor(&self, mle_idx: usize, round: usize) -> E {
        let local_num_vars = self.poly.flattened_ml_extensions[mle_idx].num_vars();
        let global_num_vars = self.global_mle_num_vars[mle_idx];
        if round <= local_num_vars {
            return E::ONE;
        }
        (local_num_vars..round).fold(E::ONE, |acc, fixed_round| {
            let challenge = self.challenges[fixed_round].elements;
            if fixed_round < global_num_vars {
                if self.worker_round_bit(mle_idx, fixed_round) == 0 {
                    acc * (E::ONE - challenge)
                } else {
                    acc * challenge
                }
            } else {
                acc * challenge
            }
        })
    }

    fn frontload_tail_start(&self, mle_idx: usize) -> usize {
        self.global_mle_num_vars[mle_idx]
    }

    fn mle_round_is_worker_bit(&self, mle_idx: usize, round: usize) -> bool {
        self.worker.is_some()
            && self.mle_has_worker_bits[mle_idx]
            && round >= self.poly.flattened_ml_extensions[mle_idx].num_vars()
            && round < self.global_mle_num_vars[mle_idx]
    }

    fn worker_round_bit(&self, mle_idx: usize, round: usize) -> usize {
        let local_num_vars = self.poly.flattened_ml_extensions[mle_idx].num_vars();
        debug_assert!(round >= local_num_vars);
        self.worker
            .map(|(worker_id, _log_num_workers)| (worker_id >> (round - local_num_vars)) & 1)
            .unwrap_or(1)
    }

    fn any_mle_round_needs_canonical_merge(&self, round: usize) -> bool {
        self.worker.is_some()
            && (0..self.mles.len()).any(|mle_idx| {
                self.mle_has_worker_bits[mle_idx]
                    && round >= self.poly.flattened_ml_extensions[mle_idx].num_vars()
            })
    }

    fn assert_uniform_product_num_vars(&self) {
        for MonomialTerms { terms } in &self.poly.products {
            for term in terms {
                let Some((&first_idx, rest)) = term.product.split_first() else {
                    continue;
                };
                let first_num_vars = self.global_mle_num_vars[first_idx];
                assert!(
                    rest.iter()
                        .all(|&idx| self.global_mle_num_vars[idx] == first_num_vars),
                    "frontload 2phase expects every MLE in a product term to have the same canonical num_vars"
                );
            }
        }
    }

    fn fold_round(&mut self, challenge: E) {
        let round = self.challenges.len() - 1;
        self.fold_round_at(round, challenge);
    }

    fn fold_round_at(&mut self, round: usize, challenge: E) {
        self.mles
            .iter_mut()
            .zip_eq(&self.poly.flattened_ml_extensions)
            .for_each(|(mle, original)| {
                if round < original.num_vars() {
                    mle.fix_variables_in_place(&[challenge]);
                }
            });
    }

    fn final_evaluations(&self) -> Vec<Vec<E>> {
        self.mles
            .iter()
            .map(|mle| {
                (0..mle.evaluations().len())
                    .map(|idx| read_eval(mle, idx))
                    .collect_vec()
            })
            .collect_vec()
    }
}

fn round_evaluations_across_workers<'a, E: ExtensionField>(
    workers: &[WorkingState<'a, E>],
    round: usize,
    max_degree: usize,
) -> Vec<E> {
    let first_worker = workers.first().expect("frontload 2phase needs workers");
    let mut evaluations = vec![E::ZERO; max_degree + 1];
    for MonomialTerms { terms } in &first_worker.poly.products {
        for term in terms {
            let term_evaluations = term_round_evaluations_across_workers(workers, term, round);
            evaluations
                .iter_mut()
                .zip_eq(term_evaluations)
                .for_each(|(acc, eval)| *acc += eval);
        }
    }
    evaluations
}

fn term_round_evaluations_across_workers<'a, E: ExtensionField>(
    workers: &[WorkingState<'a, E>],
    term: &Term<either::Either<E::BaseField, E>, usize>,
    round: usize,
) -> Vec<E> {
    let first_worker = workers.first().expect("frontload 2phase needs workers");
    let Some(layout) = TermWorkerLayout::new(first_worker, term) else {
        return workers
            .last()
            .expect("frontload 2phase needs workers")
            .term_round_evaluations(term, round);
    };
    let degree = term.product.len();
    let live_vars = layout.phase1_end.saturating_sub(round);
    let lane_count = if live_vars == 0 {
        1
    } else {
        1usize << (live_vars - 1)
    };
    let required_ones_mask = first_worker.required_future_frontload_mask(term, round, live_vars);
    let scalar = scalar_to_ext(&term.scalar);
    let mut evaluations = vec![E::ZERO; degree + 1];

    for_each_active_lane(lane_count, required_ones_mask, |lane| {
        for (z_idx, eval) in evaluations.iter_mut().enumerate() {
            let z = E::from_canonical_u64(z_idx as u64);
            let mut term_eval = E::ZERO;
            for group_key in layout.future_group_keys(workers, round) {
                let product = term
                    .product
                    .iter()
                    .map(|&idx| {
                        canonical_worker_group_value(
                            workers, layout, idx, round, lane, z, group_key,
                        )
                    })
                    .product::<E>();
                term_eval += product;
            }
            *eval += term_eval * scalar;
        }
    });

    if degree == first_worker.poly.aux_info.max_degree {
        evaluations
    } else {
        let mut extrapolated = vec![E::ZERO; first_worker.poly.aux_info.max_degree + 1];
        extrapolated[..=degree].copy_from_slice(&evaluations);
        extrapolate_from_table(&mut extrapolated, degree + 1);
        extrapolated
    }
}

#[derive(Clone, Copy)]
struct TermWorkerLayout {
    local_num_vars: usize,
    global_num_vars: usize,
    phase1_num_vars: usize,
    phase1_end: usize,
    first_mle_idx: usize,
}

impl TermWorkerLayout {
    fn new<E: ExtensionField>(
        worker: &WorkingState<'_, E>,
        term: &Term<either::Either<E::BaseField, E>, usize>,
    ) -> Option<Self> {
        let first_mle_idx = term
            .product
            .iter()
            .copied()
            .find(|&idx| worker.mle_has_worker_bits[idx])?;
        let local_num_vars = worker.poly.flattened_ml_extensions[first_mle_idx].num_vars();
        let global_num_vars = worker.global_mle_num_vars[first_mle_idx];
        let phase1_num_vars = worker.poly.aux_info.max_num_variables;
        Some(Self {
            local_num_vars,
            global_num_vars,
            phase1_num_vars,
            phase1_end: global_num_vars.min(phase1_num_vars),
            first_mle_idx,
        })
    }

    fn future_group_keys<E: ExtensionField>(
        self,
        workers: &[WorkingState<'_, E>],
        round: usize,
    ) -> Vec<usize> {
        let group_start = self.phase1_num_vars.max(round + 1).max(self.local_num_vars);
        if group_start >= self.global_num_vars {
            return vec![0];
        }

        let mut keys = workers
            .iter()
            .map(|worker| {
                (group_start..self.global_num_vars).fold(0usize, |key, var_idx| {
                    key | (worker.worker_round_bit(self.first_mle_idx, var_idx)
                        << (var_idx - self.phase1_num_vars))
                })
            })
            .collect_vec();
        keys.sort_unstable();
        keys.dedup();
        keys
    }
}

fn canonical_worker_group_value<'a, E: ExtensionField>(
    workers: &[WorkingState<'a, E>],
    layout: TermWorkerLayout,
    mle_idx: usize,
    round: usize,
    lane: usize,
    z: E,
    group_key: usize,
) -> E {
    workers
        .iter()
        .filter_map(|worker| {
            for var_idx in layout
                .phase1_num_vars
                .max(round + 1)
                .max(layout.local_num_vars)..layout.global_num_vars
            {
                let group_bit = (group_key >> (var_idx - layout.phase1_num_vars)) & 1;
                if worker.worker_round_bit(mle_idx, var_idx) != group_bit {
                    return None;
                }
            }

            for var_idx in layout.local_num_vars.max(round + 1)..layout.phase1_end {
                let lane_bit = (lane >> (var_idx - round - 1)) & 1;
                if worker.worker_round_bit(mle_idx, var_idx) != lane_bit {
                    return None;
                }
            }

            if round < layout.local_num_vars {
                return Some(
                    worker.mle_round_value(mle_idx, round, lane, z)
                        * worker.fixed_frontload_factor(mle_idx, round),
                );
            }

            let current_factor = if round < layout.global_num_vars {
                if worker.worker_round_bit(mle_idx, round) == 0 {
                    E::ONE - z
                } else {
                    z
                }
            } else {
                z
            };

            Some(
                read_eval(&worker.mles[mle_idx], 0)
                    * worker.fixed_frontload_factor(mle_idx, round)
                    * current_factor,
            )
        })
        .sum()
}

fn build_phase2_poly<'a, E: ExtensionField>(
    workers: &[WorkingState<'a, E>],
    poly_meta: &[FrontloadPolyMeta],
    local_num_vars: usize,
    log_num_workers: usize,
) -> VirtualPolynomial<'a, E> {
    let first_worker = workers.first().expect("frontload 2phase needs workers");
    let mut poly = VirtualPolynomial::new(log_num_workers);
    poly.aux_info.max_degree = first_worker.poly.aux_info.max_degree;

    for (mle_idx, meta) in poly_meta.iter().enumerate() {
        let global_num_vars = first_worker.global_mle_num_vars[mle_idx];
        let mle = match meta {
            FrontloadPolyMeta::Normal => {
                let remaining_num_vars = global_num_vars.saturating_sub(local_num_vars);
                // At the phase boundary a split Normal MLE has three canonical cases:
                // - exhausted in phase 1: all worker-bit variables were already bound, so sum
                //   every worker scalar into one compact constant.
                // - partially live in phase 2: some worker-bit variables were bound in phase 1
                //   and the suffix remains live; aggregate workers with the same live suffix.
                // - fully live in phase 2: no worker-bit variables were bound in phase 1, so keep
                //   the full worker-index MLE.
                if remaining_num_vars == 0 {
                    let value = workers
                        .iter()
                        .map(|worker| {
                            read_eval(&worker.mles[mle_idx], 0)
                                * worker.fixed_frontload_factor(mle_idx, local_num_vars)
                        })
                        .sum();
                    MultilinearExtension::from_evaluations_ext_vec(0, vec![value])
                } else if remaining_num_vars < log_num_workers {
                    let mut values = vec![E::ZERO; 1usize << remaining_num_vars];
                    let bound_worker_bits = log_num_workers - remaining_num_vars;
                    workers.iter().for_each(|worker| {
                        let (worker_id, _) = worker.worker.expect("frontload worker missing");
                        let phase2_idx = worker_id >> bound_worker_bits;
                        values[phase2_idx] += read_eval(&worker.mles[mle_idx], 0)
                            * worker.fixed_frontload_factor(mle_idx, local_num_vars);
                    });
                    MultilinearExtension::from_evaluations_ext_vec(remaining_num_vars, values)
                } else {
                    let values = workers
                        .iter()
                        .map(|worker| {
                            read_eval(&worker.mles[mle_idx], 0)
                                * worker.fixed_frontload_factor(mle_idx, local_num_vars)
                        })
                        .collect_vec();
                    MultilinearExtension::from_evaluations_ext_vec(log_num_workers, values)
                }
            }
            FrontloadPolyMeta::Phase1Only => {
                let value = read_eval(&first_worker.mles[mle_idx], 0)
                    * first_worker.fixed_frontload_factor(mle_idx, local_num_vars);
                MultilinearExtension::from_evaluations_ext_vec(0, vec![value])
            }
        };
        poly.flattened_ml_extensions.push(Arc::new(mle));
    }
    poly.products = first_worker.poly.products.clone();
    poly
}

fn normalize_phase2_final_evaluations<E: ExtensionField>(
    mut final_evaluations: Vec<Vec<E>>,
    global_mle_num_vars: &[usize],
    local_num_vars: usize,
    challenges: &[Challenge<E>],
) -> Vec<Vec<E>> {
    final_evaluations
        .iter_mut()
        .zip_eq(global_mle_num_vars)
        .for_each(|(evals, &global_num_vars)| {
            if global_num_vars >= local_num_vars {
                return;
            }
            let baked_tail = challenges[global_num_vars..local_num_vars]
                .iter()
                .fold(E::ONE, |acc, challenge| acc * challenge.elements);
            let baked_tail_inv = baked_tail.inverse();
            evals.iter_mut().for_each(|eval| *eval *= baked_tail_inv);
        });
    final_evaluations
}

fn read_eval<E: ExtensionField>(mle: &MultilinearExtension<'_, E>, idx: usize) -> E {
    field_index(mle.evaluations(), idx)
}

fn read_eval_or_zero<E: ExtensionField>(mle: &MultilinearExtension<'_, E>, idx: usize) -> E {
    field_index_or_zero(mle.evaluations(), idx)
}

fn field_index<E: ExtensionField>(evals: &FieldType<'_, E>, idx: usize) -> E {
    match evals.index(idx) {
        either::Either::Left(base) => E::from(base),
        either::Either::Right(ext) => ext,
    }
}

fn field_index_or_zero<E: ExtensionField>(evals: &FieldType<'_, E>, idx: usize) -> E {
    if idx >= evals.len() {
        E::ZERO
    } else {
        field_index(evals, idx)
    }
}

#[inline]
fn uniform_suffix_mask(original_num_vars: usize, round: usize) -> usize {
    if round >= original_num_vars {
        return 0;
    }

    let remaining_vars = original_num_vars - round;
    if remaining_vars == 1 {
        0
    } else {
        (1usize << (remaining_vars - 1)) - 1
    }
}

fn ext_round_endpoints<E: ExtensionField>(
    evals: &[E],
    original_num_vars: usize,
    round: usize,
    lane: usize,
) -> (E, E) {
    if round >= original_num_vars {
        return (E::ZERO, evals.first().copied().unwrap_or(E::ZERO));
    }

    let remaining_vars = original_num_vars - round;
    let suffix_mask = if remaining_vars == 1 {
        0
    } else {
        (1usize << (remaining_vars - 1)) - 1
    };
    let suffix = lane & suffix_mask;
    (
        evals.get(suffix << 1).copied().unwrap_or(E::ZERO),
        evals.get((suffix << 1) + 1).copied().unwrap_or(E::ZERO),
    )
}

fn scalar_to_ext<E: ExtensionField>(scalar: &either::Either<E::BaseField, E>) -> E {
    match scalar {
        either::Either::Left(base) => E::from(*base),
        either::Either::Right(ext) => *ext,
    }
}

fn for_each_active_lane(
    mut lane_count: usize,
    required_ones_mask: usize,
    mut f: impl FnMut(usize),
) {
    if lane_count == 0 {
        return;
    }
    if required_ones_mask == 0 {
        for lane in 0..lane_count {
            f(lane);
        }
        return;
    }

    lane_count = lane_count.next_power_of_two();
    let lane_mask = lane_count - 1;
    let fixed_mask = required_ones_mask & lane_mask;
    let free_mask = lane_mask & !fixed_mask;
    let mut free_bits = free_mask;
    loop {
        f(fixed_mask | free_bits);
        if free_bits == 0 {
            break;
        }
        free_bits = (free_bits - 1) & free_mask;
    }
}

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

macro_rules! frontload_sumcheck_code_gen {
    ($state:expr, $term:expr, $round:expr, $degree:expr, $($specialized_degree:literal),* $(,)?) => {
        match $degree {
            $(
                $specialized_degree => {
                    return $state
                        .term_round_evaluations_degree::<$specialized_degree>($term, $round);
                }
            )*
            _ => {}
        }
    };
}

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
                Some((worker_id, log_num_workers)),
            )
        })
        .collect_vec();
    let mut proofs = Vec::with_capacity(global_num_vars);
    let mut challenge: Option<Challenge<E>> = None;

    for round in 0..local_num_vars {
        let mut evaluations = workers
            .par_iter_mut()
            .map(|worker| {
                if let Some(challenge) = challenge {
                    worker.challenges.push(challenge);
                    worker.fold_round(challenge.elements);
                }
                worker.round_evaluations(round)
            })
            .reduce(
                || vec![E::ZERO; max_degree + 1],
                |mut acc, evals| {
                    acc.iter_mut().zip_eq(evals).for_each(|(acc, eval)| {
                        *acc += eval;
                    });
                    acc
                },
            );
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

    (
        IOPProof { proofs },
        FrontloadProverState {
            challenges: workers
                .first()
                .map(|worker| worker.challenges.clone())
                .unwrap_or_default()
                .into_iter()
                .chain(phase2_state.challenges)
                .collect(),
            final_evaluations: phase2_state.final_evaluations,
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

pub fn evaluate<E: ExtensionField>(poly: &VirtualPolynomial<'_, E>, point: &[E]) -> E {
    assert_eq!(poly.aux_info.max_num_variables, point.len());
    let mle_evals = poly
        .flattened_ml_extensions
        .iter()
        .map(|mle| {
            let local_eval = if mle.num_vars() == 0 {
                read_eval(mle, 0)
            } else {
                mle.evaluate(&point[..mle.num_vars()])
            };
            point[mle.num_vars()..]
                .iter()
                .fold(local_eval, |acc, point| acc * *point)
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
        Self::new_with_metadata(poly, global_mle_num_vars, None)
    }

    fn new_with_metadata(
        poly: VirtualPolynomial<'a, E>,
        global_mle_num_vars: Vec<usize>,
        worker: Option<(usize, usize)>,
    ) -> Self {
        let mles = poly
            .flattened_ml_extensions
            .iter()
            .map(|mle| mle.as_ref().as_owned())
            .collect_vec();
        assert_eq!(global_mle_num_vars.len(), mles.len());
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
        if degree == 1 {
            return self.term_round_evaluations_degree1(term, round);
        }
        if degree == 2 {
            return self.term_round_evaluations_degree2(term, round);
        }
        if degree == 3 {
            return self.term_round_evaluations_degree3(term, round);
        }
        if degree == 4 {
            return self.term_round_evaluations_degree4(term, round);
        }
        if degree == 5 {
            return self.term_round_evaluations_degree5(term, round);
        }
        frontload_sumcheck_code_gen!(self, term, round, degree, 3, 4, 5);

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
        let local_num_vars = self.poly.aux_info.max_num_variables;
        term_matches_frontload_tail(
            local_num_vars,
            &self.global_mle_num_vars,
            Some((worker_id, log_num_workers)),
            term,
        )
    }

    fn term_round_evaluations_degree1(
        &self,
        term: &Term<either::Either<E::BaseField, E>, usize>,
        round: usize,
    ) -> Vec<E> {
        let mle_idx = term.product[0];
        let live_vars = self.poly.flattened_ml_extensions[mle_idx]
            .num_vars()
            .saturating_sub(round);
        let lane_count = if live_vars == 0 {
            1
        } else {
            1usize << (live_vars - 1)
        };
        let required_ones_mask = self.required_future_frontload_mask(term, round, live_vars);
        let frontload_factor = self.fixed_frontload_factor(mle_idx, round);
        let scalar = scalar_to_ext(&term.scalar);

        if let FieldType::Ext(evals) = self.mles[mle_idx].evaluations() {
            let original_num_vars = self.poly.flattened_ml_extensions[mle_idx].num_vars();
            let mut eval_0 = E::ZERO;
            let mut eval_1 = E::ZERO;
            for_each_active_lane(lane_count, required_ones_mask, |lane| {
                let (v0, v1) =
                    ext_round_endpoints(evals.as_slice(), original_num_vars, round, lane);
                eval_0 += v0 * frontload_factor;
                eval_1 += v1 * frontload_factor;
            });
            let mut evaluations = vec![E::ZERO; self.poly.aux_info.max_degree + 1];
            evaluations[0] = eval_0 * scalar;
            evaluations[1] = eval_1 * scalar;
            if self.poly.aux_info.max_degree > 1 {
                extrapolate_from_table(&mut evaluations, 2);
            }
            return evaluations;
        }

        let mut eval_0 = E::ZERO;
        let mut eval_1 = E::ZERO;
        for_each_active_lane(lane_count, required_ones_mask, |lane| {
            let (v0, v1) = self.mle_round_endpoints_scaled(mle_idx, round, lane, frontload_factor);
            eval_0 += v0;
            eval_1 += v1;
        });

        let mut evaluations = vec![E::ZERO; self.poly.aux_info.max_degree + 1];
        evaluations[0] = eval_0 * scalar;
        evaluations[1] = eval_1 * scalar;
        if self.poly.aux_info.max_degree > 1 {
            extrapolate_from_table(&mut evaluations, 2);
        }
        evaluations
    }

    fn term_round_evaluations_degree2(
        &self,
        term: &Term<either::Either<E::BaseField, E>, usize>,
        round: usize,
    ) -> Vec<E> {
        let lhs_idx = term.product[0];
        let rhs_idx = term.product[1];
        let live_vars = [lhs_idx, rhs_idx]
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
        let lhs_frontload = self.fixed_frontload_factor(lhs_idx, round);
        let rhs_frontload = self.fixed_frontload_factor(rhs_idx, round);
        let scalar = scalar_to_ext(&term.scalar);

        if let (FieldType::Ext(lhs_evals), FieldType::Ext(rhs_evals)) = (
            self.mles[lhs_idx].evaluations(),
            self.mles[rhs_idx].evaluations(),
        ) {
            let lhs_original_num_vars = self.poly.flattened_ml_extensions[lhs_idx].num_vars();
            let rhs_original_num_vars = self.poly.flattened_ml_extensions[rhs_idx].num_vars();
            let mut eval_0 = E::ZERO;
            let mut eval_1 = E::ZERO;
            let mut eval_2 = E::ZERO;
            for_each_active_lane(lane_count, required_ones_mask, |lane| {
                let (lhs_0, lhs_1) =
                    ext_round_endpoints(lhs_evals.as_slice(), lhs_original_num_vars, round, lane);
                let (rhs_0, rhs_1) =
                    ext_round_endpoints(rhs_evals.as_slice(), rhs_original_num_vars, round, lane);
                let lhs_0 = lhs_0 * lhs_frontload;
                let lhs_1 = lhs_1 * lhs_frontload;
                let rhs_0 = rhs_0 * rhs_frontload;
                let rhs_1 = rhs_1 * rhs_frontload;
                let lhs_2 = lhs_1 + (lhs_1 - lhs_0);
                let rhs_2 = rhs_1 + (rhs_1 - rhs_0);
                eval_0 += lhs_0 * rhs_0;
                eval_1 += lhs_1 * rhs_1;
                eval_2 += lhs_2 * rhs_2;
            });
            let mut evaluations = vec![E::ZERO; self.poly.aux_info.max_degree + 1];
            evaluations[0] = eval_0 * scalar;
            evaluations[1] = eval_1 * scalar;
            evaluations[2] = eval_2 * scalar;
            if self.poly.aux_info.max_degree > 2 {
                extrapolate_from_table(&mut evaluations, 3);
            }
            return evaluations;
        }

        let mut eval_0 = E::ZERO;
        let mut eval_1 = E::ZERO;
        let mut eval_2 = E::ZERO;
        for_each_active_lane(lane_count, required_ones_mask, |lane| {
            let (lhs_0, lhs_1) =
                self.mle_round_endpoints_scaled(lhs_idx, round, lane, lhs_frontload);
            let (rhs_0, rhs_1) =
                self.mle_round_endpoints_scaled(rhs_idx, round, lane, rhs_frontload);
            let lhs_2 = lhs_1 + (lhs_1 - lhs_0);
            let rhs_2 = rhs_1 + (rhs_1 - rhs_0);
            eval_0 += lhs_0 * rhs_0;
            eval_1 += lhs_1 * rhs_1;
            eval_2 += lhs_2 * rhs_2;
        });

        let mut evaluations = vec![E::ZERO; self.poly.aux_info.max_degree + 1];
        evaluations[0] = eval_0 * scalar;
        evaluations[1] = eval_1 * scalar;
        evaluations[2] = eval_2 * scalar;
        if self.poly.aux_info.max_degree > 2 {
            extrapolate_from_table(&mut evaluations, 3);
        }
        evaluations
    }

    fn term_round_evaluations_degree3(
        &self,
        term: &Term<either::Either<E::BaseField, E>, usize>,
        round: usize,
    ) -> Vec<E> {
        let idx0 = term.product[0];
        let idx1 = term.product[1];
        let idx2 = term.product[2];
        let live_vars = [idx0, idx1, idx2]
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
        let factor0 = self.fixed_frontload_factor(idx0, round);
        let factor1 = self.fixed_frontload_factor(idx1, round);
        let factor2 = self.fixed_frontload_factor(idx2, round);
        let scalar = scalar_to_ext(&term.scalar);

        if let (FieldType::Ext(evals0), FieldType::Ext(evals1), FieldType::Ext(evals2)) = (
            self.mles[idx0].evaluations(),
            self.mles[idx1].evaluations(),
            self.mles[idx2].evaluations(),
        ) {
            let original_num_vars0 = self.poly.flattened_ml_extensions[idx0].num_vars();
            let original_num_vars1 = self.poly.flattened_ml_extensions[idx1].num_vars();
            let original_num_vars2 = self.poly.flattened_ml_extensions[idx2].num_vars();
            let mut eval_0 = E::ZERO;
            let mut eval_1 = E::ZERO;
            let mut eval_2 = E::ZERO;
            let mut eval_3 = E::ZERO;
            for_each_active_lane(lane_count, required_ones_mask, |lane| {
                let (a0, a1) =
                    ext_round_endpoints(evals0.as_slice(), original_num_vars0, round, lane);
                let (b0, b1) =
                    ext_round_endpoints(evals1.as_slice(), original_num_vars1, round, lane);
                let (c0, c1) =
                    ext_round_endpoints(evals2.as_slice(), original_num_vars2, round, lane);
                let a0 = a0 * factor0;
                let a1 = a1 * factor0;
                let b0 = b0 * factor1;
                let b1 = b1 * factor1;
                let c0 = c0 * factor2;
                let c1 = c1 * factor2;
                let a_delta = a1 - a0;
                let b_delta = b1 - b0;
                let c_delta = c1 - c0;
                let a2 = a1 + a_delta;
                let b2 = b1 + b_delta;
                let c2 = c1 + c_delta;
                let a3 = a2 + a_delta;
                let b3 = b2 + b_delta;
                let c3 = c2 + c_delta;
                eval_0 += a0 * b0 * c0;
                eval_1 += a1 * b1 * c1;
                eval_2 += a2 * b2 * c2;
                eval_3 += a3 * b3 * c3;
            });
            return self.finalize_degree3_evaluations(eval_0, eval_1, eval_2, eval_3, scalar);
        }

        let mut eval_0 = E::ZERO;
        let mut eval_1 = E::ZERO;
        let mut eval_2 = E::ZERO;
        let mut eval_3 = E::ZERO;
        for_each_active_lane(lane_count, required_ones_mask, |lane| {
            let (a0, a1) = self.mle_round_endpoints_scaled(idx0, round, lane, factor0);
            let (b0, b1) = self.mle_round_endpoints_scaled(idx1, round, lane, factor1);
            let (c0, c1) = self.mle_round_endpoints_scaled(idx2, round, lane, factor2);
            let a_delta = a1 - a0;
            let b_delta = b1 - b0;
            let c_delta = c1 - c0;
            let a2 = a1 + a_delta;
            let b2 = b1 + b_delta;
            let c2 = c1 + c_delta;
            let a3 = a2 + a_delta;
            let b3 = b2 + b_delta;
            let c3 = c2 + c_delta;
            eval_0 += a0 * b0 * c0;
            eval_1 += a1 * b1 * c1;
            eval_2 += a2 * b2 * c2;
            eval_3 += a3 * b3 * c3;
        });

        self.finalize_degree3_evaluations(eval_0, eval_1, eval_2, eval_3, scalar)
    }

    fn term_round_evaluations_degree4(
        &self,
        term: &Term<either::Either<E::BaseField, E>, usize>,
        round: usize,
    ) -> Vec<E> {
        let idx0 = term.product[0];
        let idx1 = term.product[1];
        let idx2 = term.product[2];
        let idx3 = term.product[3];
        let live_vars = [idx0, idx1, idx2, idx3]
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
        let factor0 = self.fixed_frontload_factor(idx0, round);
        let factor1 = self.fixed_frontload_factor(idx1, round);
        let factor2 = self.fixed_frontload_factor(idx2, round);
        let factor3 = self.fixed_frontload_factor(idx3, round);
        let scalar = scalar_to_ext(&term.scalar);

        if let (
            FieldType::Ext(evals0),
            FieldType::Ext(evals1),
            FieldType::Ext(evals2),
            FieldType::Ext(evals3),
        ) = (
            self.mles[idx0].evaluations(),
            self.mles[idx1].evaluations(),
            self.mles[idx2].evaluations(),
            self.mles[idx3].evaluations(),
        ) {
            let original_num_vars0 = self.poly.flattened_ml_extensions[idx0].num_vars();
            let original_num_vars1 = self.poly.flattened_ml_extensions[idx1].num_vars();
            let original_num_vars2 = self.poly.flattened_ml_extensions[idx2].num_vars();
            let original_num_vars3 = self.poly.flattened_ml_extensions[idx3].num_vars();
            let mut eval_0 = E::ZERO;
            let mut eval_1 = E::ZERO;
            let mut eval_2 = E::ZERO;
            let mut eval_3 = E::ZERO;
            let mut eval_4 = E::ZERO;
            for_each_active_lane(lane_count, required_ones_mask, |lane| {
                let (a0, a1) =
                    ext_round_endpoints(evals0.as_slice(), original_num_vars0, round, lane);
                let (b0, b1) =
                    ext_round_endpoints(evals1.as_slice(), original_num_vars1, round, lane);
                let (c0, c1) =
                    ext_round_endpoints(evals2.as_slice(), original_num_vars2, round, lane);
                let (d0, d1) =
                    ext_round_endpoints(evals3.as_slice(), original_num_vars3, round, lane);
                let a0 = a0 * factor0;
                let a1 = a1 * factor0;
                let b0 = b0 * factor1;
                let b1 = b1 * factor1;
                let c0 = c0 * factor2;
                let c1 = c1 * factor2;
                let d0 = d0 * factor3;
                let d1 = d1 * factor3;
                let a_delta = a1 - a0;
                let b_delta = b1 - b0;
                let c_delta = c1 - c0;
                let d_delta = d1 - d0;
                let a2 = a1 + a_delta;
                let b2 = b1 + b_delta;
                let c2 = c1 + c_delta;
                let d2 = d1 + d_delta;
                let a3 = a2 + a_delta;
                let b3 = b2 + b_delta;
                let c3 = c2 + c_delta;
                let d3 = d2 + d_delta;
                let a4 = a3 + a_delta;
                let b4 = b3 + b_delta;
                let c4 = c3 + c_delta;
                let d4 = d3 + d_delta;
                eval_0 += a0 * b0 * c0 * d0;
                eval_1 += a1 * b1 * c1 * d1;
                eval_2 += a2 * b2 * c2 * d2;
                eval_3 += a3 * b3 * c3 * d3;
                eval_4 += a4 * b4 * c4 * d4;
            });
            return self
                .finalize_degree4_evaluations(eval_0, eval_1, eval_2, eval_3, eval_4, scalar);
        }

        let mut eval_0 = E::ZERO;
        let mut eval_1 = E::ZERO;
        let mut eval_2 = E::ZERO;
        let mut eval_3 = E::ZERO;
        let mut eval_4 = E::ZERO;
        for_each_active_lane(lane_count, required_ones_mask, |lane| {
            let (a0, a1) = self.mle_round_endpoints_scaled(idx0, round, lane, factor0);
            let (b0, b1) = self.mle_round_endpoints_scaled(idx1, round, lane, factor1);
            let (c0, c1) = self.mle_round_endpoints_scaled(idx2, round, lane, factor2);
            let (d0, d1) = self.mle_round_endpoints_scaled(idx3, round, lane, factor3);
            let a_delta = a1 - a0;
            let b_delta = b1 - b0;
            let c_delta = c1 - c0;
            let d_delta = d1 - d0;
            let a2 = a1 + a_delta;
            let b2 = b1 + b_delta;
            let c2 = c1 + c_delta;
            let d2 = d1 + d_delta;
            let a3 = a2 + a_delta;
            let b3 = b2 + b_delta;
            let c3 = c2 + c_delta;
            let d3 = d2 + d_delta;
            let a4 = a3 + a_delta;
            let b4 = b3 + b_delta;
            let c4 = c3 + c_delta;
            let d4 = d3 + d_delta;
            eval_0 += a0 * b0 * c0 * d0;
            eval_1 += a1 * b1 * c1 * d1;
            eval_2 += a2 * b2 * c2 * d2;
            eval_3 += a3 * b3 * c3 * d3;
            eval_4 += a4 * b4 * c4 * d4;
        });

        self.finalize_degree4_evaluations(eval_0, eval_1, eval_2, eval_3, eval_4, scalar)
    }

    fn term_round_evaluations_degree5(
        &self,
        term: &Term<either::Either<E::BaseField, E>, usize>,
        round: usize,
    ) -> Vec<E> {
        let idx0 = term.product[0];
        let idx1 = term.product[1];
        let idx2 = term.product[2];
        let idx3 = term.product[3];
        let idx4 = term.product[4];
        let live_vars = [idx0, idx1, idx2, idx3, idx4]
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
        let factor0 = self.fixed_frontload_factor(idx0, round);
        let factor1 = self.fixed_frontload_factor(idx1, round);
        let factor2 = self.fixed_frontload_factor(idx2, round);
        let factor3 = self.fixed_frontload_factor(idx3, round);
        let factor4 = self.fixed_frontload_factor(idx4, round);
        let scalar = scalar_to_ext(&term.scalar);

        if let (
            FieldType::Ext(evals0),
            FieldType::Ext(evals1),
            FieldType::Ext(evals2),
            FieldType::Ext(evals3),
            FieldType::Ext(evals4),
        ) = (
            self.mles[idx0].evaluations(),
            self.mles[idx1].evaluations(),
            self.mles[idx2].evaluations(),
            self.mles[idx3].evaluations(),
            self.mles[idx4].evaluations(),
        ) {
            let original_num_vars0 = self.poly.flattened_ml_extensions[idx0].num_vars();
            let original_num_vars1 = self.poly.flattened_ml_extensions[idx1].num_vars();
            let original_num_vars2 = self.poly.flattened_ml_extensions[idx2].num_vars();
            let original_num_vars3 = self.poly.flattened_ml_extensions[idx3].num_vars();
            let original_num_vars4 = self.poly.flattened_ml_extensions[idx4].num_vars();
            let mut eval_0 = E::ZERO;
            let mut eval_1 = E::ZERO;
            let mut eval_2 = E::ZERO;
            let mut eval_3 = E::ZERO;
            let mut eval_4 = E::ZERO;
            let mut eval_5 = E::ZERO;
            for_each_active_lane(lane_count, required_ones_mask, |lane| {
                let (a0, a1) =
                    ext_round_endpoints(evals0.as_slice(), original_num_vars0, round, lane);
                let (b0, b1) =
                    ext_round_endpoints(evals1.as_slice(), original_num_vars1, round, lane);
                let (c0, c1) =
                    ext_round_endpoints(evals2.as_slice(), original_num_vars2, round, lane);
                let (d0, d1) =
                    ext_round_endpoints(evals3.as_slice(), original_num_vars3, round, lane);
                let (e0, e1) =
                    ext_round_endpoints(evals4.as_slice(), original_num_vars4, round, lane);
                let a0 = a0 * factor0;
                let a1 = a1 * factor0;
                let b0 = b0 * factor1;
                let b1 = b1 * factor1;
                let c0 = c0 * factor2;
                let c1 = c1 * factor2;
                let d0 = d0 * factor3;
                let d1 = d1 * factor3;
                let e0 = e0 * factor4;
                let e1 = e1 * factor4;
                let a_delta = a1 - a0;
                let b_delta = b1 - b0;
                let c_delta = c1 - c0;
                let d_delta = d1 - d0;
                let e_delta = e1 - e0;
                let a2 = a1 + a_delta;
                let b2 = b1 + b_delta;
                let c2 = c1 + c_delta;
                let d2 = d1 + d_delta;
                let e2 = e1 + e_delta;
                let a3 = a2 + a_delta;
                let b3 = b2 + b_delta;
                let c3 = c2 + c_delta;
                let d3 = d2 + d_delta;
                let e3 = e2 + e_delta;
                let a4 = a3 + a_delta;
                let b4 = b3 + b_delta;
                let c4 = c3 + c_delta;
                let d4 = d3 + d_delta;
                let e4 = e3 + e_delta;
                let a5 = a4 + a_delta;
                let b5 = b4 + b_delta;
                let c5 = c4 + c_delta;
                let d5 = d4 + d_delta;
                let e5 = e4 + e_delta;
                eval_0 += a0 * b0 * c0 * d0 * e0;
                eval_1 += a1 * b1 * c1 * d1 * e1;
                eval_2 += a2 * b2 * c2 * d2 * e2;
                eval_3 += a3 * b3 * c3 * d3 * e3;
                eval_4 += a4 * b4 * c4 * d4 * e4;
                eval_5 += a5 * b5 * c5 * d5 * e5;
            });
            return self.finalize_degree5_evaluations(
                [eval_0, eval_1, eval_2, eval_3, eval_4, eval_5],
                scalar,
            );
        }

        let mut eval_0 = E::ZERO;
        let mut eval_1 = E::ZERO;
        let mut eval_2 = E::ZERO;
        let mut eval_3 = E::ZERO;
        let mut eval_4 = E::ZERO;
        let mut eval_5 = E::ZERO;
        for_each_active_lane(lane_count, required_ones_mask, |lane| {
            let (a0, a1) = self.mle_round_endpoints_scaled(idx0, round, lane, factor0);
            let (b0, b1) = self.mle_round_endpoints_scaled(idx1, round, lane, factor1);
            let (c0, c1) = self.mle_round_endpoints_scaled(idx2, round, lane, factor2);
            let (d0, d1) = self.mle_round_endpoints_scaled(idx3, round, lane, factor3);
            let (e0, e1) = self.mle_round_endpoints_scaled(idx4, round, lane, factor4);
            let a_delta = a1 - a0;
            let b_delta = b1 - b0;
            let c_delta = c1 - c0;
            let d_delta = d1 - d0;
            let e_delta = e1 - e0;
            let a2 = a1 + a_delta;
            let b2 = b1 + b_delta;
            let c2 = c1 + c_delta;
            let d2 = d1 + d_delta;
            let e2 = e1 + e_delta;
            let a3 = a2 + a_delta;
            let b3 = b2 + b_delta;
            let c3 = c2 + c_delta;
            let d3 = d2 + d_delta;
            let e3 = e2 + e_delta;
            let a4 = a3 + a_delta;
            let b4 = b3 + b_delta;
            let c4 = c3 + c_delta;
            let d4 = d3 + d_delta;
            let e4 = e3 + e_delta;
            let a5 = a4 + a_delta;
            let b5 = b4 + b_delta;
            let c5 = c4 + c_delta;
            let d5 = d4 + d_delta;
            let e5 = e4 + e_delta;
            eval_0 += a0 * b0 * c0 * d0 * e0;
            eval_1 += a1 * b1 * c1 * d1 * e1;
            eval_2 += a2 * b2 * c2 * d2 * e2;
            eval_3 += a3 * b3 * c3 * d3 * e3;
            eval_4 += a4 * b4 * c4 * d4 * e4;
            eval_5 += a5 * b5 * c5 * d5 * e5;
        });

        self.finalize_degree5_evaluations([eval_0, eval_1, eval_2, eval_3, eval_4, eval_5], scalar)
    }

    fn finalize_degree3_evaluations(
        &self,
        eval_0: E,
        eval_1: E,
        eval_2: E,
        eval_3: E,
        scalar: E,
    ) -> Vec<E> {
        let mut evaluations = vec![E::ZERO; self.poly.aux_info.max_degree + 1];
        evaluations[0] = eval_0 * scalar;
        evaluations[1] = eval_1 * scalar;
        evaluations[2] = eval_2 * scalar;
        evaluations[3] = eval_3 * scalar;
        if self.poly.aux_info.max_degree > 3 {
            extrapolate_from_table(&mut evaluations, 4);
        }
        evaluations
    }

    fn finalize_degree4_evaluations(
        &self,
        eval_0: E,
        eval_1: E,
        eval_2: E,
        eval_3: E,
        eval_4: E,
        scalar: E,
    ) -> Vec<E> {
        let mut evaluations = vec![E::ZERO; self.poly.aux_info.max_degree + 1];
        evaluations[0] = eval_0 * scalar;
        evaluations[1] = eval_1 * scalar;
        evaluations[2] = eval_2 * scalar;
        evaluations[3] = eval_3 * scalar;
        evaluations[4] = eval_4 * scalar;
        if self.poly.aux_info.max_degree > 4 {
            extrapolate_from_table(&mut evaluations, 5);
        }
        evaluations
    }

    fn finalize_degree5_evaluations(&self, evals: [E; 6], scalar: E) -> Vec<E> {
        let mut evaluations = vec![E::ZERO; self.poly.aux_info.max_degree + 1];
        evaluations[0] = evals[0] * scalar;
        evaluations[1] = evals[1] * scalar;
        evaluations[2] = evals[2] * scalar;
        evaluations[3] = evals[3] * scalar;
        evaluations[4] = evals[4] * scalar;
        evaluations[5] = evals[5] * scalar;
        if self.poly.aux_info.max_degree > 5 {
            extrapolate_from_table(&mut evaluations, 6);
        }
        evaluations
    }

    fn term_round_evaluations_degree<const D: usize>(
        &self,
        term: &Term<either::Either<E::BaseField, E>, usize>,
        round: usize,
    ) -> Vec<E> {
        debug_assert_eq!(term.product.len(), D);
        debug_assert!((1..=5).contains(&D));
        let indices: [usize; D] = term.product.as_slice().try_into().unwrap();
        let live_vars = indices
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
        let frontload_factors = indices.map(|idx| self.fixed_frontload_factor(idx, round));
        let scalar = scalar_to_ext(&term.scalar);
        let mut evaluations = vec![E::ZERO; self.poly.aux_info.max_degree + 1];

        for_each_active_lane(lane_count, required_ones_mask, |lane| {
            let endpoints: [(E, E); D] = std::array::from_fn(|i| {
                let (v0, v1) =
                    self.mle_round_endpoints_scaled(indices[i], round, lane, frontload_factors[i]);
                (v0, v1 - v0)
            });
            for (z_idx, eval) in evaluations.iter_mut().take(D + 1).enumerate() {
                let product = endpoints
                    .iter()
                    .map(|&(v0, delta)| v0 + delta * E::from_canonical_u64(z_idx as u64))
                    .product::<E>();
                *eval += product * scalar;
            }
        });

        if D < self.poly.aux_info.max_degree {
            extrapolate_from_table(&mut evaluations, D + 1);
        }
        evaluations
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
                .any(|&idx| self.global_mle_num_vars[idx] <= var_idx);
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
        let original_num_vars = self.poly.flattened_ml_extensions[mle_idx].num_vars();
        if round >= original_num_vars {
            return (E::ZERO, read_eval(&self.mles[mle_idx], 0));
        }

        let remaining_vars = original_num_vars - round;
        let suffix_mask = if remaining_vars == 1 {
            0
        } else {
            (1usize << (remaining_vars - 1)) - 1
        };
        let suffix = lane & suffix_mask;
        let evals = self.mles[mle_idx].evaluations();
        let e0 = field_index(evals, suffix << 1);
        let e1 = field_index(evals, (suffix << 1) + 1);
        (e0, e1)
    }

    fn fixed_frontload_factor(&self, mle_idx: usize, round: usize) -> E {
        let original_num_vars = self.global_mle_num_vars[mle_idx];
        if round <= original_num_vars {
            return E::ONE;
        }
        self.challenges[original_num_vars..round]
            .iter()
            .fold(E::ONE, |acc, challenge| acc * challenge.elements)
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
        let mle = match meta {
            FrontloadPolyMeta::Normal => {
                let values = workers
                    .iter()
                    .map(|worker| read_eval(&worker.mles[mle_idx], 0))
                    .collect_vec();
                MultilinearExtension::from_evaluations_ext_vec(log_num_workers, values)
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

fn read_eval<E: ExtensionField>(mle: &MultilinearExtension<'_, E>, idx: usize) -> E {
    field_index(mle.evaluations(), idx)
}

fn field_index<E: ExtensionField>(evals: &FieldType<'_, E>, idx: usize) -> E {
    match evals.index(idx) {
        either::Either::Left(base) => E::from(base),
        either::Either::Right(ext) => ext,
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
        return (E::ZERO, evals[0]);
    }

    let remaining_vars = original_num_vars - round;
    let suffix_mask = if remaining_vars == 1 {
        0
    } else {
        (1usize << (remaining_vars - 1)) - 1
    };
    let suffix = lane & suffix_mask;
    (evals[suffix << 1], evals[(suffix << 1) + 1])
}

fn term_matches_frontload_tail<E: ExtensionField>(
    local_num_vars: usize,
    global_mle_num_vars: &[usize],
    worker: Option<(usize, usize)>,
    term: &Term<either::Either<E::BaseField, E>, usize>,
) -> bool {
    let Some((worker_id, log_num_workers)) = worker else {
        return true;
    };
    let term_num_vars = term
        .product
        .iter()
        .map(|&idx| global_mle_num_vars[idx])
        .max()
        .unwrap_or(0);
    (0..log_num_workers).all(|phase2_bit| {
        let global_var = local_num_vars + phase2_bit;
        term_num_vars > global_var || ((worker_id >> phase2_bit) & 1) == 1
    })
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

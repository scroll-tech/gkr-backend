use std::sync::Arc;

use ff_ext::ExtensionField;
use itertools::Itertools;
use multilinear_extensions::{
    mle::{FieldType, MultilinearExtension},
    monomial::Term,
    virtual_poly::{MonomialTerms, VPAuxInfo, VirtualPolynomial},
    virtual_polys::{PolyMeta, VirtualPolynomials},
};
use rayon::prelude::*;
use transcript::{Challenge, Transcript};

use crate::{
    structs::{IOPProof, IOPProverMessage},
    util::extrapolate_from_table,
};

#[derive(Clone, Debug)]
pub struct FrontLoadedProverState<E: ExtensionField> {
    pub challenges: Vec<Challenge<E>>,
    pub final_evaluations: Vec<Vec<E>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FrontLoadedPolyMeta {
    Normal,
    Phase1Only,
}

impl From<PolyMeta> for FrontLoadedPolyMeta {
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
) -> (IOPProof<E>, FrontLoadedProverState<E>) {
    prove_inner(WorkingState::new(poly), transcript, true)
}

pub fn prove_2phase<'a, E: ExtensionField>(
    virtual_poly: VirtualPolynomials<'a, E>,
    transcript: &mut impl Transcript<E>,
) -> (IOPProof<E>, FrontLoadedProverState<E>) {
    let log_num_workers = p3::util::log2_strict_usize(virtual_poly.num_threads);
    let max_degree = virtual_poly.degree();
    let (polys, poly_meta) = virtual_poly.get_batched_polys();
    let front_loaded_poly_meta = poly_meta
        .into_iter()
        .map(FrontLoadedPolyMeta::from)
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
                .zip_eq(&front_loaded_poly_meta)
                .map(|(mle, meta)| match meta {
                    FrontLoadedPolyMeta::Normal => mle.num_vars() + log_num_workers,
                    FrontLoadedPolyMeta::Phase1Only => mle.num_vars(),
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
    let mut challenge = None;

    for round in 0..local_num_vars {
        if let Some(challenge) = challenge.take() {
            for worker in &mut workers {
                worker.challenges.push(challenge);
                worker.fold_round(challenge.elements);
            }
        }

        let mut evaluations = workers
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
            );
        evaluations.remove(0);
        transcript.append_field_element_exts(&evaluations);
        proofs.push(IOPProverMessage { evaluations });
        challenge = Some(transcript.sample_and_append_challenge(b"Internal round"));
    }

    if let Some(challenge) = challenge.take() {
        for worker in &mut workers {
            worker.challenges.push(challenge);
            worker.fold_round(challenge.elements);
        }
    }

    let phase2_poly = build_phase2_poly(
        &workers,
        &front_loaded_poly_meta,
        local_num_vars,
        log_num_workers,
    );
    let (phase2_proof, phase2_state) =
        prove_inner(WorkingState::new(phase2_poly), transcript, false);
    proofs.extend(phase2_proof.proofs);

    (
        IOPProof { proofs },
        FrontLoadedProverState {
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
) -> (IOPProof<E>, FrontLoadedProverState<E>) {
    let num_vars = state.poly.aux_info.max_num_variables;
    let max_degree = state.poly.aux_info.max_degree;

    if append_header {
        transcript.append_message(&num_vars.to_le_bytes());
        transcript.append_message(&max_degree.to_le_bytes());
    }

    let mut proof = Vec::with_capacity(num_vars);
    let mut challenge = None;

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
        FrontLoadedProverState {
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
    challenges: Vec<Challenge<E>>,
    global_mle_num_vars: Vec<usize>,
    worker: Option<(usize, usize)>,
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
        Self {
            poly,
            mles,
            challenges: vec![],
            global_mle_num_vars,
            worker,
        }
    }

    fn round_evaluations(&self, round: usize) -> Vec<E> {
        self.poly
            .products
            .iter()
            .map(|MonomialTerms { terms }| {
                terms
                    .iter()
                    .map(|term| self.term_round_evaluations(term, round))
                    .fold(
                        vec![E::ZERO; self.poly.aux_info.max_degree + 1],
                        |mut acc, evals| {
                            acc.iter_mut().zip_eq(evals).for_each(|(acc, eval)| {
                                *acc += eval;
                            });
                            acc
                        },
                    )
            })
            .fold(
                vec![E::ZERO; self.poly.aux_info.max_degree + 1],
                |mut acc, evals| {
                    acc.iter_mut().zip_eq(evals).for_each(|(acc, eval)| {
                        *acc += eval;
                    });
                    acc
                },
            )
    }

    fn term_round_evaluations(
        &self,
        term: &Term<either::Either<E::BaseField, E>, usize>,
        round: usize,
    ) -> Vec<E> {
        let degree = term.product.len();
        if !self.worker_matches_front_loaded_tail(term) {
            return vec![E::ZERO; self.poly.aux_info.max_degree + 1];
        }
        if degree == 1 {
            return self.term_round_evaluations_degree1(term, round);
        }
        if degree == 2 {
            return self.term_round_evaluations_degree2(term, round);
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
        let required_ones_mask = self.required_future_front_loaded_mask(term, round, live_vars);
        let fixed_front_loaded = term
            .product
            .iter()
            .map(|&idx| self.fixed_front_loaded_factor(idx, round))
            .collect_vec();

        let evaluations = active_lanes(lane_count, required_ones_mask)
            .into_par_iter()
            .map(|lane| {
                let mut local = vec![E::ZERO; degree + 1];
                for (z_idx, eval) in local.iter_mut().enumerate() {
                    let z = E::from_canonical_u64(z_idx as u64);
                    let product = term
                        .product
                        .iter()
                        .zip_eq(&fixed_front_loaded)
                        .map(|(&idx, &front_loaded_factor)| {
                            self.mle_round_value(idx, round, lane, z) * front_loaded_factor
                        })
                        .product::<E>();
                    either::for_both!(&term.scalar, scalar => {
                        *eval = product * *scalar;
                    });
                }
                local
            })
            .reduce(
                || vec![E::ZERO; degree + 1],
                |mut acc, evals| {
                    acc.iter_mut().zip_eq(evals).for_each(|(acc, eval)| {
                        *acc += eval;
                    });
                    acc
                },
            );

        if degree == self.poly.aux_info.max_degree {
            evaluations
        } else {
            let mut extrapolated = vec![E::ZERO; self.poly.aux_info.max_degree + 1];
            extrapolated[..=degree].copy_from_slice(&evaluations);
            extrapolate_from_table(&mut extrapolated, degree + 1);
            extrapolated
        }
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
        let required_ones_mask = self.required_future_front_loaded_mask(term, round, live_vars);
        let front_loaded_factor = self.fixed_front_loaded_factor(mle_idx, round);
        let scalar = match &term.scalar {
            either::Either::Left(base) => E::from(*base),
            either::Either::Right(ext) => *ext,
        };

        let (eval_0, eval_1) = active_lanes(lane_count, required_ones_mask)
            .into_par_iter()
            .map(|lane| {
                (
                    self.mle_round_value(mle_idx, round, lane, E::ZERO) * front_loaded_factor,
                    self.mle_round_value(mle_idx, round, lane, E::ONE) * front_loaded_factor,
                )
            })
            .reduce(
                || (E::ZERO, E::ZERO),
                |(a0, a1), (b0, b1)| (a0 + b0, a1 + b1),
            );

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
        let required_ones_mask = self.required_future_front_loaded_mask(term, round, live_vars);
        let lhs_front_loaded = self.fixed_front_loaded_factor(lhs_idx, round);
        let rhs_front_loaded = self.fixed_front_loaded_factor(rhs_idx, round);
        let scalar = match &term.scalar {
            either::Either::Left(base) => E::from(*base),
            either::Either::Right(ext) => *ext,
        };

        let (eval_0, eval_1, eval_2) = active_lanes(lane_count, required_ones_mask)
            .into_par_iter()
            .map(|lane| {
                let lhs_0 = self.mle_round_value(lhs_idx, round, lane, E::ZERO) * lhs_front_loaded;
                let lhs_1 = self.mle_round_value(lhs_idx, round, lane, E::ONE) * lhs_front_loaded;
                let rhs_0 = self.mle_round_value(rhs_idx, round, lane, E::ZERO) * rhs_front_loaded;
                let rhs_1 = self.mle_round_value(rhs_idx, round, lane, E::ONE) * rhs_front_loaded;
                let lhs_2 = lhs_1 + (lhs_1 - lhs_0);
                let rhs_2 = rhs_1 + (rhs_1 - rhs_0);
                (lhs_0 * rhs_0, lhs_1 * rhs_1, lhs_2 * rhs_2)
            })
            .reduce(
                || (E::ZERO, E::ZERO, E::ZERO),
                |(a0, a1, a2), (b0, b1, b2)| (a0 + b0, a1 + b1, a2 + b2),
            );

        let mut evaluations = vec![E::ZERO; self.poly.aux_info.max_degree + 1];
        evaluations[0] = eval_0 * scalar;
        evaluations[1] = eval_1 * scalar;
        evaluations[2] = eval_2 * scalar;
        if self.poly.aux_info.max_degree > 2 {
            extrapolate_from_table(&mut evaluations, 3);
        }
        evaluations
    }

    fn required_future_front_loaded_mask(
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
            let needs_front_loaded = term
                .product
                .iter()
                .any(|&idx| self.global_mle_num_vars[idx] <= var_idx);
            if needs_front_loaded {
                mask | (1usize << (var_idx - round - 1))
            } else {
                mask
            }
        })
    }

    fn worker_matches_front_loaded_tail(
        &self,
        term: &Term<either::Either<E::BaseField, E>, usize>,
    ) -> bool {
        let Some((worker_id, log_num_workers)) = self.worker else {
            return true;
        };
        let local_num_vars = self.poly.aux_info.max_num_variables;
        let term_num_vars = term
            .product
            .iter()
            .map(|&idx| self.global_mle_num_vars[idx])
            .max()
            .unwrap_or(0);
        (0..log_num_workers).all(|phase2_bit| {
            let global_var = local_num_vars + phase2_bit;
            term_num_vars > global_var || ((worker_id >> phase2_bit) & 1) == 1
        })
    }

    fn mle_round_value(&self, mle_idx: usize, round: usize, lane: usize, z: E) -> E {
        let original_num_vars = self.poly.flattened_ml_extensions[mle_idx].num_vars();
        if round >= original_num_vars {
            return read_eval(&self.mles[mle_idx], 0) * z;
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
        e0 + z * (e1 - e0)
    }

    fn fixed_front_loaded_factor(&self, mle_idx: usize, round: usize) -> E {
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
    poly_meta: &[FrontLoadedPolyMeta],
    local_num_vars: usize,
    log_num_workers: usize,
) -> VirtualPolynomial<'a, E> {
    let first_worker = workers.first().expect("front-loaded 2phase needs workers");
    let mut poly = VirtualPolynomial::new(log_num_workers);
    poly.aux_info.max_degree = first_worker.poly.aux_info.max_degree;

    for (mle_idx, meta) in poly_meta.iter().enumerate() {
        let mle = match meta {
            FrontLoadedPolyMeta::Normal => {
                let values = workers
                    .iter()
                    .map(|worker| read_eval(&worker.mles[mle_idx], 0))
                    .collect_vec();
                MultilinearExtension::from_evaluations_ext_vec(log_num_workers, values)
            }
            FrontLoadedPolyMeta::Phase1Only => {
                let value = read_eval(&first_worker.mles[mle_idx], 0)
                    * first_worker.fixed_front_loaded_factor(mle_idx, local_num_vars);
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

fn active_lanes(lane_count: usize, required_ones_mask: usize) -> Vec<usize> {
    if required_ones_mask == 0 {
        return (0..lane_count).collect();
    }
    (0..lane_count)
        .filter(|lane| (lane & required_ones_mask) == required_ones_mask)
        .collect()
}

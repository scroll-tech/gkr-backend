use std::collections::HashSet;

use super::{
    encoding::EncodingScheme,
    structure::{BasefoldCommitPhaseProof, BasefoldSpec, MerkleTreeExt},
};
use crate::{
    Point,
    util::{
        codeword_fold_with_challenge,
        hash::write_digest_to_transcript,
        merkle_tree::{Poseidon2ExtMerkleMmcs, poseidon2_merkle_tree},
        pop_front_while,
    },
};
use ff_ext::{ExtensionField, PoseidonField};
use itertools::{Either, Itertools};
use multilinear_extensions::{Expression, virtual_polys::VirtualPolynomialsBuilder};
use p3::{
    commit::{ExtensionMmcs, Mmcs},
    field::{Field, FieldAlgebra},
    matrix::{
        Matrix,
        dense::{DenseMatrix, RowMajorMatrix},
    },
    util::log2_strict_usize,
};
use serde::{Serialize, de::DeserializeOwned};
use std::collections::VecDeque;
use sumcheck::{
    macros::{entered_span, exit_span},
    structs::{IOPProverMessage, IOPProverState},
    util::{AdditiveVec, merge_sumcheck_prover_state, optimal_sumcheck_threads},
};
use transcript::{Challenge, Transcript};

use multilinear_extensions::{
    mle::{IntoMLE, MultilinearExtension},
    virtual_poly::build_eq_x_r_vec,
};
use rayon::{
    iter::{IntoParallelIterator, IntoParallelRefMutIterator},
    prelude::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSlice,
};

use super::structure::BasefoldCommitmentWithWitness;

#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn batch_commit_phase<E: ExtensionField, Spec: BasefoldSpec<E>>(
    pp: &<Spec::EncodingScheme as EncodingScheme<E>>::ProverParameters,
    rounds: &Vec<(
        &BasefoldCommitmentWithWitness<E>,
        // for each matrix open at one point
        Vec<(Point<E>, Vec<E>)>,
    )>,
    max_num_vars: usize,
    num_rounds: usize,
    transcript: &mut impl Transcript<E>,
) -> (Vec<MerkleTreeExt<E>>, BasefoldCommitPhaseProof<E>)
where
    E::BaseField: Serialize + DeserializeOwned,
    <Poseidon2ExtMerkleMmcs<E> as Mmcs<E>>::Commitment:
        IntoIterator<Item = E::BaseField> + PartialEq,
{
    let prepare_span = entered_span!("Prepare");

    let mmcs_ext = ExtensionMmcs::<E::BaseField, E, _>::new(poseidon2_merkle_tree::<E>());
    let mmcs = poseidon2_merkle_tree::<E>();
    let mut trees: Vec<MerkleTreeExt<E>> = Vec::with_capacity(max_num_vars);
    let mut commits = Vec::with_capacity(max_num_vars);

    let total_num_polys = rounds
        .iter()
        .map(|(_, point_and_evals)| {
            point_and_evals
                .iter()
                .map(|(_, evals)| evals.len())
                .sum::<usize>()
        })
        .sum::<usize>();

    let batch_coeffs =
        &transcript.sample_and_append_challenge_pows(total_num_polys, b"batch coeffs");
    let mut batch_coeffs_iter = batch_coeffs.iter();

    // prepare
    // - codeword oracle => for FRI
    // - evals => for sumcheck
    let batch_codeword_span = entered_span!("batch_codeword");
    let mut batched_codewords: Vec<DenseMatrix<E>> = vec![];
    let mut initial_rlc_evals: Vec<MultilinearExtension<E>> = vec![];
    let mut eq: Vec<MultilinearExtension<E>> = vec![];
    for (pcs_data, point_and_evals) in rounds {
        let mats = mmcs.get_matrices(&pcs_data.codeword);
        for (i, mat) in mats.into_iter().enumerate() {
            let (point, _) = &point_and_evals[i];
            let polys = &pcs_data.polys[i];
            let num_rows = mat.height();
            let num_polys = mat.width();
            let coeffs = batch_coeffs_iter
                .by_ref()
                .take(num_polys)
                .copied()
                .collect_vec();
            let codeword = RowMajorMatrix::new(
                (0..num_rows)
                    .into_par_iter()
                    .map(|j| {
                        coeffs
                            .iter()
                            .zip_eq(mat.values[j * num_polys..(j + 1) * num_polys].iter())
                            .map(|(coeff, value)| *coeff * *value)
                            .sum::<E>()
                    })
                    .collect::<Vec<_>>(),
                2,
            );
            let num_vars = polys[0].num_vars();
            assert_eq!(1 << (num_vars + Spec::get_rate_log()), num_rows);
            let mle_base_vec = polys
                .iter()
                .map(|mle| mle.get_base_field_vec())
                .collect_vec();
            let running_evals: MultilinearExtension<_> =
                MultilinearExtension::from_evaluation_vec_smart(
                    num_vars,
                    (0..polys[0].evaluations().len())
                        .into_par_iter()
                        .map(|j| {
                            coeffs
                                .iter()
                                .zip(mle_base_vec.iter())
                                .map(|(batch_coeffs, mle)| *batch_coeffs * mle[j])
                                .sum::<E>()
                        })
                        .collect::<Vec<E>>(),
                );
            batched_codewords.push(codeword);
            initial_rlc_evals.push(running_evals);
            eq.push(build_eq_x_r_vec(point).into_mle());
        }
    }
    // sorted batch codewords by height in descending order
    let mut batched_codewords = VecDeque::from(
        batched_codewords
            .into_iter()
            .sorted_by_key(|codeword| std::cmp::Reverse(codeword.height()))
            .collect_vec(),
    );
    exit_span!(batch_codeword_span);
    exit_span!(prepare_span);

    // eq is the evaluation representation of the eq(X,r) polynomial over the hypercube
    let build_eq_span = entered_span!("Basefold::build eq");
    exit_span!(build_eq_span);

    let num_threads = optimal_sumcheck_threads(max_num_vars);
    let log2_num_threads = log2_strict_usize(num_threads);

    let mut expr_builder = VirtualPolynomialsBuilder::new(num_threads, max_num_vars);
    let eq_expr = eq
        .iter_mut()
        .map(|eq| expr_builder.lift(Either::Right(eq)))
        .collect_vec();
    let initial_rlc_evals_expr = initial_rlc_evals
        .iter_mut()
        .map(|initial_rlc_evals| expr_builder.lift(Either::Right(initial_rlc_evals)))
        .collect_vec();
    let polys = expr_builder.to_virtual_polys(
        // sumcheck formula: \sum_i \sum_b eq[point_i; b_i] * running_eval_i[b_i], |b_i| <= b and aligned on suffix
        &[eq_expr
            .into_iter()
            .zip(initial_rlc_evals_expr.clone())
            .map(|(eq, initial_rlc_evals)| eq * initial_rlc_evals)
            .sum::<Expression<E>>()],
        &[],
    );

    let (batched_polys, poly_meta) = polys.get_batched_polys();

    let mut prover_states = batched_polys
        .into_iter()
        .enumerate()
        .map(|(thread_id, poly)| {
            IOPProverState::prover_init_with_extrapolation_aux(
                thread_id == 0, // set thread_id 0 to be main worker
                poly,
                Some(log2_num_threads),
                Some(poly_meta.clone()),
            )
        })
        .collect::<Vec<_>>();
    let mut sumcheck_messages = Vec::with_capacity(num_rounds);

    let mut challenge = None;
    let mut running_codeword_opt: Option<RowMajorMatrix<E>> = None;
    let sumcheck_phase1 = entered_span!("sumcheck_phase1");
    let phase1_rounds = num_rounds.min(max_num_vars - log2_num_threads);

    for i in 0..phase1_rounds {
        challenge = basefold_one_round::<E, Spec>(
            pp,
            &mut prover_states,
            challenge,
            &mut sumcheck_messages,
            &mut running_codeword_opt,
            &mut batched_codewords,
            transcript,
            &mut trees,
            &mut commits,
            &mmcs_ext,
            i == num_rounds - 1,
        );
    }

    exit_span!(sumcheck_phase1);

    if let Some(p) = challenge {
        prover_states
            .iter_mut()
            .for_each(|prover_state| prover_state.fix_var(p.elements));
    }

    // deal with log(#thread) basefold rounds
    let merge_sumcheck_prover_state_span = entered_span!("merge_sumcheck_prover_state");
    let poly = merge_sumcheck_prover_state(&prover_states);
    let mut prover_states = vec![IOPProverState::prover_init_with_extrapolation_aux(
        true, poly, None, None,
    )];
    exit_span!(merge_sumcheck_prover_state_span);

    let mut challenge = None;

    let sumcheck_phase2 = entered_span!("sumcheck_phase2");
    let remaining_rounds = num_rounds.saturating_sub(max_num_vars - log2_num_threads);

    for i in 0..remaining_rounds {
        challenge = basefold_one_round::<E, Spec>(
            pp,
            &mut prover_states,
            challenge,
            &mut sumcheck_messages,
            &mut running_codeword_opt,
            &mut batched_codewords,
            transcript,
            &mut trees,
            &mut commits,
            &mmcs_ext,
            i == remaining_rounds - 1,
        );
    }

    exit_span!(sumcheck_phase2);

    if let Some(p) = challenge {
        prover_states[0].fix_var(p.elements);
    }

    let final_message = prover_states[0].get_mle_final_evaluations();
    // skip even index which is eq evaluations
    let final_message_indexes = initial_rlc_evals_expr
        .iter()
        .map(|expr| match expr {
            Expression::WitIn(index) => *index as usize,
            _ => unreachable!(),
        })
        .collect::<HashSet<usize>>();
    let final_message = final_message
        .into_iter()
        .enumerate()
        .filter(|(i, _)| final_message_indexes.contains(i))
        .map(|(_, m)| m)
        .collect_vec();

    if cfg!(feature = "sanity-check") {
        assert!(final_message.iter().map(|m| m.len()).all_equal());
        let final_message_agg: Vec<E> = (0..final_message[0].len())
            .map(|j| final_message.iter().map(|row| row[j]).sum())
            .collect_vec();
        // last round running oracle should be exactly the encoding of the folded polynomial.
        let basecode = <Spec::EncodingScheme as EncodingScheme<E>>::encode_slow_ext(
            p3::matrix::dense::DenseMatrix::new(final_message_agg, 1),
        );
        assert_eq!(
            basecode.values,
            mmcs_ext.get_matrices(trees.last().unwrap())[0].values
        );
        // remove last tree/commmitment which is only for debug purpose
        let _ = (trees.pop(), commits.pop());
    }
    transcript.append_field_element_exts_iter(final_message.iter().flatten());
    (
        trees,
        BasefoldCommitPhaseProof {
            sumcheck_messages,
            commits,
            final_message,
        },
    )
}

/// basefold fri round to fold codewords
#[allow(clippy::too_many_arguments)]
pub(crate) fn basefold_fri_round<E: ExtensionField, Spec: BasefoldSpec<E>>(
    pp: &<Spec::EncodingScheme as EncodingScheme<E>>::ProverParameters,
    running_codeword_opt: &mut Option<RowMajorMatrix<E>>,
    codewords: &mut VecDeque<RowMajorMatrix<E>>,
    trees: &mut Vec<MerkleTreeExt<E>>,
    commits: &mut Vec<<Poseidon2ExtMerkleMmcs<E> as Mmcs<E>>::Commitment>,
    mmcs_ext: &ExtensionMmcs<
        E::BaseField,
        E,
        <<E as ExtensionField>::BaseField as PoseidonField>::MMCS,
    >,
    challenge: E,
    is_last_round: bool,
    transcript: &mut impl Transcript<E>,
) where
    E::BaseField: Serialize + DeserializeOwned,
    <Poseidon2ExtMerkleMmcs<E> as Mmcs<E>>::Commitment:
        IntoIterator<Item = E::BaseField> + PartialEq,
{
    let target_len = running_codeword_opt
        .as_ref()
        .map(|running_codeword| running_codeword.values.len())
        .unwrap_or_else(|| {
            codewords
                .iter()
                .map(|v| v.values.len())
                .max()
                .expect("empty codeword")
        });
    let level = log2_strict_usize(target_len) - 1;
    let folding_coeffs =
        <Spec::EncodingScheme as EncodingScheme<E>>::prover_folding_coeffs_level(pp, level);
    let inv_2 = E::BaseField::from_canonical_u64(2).inverse();
    debug_assert_eq!(folding_coeffs.len(), 1 << level);

    // take codewords match with target length then fold
    let codewords_matched =
        pop_front_while(codewords, |codeword| codeword.values.len() == target_len);

    // aggregate codeword with same length
    let codeword_to_fold = (0..target_len)
        .into_par_iter()
        .map(|index| {
            running_codeword_opt
                .iter()
                .chain(codewords_matched.iter())
                .map(|codeword| codeword.values[index])
                .sum::<E>()
        })
        .collect::<Vec<E>>();

    // commit
    let codeword_as_matrix = RowMajorMatrix::new(codeword_to_fold.clone(), 2);
    let (commitment, merkle_tree) = mmcs_ext.commit_matrix(codeword_as_matrix);
    write_digest_to_transcript(&commitment, transcript);
    commits.push(commitment);
    trees.push(merkle_tree);

    // fri fold and add codewords with next target length
    *running_codeword_opt = Some(RowMajorMatrix::new(
        codeword_to_fold
            .par_chunks_exact(2)
            .zip(folding_coeffs)
            .map(|(ys, coeff)| codeword_fold_with_challenge(ys, challenge, *coeff, inv_2))
            .collect::<Vec<_>>(),
        2,
    ));

    if cfg!(feature = "sanity-check") && is_last_round {
        let (commitment, merkle_tree) =
            mmcs_ext.commit_matrix(running_codeword_opt.as_ref().map(|c| c.clone()).unwrap());
        commits.push(commitment);
        trees.push(merkle_tree);
    }
}

// do sumcheck interleaving with FRI step
#[allow(clippy::too_many_arguments)]
fn basefold_one_round<E: ExtensionField, Spec: BasefoldSpec<E>>(
    pp: &<Spec::EncodingScheme as EncodingScheme<E>>::ProverParameters,
    prover_states: &mut Vec<IOPProverState<'_, E>>,
    challenge: Option<Challenge<E>>,
    sumcheck_messages: &mut Vec<IOPProverMessage<E>>,
    running_codeword_opt: &mut Option<RowMajorMatrix<E>>,
    codewords: &mut VecDeque<RowMajorMatrix<E>>,
    transcript: &mut impl Transcript<E>,
    trees: &mut Vec<MerkleTreeExt<E>>,
    commits: &mut Vec<<Poseidon2ExtMerkleMmcs<E> as Mmcs<E>>::Commitment>,
    mmcs_ext: &ExtensionMmcs<
        E::BaseField,
        E,
        <<E as ExtensionField>::BaseField as PoseidonField>::MMCS,
    >,
    is_last_round: bool,
) -> Option<Challenge<E>>
where
    E::BaseField: Serialize + DeserializeOwned,
    <Poseidon2ExtMerkleMmcs<E> as Mmcs<E>>::Commitment:
        IntoIterator<Item = E::BaseField> + PartialEq,
{
    // 1. sumcheck part
    let sumcheck_round_span = entered_span!("basefold::sumcheck_one_round");
    let prover_msgs = prover_states
        .par_iter_mut()
        .map(|prover_state| IOPProverState::prove_round_and_update_state(prover_state, &challenge))
        .collect::<Vec<_>>();

    // for each round, we must collect #SIZE prover message
    let evaluations: AdditiveVec<E> =
        prover_msgs
            .into_iter()
            .fold(AdditiveVec::new(3), |mut acc, prover_msg| {
                acc += AdditiveVec(prover_msg.evaluations);
                acc
            });
    transcript.append_field_element_exts(&evaluations.0);
    sumcheck_messages.push(IOPProverMessage {
        evaluations: evaluations.0,
    });
    exit_span!(sumcheck_round_span);

    let next_challenge = transcript.sample_and_append_challenge(b"commit round");

    // 2. fri part
    let fri_round_span = entered_span!("basefold::fri_one_round");
    basefold_fri_round::<E, Spec>(
        pp,
        running_codeword_opt,
        codewords,
        trees,
        commits,
        mmcs_ext,
        next_challenge.elements,
        is_last_round,
        transcript,
    );
    exit_span!(fri_round_span);

    Some(next_challenge)
}

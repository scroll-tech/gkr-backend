use p3::field::PrimeCharacteristicRing;
use std::slice;

use crate::{
    Point,
    basefold::structure::{BasefoldProof, MerkleTreeExt, QueryOpeningProof},
    util::{codeword_fold_with_challenge, merkle_tree::poseidon2_merkle_tree},
};
use ff_ext::ExtensionField;
use itertools::{Itertools, izip};
use multilinear_extensions::virtual_poly::{build_eq_x_r_vec, eq_eval};
use p3::{
    commit::{BatchOpening, BatchOpeningRef, ExtensionMmcs, Mmcs},
    field::{Field, dot_product},
    fri::CommitPhaseProofStep,
    matrix::{Dimensions, dense::RowMajorMatrix},
    util::log2_strict_usize,
};
use serde::{Serialize, de::DeserializeOwned};
use sumcheck::{
    macros::{entered_span, exit_span},
    util::extrapolate_uni_poly,
};
use transcript::Transcript;

use crate::basefold::structure::QueryOpeningProofs;

use super::{
    encoding::EncodingScheme,
    structure::{BasefoldCommitment, BasefoldCommitmentWithWitness},
};

pub fn batch_query_phase<E: ExtensionField>(
    rounds: Vec<&BasefoldCommitmentWithWitness<E>>,
    trees: &[MerkleTreeExt<E>],
    num_verifier_queries: usize,
    log2_max_codeword_size: usize,
    transcript: &mut impl Transcript<E>,
) -> QueryOpeningProofs<E>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    let mmcs_ext = ExtensionMmcs::<E::BaseField, E, _>::new(poseidon2_merkle_tree::<E>());
    let mmcs = poseidon2_merkle_tree::<E>();

    // Transform the challenge queries from field elements into integers
    let queries: Vec<_> = transcript.sample_bits_and_append_vec(
        b"query indices",
        num_verifier_queries,
        log2_max_codeword_size,
    );

    queries
        .iter()
        .map(|idx| {
            let input_proofs = rounds
                .iter()
                .map(|pcs_data| {
                    // extract the even part of `idx`
                    // ---------------------------------
                    // the oracle values are committed in a row-bit-reversed format.
                    // rounding `idx` to an even value is equivalent to retrieving the "left-hand" side `j` index
                    // in the original (non-row-bit-reversed) format.
                    let idx_shift = log2_max_codeword_size - pcs_data.log2_max_codeword_size;
                    let idx = idx >> idx_shift;
                    let opened = mmcs.open_batch(idx, &pcs_data.codeword);
                    let (opened_values, opening_proof) = opened.unpack();
                    BatchOpening {
                        opened_values,
                        opening_proof,
                    }
                })
                .collect_vec();

            let (_, commit_phase_openings) =
                trees
                    .iter()
                    .fold((*idx, vec![]), |(idx, mut commit_phase_openings), tree| {
                        let leaf_idx = idx >> 1;
                        // differentiate interpolate to left or right position at next layer
                        let is_interpolate_to_right_index = (idx & 1) == 1;
                        // mask the least significant bit (LSB) for the same reason as above:
                        // 1. we only need the even part of the index.
                        // 2. since even and odd parts are concatenated in the same leaf,
                        //    the overall merkle tree height is effectively halved,
                        //    so we divide by 2.
                        let opened = mmcs_ext.open_batch(leaf_idx, tree);
                        let (mut values, opening_proof) = opened.unpack();
                        let leafs = values.pop().unwrap();
                        debug_assert_eq!(leafs.len(), 2);
                        let sibling_value = leafs[(!is_interpolate_to_right_index) as usize];
                        commit_phase_openings.push(CommitPhaseProofStep {
                            sibling_value,
                            opening_proof,
                        });
                        (idx >> 1, commit_phase_openings)
                    });
            QueryOpeningProof {
                input_proofs,
                commit_phase_openings,
            }
        })
        .collect_vec()
}

#[allow(clippy::type_complexity)]
pub fn batch_verifier_query_phase<E: ExtensionField, S: EncodingScheme<E>>(
    vp: &S::VerifierParameters,
    max_num_var: usize,
    batch_coeffs: &[E],
    fold_challenges: &[E],
    indices: &[usize],
    proof: &BasefoldProof<E>,
    rounds: &[(BasefoldCommitment<E>, Vec<(usize, (Point<E>, Vec<E>))>)],
) where
    E::BaseField: Serialize + DeserializeOwned,
{
    let inv_2 = E::BaseField::from_u64(2).inverse();
    let final_message = &proof.final_message;
    let sumcheck_messages = proof.sumcheck_proof.as_ref().unwrap();
    let encode_span = entered_span!("encode_final_codeword");
    let final_codeword = S::encode_small(
        vp,
        RowMajorMatrix::new(
            (0..final_message[0].len())
                .map(|j| final_message.iter().map(|row| row[j]).sum())
                .collect_vec(),
            1,
        ),
    );
    exit_span!(encode_span);

    let mmcs_ext = ExtensionMmcs::<E::BaseField, E, _>::new(poseidon2_merkle_tree::<E>());
    let mmcs = poseidon2_merkle_tree::<E>();
    let check_queries_span = entered_span!("check_queries");
    let log2_blowup = S::get_rate_log();
    let log2_max_codeword_size = max_num_var + log2_blowup;

    indices
        .iter()
        .zip_eq(proof.query_opening_proof.iter())
        .for_each(
            |(
                idx,
                QueryOpeningProof {
                    input_proofs,
                    commit_phase_openings: opening_ext,
                },
            )| {
                // verify base oracle query proof
                let mut idx = *idx;

                let mut reduced_openings_by_height: Vec<Option<E>> =
                    vec![None; log2_max_codeword_size + 1];
                let mut batch_coeffs_iter = batch_coeffs.iter();

                for ((commit, batch_opening), input_proof) in
                    rounds.iter().zip_eq(input_proofs.iter())
                {
                    let dimensions = batch_opening
                        .iter()
                        .map(|(num_var, (_, evals))| Dimensions {
                            width: evals.len(),
                            height: 1 << (num_var + log2_blowup),
                        })
                        .collect_vec();
                    let bits_reduced = log2_max_codeword_size - commit.log2_max_codeword_size;
                    let reduced_index = idx >> bits_reduced;
                    // verify base MMCS opening proof
                    mmcs.verify_batch(
                        &commit.commit(),
                        &dimensions,
                        reduced_index,
                        BatchOpeningRef::new(
                            &input_proof.opened_values,
                            &input_proof.opening_proof,
                        ),
                    )
                    .expect("verify mmcs opening proof failed");

                    // for each log2_height, combine codewords with randomness
                    for (mat, dimension) in
                        input_proof.opened_values.iter().zip_eq(dimensions.iter())
                    {
                        let width = mat.len();
                        assert_eq!(dimension.width, mat.len());
                        let batch_coeffs = batch_coeffs_iter
                            .by_ref()
                            .take(width)
                            .copied()
                            .collect_vec();
                        let eval = dot_product::<E, _, _>(
                            batch_coeffs.iter().copied(),
                            mat.iter().copied(),
                        );
                        let log2_height = log2_strict_usize(dimension.height);

                        if let Some(eval_acc) = reduced_openings_by_height[log2_height].as_mut() {
                            // accumulate low and high values for the same log2_height
                            *eval_acc += eval;
                        } else {
                            reduced_openings_by_height[log2_height] = Some(eval);
                        }
                    }
                }

                // fold and query
                let mut log2_height = max_num_var + log2_blowup;
                let rounds = max_num_var - S::get_basecode_msg_size_log();

                assert_eq!(rounds, fold_challenges.len());
                assert_eq!(rounds, proof.commits.len(),);
                assert_eq!(rounds, opening_ext.len(),);

                let mut folded = E::ZERO;
                for (
                    (pi_comm, r),
                    CommitPhaseProofStep {
                        sibling_value,
                        opening_proof: proof,
                    },
                ) in proof
                    .commits
                    .iter()
                    .zip_eq(fold_challenges.iter())
                    .zip_eq(opening_ext)
                {
                    let folded_idx = idx & 0x01;
                    let mut leafs = vec![*sibling_value; 2];
                    leafs[folded_idx] = folded;

                    let ro = std::mem::take(&mut reduced_openings_by_height[log2_height]);
                    if let Some(eval) = ro {
                        leafs[folded_idx] += eval;
                    }

                    let leaf_idx = idx >> 1;
                    mmcs_ext
                        .verify_batch(
                            pi_comm,
                            &[Dimensions {
                                width: 2,
                                // width is 2, thus height divide by 2 via right shift
                                height: 1 << (log2_height - 1),
                            }],
                            leaf_idx,
                            BatchOpeningRef::new(slice::from_ref(&leafs), proof),
                        )
                        .expect("verify failed");

                    let coeff = S::verifier_folding_coeffs(vp, log2_height, leaf_idx);
                    folded = codeword_fold_with_challenge(&[leafs[0], leafs[1]], *r, coeff, inv_2);
                    log2_height -= 1;
                    idx >>= 1;
                }
                assert!(
                    reduced_openings_by_height.iter().all(|v| v.is_none()),
                    "there are unused openings remain",
                );
                assert!(
                    final_codeword.values[idx] == folded,
                    "final_codeword.values[idx] value {:?} != folded {:?}",
                    final_codeword.values[idx],
                    folded
                );
            },
        );
    exit_span!(check_queries_span);

    // 1. check initial claim match with first round sumcheck value
    // we need to scale up with scalar for num_var < max_num_var
    let mut batch_coeffs_iter = batch_coeffs.iter();
    let mut expected_sum = E::ZERO;
    for round in rounds.iter() {
        for (num_var, (_, evals)) in round
            .1
            .iter()
            .filter(|(num_var, _)| *num_var >= S::get_basecode_msg_size_log())
        {
            expected_sum += evals
                .iter()
                .zip(batch_coeffs_iter.by_ref().take(evals.len()))
                .map(|(eval, coeff)| {
                    *coeff * (*eval) * E::from_u64(1 << (max_num_var - num_var) as u64)
                })
                .sum::<E>();
        }
    }

    assert_eq!(
        sumcheck_messages.len(),
        fold_challenges.len(),
        "sumcheck messages and fold challenges length mismatch"
    );

    // Reconstruct the implicit P(0) evaluation for each round and update the claim in place.
    let mut current_claim = expected_sum;
    for (msg, challenge) in sumcheck_messages.iter().zip(fold_challenges.iter()) {
        let eval_1 = msg
            .evaluations
            .first()
            .copied()
            .expect("sumcheck prover message missing evaluations");
        let eval_0 = current_claim - eval_1;
        current_claim = extrapolate_uni_poly(eval_0, &msg.evaluations, *challenge);
    }

    // check final evaluation are correct
    assert_eq!(
        current_claim,
        // \sum_i eq(p,[r,i]) * f(r,i)
        izip!(
            final_message,
            rounds.iter().flat_map(|(_, point_evals)| point_evals
                .iter()
                .filter(|(_, (point, _))| point.len() >= S::get_basecode_msg_size_log())
                .map(|(_, (point, _))| point))
        )
        .map(|(final_message, point)| {
            // coeff is the eq polynomial evaluated at the first challenge.len() variables
            let num_vars_evaluated = point.len() - S::get_basecode_msg_size_log();
            let coeff = eq_eval(
                &point[..num_vars_evaluated],
                &fold_challenges[fold_challenges.len() - num_vars_evaluated..],
            );
            // Compute eq as the partially evaluated eq polynomial
            let eq = build_eq_x_r_vec(&point[num_vars_evaluated..]);
            dot_product(
                final_message.iter().copied(),
                eq.into_iter().map(|e| e * coeff),
            )
        })
        .sum()
    );
}

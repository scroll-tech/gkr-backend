use ff_ext::ExtensionField;
use multilinear_extensions::{
    macros::{entered_span, exit_span},
    virtual_poly::VPAuxInfo,
};
use transcript::{Challenge, Transcript};

use crate::{
    structs::{IOPProof, IOPProverMessage, IOPVerifierState, SumCheckSubClaim},
    util::extrapolate_uni_poly,
};

impl<E: ExtensionField> IOPVerifierState<E> {
    pub fn verify(
        claimed_sum: E,
        proof: &IOPProof<E>,
        aux_info: &VPAuxInfo<E>,
        transcript: &mut impl Transcript<E>,
    ) -> SumCheckSubClaim<E> {
        if aux_info.max_num_variables == 0 {
            return SumCheckSubClaim {
                point: vec![],
                expected_evaluation: claimed_sum,
            };
        }
        let start = entered_span!("sum check verify");

        transcript.append_message(&aux_info.max_num_variables.to_le_bytes());
        transcript.append_message(&aux_info.max_degree.to_le_bytes());

        let mut verifier_state = IOPVerifierState::verifier_init(aux_info);
        for i in 0..aux_info.max_num_variables {
            let prover_msg = proof.proofs.get(i).expect("proof is incomplete");
            prover_msg
                .evaluations
                .iter()
                .for_each(|e| transcript.append_field_element_ext(e));
            Self::verify_round_and_update_state(&mut verifier_state, prover_msg, transcript);
        }

        let res = Self::check_and_generate_subclaim(&verifier_state, &claimed_sum);

        exit_span!(start);
        res
    }

    /// Initialize the verifier's state.
    pub fn verifier_init(index_info: &VPAuxInfo<E>) -> Self {
        let start = entered_span!("sum check verifier init");
        let verifier_state = Self {
            round: 1,
            num_vars: index_info.max_num_variables,
            max_degree: index_info.max_degree,
            finished: false,
            polynomials_received: Vec::with_capacity(index_info.max_num_variables),
            challenges: Vec::with_capacity(index_info.max_num_variables),
        };
        exit_span!(start);
        verifier_state
    }

    /// Run verifier for the current round, given a prover message.
    ///
    /// Note that `verify_round_and_update_state` only samples and stores
    /// challenges; and update the verifier's state accordingly. The actual
    /// verifications are deferred (in batch) to `check_and_generate_subclaim`
    /// at the last step.
    pub(crate) fn verify_round_and_update_state(
        &mut self,
        prover_msg: &IOPProverMessage<E>,
        transcript: &mut impl Transcript<E>,
    ) -> Challenge<E> {
        let start = entered_span!("sum check verify {}-th round and update state", self.round);

        assert!(
            !self.finished,
            "Incorrect verifier state: Verifier is already finished."
        );

        // In an interactive protocol, the verifier should
        //
        // 1. check if the received 'P(0) + P(1) = expected`.
        // 2. set `expected` to P(r)`
        //
        // When we turn the protocol to a non-interactive one, it is sufficient to defer
        // such checks to `check_and_generate_subclaim` after the last round.

        let challenge = transcript.sample_and_append_challenge(b"Internal round");
        self.challenges.push(challenge);
        self.polynomials_received
            .push(prover_msg.evaluations.to_vec());

        if self.round == self.num_vars {
            // accept and close
            self.finished = true;
        } else {
            // proceed to the next round
            self.round += 1;
        }

        exit_span!(start);
        challenge
    }

    /// This function verifies the deferred checks in the interactive version of
    /// the protocol; and generate the subclaim. Returns an error if the
    /// proof failed to verify.
    ///
    /// If the asserted sum is correct, then the multilinear polynomial
    /// evaluated at `subclaim.point` will be `subclaim.expected_evaluation`.
    /// Otherwise, it is highly unlikely that those two will be equal.
    /// Larger field size guarantees smaller soundness error.
    pub(crate) fn check_and_generate_subclaim(&self, asserted_sum: &E) -> SumCheckSubClaim<E> {
        let start = entered_span!("sum check check and generate subclaim");
        if !self.finished {
            panic!("Incorrect verifier state: Verifier has not finished.",);
        }

        if self.polynomials_received.len() != self.num_vars {
            panic!("insufficient rounds",);
        }

        // the deferred check during the interactive phase:
        // 2. set `expected` to P(r)`
        let (expected_vec, evals_0) = self
            .polynomials_received
            .iter()
            .zip(self.challenges.iter())
            .fold((vec![*asserted_sum], vec![]), |(mut claims, mut evals_0), (evaluations, challenge)| {
                let last_claim = claims.last().copied().unwrap();
                if evaluations.len() != self.max_degree {
                    panic!(
                        "incorrect number of evaluations: {} vs {}",
                        evaluations.len(),
                        self.max_degree
                    );
                }
                // https://eprint.iacr.org/2024/108.pdf sec 3.1 derive eval_0 = claim - eval_1
                let eval_0 = last_claim - evaluations.get(0).copied().unwrap();
                evals_0.push(eval_0);
                let augmented_evals =std::iter::once(eval_0).chain(evaluations.iter().copied()).collect::<Vec<_>>();
                claims.push(extrapolate_uni_poly::<E>(&augmented_evals, challenge.elements));
                (claims, evals_0)
            });

        for (i, ((evaluations, &expected), eval_0)) in self
            .polynomials_received
            .iter()
            .zip(&expected_vec)
            .zip(&evals_0)
            .enumerate()
            .take(self.num_vars)
        {
            // the deferred check during the interactive phase:
            // 1. check if the received 'P(0) + P(1) = expected`.
            if *eval_0 + evaluations[0] != expected {
                panic!(
                    "{}th round's prover message is not consistent with the claim. {:?} {:?}",
                    i,
                     *eval_0 + evaluations[0],
                    expected
                );
            }
        }
        exit_span!(start);
        SumCheckSubClaim {
            point: self.challenges.clone(),
            // the last expected value (not checked within this function) will be included in the
            // subclaim
            expected_evaluation: expected_vec[self.num_vars],
        }
    }
}

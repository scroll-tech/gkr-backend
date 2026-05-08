use crate::{
    frontload,
    structs::{IOPProverState, IOPVerifierState, SumcheckProverMode},
    util::extrapolate_uni_poly,
};
use either::Either;
use ff_ext::{BabyBearExt4, ExtensionField, FromUniformBytes, GoldilocksExt2};
use itertools::Itertools;
use multilinear_extensions::{
    mle::{MultilinearExtension, Point},
    monomial::Term,
    util::{ceil_log2, max_usable_threads},
    virtual_poly::{VPAuxInfo, VirtualPolynomial},
    virtual_polys::VirtualPolynomials,
};
use p3::field::FieldAlgebra;
use rand::{Rng, SeedableRng, rngs::StdRng, thread_rng};
use std::sync::Arc;
use transcript::{BasicTranscript, Transcript};

#[test]
fn test_frontload_mixed_size_sumcheck() {
    test_frontload_mixed_size_sumcheck_helper::<GoldilocksExt2>();
    test_frontload_mixed_size_sumcheck_helper::<BabyBearExt4>();
}

fn test_frontload_mixed_size_sumcheck_helper<E: ExtensionField>() {
    let mut rng = thread_rng();
    let num_vars = 8;
    let large = Arc::new(
        multilinear_extensions::mle::MultilinearExtension::<E>::random(num_vars, &mut rng),
    );
    let small =
        Arc::new(multilinear_extensions::mle::MultilinearExtension::<E>::random(2, &mut rng));

    let mut poly = VirtualPolynomial::new(num_vars);
    let large_idx = poly.register_mle(large);
    let small_idx = poly.register_mle(small);
    poly.aux_info.max_degree = 2;
    poly.products
        .push(multilinear_extensions::virtual_poly::MonomialTerms {
            terms: vec![Term {
                scalar: Either::Right(E::ONE),
                product: vec![large_idx, small_idx],
            }],
        });

    let asserted_sum = frontload::claimed_sum(&poly);
    let mut transcript = BasicTranscript::<E>::new(b"frontload-test");
    let (proof, _) = frontload::prove(poly.as_view(), &mut transcript);

    let mut transcript = BasicTranscript::<E>::new(b"frontload-test");
    let subclaim = IOPVerifierState::<E>::verify(
        asserted_sum,
        &proof,
        &frontload::aux_info(&poly),
        &mut transcript,
    );
    let point = subclaim
        .point
        .iter()
        .map(|challenge| challenge.elements)
        .collect_vec();

    assert_eq!(
        frontload::evaluate(&poly, &point),
        subclaim.expected_evaluation
    );
}

#[test]
fn test_frontload_2phase_sum_keeps_small_mle_compact() {
    let mut rng = thread_rng();
    let num_vars = 8;
    let num_threads = 4;
    let large = multilinear_extensions::mle::MultilinearExtension::<GoldilocksExt2>::random(
        num_vars, &mut rng,
    );
    let medium =
        multilinear_extensions::mle::MultilinearExtension::<GoldilocksExt2>::random(5, &mut rng);
    let small =
        multilinear_extensions::mle::MultilinearExtension::<GoldilocksExt2>::random(2, &mut rng);
    let poly = VirtualPolynomials::new_from_monimials(
        num_threads,
        num_vars,
        vec![
            Term {
                scalar: Either::Right(GoldilocksExt2::ONE),
                product: vec![Either::Left(&large)],
            },
            Term {
                scalar: Either::Right(GoldilocksExt2::ONE),
                product: vec![Either::Left(&medium)],
            },
            Term {
                scalar: Either::Right(GoldilocksExt2::ONE),
                product: vec![Either::Left(&small)],
            },
        ],
    );

    let mut transcript = BasicTranscript::<GoldilocksExt2>::new(b"frontload-2phase-test");
    let (proof, state) = IOPProverState::<GoldilocksExt2>::prove(poly, &mut transcript);
    assert_eq!(state.prover_mode(), SumcheckProverMode::Frontload);

    let mut direct_poly = VirtualPolynomial::new(num_vars);
    let large_idx = direct_poly.register_mle(Arc::new(large));
    let medium_idx = direct_poly.register_mle(Arc::new(medium));
    let small_idx = direct_poly.register_mle(Arc::new(small));
    direct_poly.aux_info.max_degree = 1;
    direct_poly
        .products
        .push(multilinear_extensions::virtual_poly::MonomialTerms {
            terms: vec![
                Term {
                    scalar: Either::Right(GoldilocksExt2::ONE),
                    product: vec![large_idx],
                },
                Term {
                    scalar: Either::Right(GoldilocksExt2::ONE),
                    product: vec![medium_idx],
                },
                Term {
                    scalar: Either::Right(GoldilocksExt2::ONE),
                    product: vec![small_idx],
                },
            ],
        });
    let asserted_sum = frontload::claimed_sum(&direct_poly);
    let mut transcript = BasicTranscript::<GoldilocksExt2>::new(b"frontload-2phase-test");
    let subclaim = IOPVerifierState::<GoldilocksExt2>::verify(
        asserted_sum,
        &proof,
        &frontload::aux_info(&direct_poly),
        &mut transcript,
    );
    let point = subclaim
        .point
        .iter()
        .map(|challenge| challenge.elements)
        .collect_vec();
    let mut direct_transcript = BasicTranscript::<GoldilocksExt2>::new(b"frontload-2phase-test");
    let (direct_proof, _) = frontload::prove(direct_poly.as_view(), &mut direct_transcript);
    let mut direct_verify_transcript =
        BasicTranscript::<GoldilocksExt2>::new(b"frontload-2phase-test");
    let direct_subclaim = IOPVerifierState::<GoldilocksExt2>::verify(
        asserted_sum,
        &direct_proof,
        &frontload::aux_info(&direct_poly),
        &mut direct_verify_transcript,
    );
    let direct_point = direct_subclaim
        .point
        .iter()
        .map(|challenge| challenge.elements)
        .collect_vec();
    assert_eq!(
        frontload::evaluate(&direct_poly, &direct_point),
        direct_subclaim.expected_evaluation,
        "single frontload proof failed"
    );
    for (round, (direct, two_phase)) in direct_proof.proofs.iter().zip(&proof.proofs).enumerate() {
        assert_eq!(
            direct, two_phase,
            "frontload 2phase diverged at round {round}"
        );
    }
    assert_eq!(
        frontload::evaluate(&direct_poly, &point),
        subclaim.expected_evaluation
    );
}

#[test]
fn test_frontload_small_only_sumcheck() {
    let mut rng = thread_rng();
    let num_vars = 8;
    let small = Arc::new(multilinear_extensions::mle::MultilinearExtension::<
        GoldilocksExt2,
    >::random(2, &mut rng));
    let mut poly = VirtualPolynomial::new(num_vars);
    let small_idx = poly.register_mle(small);
    poly.aux_info.max_degree = 1;
    poly.products
        .push(multilinear_extensions::virtual_poly::MonomialTerms {
            terms: vec![Term {
                scalar: Either::Right(GoldilocksExt2::ONE),
                product: vec![small_idx],
            }],
        });
    let asserted_sum = frontload::claimed_sum(&poly);
    let mut transcript = BasicTranscript::<GoldilocksExt2>::new(b"frontload-small-only");
    let (proof, _) = frontload::prove(poly.as_view(), &mut transcript);
    let mut transcript = BasicTranscript::<GoldilocksExt2>::new(b"frontload-small-only");
    let subclaim = IOPVerifierState::<GoldilocksExt2>::verify(
        asserted_sum,
        &proof,
        &frontload::aux_info(&poly),
        &mut transcript,
    );
    let point = subclaim
        .point
        .iter()
        .map(|challenge| challenge.elements)
        .collect_vec();
    assert_eq!(
        frontload::evaluate(&poly, &point),
        subclaim.expected_evaluation
    );
}

#[test]
fn test_random_monimials_use_frontload_sum() {
    let mut rng = thread_rng();
    let nv = vec![2, 4, 6];
    let degree = 2;
    let num_products = 2;
    let (monimials, asserted_sum) = VirtualPolynomials::<GoldilocksExt2>::random_monimials(
        &nv,
        (degree, degree + 1),
        num_products,
        &mut rng,
    );
    let max_num_variables = *nv.iter().max().unwrap();

    // Build a single-worker VirtualPolynomial for natural frontload evaluation check.
    // Must be built before the mutable borrow in new_from_monimials below.
    let mut direct_poly = VirtualPolynomial::new(max_num_variables);
    direct_poly.aux_info.max_degree = degree;
    for term in &monimials {
        let indices: Vec<usize> = term
            .product
            .iter()
            .map(|mle| direct_poly.register_mle(Arc::new(mle.clone())))
            .collect_vec();
        direct_poly
            .products
            .push(multilinear_extensions::virtual_poly::MonomialTerms {
                terms: vec![Term {
                    scalar: Either::Right(term.scalar),
                    product: indices,
                }],
            });
    }

    let poly = VirtualPolynomials::<GoldilocksExt2>::new_from_monimials(
        4,
        max_num_variables,
        monimials
            .iter()
            .map(|Term { scalar, product }| Term {
                scalar: Either::Right(*scalar),
                product: product.iter().map(Either::Left).collect_vec(),
            })
            .collect_vec(),
    );

    let mut transcript = BasicTranscript::<GoldilocksExt2>::new(b"frontload-random-monimials");
    let (proof, _) = IOPProverState::<GoldilocksExt2>::prove(poly, &mut transcript);
    let mut transcript = BasicTranscript::<GoldilocksExt2>::new(b"frontload-random-monimials");
    let subclaim = IOPVerifierState::<GoldilocksExt2>::verify(
        asserted_sum,
        &proof,
        &VPAuxInfo {
            max_degree: degree,
            max_num_variables,
            ..Default::default()
        },
        &mut transcript,
    );
    assert_eq!(subclaim.point.len(), max_num_variables);
    let point = subclaim
        .point
        .iter()
        .map(|challenge| challenge.elements)
        .collect_vec();
    assert_eq!(
        frontload::evaluate(&direct_poly, &point),
        subclaim.expected_evaluation,
        "frontload 2phase final evaluation mismatch: natural frontload evaluation \
         must agree with the verifier's expected evaluation"
    );
}

#[test]
fn test_frontload_2phase_mle_category_combinations() {
    let cases = [
        ("phase1_only", 2),
        ("normal_exhausted_with_tail", 3),
        ("normal_exhausted_at_phase1_end", 4),
        ("normal_not_exhausted", 6),
    ];
    let degree = 2;
    let max_num_variables = 6;
    let num_threads = 4;
    let mut rng = StdRng::seed_from_u64(50);

    for mask in 1usize..(1 << cases.len()) {
        let selected = cases
            .iter()
            .enumerate()
            .filter(|(idx, _)| ((mask >> idx) & 1) == 1)
            .collect_vec();
        let mut asserted_sum = GoldilocksExt2::ZERO;
        let monomials = selected
            .iter()
            .map(|(_, (_, num_variables))| {
                let (product, product_sum) =
                    MultilinearExtension::random_mle_list(*num_variables, degree, &mut rng);
                let scalar = GoldilocksExt2::random(&mut rng);
                asserted_sum += product_sum * scalar;
                Term { scalar, product }
            })
            .collect_vec();

        let poly = VirtualPolynomials::<GoldilocksExt2>::new_from_monimials(
            num_threads,
            max_num_variables,
            monomials
                .iter()
                .map(|Term { scalar, product }| Term {
                    scalar: Either::Right(*scalar),
                    product: product.iter().map(Either::Left).collect_vec(),
                })
                .collect_vec(),
        );

        let mut transcript =
            BasicTranscript::<GoldilocksExt2>::new(b"frontload-category-combinations");
        let (proof, _) = IOPProverState::<GoldilocksExt2>::prove(poly, &mut transcript);
        let selected_names = selected.iter().map(|(_, (name, _))| *name).join(", ");
        let mut transcript =
            BasicTranscript::<GoldilocksExt2>::new(b"frontload-category-combinations");
        let subclaim = IOPVerifierState::<GoldilocksExt2>::verify(
            asserted_sum,
            &proof,
            &VPAuxInfo {
                max_degree: degree,
                max_num_variables,
                ..Default::default()
            },
            &mut transcript,
        );
        let point = subclaim
            .point
            .iter()
            .map(|challenge| challenge.elements)
            .collect_vec();
        let mut direct_poly = VirtualPolynomial::new(max_num_variables);
        direct_poly.aux_info.max_degree = degree;
        for Term { scalar, product } in &monomials {
            let indices = product
                .iter()
                .map(|mle| direct_poly.register_mle(Arc::new(mle.clone())))
                .collect_vec();
            direct_poly
                .products
                .push(multilinear_extensions::virtual_poly::MonomialTerms {
                    terms: vec![Term {
                        scalar: Either::Right(*scalar),
                        product: indices,
                    }],
                });
        }
        assert_eq!(
            frontload::evaluate(&direct_poly, &point),
            subclaim.expected_evaluation,
            "frontload 2phase failed for {selected_names}"
        );
    }
}

// test polynomial mixed with different num_var
#[test]
fn test_sumcheck_with_different_degree() {
    let log_max_thread = ceil_log2(max_usable_threads());
    let nv = vec![1, 2, 3, 4];
    for num_threads in 1..log_max_thread {
        test_sumcheck_with_different_degree_helper::<GoldilocksExt2>(1 << num_threads, &nv);
    }
}

fn test_sumcheck_with_different_degree_helper<E: ExtensionField>(num_threads: usize, nv: &[usize]) {
    let mut rng = thread_rng();
    let degree = 2;
    let num_multiplicands_range = (degree, degree + 1);
    let num_products = 1;
    let mut transcript = BasicTranscript::<E>::new(b"test");

    let max_num_variables = *nv.iter().max().unwrap();
    let (mut monimials, asserted_sum) = VirtualPolynomials::<E>::random_suffixload_monimials(
        nv,
        num_multiplicands_range,
        num_products,
        &mut rng,
    );

    let poly = VirtualPolynomials::<E>::new_from_monimials(
        num_threads,
        max_num_variables,
        monimials
            .iter_mut()
            .map(|Term { scalar, product }| Term {
                scalar: Either::Right(*scalar),
                product: product.iter_mut().map(Either::Right).collect_vec(),
            })
            .collect_vec(),
    );

    let (proof, _) = IOPProverState::<E>::prove_suffix(poly.as_view(), &mut transcript);
    let mut transcript = BasicTranscript::new(b"test");
    let subclaim = IOPVerifierState::<E>::verify(
        asserted_sum,
        &proof,
        &VPAuxInfo {
            max_degree: degree,
            max_num_variables,
            ..Default::default()
        },
        &mut transcript,
    );
    let r: Point<E> = subclaim
        .point
        .iter()
        .map(|c| c.elements)
        .collect::<Vec<_>>();
    assert_eq!(r.len(), max_num_variables);
    // r are right alignment
    assert!(
        poly.evaluate_slow(&r) == subclaim.expected_evaluation,
        "wrong subclaim"
    );

    // test in-place work
    let mut transcript = BasicTranscript::<E>::new(b"test");
    let (proof_mut, _) = IOPProverState::<E>::prove_suffix(poly, &mut transcript);
    assert_eq!(proof, proof_mut, "different proof");
}

#[test]
fn test_runtime_prover_modes_are_compatible() {
    test_runtime_prover_modes_are_compatible_helper::<GoldilocksExt2>();
    test_runtime_prover_modes_are_compatible_helper::<BabyBearExt4>();
}

fn test_runtime_prover_modes_are_compatible_helper<E: ExtensionField>() {
    let mut rng = thread_rng();
    let nv = vec![8];
    let degree = 4;
    let num_products = 4;

    let max_num_variables = *nv.iter().max().unwrap();
    let (mut monimials, asserted_sum) = VirtualPolynomials::<E>::random_suffixload_monimials(
        &nv,
        (degree, degree + 1),
        num_products,
        &mut rng,
    );
    let poly = VirtualPolynomials::<E>::new_from_monimials(
        1,
        max_num_variables,
        monimials
            .iter_mut()
            .map(|Term { scalar, product }| Term {
                scalar: Either::Right(*scalar),
                product: product.iter_mut().map(Either::Right).collect_vec(),
            })
            .collect_vec(),
    );

    let mut transcript_legacy = BasicTranscript::<E>::new(b"mode-test");
    let (proof_legacy, _) = IOPProverState::<E>::prove_with_mode(
        poly.as_view(),
        &mut transcript_legacy,
        SumcheckProverMode::LegacyStable,
    );

    let mut transcript_reduced = BasicTranscript::<E>::new(b"mode-test");
    let (proof_reduced, _) = IOPProverState::<E>::prove_with_mode(
        poly.as_view(),
        &mut transcript_reduced,
        SumcheckProverMode::ReducedPeakMemory,
    );

    let mut verifier_transcript_legacy = BasicTranscript::<E>::new(b"mode-test");
    let legacy_subclaim = IOPVerifierState::<E>::verify(
        asserted_sum,
        &proof_legacy,
        &VPAuxInfo {
            max_degree: degree,
            max_num_variables,
            ..Default::default()
        },
        &mut verifier_transcript_legacy,
    );

    let mut verifier_transcript_reduced = BasicTranscript::<E>::new(b"mode-test");
    let reduced_subclaim = IOPVerifierState::<E>::verify(
        asserted_sum,
        &proof_reduced,
        &VPAuxInfo {
            max_degree: degree,
            max_num_variables,
            ..Default::default()
        },
        &mut verifier_transcript_reduced,
    );

    assert_eq!(legacy_subclaim, reduced_subclaim);
}

#[test]
fn test_compact_mle_matches_zero_padded_sumcheck() {
    let eval = vec![
        GoldilocksExt2::from_canonical_u32(2),
        GoldilocksExt2::from_canonical_u32(3),
        GoldilocksExt2::from_canonical_u32(5),
        GoldilocksExt2::from_canonical_u32(7),
        GoldilocksExt2::from_canonical_u32(11),
    ];
    let compact = MultilinearExtension::from_evaluations_ext_vec_compact(3, eval.clone());
    let mut padded_eval = eval;
    padded_eval.resize(1 << 3, GoldilocksExt2::ZERO);
    let padded = MultilinearExtension::from_evaluations_ext_vec(3, padded_eval);

    let compact_poly = VirtualPolynomials::new_from_monimials(
        1,
        3,
        vec![Term {
            scalar: Either::Right(GoldilocksExt2::ONE),
            product: vec![Either::Left(&compact)],
        }],
    );
    let padded_poly = VirtualPolynomials::new_from_monimials(
        1,
        3,
        vec![Term {
            scalar: Either::Right(GoldilocksExt2::ONE),
            product: vec![Either::Left(&padded)],
        }],
    );

    let mut compact_transcript = BasicTranscript::<GoldilocksExt2>::new(b"compact");
    let (compact_proof, compact_state) =
        IOPProverState::<GoldilocksExt2>::prove(compact_poly, &mut compact_transcript);

    let mut padded_transcript = BasicTranscript::<GoldilocksExt2>::new(b"compact");
    let (padded_proof, padded_state) =
        IOPProverState::<GoldilocksExt2>::prove(padded_poly, &mut padded_transcript);

    assert_eq!(compact_proof, padded_proof);
    let compact_final = compact_state.get_mle_final_evaluations();
    let padded_final = padded_state.get_mle_final_evaluations();
    assert_eq!(compact_final.len(), padded_final.len());
    for (compact_eval, padded_eval) in compact_final.iter().zip(padded_final.iter()) {
        assert_eq!(compact_eval, &padded_eval[..compact_eval.len()]);
    }
}

fn test_sumcheck<E: ExtensionField>(
    nv: usize,
    num_multiplicands_range: (usize, usize),
    num_products: usize,
) {
    let mut rng = thread_rng();
    let mut transcript = BasicTranscript::new(b"test");

    let (poly, asserted_sum) =
        VirtualPolynomial::<E>::random(&[nv], num_multiplicands_range, num_products, &mut rng);
    let poly_info = poly.aux_info.clone();
    #[allow(deprecated)]
    let (proof, _) = IOPProverState::<E>::prove_parallel(poly.as_view(), &mut transcript);

    let mut transcript = BasicTranscript::new(b"test");
    let subclaim = IOPVerifierState::<E>::verify(asserted_sum, &proof, &poly_info, &mut transcript);
    assert!(
        poly.evaluate(
            subclaim
                .point
                .iter()
                .map(|c| c.elements)
                .collect::<Vec<_>>()
                .as_ref()
        ) == subclaim.expected_evaluation,
        "wrong subclaim"
    );
}

fn test_sumcheck_internal<E: ExtensionField>(
    nv: usize,
    num_multiplicands_range: (usize, usize),
    num_products: usize,
) {
    let mut rng = thread_rng();
    let (poly, asserted_sum) =
        VirtualPolynomial::<E>::random(&[nv], num_multiplicands_range, num_products, &mut rng);
    let (poly_info, num_variables) = (poly.aux_info.clone(), poly.aux_info.max_num_variables);
    #[allow(deprecated)]
    let mut prover_state = IOPProverState::prover_init_parallel(poly.as_view());
    let mut verifier_state = IOPVerifierState::verifier_init(&poly_info);
    let mut challenge = None;

    let mut transcript = BasicTranscript::new(b"test");

    transcript.append_message(b"initializing transcript for testing");

    for _ in 0..num_variables {
        let prover_message =
            IOPProverState::prove_round_and_update_state(&mut prover_state, &challenge);

        challenge = Some(IOPVerifierState::verify_round_and_update_state(
            &mut verifier_state,
            &prover_message,
            &mut transcript,
        ));
    }
    // pushing the last challenge point to the state
    if let Some(p) = challenge {
        prover_state.push_challenges(vec![p]);
        // fix last challenge to collect final evaluation
        prover_state.fix_var(p.elements);
    };
    let subclaim = IOPVerifierState::check_and_generate_subclaim(&verifier_state, &asserted_sum);
    assert!(
        poly.evaluate(
            subclaim
                .point
                .iter()
                .map(|c| c.elements)
                .collect::<Vec<_>>()
                .as_ref()
        ) == subclaim.expected_evaluation,
        "wrong subclaim"
    );
}

#[test]
fn test_trivial_polynomial() {
    test_trivial_polynomial_helper::<GoldilocksExt2>();
    test_trivial_polynomial_helper::<BabyBearExt4>();
}

fn test_trivial_polynomial_helper<E: ExtensionField>() {
    let nv = 1;
    let num_multiplicands_range = (3, 5);
    let num_products = 5;

    test_sumcheck::<E>(nv, num_multiplicands_range, num_products);
    test_sumcheck_internal::<E>(nv, num_multiplicands_range, num_products);
}

#[test]
fn test_normal_polynomial() {
    test_normal_polynomial_helper::<GoldilocksExt2>();
    test_normal_polynomial_helper::<BabyBearExt4>();
}

fn test_normal_polynomial_helper<E: ExtensionField>() {
    let nv = 12;
    let num_multiplicands_range = (3, 5);
    let num_products = 5;

    test_sumcheck::<E>(nv, num_multiplicands_range, num_products);
    test_sumcheck_internal::<E>(nv, num_multiplicands_range, num_products);
}

struct DensePolynomial(Vec<GoldilocksExt2>);

impl DensePolynomial {
    fn rand_coeffs<R: Rng>(degree: usize, rng: &mut R) -> Self {
        Self(
            (0..=degree)
                .map(|_| GoldilocksExt2::random(&mut *rng))
                .collect(),
        )
    }

    fn evaluate(&self, p: &GoldilocksExt2) -> GoldilocksExt2 {
        let mut powers_of_p = *p;
        let mut res = self.0[0];
        for &c in self.0.iter().skip(1) {
            res += powers_of_p * c;
            powers_of_p *= *p;
        }
        res
    }
}

#[test]
fn test_extrapolation() {
    fn run_extrapolation_test(degree: usize) {
        let mut prng = rand::thread_rng();
        let poly = DensePolynomial::rand_coeffs(degree, &mut prng);
        let evals = (0..=degree)
            .map(|i| poly.evaluate(&GoldilocksExt2::from_canonical_u64(i as u64)))
            .collect::<Vec<_>>();
        let query = GoldilocksExt2::random(&mut prng);
        assert_eq!(
            poly.evaluate(&query),
            extrapolate_uni_poly(evals[0], &evals[1..], query)
        );
    }

    run_extrapolation_test(1);
    run_extrapolation_test(2);
    run_extrapolation_test(3);
    run_extrapolation_test(4);
}

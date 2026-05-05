#![allow(clippy::manual_memcpy)]
#![allow(clippy::needless_range_loop)]

use std::{sync::Arc, time::Duration};

use criterion::*;
use either::Either;
use ff_ext::{ExtensionField, GoldilocksExt2};
use itertools::Itertools;
use p3::field::FieldAlgebra;
use rand::{Rng, thread_rng};
use sumcheck::{
    front_loaded,
    structs::{IOPProverState, SumcheckProverMode},
};

use multilinear_extensions::{
    Expression,
    mle::MultilinearExtension,
    monomial::Term,
    op_mle,
    util::max_usable_threads,
    virtual_poly::VirtualPolynomial,
    virtual_polys::{VirtualPolynomials, VirtualPolynomialsBuilder},
};
use transcript::BasicTranscript as Transcript;

criterion_group!(
    benches,
    sumcheck_fn,
    devirgo_sumcheck_fn,
    devirgo_sumcheck_reduced_peak_memory_fn,
    front_loaded_mixed_sumcheck_fn,
    suffix_phase2_expanded_mixed_sumcheck_fn,
    mixed_sum_front_loaded_vs_suffix_fn,
    mixed_sum_three_terms_front_loaded_vs_suffix_fn,
    mixed_product_sum_front_loaded_vs_suffix_fn,
    ceno_dense_uniform_front_loaded_vs_suffix_fn,
);
criterion_main!(benches);

const NUM_SAMPLES: usize = 10;
const NUM_DEGREE: usize = 3;
const NV: [usize; 2] = [25, 26];

/// transpose 2d vector without clone
pub fn transpose<T>(v: Vec<Vec<T>>) -> Vec<Vec<T>> {
    assert!(!v.is_empty());
    let len = v[0].len();
    let mut iters: Vec<_> = v.into_iter().map(|n| n.into_iter()).collect();
    (0..len)
        .map(|_| {
            iters
                .iter_mut()
                .map(|n| n.next().unwrap())
                .collect::<Vec<T>>()
        })
        .collect()
}

fn prepare_input<'a, E: ExtensionField>(nv: usize) -> (E, Vec<MultilinearExtension<'a, E>>) {
    let mut rng = thread_rng();
    let fs = (0..NUM_DEGREE)
        .map(|_| MultilinearExtension::<E>::random(nv, &mut rng))
        .collect_vec();

    let asserted_sum = fs
        .iter()
        .fold(vec![E::ONE; 1 << nv], |mut acc, f| {
            op_mle!(f, |f| {
                (0..f.len()).zip(acc.iter_mut()).for_each(|(i, acc)| {
                    *acc *= f[i];
                });
                acc
            })
        })
        .iter()
        .cloned()
        .sum::<E>();

    (asserted_sum, fs)
}

fn sumcheck_fn(c: &mut Criterion) {
    type E = GoldilocksExt2;

    for nv in NV {
        // expand more input size once runtime is acceptable
        let mut group = c.benchmark_group(format!("sumcheck_nv_{}", nv));
        group.sample_size(NUM_SAMPLES);

        // Benchmark the proving time
        group.bench_function(
            BenchmarkId::new("prove_sumcheck", format!("sumcheck_nv_{}", nv)),
            |b| {
                b.iter_custom(|iters| {
                    let mut time = Duration::new(0, 0);
                    for _ in 0..iters {
                        let mut prover_transcript = Transcript::new(b"test");
                        let (_, fs) = { prepare_input(nv) };
                        let fs = fs.into_iter().map(Arc::new).collect_vec();

                        let virtual_poly_v1 = VirtualPolynomial::new_from_product(fs, E::ONE);
                        let instant = std::time::Instant::now();
                        #[allow(deprecated)]
                        let (_sumcheck_proof_v1, _) = IOPProverState::<E>::prove_parallel(
                            virtual_poly_v1,
                            &mut prover_transcript,
                        );
                        let elapsed = instant.elapsed();
                        time += elapsed;
                    }
                    time
                });
            },
        );

        group.finish();
    }
}

fn devirgo_sumcheck_fn(c: &mut Criterion) {
    type E = GoldilocksExt2;

    let threads = max_usable_threads();
    for nv in NV {
        // expand more input size once runtime is acceptable
        let mut group = c.benchmark_group(format!("devirgo_nv_{}", nv));
        group.sample_size(NUM_SAMPLES);

        // Benchmark the proving time
        group.bench_function(
            BenchmarkId::new("prove_sumcheck", format!("devirgo_nv_{}", nv)),
            |b| {
                b.iter_custom(|iters| {
                    let mut time = Duration::new(0, 0);
                    for _ in 0..iters {
                        let mut prover_transcript = Transcript::new(b"test");
                        let (_, fs) = { prepare_input(nv) };

                        let virtual_poly_v2 = VirtualPolynomials::new_from_monimials(
                            threads,
                            nv,
                            vec![Term {
                                scalar: Either::Right(E::ONE),
                                product: fs.iter().map(Either::Left).collect_vec(),
                            }],
                        );
                        let instant = std::time::Instant::now();
                        let (_sumcheck_proof_v2, _) =
                            IOPProverState::<E>::prove(virtual_poly_v2, &mut prover_transcript);
                        let elapsed = instant.elapsed();
                        time += elapsed;
                    }
                    time
                });
            },
        );

        // Benchmark the proving time ext
        group.bench_function(
            BenchmarkId::new("prove_sumcheck_ext", format!("devirgo_nv_{}", nv)),
            |b| {
                b.iter_custom(|iters| {
                    let mut time = Duration::new(0, 0);
                    for _ in 0..iters {
                        let mut prover_transcript = Transcript::new(b"test");
                        let (_, fs) = { prepare_input(nv) };
                        let fs = fs
                            .into_iter()
                            .map(|mle: MultilinearExtension<'_, E>| {
                                MultilinearExtension::from_evaluation_vec_smart(
                                    mle.num_vars(),
                                    mle.get_base_field_vec()
                                        .iter()
                                        .map(E::from_ref_base)
                                        .collect_vec(),
                                )
                            })
                            .collect_vec();

                        let virtual_poly_v2 = VirtualPolynomials::new_from_monimials(
                            threads,
                            nv,
                            vec![Term {
                                scalar: Either::Right(E::ONE),
                                product: fs.iter().map(Either::Left).collect_vec(),
                            }],
                        );
                        let instant = std::time::Instant::now();
                        let (_sumcheck_proof_v2, _) =
                            IOPProverState::<E>::prove(virtual_poly_v2, &mut prover_transcript);
                        let elapsed = instant.elapsed();
                        time += elapsed;
                    }
                    time
                });
            },
        );

        // Benchmark the proving time
        group.bench_function(
            BenchmarkId::new("prove_sumcheck_ext_in_place", format!("devirgo_nv_{}", nv)),
            |b| {
                b.iter_custom(|iters| {
                    let mut time = Duration::new(0, 0);
                    for _ in 0..iters {
                        let mut prover_transcript = Transcript::new(b"test");
                        let (_, fs) = { prepare_input(nv) };
                        let mut fs = fs
                            .into_iter()
                            .map(|mle: MultilinearExtension<'_, E>| {
                                MultilinearExtension::from_evaluation_vec_smart(
                                    mle.num_vars(),
                                    mle.get_base_field_vec()
                                        .iter()
                                        .map(E::from_ref_base)
                                        .collect_vec(),
                                )
                            })
                            .collect_vec();

                        let virtual_poly_v2 = VirtualPolynomials::new_from_monimials(
                            threads,
                            nv,
                            vec![Term {
                                scalar: Either::Right(E::ONE),
                                product: fs.iter_mut().map(Either::Right).collect_vec(),
                            }],
                        );
                        let instant = std::time::Instant::now();
                        let (_sumcheck_proof_v2, _) =
                            IOPProverState::<E>::prove(virtual_poly_v2, &mut prover_transcript);
                        let elapsed = instant.elapsed();
                        time += elapsed;
                    }
                    time
                });
            },
        );

        group.finish();
    }
}

fn front_loaded_mixed_poly<'a, E: ExtensionField>(
    large_nv: usize,
    small_nv: usize,
) -> VirtualPolynomial<'a, E> {
    let mut rng = thread_rng();
    let large = Arc::new(MultilinearExtension::<E>::random(large_nv, &mut rng));
    let small = Arc::new(MultilinearExtension::<E>::random(small_nv, &mut rng));

    let mut poly = VirtualPolynomial::new(large_nv);
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
    poly
}

fn front_loaded_mixed_sumcheck_fn(c: &mut Criterion) {
    type E = GoldilocksExt2;

    let mut group = c.benchmark_group("front_loaded_mixed_nv_22_2");
    group.sample_size(NUM_SAMPLES);
    group.bench_function("prove_front_loaded_mixed", |b| {
        b.iter_custom(|iters| {
            let mut time = Duration::new(0, 0);
            for _ in 0..iters {
                let mut prover_transcript = Transcript::new(b"front-loaded-bench");
                let poly = front_loaded_mixed_poly::<E>(22, 2);
                let instant = std::time::Instant::now();
                let (_proof, _state) = front_loaded::prove(poly, &mut prover_transcript);
                time += instant.elapsed();
            }
            time
        });
    });
    group.finish();
}

fn suffix_phase2_expanded_mixed_sumcheck_fn(c: &mut Criterion) {
    type E = GoldilocksExt2;

    let threads = max_usable_threads();
    let mut group = c.benchmark_group("suffix_phase2_expanded_mixed_nv_22_2");
    group.sample_size(NUM_SAMPLES);
    group.bench_function("prove_suffix_phase2_expanded_mixed", |b| {
        b.iter_custom(|iters| {
            let mut time = Duration::new(0, 0);
            for _ in 0..iters {
                let mut prover_transcript = Transcript::new(b"suffix-phase2-bench");
                let mut rng = thread_rng();
                let large = MultilinearExtension::<E>::random(22, &mut rng);
                let small = MultilinearExtension::<E>::random(2, &mut rng);
                let small_values = (0..(1usize << 22))
                    .map(
                        |idx| match small.evaluations().index(idx & ((1usize << 2) - 1)) {
                            Either::Left(base) => E::from(base),
                            Either::Right(ext) => ext,
                        },
                    )
                    .collect_vec();
                let expanded_small =
                    MultilinearExtension::from_evaluations_ext_vec(22, small_values);
                let poly = VirtualPolynomials::new_from_monimials(
                    threads,
                    22,
                    vec![Term {
                        scalar: Either::Right(E::ONE),
                        product: vec![Either::Left(&large), Either::Left(&expanded_small)],
                    }],
                );
                let instant = std::time::Instant::now();
                let (_proof, _state) =
                    IOPProverState::<E>::prove_suffix(poly, &mut prover_transcript);
                time += instant.elapsed();
            }
            time
        });
    });
    group.finish();
}

fn mixed_sum_poly<'a, E: ExtensionField>(
    threads: usize,
    max_num_variables: usize,
    mles: &'a [MultilinearExtension<E>],
) -> VirtualPolynomials<'a, E> {
    VirtualPolynomials::new_from_monimials(
        threads,
        max_num_variables,
        mles.iter()
            .map(|mle| Term {
                scalar: Either::Right(E::ONE),
                product: vec![Either::Left(mle)],
            })
            .collect_vec(),
    )
}

fn mixed_sum_front_loaded_vs_suffix_case(
    c: &mut Criterion,
    group_name: &'static str,
    front_loaded_name: &'static str,
    suffix_name: &'static str,
    num_variables: &'static [usize],
) {
    type E = GoldilocksExt2;

    let threads = max_usable_threads();
    let max_num_variables = *num_variables.iter().max().unwrap();
    let mut group = c.benchmark_group(group_name);
    group.sample_size(NUM_SAMPLES);

    group.bench_function(front_loaded_name, |b| {
        b.iter_custom(|iters| {
            let mut time = Duration::new(0, 0);
            for _ in 0..iters {
                let mut prover_transcript = Transcript::new(b"front-loaded-sum-bench");
                let mut rng = thread_rng();
                let mles = num_variables
                    .iter()
                    .map(|&num_variables| {
                        MultilinearExtension::<E>::random(num_variables, &mut rng)
                    })
                    .collect_vec();
                let poly = mixed_sum_poly(threads, max_num_variables, &mles);
                let instant = std::time::Instant::now();
                let (_proof, _state) = front_loaded::prove_2phase(poly, &mut prover_transcript);
                time += instant.elapsed();
            }
            time
        });
    });

    group.bench_function(suffix_name, |b| {
        b.iter_custom(|iters| {
            let mut time = Duration::new(0, 0);
            for _ in 0..iters {
                let mut prover_transcript = Transcript::new(b"suffix-sum-bench");
                let mut rng = thread_rng();
                let mles = num_variables
                    .iter()
                    .map(|&num_variables| {
                        MultilinearExtension::<E>::random(num_variables, &mut rng)
                    })
                    .collect_vec();
                let poly = mixed_sum_poly(threads, max_num_variables, &mles);
                let instant = std::time::Instant::now();
                let (_proof, _state) =
                    IOPProverState::<E>::prove_suffix(poly, &mut prover_transcript);
                time += instant.elapsed();
            }
            time
        });
    });

    group.finish();
}

fn mixed_sum_front_loaded_vs_suffix_fn(c: &mut Criterion) {
    mixed_sum_front_loaded_vs_suffix_case(
        c,
        "mixed_sum_nv_22_2",
        "front_loaded_compact_a_plus_b",
        "suffix_phase2_a_plus_b",
        &[22, 2],
    );
}

fn mixed_sum_three_terms_front_loaded_vs_suffix_fn(c: &mut Criterion) {
    mixed_sum_front_loaded_vs_suffix_case(
        c,
        "mixed_sum_nv_22_16_2",
        "front_loaded_compact_a_plus_b_plus_c",
        "suffix_phase2_a_plus_b_plus_c",
        &[22, 16, 2],
    );
}

fn mixed_product_sum_poly<'a, E: ExtensionField>(
    threads: usize,
    max_num_variables: usize,
    num_variables: &[usize],
    product_degree: usize,
    mles: &'a [MultilinearExtension<E>],
) -> VirtualPolynomials<'a, E> {
    VirtualPolynomials::new_from_monimials(
        threads,
        max_num_variables,
        num_variables
            .iter()
            .enumerate()
            .map(|(term_idx, _)| {
                let product_start = term_idx * product_degree;
                Term {
                    scalar: Either::Right(E::ONE),
                    product: mles[product_start..product_start + product_degree]
                        .iter()
                        .map(Either::Left)
                        .collect_vec(),
                }
            })
            .collect_vec(),
    )
}

fn mixed_product_sum_mles<E: ExtensionField, R: Rng>(
    num_variables: &[usize],
    product_degree: usize,
    rng: &mut R,
) -> Vec<MultilinearExtension<'static, E>> {
    let mut mles = Vec::with_capacity(num_variables.len() * product_degree);
    for &num_variables in num_variables {
        for _ in 0..product_degree {
            mles.push(MultilinearExtension::<E>::random(num_variables, rng));
        }
    }
    mles
}

fn mixed_product_sum_front_loaded_vs_suffix_case(
    c: &mut Criterion,
    group_name: &'static str,
    front_loaded_name: &'static str,
    suffix_name: &'static str,
    num_variables: &'static [usize],
    product_degree: usize,
) {
    type E = GoldilocksExt2;

    let threads = max_usable_threads();
    let max_num_variables = *num_variables.iter().max().unwrap();
    let mut group = c.benchmark_group(group_name);
    group.sample_size(NUM_SAMPLES);

    group.bench_function(front_loaded_name, |b| {
        b.iter_custom(|iters| {
            let mut time = Duration::new(0, 0);
            for _ in 0..iters {
                let mut prover_transcript = Transcript::<E>::new(b"front-loaded-product-sum-bench");
                let mut rng = thread_rng();
                let mles = mixed_product_sum_mles::<E, _>(num_variables, product_degree, &mut rng);
                let poly = mixed_product_sum_poly(
                    threads,
                    max_num_variables,
                    num_variables,
                    product_degree,
                    &mles,
                );
                let instant = std::time::Instant::now();
                let (_proof, _state) = front_loaded::prove_2phase(poly, &mut prover_transcript);
                time += instant.elapsed();
            }
            time
        });
    });

    group.bench_function(suffix_name, |b| {
        b.iter_custom(|iters| {
            let mut time = Duration::new(0, 0);
            for _ in 0..iters {
                let mut prover_transcript = Transcript::<E>::new(b"suffix-product-sum-bench");
                let mut rng = thread_rng();
                let mles = mixed_product_sum_mles::<E, _>(num_variables, product_degree, &mut rng);
                let poly = mixed_product_sum_poly(
                    threads,
                    max_num_variables,
                    num_variables,
                    product_degree,
                    &mles,
                );
                let instant = std::time::Instant::now();
                let (_proof, _state) =
                    IOPProverState::<E>::prove_suffix(poly, &mut prover_transcript);
                time += instant.elapsed();
            }
            time
        });
    });

    group.finish();
}

fn mixed_product_sum_front_loaded_vs_suffix_fn(c: &mut Criterion) {
    mixed_product_sum_front_loaded_vs_suffix_case(
        c,
        "mixed_product_sum_nv_22_16_2",
        "front_loaded_compact_product_sum",
        "suffix_phase2_product_sum",
        &[22, 16, 2],
        2,
    );
    mixed_product_sum_front_loaded_vs_suffix_case(
        c,
        "mixed_product3_sum_nv_22_16_2",
        "front_loaded_compact_product3_sum",
        "suffix_phase2_product3_sum",
        &[22, 16, 2],
        3,
    );
    mixed_product_sum_front_loaded_vs_suffix_case(
        c,
        "mixed_product4_sum_nv_22_16_2",
        "front_loaded_compact_product4_sum",
        "suffix_phase2_product4_sum",
        &[22, 16, 2],
        4,
    );
}

#[derive(Clone, Copy)]
struct DenseUniformShape {
    name: &'static str,
    num_variables: usize,
    num_mles: usize,
    degree_hist: &'static [(usize, usize)],
}

const CENO_DENSE_UNIFORM_SHAPES: &[DenseUniformShape] = &[
    DenseUniformShape {
        name: "keccak_tower_nv_11_mles_4705_terms_3475",
        num_variables: 11,
        num_mles: 4705,
        degree_hist: &[(3, 3475)],
    },
    DenseUniformShape {
        name: "keccak_main_nv_12_mles_1700_terms_2785",
        num_variables: 12,
        num_mles: 1700,
        degree_hist: &[(1, 3), (2, 2754), (3, 28)],
    },
    DenseUniformShape {
        name: "weierstrass_main_nv_10_mles_2193_terms_5297",
        num_variables: 10,
        num_mles: 2193,
        degree_hist: &[(1, 1), (2, 2192), (3, 3104)],
    },
    DenseUniformShape {
        name: "shard_ram_main_nv_13_mles_382_terms_992",
        num_variables: 13,
        num_mles: 382,
        degree_hist: &[(1, 3), (2, 404), (3, 303), (4, 282)],
    },
];

fn dense_uniform_mles<E: ExtensionField, R: Rng>(
    shape: DenseUniformShape,
    rng: &mut R,
) -> Vec<MultilinearExtension<'static, E>> {
    (0..shape.num_mles)
        .map(|_| MultilinearExtension::<E>::random(shape.num_variables, rng))
        .collect_vec()
}

fn dense_uniform_poly<'a, E: ExtensionField>(
    threads: usize,
    shape: DenseUniformShape,
    mles: &'a [MultilinearExtension<E>],
) -> VirtualPolynomials<'a, E> {
    let mut builder = VirtualPolynomialsBuilder::new_with_mles(
        threads,
        shape.num_variables,
        mles.iter().map(Either::Left).collect_vec(),
    );
    let mle_exprs = mles
        .iter()
        .map(|mle| builder.lift(Either::Left(mle)))
        .collect_vec();
    let mut terms = Vec::with_capacity(shape.degree_hist.iter().map(|(_, count)| count).sum());
    let mut term_offset = 0usize;

    for &(degree, count) in shape.degree_hist {
        for term_idx in 0..count {
            let product = (0..degree)
                .map(|factor_idx| {
                    let mle_idx = (term_offset + term_idx * degree + factor_idx) % shape.num_mles;
                    mle_exprs[mle_idx].clone()
                })
                .collect_vec();
            terms.push(Term {
                scalar: Expression::ONE,
                product,
            });
        }
        term_offset += count * degree;
    }

    builder.to_virtual_polys_with_monomial_terms(&terms, &[], &[])
}

fn ceno_dense_uniform_front_loaded_vs_suffix_fn(c: &mut Criterion) {
    type E = GoldilocksExt2;

    let threads = max_usable_threads();
    for shape in CENO_DENSE_UNIFORM_SHAPES {
        let mut group = c.benchmark_group(format!("ceno_dense_uniform_{}", shape.name));
        group.sample_size(NUM_SAMPLES);

        group.bench_function("front_loaded", |b| {
            b.iter_custom(|iters| {
                let mut time = Duration::new(0, 0);
                for _ in 0..iters {
                    let mut prover_transcript = Transcript::<E>::new(b"front-loaded-dense-bench");
                    let mut rng = thread_rng();
                    let mles = dense_uniform_mles::<E, _>(*shape, &mut rng);
                    let poly = dense_uniform_poly(threads, *shape, &mles);
                    let instant = std::time::Instant::now();
                    let (_proof, _state) = front_loaded::prove_2phase(poly, &mut prover_transcript);
                    time += instant.elapsed();
                }
                time
            });
        });

        group.bench_function("suffix", |b| {
            b.iter_custom(|iters| {
                let mut time = Duration::new(0, 0);
                for _ in 0..iters {
                    let mut prover_transcript = Transcript::<E>::new(b"suffix-dense-bench");
                    let mut rng = thread_rng();
                    let mles = dense_uniform_mles::<E, _>(*shape, &mut rng);
                    let poly = dense_uniform_poly(threads, *shape, &mles);
                    let instant = std::time::Instant::now();
                    let (_proof, _state) =
                        IOPProverState::<E>::prove_suffix(poly, &mut prover_transcript);
                    time += instant.elapsed();
                }
                time
            });
        });

        group.finish();
    }
}

fn devirgo_sumcheck_reduced_peak_memory_fn(c: &mut Criterion) {
    type E = GoldilocksExt2;

    let threads = max_usable_threads();
    for nv in NV {
        let mut group = c.benchmark_group(format!("devirgo_reduced_peak_memory_nv_{}", nv));
        group.sample_size(NUM_SAMPLES);

        group.bench_function(
            BenchmarkId::new(
                "prove_sumcheck",
                format!("devirgo_reduced_peak_memory_nv_{}", nv),
            ),
            |b| {
                b.iter_custom(|iters| {
                    let mut time = Duration::new(0, 0);
                    for _ in 0..iters {
                        let mut prover_transcript = Transcript::new(b"test");
                        let (_, fs) = { prepare_input(nv) };

                        let virtual_poly = VirtualPolynomials::new_from_monimials(
                            threads,
                            nv,
                            vec![Term {
                                scalar: Either::Right(E::ONE),
                                product: fs.iter().map(Either::Left).collect_vec(),
                            }],
                        );
                        let instant = std::time::Instant::now();
                        let _ = IOPProverState::<E>::prove_with_mode(
                            virtual_poly,
                            &mut prover_transcript,
                            SumcheckProverMode::ReducedPeakMemory,
                        );
                        time += instant.elapsed();
                    }
                    time
                });
            },
        );

        group.bench_function(
            BenchmarkId::new(
                "prove_sumcheck_ext",
                format!("devirgo_reduced_peak_memory_nv_{}", nv),
            ),
            |b| {
                b.iter_custom(|iters| {
                    let mut time = Duration::new(0, 0);
                    for _ in 0..iters {
                        let mut prover_transcript = Transcript::new(b"test");
                        let (_, fs) = { prepare_input(nv) };
                        let fs = fs
                            .into_iter()
                            .map(|mle: MultilinearExtension<'_, E>| {
                                MultilinearExtension::from_evaluation_vec_smart(
                                    mle.num_vars(),
                                    mle.get_base_field_vec()
                                        .iter()
                                        .map(E::from_ref_base)
                                        .collect_vec(),
                                )
                            })
                            .collect_vec();

                        let virtual_poly = VirtualPolynomials::new_from_monimials(
                            threads,
                            nv,
                            vec![Term {
                                scalar: Either::Right(E::ONE),
                                product: fs.iter().map(Either::Left).collect_vec(),
                            }],
                        );
                        let instant = std::time::Instant::now();
                        let _ = IOPProverState::<E>::prove_with_mode(
                            virtual_poly,
                            &mut prover_transcript,
                            SumcheckProverMode::ReducedPeakMemory,
                        );
                        time += instant.elapsed();
                    }
                    time
                });
            },
        );

        group.bench_function(
            BenchmarkId::new(
                "prove_sumcheck_ext_in_place",
                format!("devirgo_reduced_peak_memory_nv_{}", nv),
            ),
            |b| {
                b.iter_custom(|iters| {
                    let mut time = Duration::new(0, 0);
                    for _ in 0..iters {
                        let mut prover_transcript = Transcript::new(b"test");
                        let (_, fs) = { prepare_input(nv) };
                        let mut fs = fs
                            .into_iter()
                            .map(|mle: MultilinearExtension<'_, E>| {
                                MultilinearExtension::from_evaluation_vec_smart(
                                    mle.num_vars(),
                                    mle.get_base_field_vec()
                                        .iter()
                                        .map(E::from_ref_base)
                                        .collect_vec(),
                                )
                            })
                            .collect_vec();

                        let virtual_poly = VirtualPolynomials::new_from_monimials(
                            threads,
                            nv,
                            vec![Term {
                                scalar: Either::Right(E::ONE),
                                product: fs.iter_mut().map(Either::Right).collect_vec(),
                            }],
                        );
                        let instant = std::time::Instant::now();
                        let _ = IOPProverState::<E>::prove_with_mode(
                            virtual_poly,
                            &mut prover_transcript,
                            SumcheckProverMode::ReducedPeakMemory,
                        );
                        time += instant.elapsed();
                    }
                    time
                });
            },
        );

        group.finish();
    }
}

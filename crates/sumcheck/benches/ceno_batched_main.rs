use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ff_ext::BabyBearExt4;
use itertools::Itertools;
use multilinear_extensions::{
    Expression,
    mle::MultilinearExtension,
    util::max_usable_threads,
    virtual_polys::{VirtualPolynomials, VirtualPolynomialsBuilder},
};
use p3::field::FieldAlgebra;
use rand::{SeedableRng, rngs::StdRng};
use sumcheck::structs::{IOPProverState, SumcheckProverMode};
use transcript::BasicTranscript as Transcript;

type E = BabyBearExt4;

#[derive(Clone, Copy)]
struct ChipShape {
    num_vars: usize,
    num_mles: usize,
    terms: usize,
}

struct TermShape {
    scalar: E,
    product: Vec<usize>,
}

struct CenoBatchedMainCase<'a> {
    max_num_vars: usize,
    mles: Vec<MultilinearExtension<'a, E>>,
    terms: Vec<TermShape>,
}

impl CenoBatchedMainCase<'_> {
    fn virtual_polys(&self, mode: SumcheckProverMode) -> VirtualPolynomials<'_, E> {
        let threads = match mode {
            SumcheckProverMode::Frontload => max_usable_threads(),
            SumcheckProverMode::LegacyStable | SumcheckProverMode::ReducedPeakMemory => {
                max_usable_threads()
            }
        };
        let mut builder = VirtualPolynomialsBuilder::new(threads, self.max_num_vars);
        let lifted = self
            .mles
            .iter()
            .map(|mle| builder.lift(either::Either::Left(mle)))
            .collect_vec();

        let mut global_expr = Expression::ZERO;
        for term in &self.terms {
            let product_expr = term
                .product
                .iter()
                .map(|&mle_idx| lifted[mle_idx].clone())
                .fold(Expression::ONE, |acc, expr| acc * expr);
            global_expr +=
                product_expr * Expression::Constant(itertools::Either::Right(term.scalar));
        }

        builder.to_virtual_polys(&[global_expr], &[])
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn chip_shapes(scale: &str) -> Vec<ChipShape> {
    match scale {
        "tiny" => vec![
            ChipShape {
                num_vars: 16,
                num_mles: 12,
                terms: 24,
            },
            ChipShape {
                num_vars: 14,
                num_mles: 12,
                terms: 24,
            },
            ChipShape {
                num_vars: 12,
                num_mles: 8,
                terms: 16,
            },
        ],
        "large" => vec![
            ChipShape {
                num_vars: 22,
                num_mles: 16,
                terms: 48,
            },
            ChipShape {
                num_vars: 21,
                num_mles: 16,
                terms: 48,
            },
            ChipShape {
                num_vars: 20,
                num_mles: 16,
                terms: 48,
            },
            ChipShape {
                num_vars: 18,
                num_mles: 12,
                terms: 36,
            },
            ChipShape {
                num_vars: 16,
                num_mles: 12,
                terms: 36,
            },
        ],
        "ceno-ish" => vec![
            ChipShape {
                num_vars: 24,
                num_mles: 24,
                terms: 72,
            },
            ChipShape {
                num_vars: 23,
                num_mles: 24,
                terms: 72,
            },
            ChipShape {
                num_vars: 22,
                num_mles: 24,
                terms: 72,
            },
            ChipShape {
                num_vars: 20,
                num_mles: 20,
                terms: 60,
            },
            ChipShape {
                num_vars: 18,
                num_mles: 16,
                terms: 48,
            },
            ChipShape {
                num_vars: 16,
                num_mles: 16,
                terms: 48,
            },
        ],
        _ => vec![
            ChipShape {
                num_vars: 20,
                num_mles: 14,
                terms: 40,
            },
            ChipShape {
                num_vars: 19,
                num_mles: 14,
                terms: 40,
            },
            ChipShape {
                num_vars: 18,
                num_mles: 14,
                terms: 40,
            },
            ChipShape {
                num_vars: 16,
                num_mles: 10,
                terms: 30,
            },
        ],
    }
}

fn build_case<'a>(scale: &str, max_vars_override: Option<usize>) -> CenoBatchedMainCase<'a> {
    let mut shapes = chip_shapes(scale);
    if let Some(max_num_vars) = max_vars_override {
        let current_max = shapes
            .iter()
            .map(|shape| shape.num_vars)
            .max()
            .expect("shape list is non-empty");
        let delta = max_num_vars as isize - current_max as isize;
        for shape in &mut shapes {
            shape.num_vars = (shape.num_vars as isize + delta).max(1) as usize;
        }
    }

    let max_num_vars = shapes
        .iter()
        .map(|shape| shape.num_vars)
        .max()
        .expect("shape list is non-empty");
    let mut rng = StdRng::seed_from_u64(0xCE_00_BA_7C_ED);
    let mut mles = Vec::new();
    let mut terms = Vec::new();

    for (chip_idx, shape) in shapes.into_iter().enumerate() {
        let mle_start = mles.len();
        mles.extend(
            (0..shape.num_mles)
                .map(|_| MultilinearExtension::<E>::random(shape.num_vars, &mut rng)),
        );

        for term_idx in 0..shape.terms {
            let degree = match term_idx % 8 {
                0 => 1,
                1 | 2 => 2,
                3..=5 => 3,
                _ => 4,
            };
            let product = (0..degree)
                .map(|offset| mle_start + ((term_idx * 3 + offset * 5) % shape.num_mles))
                .collect_vec();
            let scalar = E::from_canonical_u64(((chip_idx + 1) * 17 + term_idx + 1) as u64);
            terms.push(TermShape { scalar, product });
        }
    }

    CenoBatchedMainCase {
        max_num_vars,
        mles,
        terms,
    }
}

fn mode_name(mode: SumcheckProverMode) -> &'static str {
    match mode {
        SumcheckProverMode::Frontload => "frontload",
        SumcheckProverMode::LegacyStable => "suffix_legacy",
        SumcheckProverMode::ReducedPeakMemory => "suffix_reduced_peak",
    }
}

fn ceno_batched_main(c: &mut Criterion) {
    let scale = std::env::var("GKR_CENO_BENCH_SCALE").unwrap_or_else(|_| "default".to_string());
    let sample_size = env_usize("GKR_CENO_BENCH_SAMPLES", 10);
    let max_vars_override = std::env::var("GKR_CENO_BENCH_MAX_VARS")
        .ok()
        .and_then(|value| value.parse().ok());
    let include_suffix = std::env::var_os("GKR_CENO_BENCH_SUFFIX").is_some();

    let case = build_case(&scale, max_vars_override);
    let mut group = c.benchmark_group(format!(
        "ceno_batched_main/{}/max_vars_{}/mles_{}/terms_{}",
        scale,
        case.max_num_vars,
        case.mles.len(),
        case.terms.len()
    ));
    group.sample_size(sample_size);

    let mut modes = vec![SumcheckProverMode::Frontload];
    if include_suffix {
        modes.push(SumcheckProverMode::LegacyStable);
    }

    for mode in modes {
        group.bench_function(BenchmarkId::from_parameter(mode_name(mode)), |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let mut transcript = Transcript::new(b"ceno-batched-main");
                    let virtual_polys = case.virtual_polys(mode);
                    let start = std::time::Instant::now();
                    let (proof, state) =
                        IOPProverState::<E>::prove_with_mode(virtual_polys, &mut transcript, mode);
                    criterion::black_box((
                        proof.proofs.len(),
                        state.collect_raw_challenges().len(),
                    ));
                    total += start.elapsed();
                }
                total
            });
        });
    }

    group.finish();
}

criterion_group!(benches, ceno_batched_main);
criterion_main!(benches);

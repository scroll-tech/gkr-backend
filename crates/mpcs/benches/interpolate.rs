use std::time::Duration;

use criterion::*;

use ff_ext::{FromUniformBytes, GoldilocksExt2};
use itertools::Itertools;
use mpcs::util::arithmetic::interpolate_field_type_over_boolean_hypercube;

use multilinear_extensions::mle::{FieldType, MultilinearExtension};
use p3::maybe_rayon::prelude::*;
use rand::{SeedableRng, rngs::OsRng};
use rand_chacha::ChaCha8Rng;

type E = GoldilocksExt2;

const NUM_SAMPLES: usize = 10;
const NUM_VARS_START: usize = 15;
const NUM_VARS_END: usize = 20;
const BATCH_SIZE_LOG_START: usize = 3;
const BATCH_SIZE_LOG_END: usize = 5;

fn bench_encoding(c: &mut Criterion, is_base: bool) {
    let mut group = c.benchmark_group(format!(
        "interpolate_{}",
        if is_base { "base" } else { "extension" }
    ));
    group.sample_size(NUM_SAMPLES);
    // Challenge is over extension field, poly over the base field
    for num_vars in NUM_VARS_START..=NUM_VARS_END {
        for batch_size_log in BATCH_SIZE_LOG_START..=BATCH_SIZE_LOG_END {
            let batch_size = 1 << batch_size_log;
            let rng = ChaCha8Rng::from_seed([0u8; 32]);
            let polys = (0..batch_size)
                .map(|_| {
                    if is_base {
                        MultilinearExtension::random(num_vars, &mut rng.clone())
                    } else {
                        MultilinearExtension::from_evaluations_ext_vec(
                            num_vars,
                            (0..1 << num_vars).map(|_| E::random(&mut OsRng)).collect(),
                        )
                    }
                })
                .collect_vec();

            group.bench_function(
                BenchmarkId::new("batch_encode", format!("{}-{}", num_vars, batch_size)),
                |b| {
                    b.iter(|| {
                        polys
                            .par_iter()
                            .map(|poly| {
                                // Switch to coefficient form
                                let mut coeffs = poly.evaluations.clone();
                                interpolate_field_type_over_boolean_hypercube(&mut coeffs);
                                coeffs
                            })
                            .collect::<Vec<FieldType<E>>>();
                    })
                },
            );
        }
    }
}

fn bench_encoding_goldilocks_2(c: &mut Criterion) {
    bench_encoding(c, false);
}

fn bench_encoding_base(c: &mut Criterion) {
    bench_encoding(c, true);
}

criterion_group! {
  name = bench_basefold;
  config = Criterion::default().warm_up_time(Duration::from_millis(3000));
  targets = bench_encoding_base, bench_encoding_goldilocks_2,
}

criterion_main!(bench_basefold);

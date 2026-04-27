use criterion::*;
use ff_ext::{BabyBearExt4, ExtensionField, FieldFrom, FromUniformBytes};
use mpcs::{JaggedSumcheckInput, assist_sumcheck_prove, jagged_sumcheck_prove};
use multilinear_extensions::{util::ceil_log2, virtual_poly::build_eq_x_r_vec};
use rand::thread_rng;
use transcript::BasicTranscript;

type E = BabyBearExt4;
type F = <E as ExtensionField>::BaseField;

const NUM_SAMPLES: usize = 10;

fn bench_jagged_sumcheck(c: &mut Criterion) {
    let mut group = c.benchmark_group("jagged_sumcheck");
    group.sample_size(NUM_SAMPLES);

    // (num_giga_vars, num_polys, poly_height_log2)
    let configs: Vec<(usize, usize, usize)> = (25..=31)
        .map(|n| {
            let s = 21usize;
            let num_polys = 1usize << (n - s);
            (n, num_polys, s)
        })
        .collect();

    for (num_giga_vars, num_polys, s) in configs {
        let poly_height = 1usize << s;
        let total_evals = num_polys * poly_height;

        let mut rng = thread_rng();

        let q_evals: Vec<F> = (0..total_evals)
            .map(|i| F::from_v((i as u64 * 13 + 7) % (1 << 30)))
            .collect();

        let cumulative_heights: Vec<usize> = (0..=num_polys).map(|i| i * poly_height).collect();

        let z_row: Vec<E> = (0..s).map(|_| E::random(&mut rng)).collect();
        let z_col_vars = (num_polys as f64).log2().ceil() as usize;
        let z_col: Vec<E> = (0..z_col_vars).map(|_| E::random(&mut rng)).collect();

        let input = JaggedSumcheckInput {
            q_evals: &q_evals,
            num_giga_vars,
            cumulative_heights: &cumulative_heights,
            eq_row: build_eq_x_r_vec(&z_row),
            eq_col: build_eq_x_r_vec(&z_col),
        };

        group.bench_function(
            BenchmarkId::new("prove", format!("n={}", num_giga_vars)),
            |b| {
                b.iter(|| {
                    let mut transcript = BasicTranscript::<E>::new(b"jagged_bench");
                    jagged_sumcheck_prove(black_box(&input), &mut transcript, None)
                })
            },
        );
    }

    group.finish();
}

fn bench_assist_sumcheck(c: &mut Criterion) {
    let mut group = c.benchmark_group("assist_sumcheck");
    group.sample_size(NUM_SAMPLES);

    let poly_height_log2 = 20usize;
    let poly_height = 1usize << poly_height_log2;

    for num_polys in [1000, 2000, 4000, 8000] {
        let mut rng = thread_rng();

        let total_evals = num_polys * poly_height;
        let num_giga_vars = ceil_log2(total_evals);
        let n_robp = num_giga_vars + if total_evals.is_power_of_two() { 1 } else { 0 };

        let cumulative_heights: Vec<usize> = (0..=num_polys).map(|i| i * poly_height).collect();

        let z_row: Vec<E> = (0..n_robp).map(|_| E::random(&mut rng)).collect();
        let rho: Vec<E> = (0..n_robp).map(|_| E::random(&mut rng)).collect();
        let num_col_vars = ceil_log2(num_polys).max(1);
        let z_col: Vec<E> = (0..num_col_vars).map(|_| E::random(&mut rng)).collect();
        let eq_col = build_eq_x_r_vec(&z_col);

        group.bench_function(BenchmarkId::new("prove", format!("K={}", num_polys)), |b| {
            b.iter(|| {
                let mut transcript = BasicTranscript::<E>::new(b"assist_bench");
                assist_sumcheck_prove(
                    black_box(&z_row),
                    black_box(&rho),
                    black_box(&eq_col),
                    black_box(&cumulative_heights),
                    black_box(n_robp),
                    &mut transcript,
                )
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_jagged_sumcheck, bench_assist_sumcheck);
criterion_main!(benches);

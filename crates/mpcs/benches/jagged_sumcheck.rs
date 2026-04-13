use criterion::*;
use ff_ext::{ExtensionField, FieldFrom, FromUniformBytes, GoldilocksExt2};
use mpcs::{JaggedSumcheckInput, jagged_sumcheck_prove};
use multilinear_extensions::virtual_poly::build_eq_x_r_vec;
use rand::thread_rng;
use transcript::BasicTranscript;

type E = GoldilocksExt2;
type F = <E as ExtensionField>::BaseField;

const NUM_SAMPLES: usize = 10;

fn bench_jagged_sumcheck(c: &mut Criterion) {
    let mut group = c.benchmark_group("jagged_sumcheck");
    group.sample_size(NUM_SAMPLES);

    // (num_giga_vars, num_polys, poly_height_log2)
    let configs: Vec<(usize, usize, usize)> = vec![
        (20, 1 << 5, 15),  // n=20: 32 polys * 2^15 = 2^20
        (25, 1 << 10, 15), // n=25: 1024 polys * 2^15 = 2^25
    ];

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
                    jagged_sumcheck_prove(black_box(&input), &mut transcript)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_jagged_sumcheck);
criterion_main!(benches);

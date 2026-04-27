use std::time::Duration;

use criterion::*;
use ff_ext::{FromUniformBytes, GoldilocksExt2};
use mpcs::{
    Basefold, BasefoldRSParams, PolynomialCommitmentScheme, SecurityLevel, jagged_batch_open,
    jagged_commit,
};
use multilinear_extensions::{mle::MultilinearExtension, util::ceil_log2};
use p3::{field::FieldAlgebra, goldilocks::Goldilocks, matrix::Matrix};
use rand::{Rng, thread_rng};
use transcript::BasicTranscript;
use witness::{InstancePaddingStrategy, RowMajorMatrix};

type E = GoldilocksExt2;
type F = Goldilocks;
type Pcs = Basefold<E, BasefoldRSParams>;

const NUM_SAMPLES: usize = 10;
const NUM_MATRICES: usize = 30;
const NUM_COLS: usize = 32;

fn make_rmm(num_rows: usize, num_cols: usize) -> RowMajorMatrix<F> {
    let values: Vec<F> = (0..num_rows * num_cols)
        .map(|i| F::from_canonical_u32(((i as u64 * 13 + 7) % (1 << 30)) as u32))
        .collect();
    RowMajorMatrix::<F>::new_by_values(values, num_cols, InstancePaddingStrategy::Default)
}

fn sample_heights(rng: &mut impl Rng, num_matrices: usize) -> Vec<usize> {
    let log_range: Vec<usize> = (16..=22).collect();
    (0..num_matrices)
        .map(|_| 1usize << log_range[rng.gen_range(0..log_range.len())])
        .collect()
}

fn eval_column_poly_at_point(col_evals: &[F], point: &[E]) -> E {
    let s = point.len();
    assert_eq!(col_evals.len(), 1 << s);
    let mle = MultilinearExtension::from_evaluations_vec(s, col_evals.to_vec());
    mle.evaluate(point)
}

fn bench_jagged_pcs(c: &mut Criterion) {
    let mut group = c.benchmark_group("jagged_pcs");
    group.sample_size(NUM_SAMPLES);

    let mut rng = thread_rng();

    let heights = sample_heights(&mut rng, NUM_MATRICES);
    println!(
        "Matrix heights (log2): {:?}",
        heights.iter().map(|h| ceil_log2(*h)).collect::<Vec<_>>()
    );

    let rmms: Vec<_> = heights.iter().map(|&h| make_rmm(h, NUM_COLS)).collect();

    let log_heights: Vec<usize> = rmms.iter().map(|rmm| ceil_log2(rmm.height())).collect();
    let max_s = *log_heights.iter().max().unwrap();
    let total_evals: usize = rmms.iter().map(|rmm| rmm.height() * rmm.width()).sum();
    let num_giga_vars = ceil_log2(total_evals);
    let num_polys = NUM_MATRICES * NUM_COLS;

    println!(
        "num_matrices={}, num_cols={}, num_polys={}, total_evals={}, num_giga_vars={}, max_s={}",
        NUM_MATRICES, NUM_COLS, num_polys, total_evals, num_giga_vars, max_s
    );

    let poly_size = 1usize << num_giga_vars;
    let param = Pcs::setup(poly_size, SecurityLevel::Conjecture100bits).unwrap();
    let (pp, vp) = Pcs::trim(param, poly_size).unwrap();

    // --- Bench commit ---
    let comm = jagged_commit::<E, Pcs>(&pp, rmms.clone()).expect("commit failed");

    group.bench_function(
        BenchmarkId::new("commit", format!("{}x{}", NUM_MATRICES, NUM_COLS)),
        |b| {
            b.iter_custom(|iters| {
                let mut time = Duration::new(0, 0);
                for _ in 0..iters {
                    let rmms_clone = rmms.clone();
                    let instant = std::time::Instant::now();
                    let _ = jagged_commit::<E, Pcs>(&pp, rmms_clone).unwrap();
                    time += instant.elapsed();
                }
                time
            })
        },
    );

    // --- Prepare evaluation data ---
    // Extract column polynomials (before bit-reversal) for computing true evaluations.
    let col_polys: Vec<Vec<F>> = rmms
        .iter()
        .flat_map(|rmm| {
            let h = rmm.height();
            let w = rmm.width();
            (0..w).map(move |c| (0..h).map(|r| rmm.values[r * w + c]).collect())
        })
        .collect();

    let point: Vec<E> = (0..max_s).map(|_| E::random(&mut rng)).collect();

    let evals: Vec<E> = col_polys
        .iter()
        .zip(
            log_heights
                .iter()
                .flat_map(|&s| std::iter::repeat_n(s, NUM_COLS)),
        )
        .map(|(col, s_i)| eval_column_poly_at_point(col, &point[(max_s - s_i)..]))
        .collect();

    // --- Bench batch open ---
    group.bench_function(
        BenchmarkId::new("batch_open", format!("{}x{}", NUM_MATRICES, NUM_COLS)),
        |b| {
            b.iter_batched(
                || {
                    let mut transcript = BasicTranscript::<E>::new(b"jagged_bench");
                    Pcs::write_commitment(&comm.to_commitment().inner, &mut transcript).unwrap();
                    transcript
                },
                |mut transcript| {
                    jagged_batch_open::<E, Pcs>(&pp, &comm, &point, &evals, &mut transcript)
                        .unwrap();
                },
                BatchSize::SmallInput,
            );
        },
    );

    // --- Bench batch verify ---
    let mut transcript_p = BasicTranscript::<E>::new(b"jagged_bench");
    Pcs::write_commitment(&comm.to_commitment().inner, &mut transcript_p).unwrap();
    let proof = jagged_batch_open::<E, Pcs>(&pp, &comm, &point, &evals, &mut transcript_p).unwrap();
    let pure_comm = comm.to_commitment();

    group.bench_function(
        BenchmarkId::new("batch_verify", format!("{}x{}", NUM_MATRICES, NUM_COLS)),
        |b| {
            b.iter_batched(
                || {
                    let mut transcript = BasicTranscript::<E>::new(b"jagged_bench");
                    Pcs::write_commitment(&pure_comm.inner, &mut transcript).unwrap();
                    transcript
                },
                |mut transcript| {
                    mpcs::jagged_batch_verify::<E, Pcs>(
                        &vp,
                        &pure_comm,
                        &point,
                        &evals,
                        &proof,
                        &mut transcript,
                    )
                    .unwrap();
                },
                BatchSize::SmallInput,
            );
        },
    );

    group.finish();
}

criterion_group!(benches, bench_jagged_pcs);
criterion_main!(benches);

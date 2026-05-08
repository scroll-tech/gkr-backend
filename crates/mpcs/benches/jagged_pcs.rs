use std::time::Duration;

use criterion::*;
use ff_ext::{BabyBearExt4, FromUniformBytes};
use mpcs::{
    Basefold, BasefoldRSParams, PolynomialCommitmentScheme, SecurityLevel, jagged_batch_open,
    jagged_commit,
};
use multilinear_extensions::{util::ceil_log2, virtual_poly::build_eq_x_r_vec_sequential};
use p3::{
    babybear::BabyBear,
    field::FieldAlgebra,
    matrix::{Matrix, dense::RowMajorMatrix},
    maybe_rayon::prelude::*,
};
use rand::{Rng, thread_rng};
use transcript::BasicTranscript;

type E = BabyBearExt4;
type F = BabyBear;
type Pcs = Basefold<E, BasefoldRSParams>;

const NUM_SAMPLES: usize = 10;
const NUM_MATRICES: usize = 30;
const NUM_COLS: usize = 32;

fn make_rmm(num_rows: usize, num_cols: usize) -> RowMajorMatrix<F> {
    let values: Vec<F> = (0..num_rows * num_cols)
        .into_par_iter()
        .map(|i| F::from_canonical_u32(((i as u64 * 13 + 7) % (1 << 30)) as u32))
        .collect();
    RowMajorMatrix::new(values, num_cols)
}

fn sample_heights(rng: &mut impl Rng, num_matrices: usize) -> Vec<usize> {
    (0..num_matrices)
        .map(|_| {
            let log = rng.gen_range(16u32..=22);
            let base = 1usize << log;
            // Jitter within ±25% of the power-of-two base.
            let lo = base - base / 4;
            let hi = base + base / 4;
            rng.gen_range(lo..=hi)
        })
        .collect()
}

fn eval_all_columns_at_point(rmm: &RowMajorMatrix<F>, point: &[E]) -> Vec<E> {
    let w = rmm.width();
    let eq = build_eq_x_r_vec_sequential(point);
    let mut col_evals = vec![E::ZERO; w];
    for (eq_r, row) in eq.iter().zip(rmm.rows()) {
        for (col_eval, val) in col_evals.iter_mut().zip(row) {
            *col_eval += *eq_r * val;
        }
    }
    col_evals
}

fn bench_jagged_pcs(c: &mut Criterion) {
    let mut group = c.benchmark_group("jagged_pcs");
    group.sample_size(NUM_SAMPLES);

    let mut rng = thread_rng();

    let heights = sample_heights(&mut rng, NUM_MATRICES);
    println!("Matrix heights: {:?}", heights);

    let rmms: Vec<_> = heights.iter().map(|&h| make_rmm(h, NUM_COLS)).collect();

    let log_heights: Vec<usize> = rmms.iter().map(|rmm| ceil_log2(rmm.height())).collect();
    let max_s = *log_heights.iter().max().unwrap();
    let total_evals: usize = rmms.iter().map(|rmm| rmm.height() * rmm.width()).sum();
    let num_giga_vars = ceil_log2(total_evals);

    println!(
        "num_matrices={}, num_cols={}, total_evals={}, num_giga_vars={}, max_s={}",
        NUM_MATRICES, NUM_COLS, total_evals, num_giga_vars, max_s
    );

    let point: Vec<E> = (0..max_s).map(|_| E::random(&mut rng)).collect();

    let evals: Vec<E> = rmms
        .iter()
        .zip(log_heights.iter())
        .flat_map(|(rmm, &s_i)| eval_all_columns_at_point(rmm, &point[(max_s - s_i)..]))
        .collect();

    // BabyBear two-adicity is 27; RS rate_log=1 needs level+1 ≤ 27, so max poly_size is 2^25.
    let max_log_h = 25usize;
    let reshape_log_heights: Vec<usize> = {
        let start = num_giga_vars.min(max_log_h);
        let mut vals = vec![start];
        for step in [2, 4, 6] {
            let v = start.saturating_sub(step);
            if v > 0 && !vals.contains(&v) {
                vals.push(v);
            }
        }
        vals
    };

    for &log_h in &reshape_log_heights {
        let h = 1usize << log_h;
        let w = total_evals.div_ceil(h);
        let label = format!("log_h={log_h}_w={w}");

        let poly_size = 1usize << log_h;
        let param = Pcs::setup(poly_size, SecurityLevel::Conjecture100bits).unwrap();
        let (pp, vp) = Pcs::trim(param, poly_size).unwrap();

        let comm = jagged_commit::<E, Pcs>(&pp, rmms.clone(), log_h).expect("commit failed");

        group.bench_function(BenchmarkId::new("commit", &label), |b| {
            b.iter_custom(|iters| {
                let mut time = Duration::new(0, 0);
                for _ in 0..iters {
                    let rmms_clone = rmms.clone();
                    let instant = std::time::Instant::now();
                    let _ = jagged_commit::<E, Pcs>(&pp, rmms_clone, log_h).unwrap();
                    time += instant.elapsed();
                }
                time
            })
        });

        group.bench_function(BenchmarkId::new("batch_open", &label), |b| {
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
        });

        let mut transcript_p = BasicTranscript::<E>::new(b"jagged_bench");
        Pcs::write_commitment(&comm.to_commitment().inner, &mut transcript_p).unwrap();
        let proof =
            jagged_batch_open::<E, Pcs>(&pp, &comm, &point, &evals, &mut transcript_p).unwrap();
        let proof_size = bincode::serialize(&proof).map(|v| v.len()).unwrap_or(0);
        println!("{label}: proof_size={proof_size} bytes, col_evals.len={w}");

        group.bench_function(BenchmarkId::new("batch_verify", &label), |b| {
            b.iter_batched(
                || {
                    let mut transcript = BasicTranscript::<E>::new(b"jagged_bench");
                    Pcs::write_commitment(&comm.to_commitment().inner, &mut transcript).unwrap();
                    transcript
                },
                |mut transcript| {
                    mpcs::jagged_batch_verify::<E, Pcs>(
                        &vp,
                        &comm.to_commitment(),
                        &point,
                        &evals,
                        &proof,
                        &mut transcript,
                    )
                    .unwrap();
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, bench_jagged_pcs);
criterion_main!(benches);

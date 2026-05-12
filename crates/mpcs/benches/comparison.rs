use std::time::Duration;

use criterion::*;
use ff_ext::{BabyBearExt4, FromUniformBytes};
use mpcs::{
    Basefold, BasefoldRSParams, PolynomialCommitmentScheme, SecurityLevel, jagged_batch_open,
    jagged_batch_verify, jagged_commit,
};
use multilinear_extensions::{util::ceil_log2, virtual_poly::build_eq_x_r_vec_sequential};
use p3::{
    babybear::BabyBear,
    field::FieldAlgebra,
    matrix::{Matrix, dense::RowMajorMatrix},
    maybe_rayon::prelude::*,
};
use rand::{Rng, thread_rng};
use transcript::{BasicTranscript, Transcript};
use witness::{InstancePaddingStrategy, RowMajorMatrix as WitnessRowMajorMatrix};

type E = BabyBearExt4;
type F = BabyBear;
type Pcs = Basefold<E, BasefoldRSParams>;

const NUM_SAMPLES: usize = 10;
const NUM_MATRICES: usize = 35;
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

fn bench_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("jagged_vs_direct");
    group.sample_size(NUM_SAMPLES);

    let mut rng = thread_rng();
    let heights = sample_heights(&mut rng, NUM_MATRICES);
    let log_heights: Vec<usize> = heights.iter().map(|h| ceil_log2(*h)).collect();
    let max_s = *log_heights.iter().max().unwrap();

    println!("Matrix heights: {:?}", heights);

    let rmms: Vec<_> = heights.iter().map(|&h| make_rmm(h, NUM_COLS)).collect();
    let total_evals: usize = rmms.iter().map(|rmm| rmm.height() * rmm.width()).sum();
    let num_giga_vars = ceil_log2(total_evals);

    println!(
        "num_matrices={NUM_MATRICES}, num_cols={NUM_COLS}, \
         total_evals={total_evals}, num_giga_vars={num_giga_vars}, max_s={max_s}"
    );

    let point: Vec<E> = (0..max_s).map(|_| E::random(&mut rng)).collect();

    let evals: Vec<E> = rmms
        .iter()
        .zip(log_heights.iter())
        .flat_map(|(rmm, &s_i)| eval_all_columns_at_point(rmm, &point[..s_i]))
        .collect();

    // Per-matrix points and evals (used by direct batch_open).
    let per_matrix_point_evals: Vec<(Vec<E>, Vec<E>)> = log_heights
        .iter()
        .enumerate()
        .map(|(i, &s_i)| {
            let matrix_point = point[..s_i].to_vec();
            let matrix_evals = evals[i * NUM_COLS..(i + 1) * NUM_COLS].to_vec();
            (matrix_point, matrix_evals)
        })
        .collect();

    // ======================== Jagged PCS ========================
    // BabyBear two-adicity is 27; RS rate_log=1 needs level+1 ≤ 27, so max poly_size is 2^25.
    let reshape_log_height = num_giga_vars.saturating_sub(4).min(25);
    let jagged_poly_size = 1usize << reshape_log_height;
    let jagged_param = Pcs::setup(jagged_poly_size, SecurityLevel::Conjecture100bits).unwrap();
    let (jagged_pp, jagged_vp) = Pcs::trim(jagged_param, jagged_poly_size).unwrap();

    group.bench_function("jagged/commit", |b| {
        b.iter_custom(|iters| {
            let mut time = Duration::new(0, 0);
            for _ in 0..iters {
                let rmms_clone = rmms.clone();
                let instant = std::time::Instant::now();
                let _ =
                    jagged_commit::<E, Pcs>(&jagged_pp, rmms_clone, reshape_log_height).unwrap();
                time += instant.elapsed();
            }
            time
        })
    });

    let t0 = std::time::Instant::now();
    let jagged_comm =
        jagged_commit::<E, Pcs>(&jagged_pp, rmms.clone(), reshape_log_height).unwrap();
    println!("jagged_commit: {:?}", t0.elapsed());
    let jagged_pure_comm = jagged_comm.to_commitment();

    group.bench_function("jagged/batch_open", |b| {
        b.iter_batched(
            || {
                let mut t = BasicTranscript::<E>::new(b"bench");
                Pcs::write_commitment(&jagged_pure_comm.inner, &mut t).unwrap();
                t
            },
            |mut t| {
                jagged_batch_open::<E, Pcs>(&jagged_pp, &jagged_comm, &point, &evals, &mut t)
                    .unwrap();
            },
            BatchSize::SmallInput,
        );
    });

    let jagged_proof = {
        let mut t = BasicTranscript::<E>::new(b"bench");
        Pcs::write_commitment(&jagged_pure_comm.inner, &mut t).unwrap();
        let t0 = std::time::Instant::now();
        let proof =
            jagged_batch_open::<E, Pcs>(&jagged_pp, &jagged_comm, &point, &evals, &mut t).unwrap();
        println!("jagged_batch_open: {:?}", t0.elapsed());
        proof
    };
    let jagged_proof_size = bincode::serialize(&jagged_proof)
        .map(|v| v.len())
        .unwrap_or(0);

    {
        let mut t = BasicTranscript::<E>::new(b"bench");
        Pcs::write_commitment(&jagged_pure_comm.inner, &mut t).unwrap();
        let t0 = std::time::Instant::now();
        jagged_batch_verify::<E, Pcs>(
            &jagged_vp,
            &jagged_pure_comm,
            &point,
            &evals,
            &jagged_proof,
            &mut t,
        )
        .unwrap();
        println!("jagged_batch_verify: {:?}", t0.elapsed());
    }

    group.bench_function("jagged/batch_verify", |b| {
        b.iter_batched(
            || {
                let mut t = BasicTranscript::<E>::new(b"bench");
                Pcs::write_commitment(&jagged_pure_comm.inner, &mut t).unwrap();
                t
            },
            |mut t| {
                jagged_batch_verify::<E, Pcs>(
                    &jagged_vp,
                    &jagged_pure_comm,
                    &point,
                    &evals,
                    &jagged_proof,
                    &mut t,
                )
                .unwrap();
            },
            BatchSize::SmallInput,
        );
    });

    // ======================== Direct Inner PCS ========================
    // Pcs::batch_commit expects witness::RowMajorMatrix, so convert.
    let to_witness = |rmms: &[RowMajorMatrix<F>]| -> Vec<WitnessRowMajorMatrix<F>> {
        rmms.iter()
            .map(|rmm| {
                WitnessRowMajorMatrix::new_by_values(
                    rmm.values.clone(),
                    rmm.width(),
                    InstancePaddingStrategy::Default,
                )
            })
            .collect()
    };

    let direct_poly_size = 1usize << max_s;
    let direct_param = Pcs::setup(direct_poly_size, SecurityLevel::Conjecture100bits).unwrap();
    let (direct_pp, direct_vp) = Pcs::trim(direct_param, direct_poly_size).unwrap();

    group.bench_function("direct/commit", |b| {
        b.iter_custom(|iters| {
            let mut time = Duration::new(0, 0);
            for _ in 0..iters {
                let w_rmms = to_witness(&rmms);
                let instant = std::time::Instant::now();
                let _ = Pcs::batch_commit(&direct_pp, w_rmms).unwrap();
                time += instant.elapsed();
            }
            time
        })
    });

    let t0 = std::time::Instant::now();
    let direct_comm = Pcs::batch_commit(&direct_pp, to_witness(&rmms)).unwrap();
    println!("direct_commit: {:?}", t0.elapsed());
    let direct_pure_comm = Pcs::get_pure_commitment(&direct_comm);

    let make_direct_transcript = |comm: &<Pcs as PolynomialCommitmentScheme<E>>::Commitment| {
        let mut t = BasicTranscript::<E>::new(b"bench");
        Pcs::write_commitment(comm, &mut t).unwrap();
        for (_, matrix_evals) in &per_matrix_point_evals {
            t.append_field_element_exts(matrix_evals);
        }
        t
    };

    group.bench_function("direct/batch_open", |b| {
        b.iter_batched(
            || make_direct_transcript(&direct_pure_comm),
            |mut t| {
                Pcs::batch_open(
                    &direct_pp,
                    vec![(&direct_comm, per_matrix_point_evals.clone())],
                    &mut t,
                )
                .unwrap();
            },
            BatchSize::SmallInput,
        );
    });

    let direct_proof = {
        let mut t = make_direct_transcript(&direct_pure_comm);
        let t0 = std::time::Instant::now();
        let proof = Pcs::batch_open(
            &direct_pp,
            vec![(&direct_comm, per_matrix_point_evals.clone())],
            &mut t,
        )
        .unwrap();
        println!("direct_batch_open: {:?}", t0.elapsed());
        proof
    };
    let direct_proof_size = bincode::serialize(&direct_proof)
        .map(|v| v.len())
        .unwrap_or(0);

    let direct_verify_rounds: Vec<_> = log_heights
        .iter()
        .enumerate()
        .map(|(i, &s_i)| {
            let matrix_point = point[..s_i].to_vec();
            let matrix_evals = evals[i * NUM_COLS..(i + 1) * NUM_COLS].to_vec();
            (s_i, (matrix_point, matrix_evals))
        })
        .collect();

    {
        let mut t = make_direct_transcript(&direct_pure_comm);
        let t0 = std::time::Instant::now();
        Pcs::batch_verify(
            &direct_vp,
            vec![(direct_pure_comm.clone(), direct_verify_rounds.clone())],
            &direct_proof,
            &mut t,
        )
        .unwrap();
        println!("direct_batch_verify: {:?}", t0.elapsed());
    }

    group.bench_function("direct/batch_verify", |b| {
        b.iter_batched(
            || make_direct_transcript(&direct_pure_comm),
            |mut t| {
                Pcs::batch_verify(
                    &direct_vp,
                    vec![(direct_pure_comm.clone(), direct_verify_rounds.clone())],
                    &direct_proof,
                    &mut t,
                )
                .unwrap();
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();

    println!("\n=== Proof Size Comparison ===");
    println!(
        "Jagged PCS:  {jagged_proof_size:>10} bytes  ({:.1} KB)",
        jagged_proof_size as f64 / 1024.0
    );
    println!(
        "Direct PCS:  {direct_proof_size:>10} bytes  ({:.1} KB)",
        direct_proof_size as f64 / 1024.0
    );
    println!(
        "Ratio (direct / jagged): {:.2}x",
        direct_proof_size as f64 / jagged_proof_size as f64
    );
}

criterion_group! {
    name = benches;
    config = Criterion::default().warm_up_time(Duration::from_millis(3000));
    targets = bench_comparison,
}
criterion_main!(benches);

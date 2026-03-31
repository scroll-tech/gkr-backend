#![allow(clippy::manual_memcpy)]
#![allow(clippy::needless_range_loop)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use either::Either;
use ff_ext::{BabyBearExt4, ExtensionField};
use itertools::Itertools;
use multilinear_extensions::{
    mle::MultilinearExtension, monomial::Term, util::max_usable_threads,
    virtual_polys::VirtualPolynomials,
};
use p3::field::FieldAlgebra;
use rand::thread_rng;
use sumcheck::structs::{IOPProverState, SumcheckProverMode};
use transcript::BasicTranscript as Transcript;

// ---------------------------------------------------------------------------
// Tracking allocator
// ---------------------------------------------------------------------------

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

struct TrackingAllocator;

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            let prev = ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
            let new_total = prev + layout.size();
            let mut peak = PEAK.load(Ordering::Relaxed);
            while new_total > peak {
                match PEAK.compare_exchange_weak(
                    peak,
                    new_total,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(p) => peak = p,
                }
            }
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        ALLOCATED.fetch_sub(layout.size(), Ordering::Relaxed);
    }
}

#[global_allocator]
static ALLOC: TrackingAllocator = TrackingAllocator;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Reset the peak counter to the current live allocation level, then run `f`,
/// and return the highest additional bytes allocated above that baseline.
fn measure_peak<F: FnOnce()>(f: F) -> usize {
    let baseline = ALLOCATED.load(Ordering::SeqCst);
    PEAK.store(baseline, Ordering::SeqCst);
    f();
    PEAK.load(Ordering::SeqCst).saturating_sub(baseline)
}

fn mb(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

const NUM_DEGREE: usize = 3;

// s = \prod_i fi
fn prepare_input<'a, E: ExtensionField>(nv: usize) -> Vec<MultilinearExtension<'a, E>> {
    let mut rng = thread_rng();
    (0..NUM_DEGREE)
        .map(|_| MultilinearExtension::<E>::random(nv, &mut rng))
        .collect_vec()
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    type E = BabyBearExt4;

    let threads = max_usable_threads();

    // Warm up the rayon thread pool so its internal allocations don't skew results.
    let _ = rayon::current_num_threads();

    const NV: &[usize] = &[25, 26];
    const RUNS: usize = 3;

    println!(
        "{:<8} {:<30} {:>12} {:>12} {:>12}",
        "nv", "mode", "min (MB)", "max (MB)", "median (MB)"
    );
    println!("{}", "-".repeat(80));

    for &nv in NV {
        for mode in [
            SumcheckProverMode::LegacyStable,
            SumcheckProverMode::ReducedPeakMemory,
        ] {
            let mut peaks: Vec<usize> = Vec::with_capacity(RUNS);

            for _ in 0..RUNS {
                // Prepare inputs outside the measurement window.
                let fs = prepare_input::<E>(nv);

                let peak = measure_peak(|| {
                    let mut transcript = Transcript::new(b"memory-bench");

                    // prove the sumcheck for s = \sum_b \prod_i fi(b)
                    let virtual_poly = VirtualPolynomials::new_from_monimials(
                        threads,
                        nv,
                        vec![Term {
                            scalar: Either::Right(E::ONE),
                            product: fs.iter().map(Either::Left).collect_vec(),
                        }],
                    );

                    let _ = IOPProverState::<E>::prove_with_mode(
                        virtual_poly,
                        &mut transcript,
                        mode,
                    );
                });

                peaks.push(peak);
            }

            peaks.sort_unstable();
            let min = peaks[0];
            let max = *peaks.last().unwrap();
            let median = peaks[RUNS / 2];

            let mode_name = match mode {
                SumcheckProverMode::LegacyStable => "LegacyStable",
                SumcheckProverMode::ReducedPeakMemory => "ReducedPeakMemory",
            };

            println!(
                "{:<8} {:<30} {:>12.1} {:>12.1} {:>12.1}",
                nv,
                mode_name,
                mb(min),
                mb(max),
                mb(median),
            );
        }

        println!();
    }
}

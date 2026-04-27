use ff_ext::ExtensionField;

// ---------------------------------------------------------------------------
// Succinct evaluation of ĝ(z₁, z₂, z₃, z₄) where g(a,b,c,d) = [a+c=b ∧ b<d]
//
// g is computable by a width-4 ROBP with state (carry, lt) ∈ {0,1}²
// (Claim 3.2.2 of the jagged PCS paper). The ROBP reads one symbol
// σᵢ = (aᵢ, bᵢ, cᵢ, dᵢ) ∈ {0,1}⁴ per step.
//
// Both functions below compute the same value:
//   ĝ(z₁,z₂,z₃,z₄) = e₁ᵀ · T₁ · T₂ · … · Tₙ · u
// where Tᵢ = Σ_{σ ∈ {0,1}⁴} eq(ζᵢ, σ) · M^(σ) is the eq-weighted
// transition matrix at step i, e₁ is the initial state vector, and u is
// the sink label vector.
//
// Since multilinear variables can be bound in any order, we fix z₁ and z₂
// first (to z_row and ρ), which reduces the per-step alphabet from {0,1}⁴
// to {0,1}² and gives the 4 matrices M_i^{(c,d)}.  The remaining variables
// z₃, z₄ are then interleaved as (z₃[0], z₄[0], z₃[1], …) to match the
// ROBP step order, enabling the forward/backward decomposition.
// ---------------------------------------------------------------------------

pub const ROBP_WIDTH: usize = 4;

pub type StateVec<E> = [E; ROBP_WIDTH];
pub type TransitionMatrix<E> = [[E; ROBP_WIDTH]; ROBP_WIDTH];

/// Sink label vector: accept at state (carry=0, lt=1) = index 1.
pub fn sink_labels<E: ExtensionField>() -> StateVec<E> {
    let mut u = [E::ZERO; ROBP_WIDTH];
    u[1] = E::ONE;
    u
}

/// Initial forward vector: source state (carry=0, lt=0) = index 0.
pub fn source_vec<E: ExtensionField>() -> StateVec<E> {
    let mut v = [E::ZERO; ROBP_WIDTH];
    v[0] = E::ONE;
    v
}

#[inline]
pub fn dot4<E: ExtensionField>(a: &StateVec<E>, b: &StateVec<E>) -> E {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

/// Row-vector × matrix: out[j] = Σ_i v[i] * m[i][j].
#[inline]
pub fn vec_mat_mul<E: ExtensionField>(v: &StateVec<E>, m: &TransitionMatrix<E>) -> StateVec<E> {
    std::array::from_fn(|j| v[0] * m[0][j] + v[1] * m[1][j] + v[2] * m[2][j] + v[3] * m[3][j])
}

/// Matrix × column-vector: out[i] = Σ_j m[i][j] * v[j].
#[inline]
pub fn mat_vec_mul<E: ExtensionField>(m: &TransitionMatrix<E>, v: &StateVec<E>) -> StateVec<E> {
    std::array::from_fn(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2] + m[i][3] * v[3])
}

/// Raw ROBP transition table for the indicator g(a,b,c,d) = [a+c=b ∧ b<d].
///
/// `ROBP_TRANSITION[from_state][symbol]` = `to_state`, where:
///   - state = carry * 2 + lt
///   - symbol = a * 8 + b * 4 + c * 2 + d
///
/// A value of `REJECT` (0xFF) means the transition leads to a rejecting sink
/// (inconsistent addition: LSB of a + c + carry_in ≠ b).
const REJECT: u8 = 0xFF;

static ROBP_TRANSITION: [[u8; 16]; ROBP_WIDTH] = {
    let mut table = [[REJECT; 16]; ROBP_WIDTH];
    let mut from = 0u8;
    while from < 4 {
        let carry_in = from >> 1;
        let lt_in = from & 1;
        let mut sym = 0u8;
        while sym < 16 {
            let a = sym >> 3;
            let b = (sym >> 2) & 1;
            let c = (sym >> 1) & 1;
            let d = sym & 1;
            let sum = a + c + carry_in;
            if sum & 1 == b {
                let carry_out = sum >> 1;
                let lt_out = if b < d { 1 } else if b == d { lt_in } else { 0 };
                table[from as usize][sym as usize] = carry_out * 2 + lt_out;
            }
            sym += 1;
        }
        from += 1;
    }
    table
};

/// Per-symbol transition matrices M_i^{(c,d)} at one ROBP step, with
/// z1 (= z_row[i]) and z2 (= ρ[i]) eq-weights baked in.
///
/// Returns 4 matrices indexed by `c * 2 + d` for `(c, d) ∈ {0,1}²`.
/// Matrix entry `m[from][to]` gives the transition weight from `from` to `to`
/// when reading symbol `(c, d)`.
///
/// Derived from the raw ROBP transition table via:
///   M_i^{(c,d)}[from][to] = Σ_{a,b} eq₁(z1,a) · eq₁(z2,b) · [transition(from,(a,b,c,d)) = to]
///
/// The full eq-weighted transition matrix is recovered as:
///   T_i = Σ_{c,d} eq₁(z3, c) · eq₁(z4, d) · M_i^{(c,d)}
pub fn symbol_transition_matrices<E: ExtensionField>(z1i: E, z2i: E) -> [TransitionMatrix<E>; 4] {
    let eq_ab: [E; 4] = {
        let (nz1, nz2) = (E::ONE - z1i, E::ONE - z2i);
        [nz1 * nz2, nz1 * z2i, z1i * nz2, z1i * z2i]
    };

    let mut mats = [[[E::ZERO; ROBP_WIDTH]; ROBP_WIDTH]; 4];
    for cd in 0..4u8 {
        for from in 0..ROBP_WIDTH {
            for ab in 0..4u8 {
                let sym = (ab << 2) | cd;
                let to = ROBP_TRANSITION[from][sym as usize];
                if to != REJECT {
                    mats[cd as usize][from][to as usize] += eq_ab[ab as usize];
                }
            }
        }
    }
    mats
}

/// Compute the eq-weighted transition matrix at step `i`.
///
/// Returns `(T_same, T_lt1, T_lt0)`: three 2×2 matrices (indexed by carry)
/// that describe how the 4 ROBP states transition, grouped by effect on `lt`:
///   - `T_same[ci][co]`: weight of symbols that preserve `lt`
///   - `T_lt1[ci][co]`:  weight of symbols that force `lt = 1`
///   - `T_lt0[ci][co]`:  weight of symbols that force `lt = 0`
///
/// For each (carry_in, carry_out) pair, we enumerate all consistent symbols
/// (where `a + c + carry_in` has LSB = `b` and MSB = `carry_out`) and sum
/// their eq-weights. Inconsistent symbols are absent — they correspond to
/// transitions to a rejecting sink with label 0.
#[inline]
fn transition_weights<E: ExtensionField>(
    z1i: E,
    z2i: E,
    z3i: E,
    z4i: E,
) -> [(usize, usize, E, E, E); 4] {
    let (nz1, nz2, nz3, nz4) = (E::ONE - z1i, E::ONE - z2i, E::ONE - z3i, E::ONE - z4i);

    // eq-weights for each (a,b) and (c,d) bit combination:
    //   abXY = eq₁(z1ᵢ, X) · eq₁(z2ᵢ, Y)
    //   cdXY = eq₁(z3ᵢ, X) · eq₁(z4ᵢ, Y)
    // The eq-weight for symbol (a,b,c,d) = abAB · cdCD.
    let ab00 = nz1 * nz2;
    let ab01 = nz1 * z2i;
    let ab10 = z1i * nz2;
    let ab11 = z1i * z2i;
    let cd00 = nz3 * nz4;
    let cd01 = nz3 * z4i;
    let cd10 = z3i * nz4;
    let cd11 = z3i * z4i;

    // Each entry: (carry_in, carry_out, w_same, w_lt1, w_lt0)
    //   w_same: symbols where b = d  (lt preserved)
    //   w_lt1:  symbols where b = 0, d = 1 (lt forced to 1)
    //   w_lt0:  symbols where b = 1, d = 0 (lt forced to 0)
    [
        // carry_in=0, carry_out=0: consistent (a,c) ∈ {(0,0),(0,1),(1,0)}; b = a⊕c
        (
            0,
            0,
            ab00 * cd00 + ab01 * cd11 + ab11 * cd01, // same: (0,0,0,0),(0,1,1,1),(1,1,0,1)
            ab00 * cd01,                             // lt=1: (0,0,0,1)
            ab01 * cd10 + ab11 * cd00,               // lt=0: (0,1,1,0),(1,1,0,0)
        ),
        // carry_in=0, carry_out=1: consistent (a,c) = (1,1); b = 0
        (0, 1, ab10 * cd10, ab10 * cd11, E::ZERO),
        // carry_in=1, carry_out=0: consistent (a,c) = (0,0); b = 1
        (1, 0, ab01 * cd01, E::ZERO, ab01 * cd00),
        // carry_in=1, carry_out=1: consistent (a,c) ∈ {(0,1),(1,0),(1,1)}; b = a⊕c⊕1
        (
            1,
            1,
            ab00 * cd10 + ab10 * cd00 + ab11 * cd11, // same: (0,0,1,0),(1,0,0,0),(1,1,1,1)
            ab00 * cd11 + ab10 * cd01,               // lt=1: (0,0,1,1),(1,0,0,1)
            ab11 * cd10,                             // lt=0: (1,1,1,0)
        ),
    ]
}

/// Evaluate ĝ(z₁, z₂, z₃, z₄) using **forward** propagation (source → sinks).
///
/// Maintains a state vector `α` of 4 weights, where `α[carry*2 + lt]` is the
/// total eq-weight of all paths from the source to state `(carry, lt)`. At each
/// step, all 16 transitions fire simultaneously, weighted by `eq(ζᵢ, σ)`.
///
/// Computes: `(…((e₁ᵀ · T₁) · T₂) · … · Tₙ) · u`
///
/// This follows from the MLE definition directly:
///   ĝ(z) = Σ_{sink v} label(v) · (Σ_{paths to v} Π_i eq(ζᵢ, σᵢ))
/// The forward pass accumulates the path-weight sums layer by layer.
pub fn evaluate_g_forward<E: ExtensionField>(z1: &[E], z2: &[E], z3: &[E], z4: &[E]) -> E {
    let n = z1.len();
    assert_eq!(z2.len(), n);
    assert_eq!(z3.len(), n);
    assert_eq!(z4.len(), n);

    // state[carry * 2 + lt]: weight on ROBP state (carry, lt).
    // Initial: (carry=0, lt=0) with weight 1.
    let mut state = [E::ZERO; 4];
    state[0] = E::ONE;

    for i in 0..n {
        let transitions = transition_weights(z1[i], z2[i], z3[i], z4[i]);

        let mut new_state = [E::ZERO; 4];
        for &(ci, co, w_same, w_lt1, w_lt0) in &transitions {
            let s0 = state[ci * 2]; // weight on (carry=ci, lt=0)
            let s1 = state[ci * 2 + 1]; // weight on (carry=ci, lt=1)
            let total = s0 + s1;
            // lt preserved: w_same keeps lt=0 → lt=0 and lt=1 → lt=1
            // lt forced to 0: w_lt0 sends both lt=0 and lt=1 → lt=0
            // lt forced to 1: w_lt1 sends both lt=0 and lt=1 → lt=1
            new_state[co * 2] += w_same * s0 + w_lt0 * total;
            new_state[co * 2 + 1] += w_same * s1 + w_lt1 * total;
        }
        state = new_state;
    }

    // Accept state: (carry=0, lt=1) → index 1.
    // g requires carry=0 (exact addition a+c=b) and lt=1 (b < d).
    state[1]
}

/// Evaluate ĝ(z₁, z₂, z₃, z₄) using **backward** propagation (sinks → source).
///
/// Processes the ROBP in reverse (Lemma 4.2, Claim 4.2.1 of the jagged PCS paper):
///   ĝ_v(ζ, z) = Σ_{σ ∈ {0,1}⁴} eq(ζ, σ) · ĝ_{Γ(v,σ)}(z)
///
/// Maintains a 4-element vector `val` where `val[carry*2 + lt]` is ĝ evaluated
/// at the corresponding sink, for the suffix of variables processed so far.
/// Starting from sink labels, we work backward to the source.
///
/// Computes: `e₁ᵀ · (T₁ · (T₂ · … · (Tₙ · u)))`
///
/// This is the paper's algorithm. It naturally extends to symbolic evaluation
/// (Lemma 4.5): by not multiplying with `u` at the end, we get a result vector
/// that works for any sink labels — useful for batch evaluation in the jagged
/// assist sumcheck.
pub fn evaluate_g_backward<E: ExtensionField>(z1: &[E], z2: &[E], z3: &[E], z4: &[E]) -> E {
    let n = z1.len();
    assert_eq!(z2.len(), n);
    assert_eq!(z3.len(), n);
    assert_eq!(z4.len(), n);

    // Sink labels: u[carry*2 + lt] = label of sink state (carry, lt).
    // g accepts iff carry=0 AND lt=1, so only state (0,1) has label 1.
    let mut val = [E::ZERO; 4];
    val[1] = E::ONE; // (carry=0, lt=1) → accept

    // Process layers in reverse: from layer n-1 down to 0.
    // After processing layer i, val[v] = ĝ_v(ζᵢ, ζᵢ₊₁, …, ζₙ₋₁) for the
    // suffix z[i..n], where ĝ_v is the MLE starting at state v.
    for i in (0..n).rev() {
        let transitions = transition_weights(z1[i], z2[i], z3[i], z4[i]);

        // For each state v at layer i, compute:
        //   new_val[v] = Σ_{σ} eq(ζᵢ, σ) · val[Γ(v, σ)]
        // Grouped by (carry_in, carry_out) and lt effect:
        //   new_val[ci*2 + lt_in] += w_same * val[co*2 + lt_in]     (lt preserved)
        //                          + w_lt1  * val[co*2 + 1]          (lt forced to 1)
        //                          + w_lt0  * val[co*2 + 0]          (lt forced to 0)
        let mut new_val = [E::ZERO; 4];
        for &(ci, co, w_same, w_lt1, w_lt0) in &transitions {
            let v0 = val[co * 2]; // val at (carry=co, lt=0)
            let v1 = val[co * 2 + 1]; // val at (carry=co, lt=1)
            // State (ci, lt=0): lt preserved → goes to val[co, lt=0];
            //                   lt forced to 1 → goes to val[co, lt=1];
            //                   lt forced to 0 → goes to val[co, lt=0].
            new_val[ci * 2] += w_same * v0 + w_lt1 * v1 + w_lt0 * v0;
            // State (ci, lt=1): lt preserved → goes to val[co, lt=1];
            //                   lt forced to 1 → goes to val[co, lt=1];
            //                   lt forced to 0 → goes to val[co, lt=0].
            new_val[ci * 2 + 1] += w_same * v1 + w_lt1 * v1 + w_lt0 * v0;
        }
        val = new_val;
    }

    // val[0] = ĝ_source(z) = ĝ(z₁, z₂, z₃, z₄).
    // Source is state (carry=0, lt=0) → index 0.
    val[0]
}

/// Evaluate ĝ(z₁, z₂, z₃, z₄) — delegates to the backward algorithm.
pub fn evaluate_g<E: ExtensionField>(z1: &[E], z2: &[E], z3: &[E], z4: &[E]) -> E {
    evaluate_g_backward(z1, z2, z3, z4)
}

#[cfg(test)]
mod tests {
    use ff_ext::{BabyBearExt4, FromUniformBytes};
    use multilinear_extensions::virtual_poly::build_eq_x_r_vec;
    use p3::field::FieldAlgebra;
    use rand::thread_rng;

    use super::*;

    type E = BabyBearExt4;

    /// Brute-force MLE evaluation of g(a,b,c,d) = [a+c=b AND b<d].
    fn evaluate_g_bruteforce(z1: &[E], z2: &[E], z3: &[E], z4: &[E]) -> E {
        let n = z1.len();
        let size = 1usize << n;
        let eq1 = build_eq_x_r_vec(z1);
        let eq2 = build_eq_x_r_vec(z2);
        let eq3 = build_eq_x_r_vec(z3);
        let eq4 = build_eq_x_r_vec(z4);

        let mut sum = E::ZERO;
        for (a, eq1a) in eq1.iter().enumerate().take(size) {
            for (c, eq3c) in eq3.iter().enumerate().take(size) {
                let b = a + c;
                if b >= size {
                    continue;
                }
                for eq4d in eq4.iter().take(size).skip(b + 1) {
                    sum += *eq1a * eq2[b] * *eq3c * *eq4d;
                }
            }
        }
        sum
    }

    #[test]
    fn test_evaluate_g_forward() {
        let mut rng = thread_rng();
        for n in 1..=4 {
            let z1: Vec<E> = (0..n).map(|_| E::random(&mut rng)).collect();
            let z2: Vec<E> = (0..n).map(|_| E::random(&mut rng)).collect();
            let z3: Vec<E> = (0..n).map(|_| E::random(&mut rng)).collect();
            let z4: Vec<E> = (0..n).map(|_| E::random(&mut rng)).collect();

            let expected = evaluate_g_bruteforce(&z1, &z2, &z3, &z4);
            let result = evaluate_g_forward(&z1, &z2, &z3, &z4);
            assert_eq!(result, expected, "forward mismatch at n={n}");
        }
    }

    #[test]
    fn test_evaluate_g_backward() {
        let mut rng = thread_rng();
        for n in 1..=4 {
            let z1: Vec<E> = (0..n).map(|_| E::random(&mut rng)).collect();
            let z2: Vec<E> = (0..n).map(|_| E::random(&mut rng)).collect();
            let z3: Vec<E> = (0..n).map(|_| E::random(&mut rng)).collect();
            let z4: Vec<E> = (0..n).map(|_| E::random(&mut rng)).collect();

            let expected = evaluate_g_bruteforce(&z1, &z2, &z3, &z4);
            let result = evaluate_g_backward(&z1, &z2, &z3, &z4);
            assert_eq!(result, expected, "backward mismatch at n={n}");
        }
    }

    #[test]
    fn test_forward_backward_agree() {
        let mut rng = thread_rng();
        for n in 1..=6 {
            let z1: Vec<E> = (0..n).map(|_| E::random(&mut rng)).collect();
            let z2: Vec<E> = (0..n).map(|_| E::random(&mut rng)).collect();
            let z3: Vec<E> = (0..n).map(|_| E::random(&mut rng)).collect();
            let z4: Vec<E> = (0..n).map(|_| E::random(&mut rng)).collect();

            let fwd = evaluate_g_forward(&z1, &z2, &z3, &z4);
            let bwd = evaluate_g_backward(&z1, &z2, &z3, &z4);
            assert_eq!(fwd, bwd, "forward != backward at n={n}");
        }
    }

    /// Verify that recombining per-symbol matrices reproduces transition_weights.
    #[test]
    fn test_symbol_matrices_match_transition_weights() {
        let mut rng = thread_rng();
        for _ in 0..20 {
            let z1i = E::random(&mut rng);
            let z2i = E::random(&mut rng);
            let z3i = E::random(&mut rng);
            let z4i = E::random(&mut rng);

            let mats = symbol_transition_matrices(z1i, z2i);
            let (nz3, nz4) = (E::ONE - z3i, E::ONE - z4i);
            let cd_weights = [nz3 * nz4, nz3 * z4i, z3i * nz4, z3i * z4i];

            // Reconstruct T = Σ_{c,d} eq₁(z3,c)·eq₁(z4,d)·M^{(c,d)}
            let mut t_recon = [[E::ZERO; 4]; 4];
            for cd in 0..4 {
                for i in 0..4 {
                    for j in 0..4 {
                        t_recon[i][j] += cd_weights[cd] * mats[cd][i][j];
                    }
                }
            }

            // Compare against transition_weights-based backward step.
            let transitions = transition_weights(z1i, z2i, z3i, z4i);
            let mut t_expected = [[E::ZERO; 4]; 4];
            for &(ci, co, w_same, w_lt1, w_lt0) in &transitions {
                // M[from][to] convention, matching backward pass.
                t_expected[ci * 2][co * 2] += w_same + w_lt0;
                t_expected[ci * 2][co * 2 + 1] += w_lt1;
                t_expected[ci * 2 + 1][co * 2] += w_lt0;
                t_expected[ci * 2 + 1][co * 2 + 1] += w_same + w_lt1;
            }

            assert_eq!(
                t_recon, t_expected,
                "symbol matrices don't match transition_weights"
            );
        }
    }

    /// Verify backward precomputation: using per-symbol matrices with Boolean
    /// (c, d) reproduces evaluate_g.
    #[test]
    fn test_backward_via_symbol_matrices() {
        let mut rng = thread_rng();
        for n in 1..=5 {
            let z1: Vec<E> = (0..n).map(|_| E::random(&mut rng)).collect();
            let z2: Vec<E> = (0..n).map(|_| E::random(&mut rng)).collect();
            let z3: Vec<E> = (0..n).map(|_| E::random(&mut rng)).collect();
            let z4: Vec<E> = (0..n).map(|_| E::random(&mut rng)).collect();

            let expected = evaluate_g_backward(&z1, &z2, &z3, &z4);

            // Compute using per-symbol matrices with Boolean (c, d) from z3/z4.
            // Round each z3[i]/z4[i] to the nearest bit for this test — instead,
            // use evaluate_g with actual field elements and compare using the
            // matrix product approach.
            let step_mats: Vec<_> = (0..n)
                .map(|i| symbol_transition_matrices(z1[i], z2[i]))
                .collect();

            // Build T_i = Σ_{c,d} eq1(z3[i],c)*eq1(z4[i],d)*M^{(c,d)}
            // and multiply backward.
            let u = sink_labels::<E>();
            let mut val = u;
            for i in (0..n).rev() {
                let (nz3, nz4) = (E::ONE - z3[i], E::ONE - z4[i]);
                let cd_w = [nz3 * nz4, nz3 * z4[i], z3[i] * nz4, z3[i] * z4[i]];
                let mut t_i = [[E::ZERO; 4]; 4];
                for cd in 0..4 {
                    for r in 0..4 {
                        for c in 0..4 {
                            t_i[r][c] += cd_w[cd] * step_mats[i][cd][r][c];
                        }
                    }
                }
                val = mat_vec_mul(&t_i, &val);
            }
            let result = val[0]; // source state
            assert_eq!(
                result, expected,
                "symbol-matrix backward != evaluate_g at n={n}"
            );
        }
    }
}

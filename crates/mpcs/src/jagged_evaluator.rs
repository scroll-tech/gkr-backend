use ff_ext::ExtensionField;

// ---------------------------------------------------------------------------
// Succinct evaluation of g(a, b, c, d) = [a + c = b AND b < d]
// ---------------------------------------------------------------------------

/// Evaluate the multilinear extension of the indicator function
/// `g(a, b, c, d) = [a + c = b AND b < d]` at a random point `(z1, z2, z3, z4)`.
///
/// Each argument is a slice of `n` field elements (one per bit, LSB first).
/// Uses a width-4 ROBP (read-once branching program) from Section 3.2 of the
/// jagged PCS paper, with state `(carry, lt)` where `carry` tracks addition
/// carry and `lt` tracks whether `b < d`. Evaluated per Lemma 4.2.
/// Complexity: O(n).
pub fn evaluate_g<E: ExtensionField>(z1: &[E], z2: &[E], z3: &[E], z4: &[E]) -> E {
    let n = z1.len();
    assert_eq!(z2.len(), n);
    assert_eq!(z3.len(), n);
    assert_eq!(z4.len(), n);

    // State = (carry, lt) ∈ {0,1}², index = carry * 2 + lt.
    // Initial: (carry=0, lt=0) → index 0.
    let mut state = [E::ZERO; 4];
    state[0] = E::ONE;

    for i in 0..n {
        let (z1i, z2i, z3i, z4i) = (z1[i], z2[i], z3[i], z4[i]);
        let (nz1, nz2, nz3, nz4) = (E::ONE - z1i, E::ONE - z2i, E::ONE - z3i, E::ONE - z4i);

        // Pairwise products for (a,b) and (c,d) bit combinations.
        let ab00 = nz1 * nz2;
        let ab01 = nz1 * z2i;
        let ab10 = z1i * nz2;
        let ab11 = z1i * z2i;
        let cd00 = nz3 * nz4;
        let cd01 = nz3 * z4i;
        let cd10 = z3i * nz4;
        let cd11 = z3i * z4i;

        // Transition weights: (carry_in, carry_out, w_same, w_lt1, w_lt0).
        // w_same: preserves lt; w_lt1: forces lt=1; w_lt0: forces lt=0.
        let transitions: [(usize, usize, E, E, E); 4] = [
            (
                0,
                0,
                ab00 * cd00 + ab01 * cd11 + ab11 * cd01,
                ab00 * cd01,
                ab01 * cd10 + ab11 * cd00,
            ),
            (0, 1, ab10 * cd10, ab10 * cd11, E::ZERO),
            (1, 0, ab01 * cd01, E::ZERO, ab01 * cd00),
            (
                1,
                1,
                ab00 * cd10 + ab10 * cd00 + ab11 * cd11,
                ab00 * cd11 + ab10 * cd01,
                ab11 * cd10,
            ),
        ];

        let mut new_state = [E::ZERO; 4];
        for &(ci, co, w_same, w_lt1, w_lt0) in &transitions {
            let total = state[ci * 2] + state[ci * 2 + 1];
            new_state[co * 2] += w_same * state[ci * 2] + w_lt0 * total;
            new_state[co * 2 + 1] += w_same * state[ci * 2 + 1] + w_lt1 * total;
        }
        state = new_state;
    }

    // Accept: (carry=0, lt=1) → index 1.
    state[1]
}

#[cfg(test)]
mod tests {
    use ff_ext::{BabyBearExt4, FromUniformBytes};
    use multilinear_extensions::virtual_poly::build_eq_x_r_vec;
    use p3::field::FieldAlgebra;
    use rand::thread_rng;

    use super::evaluate_g;

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
    fn test_evaluate_g_small() {
        let mut rng = thread_rng();

        for n in 1..=4 {
            let z1: Vec<E> = (0..n).map(|_| E::random(&mut rng)).collect();
            let z2: Vec<E> = (0..n).map(|_| E::random(&mut rng)).collect();
            let z3: Vec<E> = (0..n).map(|_| E::random(&mut rng)).collect();
            let z4: Vec<E> = (0..n).map(|_| E::random(&mut rng)).collect();

            let expected = evaluate_g_bruteforce(&z1, &z2, &z3, &z4);
            let result = evaluate_g(&z1, &z2, &z3, &z4);
            assert_eq!(result, expected, "evaluate_g mismatch at n={n}");
        }
    }
}

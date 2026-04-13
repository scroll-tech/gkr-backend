# Jagged Sumcheck Prover — M-Table Algorithm Plan

Based on [ceno#1289 comment](https://github.com/scroll-tech/ceno/issues/1289#issuecomment-4234003987)
and the "Time-Space Trade-Offs for Sumcheck" paper (eprint 2025/1473).

## Problem

Prove `v = sum_b q'(b) * f(b)` where:
- `q'` is the giga polynomial (n ~ 31 variables, 2^n evaluations, padded)
- `f(b) = eq(z_row, row(b)) * eq(z_c, col(b))`
- The sumcheck is **degree 2** (product of two MLEs)
- Neither q' nor f can be materialized fully in memory

## f(b) Definition

All polynomials have the same height `h = 2^s`. Given cumulative heights `t[j]`:
```
col(b) = j        where t[j] <= b < t[j+1]
row(b) = b - t[j]
f(b) = eq(z_row, row(b)) * eq(z_c, col(b))
```
- `z_row` is a single shared evaluation point with `s = log2(h)` components
- `z_c` encodes the polynomial index
- For b >= total_evaluations (padding region): f(b) = 0, q'(b) = 0

### Precomputation (once, before any streaming)
- `eq_row = build_eq_x_r_vec(z_row)` — size `h`, computed once
- `eq_col = build_eq_x_r_vec(z_c)` — size `2^{len(z_c)}`, computed once
- Then `f(b) = eq_row[row(b)] * eq_col[col(b)]`

## Algorithm Overview

Split sumcheck into two phases:
1. **Streaming phase** (rounds 0..K-1, K=16): use M-table streaming, 6 total passes over the data
2. **Standard phase** (rounds K..n-1): materialize bound MLEs (2^{n-K} entries each), run standard sumcheck

### Epoch Schedule (K=15)

Rounds are numbered 1..n. Each epoch j' computes j' rounds (rounds j' through 2j'-1).
Epoch j' requires j'-1 prior challenges from earlier epochs.

| Epoch   | Rounds     | Prior challenges | M-table size | Streaming pass |
|---------|------------|-----------------|-------------|----------------|
| j'=1    | j=1        | 0               | 2^2 = 4     | 1 pass         |
| j'=2    | j=2,3      | 1 (r_1)         | 2^4 = 16    | 1 pass         |
| j'=4    | j=4,5,6,7  | 3 (r_1..r_3)    | 2^8 = 256   | 1 pass         |
| j'=8    | j=8..15    | 7 (r_1..r_7)    | 2^16 = 65K  | 1 pass         |
| Bind    | —          | 15 (r_1..r_15)  | —           | 1 pass         |
| **Total** | 1..15    |                 |             | **5 passes**   |

After streaming: standard sumcheck on MLEs of size 2^{n-15} for rounds 16..n.

## Data Structures

```rust
/// M-table: M[beta1 * 2^{j'} + beta2] for beta1, beta2 in {0,1}^{j'}
/// Size: 2^{2*epoch_size} extension field elements
type MTable<E> = Vec<E>;
```

## Building M-Table for Epoch j'

After collecting challenges R_{j'} = (r_1, ..., r_{j'-1}), build:
```
M_{j'}[beta1, beta2] = sum_b Q_bound(beta1, b) * F_bound(beta2, b)
```
where:
- `Q_bound(beta, b) = sum_{a in {0,1}^{j'-1}} eq(R_{j'}, a) * q'[a || beta || b]`
- `F_bound(beta, b) = sum_{a in {0,1}^{j'-1}} eq(R_{j'}, a) * f[a || beta || b]`

### Streaming Implementation

Precompute once (shared across all epochs):
- `eq_row = build_eq_x_r_vec(z_row)` — size `h`
- `eq_col = build_eq_x_r_vec(z_col)` — size `2^{len(z_col)}`

Precompute per epoch:
- `eq_R = build_eq_x_r_vec(R_{j'})` — 2^{j'-1} entries (trivially [1] for j'=1)

(`eq_row` and `eq_col` are already computed once in the precomputation step above.)

Stream through the giga polynomial in **chunks** of size `2^{j'-1 + j'} = 2^{2j'-1}`.
Each chunk corresponds to one value of b (the "tail" bits).
Within a chunk, the index decomposes as: `idx = a + beta * 2^{j'-1} + b * 2^{2j'-1}` (little-endian).

```
for b_idx in 0..2^{n - 2*j' + 1}:
    chunk_start = b_idx * 2^{2*j' - 1}
    Q_bound = [0; 2^{j'}]   // indexed by beta
    F_bound = [0; 2^{j'}]

    for beta in 0..2^{j'}:
        for a in 0..2^{j'-1}:
            giga_idx = chunk_start + beta * 2^{j'-1} + a
            if giga_idx >= total_evaluations: continue

            j = col(giga_idx)      // binary search in cumulative_heights
            local = giga_idx - t[j]
            q_val = q_evals[giga_idx]
            f_val = eq_row[local] * eq_col[j]

            Q_bound[beta] += eq_R[a] * q_val
            F_bound[beta] += eq_R[a] * f_val

    // Outer product accumulation
    for beta1 in 0..2^{j'}:
        for beta2 in 0..2^{j'}:
            M[beta1 * 2^{j'} + beta2] += Q_bound[beta1] * F_bound[beta2]
```

**col(b) lookup**: binary search in `cumulative_heights` to find j with `t[j] <= b < t[j+1]`.
Since we stream sequentially, maintain a cursor to avoid repeated binary searches.

## Computing Round j from M-Table

For round j within epoch j' (where j' <= j < 2*j'), let d = j - j' (intra-epoch offset).
Challenges within epoch: `r_{j'}, ..., r_{j-1}` (d challenges, empty for d=0).

Precompute: `eq_intra = build_eq_x_r_vec(&[r_{j'}, ..., r_{j-1}])` — 2^d entries.

The round univariate h_j(x) for x in {0, 1, 2}:

```
h_j(x) = sum_{a in {0,1}^d} sum_{c in {0,1}^d}
    eq_intra[a] * eq_intra[c] * g_x(a, c)
```

where, letting `pad = j' - d - 1` zero bits:
```
beta1(a, x_bit) = a || x_bit || 0^pad    (j' bits total)
beta2(c, x_bit) = c || x_bit || 0^pad    (j' bits total)
```

For x=0 and x=1:
```
g_x(a, c) = M[beta1(a, x), beta2(c, x)]
```

For x=2, bilinear interpolation:
```
M00 = M[beta1(a,0), beta2(c,0)]
M10 = M[beta1(a,1), beta2(c,0)]
M01 = M[beta1(a,0), beta2(c,1)]
M11 = M[beta1(a,1), beta2(c,1)]

g_2(a,c) = M00 - 2*M10 - 2*M01 + 4*M11
```

(From bilinear interpolation: `(1-2)^2 * M00 + 2(1-2)*M10 + (1-2)*2*M01 + 4*M11 = M00 - 2M10 - 2M01 + 4M11`)

### Bit Layout for M-Table Indexing

beta1 is j' bits. In the first round of the epoch (d=0), x occupies bit position 0 and the remaining j'-1 bits are zero:
```
beta1 = x_bit  (for d=0)
beta1 = a[0] || a[1] || ... || a[d-1] || x_bit || 0...0  (for d>0)
```

In little-endian: `beta1_val = a_val + x_bit * 2^d + 0` (the pad bits are the most significant).

Similarly for beta2 with c.

## Phase 2: Bind and Materialize (Streaming Pass #6)

After K=15 streaming rounds, we have challenges R = (r_1, ..., r_15).
One final streaming pass to compute bound MLEs:

```
q_bound[idx] = sum_{a in {0,1}^K} eq(R_K, a) * q'[a + idx * 2^K]
f_bound[idx] = sum_{a in {0,1}^K} eq(R_K, a) * f[a + idx * 2^K]
```

for idx = 0..2^{n-K}-1. Each result is an extension-field MLE of size 2^{n-K}.

Then construct `VirtualPolynomial` from `q_bound * f_bound` and run `IOPProverState::prove()` for rounds K+1..n. The proof messages from both phases are concatenated into a single `IOPProof`.

## Data Types and Function Signatures

```rust
/// All inputs needed for the jagged sumcheck.
pub struct JaggedSumcheckInput<'a, E: ExtensionField> {
    pub q_evals: &'a [E::BaseField],    // giga polynomial evaluations (concatenated)
    pub num_giga_vars: usize,            // n = log2(padded_total_size)
    pub cumulative_heights: &'a [usize], // t[j], length num_polys + 1
    pub z_row: &'a [E],                  // shared row evaluation point (s components)
    pub z_col: &'a [E],                  // column challenge point
}

/// Main entry point: run the full jagged sumcheck
pub fn jagged_sumcheck_prove<E: ExtensionField>(
    input: &JaggedSumcheckInput<E>,
    transcript: &mut impl Transcript<E>,
) -> (IOPProof<E>, Vec<E>)             // proof + challenges

/// Build M-table for epoch j'
fn build_m_table<E>(
    input: &JaggedSumcheckInput<E>,
    challenges: &[E],   // R_{j'} = (r_1, ..., r_{j'-1})
    epoch_size: usize,   // j'
) -> Vec<E>

/// Extract round univariate h_j(x) from M-table
fn compute_round_from_m<E>(
    m_table: &[E],
    epoch_size: usize,
    intra_challenges: &[E],  // r_{j'}..r_{j-1}
) -> [E; 3]

/// Bind first K variables and materialize reduced MLEs
fn bind_and_materialize<E>(
    input: &JaggedSumcheckInput<E>,
    challenges: &[E],   // R_K = (r_1, ..., r_K)
) -> (Vec<E>, Vec<E>)   // (q_bound, f_bound)
```

## Proof Format

Each round sends `evaluations = [h(1), h(2)]` (h(0) is derived by verifier as `claim - h(1)`).
This matches the existing `IOPProverMessage` convention. The streaming phase produces K=16
messages, and the standard phase produces n-K messages. All are concatenated into one `IOPProof`.

## Complexity

- **Time**: O(6N) where N = 2^n (5 streaming passes + 1 bind pass, each O(N))
- **Space**: O(2^K + 2^{n-K}) = O(2^16 + 2^15) ~ 100K extension field elements ~ 1.5 MB
- **Proof size**: n round messages, each with 2 field elements = 2n field elements total

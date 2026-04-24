use crate::PolynomialCommitmentScheme;
use ::sumcheck::structs::IOPProof;
use ff_ext::ExtensionField;
use serde::{Deserialize, Serialize};

/// Commitment to a jagged polynomial `q'`, together with all witness data needed
/// for opening proofs.
///
/// Generic over the inner PCS `InnerPcs` so that any `PolynomialCommitmentScheme` can
/// serve as the underlying commitment engine.
pub struct JaggedCommitmentWithWitness<E: ExtensionField, InnerPcs: PolynomialCommitmentScheme<E>> {
    /// Commitment (with witness) to the "giga" polynomial `q'` via `InnerPcs`.
    pub inner: InnerPcs::CommitmentWithWitness,
    /// Cumulative height sequence `t`:
    /// - `t[0] = 0`
    /// - `t[i+1] = t[i] + poly_heights[i]`
    /// - Length: `num_polys + 1`
    pub cumulative_heights: Vec<usize>,
    /// Number of evaluations `h_i = 2^(num_vars_i)` for each polynomial `p_i`.
    /// Length: `num_polys`.
    pub poly_heights: Vec<usize>,
}

/// The pure commitment (without witness data) for a jagged polynomial `q'`.
/// This is what the verifier receives.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "", deserialize = ""))]
pub struct JaggedCommitment<E: ExtensionField, InnerPcs: PolynomialCommitmentScheme<E>> {
    /// Pure commitment to the underlying giga polynomial `q'`.
    pub inner: InnerPcs::Commitment,
    /// Cumulative height sequence `t` (verifier needs this to evaluate `f(b)`).
    pub cumulative_heights: Vec<usize>,
}

impl<E: ExtensionField, InnerPcs: PolynomialCommitmentScheme<E>>
    JaggedCommitmentWithWitness<E, InnerPcs>
{
    /// Extract the pure commitment (without witness data).
    pub fn to_commitment(&self) -> JaggedCommitment<E, InnerPcs> {
        JaggedCommitment {
            inner: InnerPcs::get_pure_commitment(&self.inner),
            cumulative_heights: self.cumulative_heights.clone(),
        }
    }

    /// Total number of polynomials packed into `q'`.
    pub fn num_polys(&self) -> usize {
        self.poly_heights.len()
    }

    /// Total number of evaluations in the *unpadded* concatenated polynomial
    /// (= `t[num_polys]` = `cumulative_heights.last()`).
    pub fn total_evaluations(&self) -> usize {
        self.cumulative_heights.last().copied().unwrap_or(0)
    }
}

/// Proof for the jagged batch opening protocol.
///
/// Contains a sumcheck proof (reducing K column evaluation claims to a single
/// point on q'), the evaluation q'(ρ), and an inner PCS opening proof.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "", deserialize = ""))]
pub struct JaggedBatchOpenProof<E: ExtensionField, InnerPcs: PolynomialCommitmentScheme<E>> {
    pub sumcheck_proof: IOPProof<E>,
    pub q_eval: E,
    pub inner_proof: InnerPcs::Proof,
}

/// Convert a `usize` to its little-endian binary representation as field elements.
pub(super) fn int_to_field_bits<E: ExtensionField>(val: usize, num_bits: usize) -> Vec<E> {
    (0..num_bits)
        .map(|i| if (val >> i) & 1 == 1 { E::ONE } else { E::ZERO })
        .collect()
}

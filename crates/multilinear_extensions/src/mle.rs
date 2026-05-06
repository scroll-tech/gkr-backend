use std::{any::TypeId, borrow::Cow, mem, ops::Range, sync::Arc};

use crate::{
    field_type_mut_map,
    macros::{entered_span, exit_span},
    op_mle,
    smart_slice::SmartSlice,
    util::{ceil_log2, largest_even_below, largest_multiple_of_four_below},
};
use either::Either;
use ff_ext::{ExtensionField, FromUniformBytes};
#[cfg(not(feature = "parallel"))]
use itertools::Itertools;
use p3::{
    field::{Field, FieldAlgebra},
    maybe_rayon::prelude::*,
};
use rand::Rng;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::fmt::Debug;

/// A point is a vector of num_var length
pub type Point<F> = Vec<F>;

/// A point and the evaluation of this point.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct PointAndEval<F> {
    pub point: Point<F>,
    pub eval: F,
}

impl<F> PointAndEval<F> {
    pub fn new(point: Point<F>, eval: F) -> Self {
        Self { point, eval }
    }
}

impl<E: ExtensionField> Debug for MultilinearExtension<'_, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}", self.evaluations())
    }
}

impl<E: ExtensionField> From<Vec<Vec<E::BaseField>>> for MultilinearExtension<'_, E> {
    fn from(val: Vec<Vec<E::BaseField>>) -> Self {
        let per_instance_size = val[0].len();
        let next_pow2_per_instance_size = ceil_log2(per_instance_size);
        let evaluations = val
            .into_iter()
            .enumerate()
            .flat_map(|(i, mut instance)| {
                assert_eq!(
                    instance.len(),
                    per_instance_size,
                    "{}th instance with length {} != {} ",
                    i,
                    instance.len(),
                    per_instance_size
                );
                instance.resize(1 << next_pow2_per_instance_size, E::BaseField::ZERO);
                instance
            })
            .collect::<Vec<E::BaseField>>();
        assert!(evaluations.len().is_power_of_two());
        let num_vars = ceil_log2(evaluations.len());
        MultilinearExtension::from_evaluations_vec(num_vars, evaluations)
    }
}

/// this is to avoid conflict implementation for Into of Vec<Vec<E::BaseField>>
pub trait IntoMLE<T>: Sized {
    /// Converts this type into the (usually inferred) input type.
    fn into_mle(self) -> T;
}

impl<'a, F: Field, E: ExtensionField> IntoMLE<MultilinearExtension<'a, E>> for Vec<F> {
    fn into_mle(self) -> MultilinearExtension<'a, E> {
        let next_pow2 = self.len().next_power_of_two();
        assert!(self.len().is_power_of_two(), "{}", self.len());
        MultilinearExtension::from_evaluation_vec_smart::<F>(ceil_log2(next_pow2), self)
    }
}
pub trait IntoMLEs<T>: Sized {
    /// Converts this type into the (usually inferred) input type.
    fn into_mles(self) -> Vec<T>;
}

impl<'a, F: Field, E: ExtensionField<BaseField = F>> IntoMLEs<MultilinearExtension<'a, E>>
    for Vec<Vec<F>>
{
    fn into_mles(self) -> Vec<MultilinearExtension<'a, E>> {
        self.into_iter().map(|v| v.into_mle()).collect()
    }
}

#[derive(Clone, Default, Eq, Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "E::BaseField: Serialize",
    deserialize = "E::BaseField: DeserializeOwned"
))]
/// Differentiate inner vector on base/extension field.
pub enum FieldType<'a, E: ExtensionField> {
    Base(SmartSlice<'a, E::BaseField>),
    Ext(SmartSlice<'a, E>),
    #[default]
    Unreachable,
}

impl<'a, E: ExtensionField> FieldType<'a, E> {
    pub fn len(&self) -> usize {
        match self {
            FieldType::Base(content) => content.len(),
            FieldType::Ext(content) => content.len(),
            FieldType::Unreachable => 0,
        }
    }

    pub fn as_borrowed_view(&self) -> Self {
        match self {
            FieldType::Base(SmartSlice::Borrowed(slice)) => {
                FieldType::Base(SmartSlice::Borrowed(slice))
            }
            FieldType::Ext(SmartSlice::Borrowed(slice)) => {
                FieldType::Ext(SmartSlice::Borrowed(slice))
            }
            invalid_type => panic!("invalid type {:?}", invalid_type),
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            FieldType::Base(content) => content.is_empty(),
            FieldType::Ext(content) => content.is_empty(),
            FieldType::Unreachable => true,
        }
    }

    pub fn variant_name(&self) -> &'static str {
        match self {
            FieldType::Base(_) => "Base",
            FieldType::Ext(_) => "Ext",
            FieldType::Unreachable => "Unreachable",
        }
    }

    pub fn is_zero(&self) -> bool {
        match self {
            FieldType::Base(content) => content.par_iter().all(|x| x.is_zero()),
            FieldType::Ext(content) => content.par_iter().all(|x| x.is_zero()),
            FieldType::Unreachable => true,
        }
    }

    pub fn zero(num_vars: usize) -> Self {
        FieldType::Base(SmartSlice::Owned(
            (0..1 << num_vars)
                .into_par_iter()
                .map(|_| E::BaseField::ZERO)
                .collect(),
        ))
    }

    pub fn constant(num_vars: usize, value: &E::BaseField) -> Self {
        FieldType::Base(SmartSlice::Owned(
            (0..1 << num_vars).into_par_iter().map(|_| *value).collect(),
        ))
    }

    pub fn sum(&self) -> FieldType<'a, E> {
        match self {
            FieldType::Base(slice) => {
                let sum = slice.par_iter().cloned().sum::<E::BaseField>();
                FieldType::Base(SmartSlice::Owned(vec![sum]))
            }
            FieldType::Ext(slice) => {
                let sum = slice.par_iter().cloned().sum::<E>();
                FieldType::Ext(SmartSlice::Owned(vec![sum]))
            }
            FieldType::Unreachable => FieldType::Unreachable,
        }
    }

    pub fn select_prefix(self, prefix_len: usize) -> Self {
        field_type_mut_map!(self, |slice| {
            slice[prefix_len..]
                .par_iter_mut()
                .for_each(|elem| *elem = Default::default());
            slice
        })
    }

    // pick indice within chunk, and fill default for others value
    pub fn pick_indices_within_chunk(
        self,
        chunk_size: usize,
        valid_chunk_index: usize,
        indices: &[usize],
    ) -> Self {
        field_type_mut_map!(self, |slice| {
            slice
                .par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(chunk_index, chunk)| {
                    if chunk_index >= valid_chunk_index {
                        // Entire chunk is invalid — fill all with default
                        chunk.fill(Default::default());
                        return;
                    }

                    // Only keep values at `indices`, zero out others
                    let mut indices_iter = indices.iter().copied();
                    let mut next_idx = indices_iter.next();

                    for (i, value) in chunk.iter_mut().enumerate() {
                        if Some(i) == next_idx {
                            next_idx = indices_iter.next(); // keep this one
                        } else {
                            *value = Default::default(); // reset others
                        }
                    }
                });
            slice
        })
    }

    #[inline(always)]
    pub fn index(&self, index: usize) -> Either<E::BaseField, E> {
        match self {
            FieldType::Base(slice) => Either::Left(slice[index]),
            FieldType::Ext(slice) => Either::Right(slice[index]),
            FieldType::Unreachable => unreachable!(),
        }
    }

    pub fn as_slice(&self, range: Range<usize>) -> Either<&[E::BaseField], &[E]> {
        match self {
            FieldType::Base(slice) => Either::Left(&slice[range]),
            FieldType::Ext(slice) => Either::Right(&slice[range]),
            FieldType::Unreachable => unreachable!(),
        }
    }
}

impl<'a, E: ExtensionField> PartialEq for FieldType<'a, E> {
    /// compares the contents of two slices
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (FieldType::Base(a), FieldType::Base(b)) => a == b,
            (FieldType::Ext(a), FieldType::Ext(b)) => a == b,
            (FieldType::Base(a), FieldType::Ext(b)) | (FieldType::Ext(b), FieldType::Base(a)) => a
                .par_iter()
                .zip_eq(b.par_iter())
                .all(|(a, b)| E::from_base(*a) == *b),
            _ => self.is_zero() && other.is_zero(),
        }
    }
}

impl<'a, E: ExtensionField> IntoMLE<MultilinearExtension<'a, E>> for FieldType<'a, E> {
    fn into_mle(self) -> MultilinearExtension<'a, E> {
        let num_vars = ceil_log2(self.len());
        MultilinearExtension {
            evaluations: self,
            num_vars,
        }
    }
}

/// Stores a multilinear polynomial in dense evaluation form.
#[derive(Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(bound(
    serialize = "E::BaseField: Serialize",
    deserialize = "E::BaseField: DeserializeOwned"
))]
pub struct MultilinearExtension<'a, E: ExtensionField> {
    /// The evaluation over {0,1}^`num_vars`
    pub evaluations: FieldType<'a, E>,
    /// Number of variables
    pub num_vars: usize,
}

pub type ArcMultilinearExtension<'a, E> = Arc<MultilinearExtension<'a, E>>;

fn cast_vec<A, B>(mut vec: Vec<A>) -> Vec<B> {
    let length = vec.len();
    let capacity = vec.capacity();
    let ptr = vec.as_mut_ptr();
    // Prevent `vec` from dropping its contents
    mem::forget(vec);

    // Convert the pointer to the new type
    let new_ptr = ptr as *mut B;

    // Create a new vector with the same length and capacity, but different type
    unsafe { Vec::from_raw_parts(new_ptr, length, capacity) }
}

macro_rules! split_eval_chunks {
    ($chunk_fn:ident, $smart_variant:ident, $variant:ident, $slice:expr, $chunk_size:expr, $num_vars:expr) => {
        $slice
            .$chunk_fn($chunk_size)
            .map(|chunk| MultilinearExtension {
                evaluations: FieldType::$variant(SmartSlice::$smart_variant(chunk)),
                num_vars: $num_vars,
            })
            .collect::<Vec<_>>()
    };
}

impl<'a, E: ExtensionField> MultilinearExtension<'a, E> {
    #[inline(always)]
    fn assert_occupied_len(num_vars: usize, occupied_len: usize) {
        let max_len = 1usize << num_vars;
        assert!(
            occupied_len > 0 && occupied_len <= max_len,
            "occupied length {} exceeds logical length {} for num_vars {}",
            occupied_len,
            max_len,
            num_vars
        );
    }

    /// Returns `Right(&mut self)` if mutable access is possible, otherwise `Left(&self)`
    pub fn to_either(&mut self) -> Either<&Self, &mut Self> {
        if self.is_mut() {
            Either::Right(self)
        } else {
            Either::Left(self)
        }
    }

    /// returns true if the evaluations are either mutably borrowed or owned (i.e., mutable access is possible)
    pub fn is_mut(&self) -> bool {
        match &self.evaluations {
            FieldType::Base(slice) => {
                matches!(slice, SmartSlice::BorrowedMut(_) | SmartSlice::Owned(_))
            }
            FieldType::Ext(slice) => {
                matches!(slice, SmartSlice::BorrowedMut(_) | SmartSlice::Owned(_))
            }
            FieldType::Unreachable => false,
        }
    }

    /// This function can tell T being Field or ExtensionField and invoke respective function
    pub fn from_evaluation_vec_smart<T: Clone + 'static>(
        num_vars: usize,
        evaluations: Vec<T>,
    ) -> Self {
        if TypeId::of::<T>() == TypeId::of::<E>() {
            return Self::from_evaluations_ext_vec(num_vars, cast_vec(evaluations));
        }

        if TypeId::of::<T>() == TypeId::of::<E::BaseField>() {
            return Self::from_evaluations_vec(num_vars, cast_vec(evaluations));
        }

        unimplemented!("type not support")
    }

    pub fn from_evaluation_vec_smart_compact<T: Clone + 'static>(
        num_vars: usize,
        evaluations: Vec<T>,
    ) -> Self {
        if TypeId::of::<T>() == TypeId::of::<E>() {
            return Self::from_evaluations_ext_vec_compact(num_vars, cast_vec(evaluations));
        }

        if TypeId::of::<T>() == TypeId::of::<E::BaseField>() {
            return Self::from_evaluations_vec_compact(num_vars, cast_vec(evaluations));
        }

        unimplemented!("type not support")
    }

    /// Create vector from field type
    pub fn from_field_type(num_vars: usize, field_type: FieldType<'a, E>) -> Self {
        Self::assert_occupied_len(num_vars, field_type.len());
        Self {
            num_vars,
            evaluations: field_type,
        }
    }

    /// Create vector from field type
    pub fn from_field_type_borrowed(num_vars: usize, field_type: &FieldType<'a, E>) -> Self {
        Self::assert_occupied_len(num_vars, field_type.len());
        Self {
            num_vars,
            evaluations: field_type.as_borrowed_view(),
        }
    }

    /// Construct a new polynomial from a list of evaluations where the index
    /// represents a point in {0,1}^`num_vars` in little endian form. For
    /// example, `0b1011` represents `P(1,1,0,1)`
    pub fn from_evaluations_slice(num_vars: usize, evaluations: &'a [E::BaseField]) -> Self {
        assert_eq!(
            evaluations.len(),
            1 << num_vars,
            "The size of evaluations should be 2^num_vars."
        );

        Self {
            num_vars,
            evaluations: FieldType::Base(SmartSlice::Borrowed(evaluations)),
        }
    }

    /// Construct a new polynomial from a list of evaluations where the index
    /// represents a point in {0,1}^`num_vars` in little endian form. For
    /// example, `0b1011` represents `P(1,1,0,1)`
    pub fn from_evaluations_vec(num_vars: usize, evaluations: Vec<E::BaseField>) -> Self {
        // assert that the number of variables matches the size of evaluations
        assert_eq!(
            evaluations.len(),
            1 << num_vars,
            "The size of evaluations should be 2^num_vars."
        );

        Self {
            num_vars,
            evaluations: FieldType::Base(SmartSlice::Owned(evaluations)),
        }
    }

    pub fn from_evaluations_vec_compact(num_vars: usize, evaluations: Vec<E::BaseField>) -> Self {
        Self::assert_occupied_len(num_vars, evaluations.len());

        Self {
            num_vars,
            evaluations: FieldType::Base(SmartSlice::Owned(evaluations)),
        }
    }

    /// Identical to [`from_evaluations_slice`], with and exception that evaluation vector is in
    /// extension field
    pub fn from_evaluations_ext_slice(num_vars: usize, evaluations: &'a [E]) -> Self {
        assert_eq!(
            evaluations.len(),
            1 << num_vars,
            "The size of evaluations should be 2^num_vars."
        );

        Self {
            num_vars,
            evaluations: FieldType::Ext(SmartSlice::Borrowed(evaluations)),
        }
    }

    /// Identical to [`from_evaluations_vec`], with and exception that evaluation vector is in
    /// extension field
    pub fn from_evaluations_ext_vec(num_vars: usize, evaluations: Vec<E>) -> Self {
        // assert that the number of variables matches the size of evaluations
        assert_eq!(
            evaluations.len(),
            1 << num_vars,
            "The size of evaluations should be 2^num_vars."
        );

        Self {
            num_vars,
            evaluations: FieldType::Ext(SmartSlice::Owned(evaluations)),
        }
    }

    pub fn from_evaluations_ext_vec_compact(num_vars: usize, evaluations: Vec<E>) -> Self {
        Self::assert_occupied_len(num_vars, evaluations.len());

        Self {
            num_vars,
            evaluations: FieldType::Ext(SmartSlice::Owned(evaluations)),
        }
    }

    /// Generate a random evaluation of a multilinear poly
    pub fn random<R: Rng>(nv: usize, rng: &mut R) -> Self {
        let eval = (0..1 << nv)
            .map(|_| E::BaseField::random(&mut *rng))
            .collect();
        MultilinearExtension::from_evaluations_vec(nv, eval)
    }

    /// Sample a random list of multilinear polynomials.
    /// Returns
    /// - the list of polynomials,
    /// - its sum of polynomial evaluations over the boolean hypercube.
    pub fn random_mle_list<R: Rng>(
        nv: usize,
        degree: usize,
        rng: &mut R,
    ) -> (Vec<MultilinearExtension<'a, E>>, E) {
        let start = entered_span!("sample random mle list");
        let mut multiplicands = Vec::with_capacity(degree);
        for _ in 0..degree {
            multiplicands.push(Vec::with_capacity(1 << nv))
        }
        let mut sum = E::ZERO;

        for _ in 0..(1 << nv) {
            let mut product = E::ONE;

            for e in multiplicands.iter_mut() {
                let val = E::BaseField::random(&mut *rng);
                e.push(val);
                product *= val
            }
            sum += product;
        }

        let list = multiplicands
            .into_iter()
            .map(|x| MultilinearExtension::from_evaluations_vec(nv, x))
            .collect();

        exit_span!(start);
        (list, sum)
    }

    // Build a randomize list of mle-s whose sum is zero.
    pub fn random_zero_mle_list<R: Rng>(
        nv: usize,
        degree: usize,
        rng: &mut R,
    ) -> Vec<ArcMultilinearExtension<'_, E>> {
        let start = entered_span!("sample random zero mle list");

        let mut multiplicands = Vec::with_capacity(degree);
        for _ in 0..degree {
            multiplicands.push(Vec::with_capacity(1 << nv))
        }
        for _ in 0..(1 << nv) {
            multiplicands[0].push(E::BaseField::ZERO);
            for e in multiplicands.iter_mut().skip(1) {
                e.push(E::BaseField::random(&mut *rng));
            }
        }

        let list = multiplicands
            .into_iter()
            .map(|x| MultilinearExtension::from_evaluations_vec(nv, x).into())
            .collect();

        exit_span!(start);
        list
    }

    pub fn fix_variables(&self, partial_point: &[E]) -> Self {
        // TODO: return error.
        assert!(
            partial_point.len() <= self.num_vars(),
            "invalid size of partial point"
        );
        let mut poly = Cow::Borrowed(self);

        // evaluate single variable of partial point from left to right
        // `Cow` type here to skip first evaluation vector copy
        for point in partial_point.iter() {
            match &mut poly {
                poly @ Cow::Borrowed(_) => {
                    *poly = Cow::Owned(match self.evaluations() {
                        FieldType::Base(evaluations) => {
                            let pair_len = largest_even_below(evaluations.len());
                            let mut folded = evaluations[..pair_len]
                                .chunks_exact(2)
                                .map(|buf| Self::eval_pair_base(buf[0], buf[1], *point))
                                .collect::<Vec<_>>();
                            if pair_len < evaluations.len() {
                                folded
                                    .push(Self::eval_pair_base_tail(evaluations[pair_len], *point));
                            }
                            MultilinearExtension::from_evaluations_ext_vec_compact(
                                self.num_vars() - 1,
                                folded,
                            )
                        }
                        FieldType::Ext(evaluations) => {
                            let pair_len = largest_even_below(evaluations.len());
                            let mut folded = evaluations[..pair_len]
                                .chunks_exact(2)
                                .map(|buf| Self::eval_pair_ext(buf[0], buf[1], *point))
                                .collect::<Vec<_>>();
                            if pair_len < evaluations.len() {
                                folded
                                    .push(Self::eval_pair_ext_tail(evaluations[pair_len], *point));
                            }
                            MultilinearExtension::from_evaluations_ext_vec_compact(
                                self.num_vars() - 1,
                                folded,
                            )
                        }
                        FieldType::Unreachable => unreachable!(),
                    });
                }
                Cow::Owned(poly) => poly.fix_variables_in_place(&[*point]),
            }
        }
        assert!(poly.num_vars == self.num_vars() - partial_point.len(),);
        poly.into_owned()
    }

    /// Reduce the number of variables of `self` by fixing the
    /// `partial_point.len()` variables at `partial_point` in place
    pub fn fix_variables_in_place(&mut self, partial_point: &[E]) {
        assert!(self.is_mut());
        assert!(
            partial_point.len() <= self.num_vars(),
            "partial point len {} >= num_vars {}",
            partial_point.len(),
            self.num_vars()
        );
        let nv = self.num_vars();
        let mut new_len = self.occupied_len();
        // evaluate single variable of partial point from left to right
        for point in partial_point.iter() {
            // override buf[b1, b2,..bt, 0] = (1-point) * buf[b1, b2,..bt, 0] + point * buf[b1,
            // b2,..bt, 1]
            match &mut self.evaluations {
                FieldType::Base(slice) => {
                    let pair_len = largest_even_below(slice.len());
                    let mut slice_ext = slice[..pair_len]
                        .chunks_exact(2)
                        .map(|buf| Self::eval_pair_base(buf[0], buf[1], *point))
                        .collect::<Vec<_>>();
                    if pair_len < slice.len() {
                        slice_ext.push(Self::eval_pair_base_tail(slice[pair_len], *point));
                    }
                    let _ = mem::replace(
                        &mut self.evaluations,
                        FieldType::Ext(SmartSlice::Owned(slice_ext)),
                    );
                }
                FieldType::Ext(slice) => {
                    let slice_mut = slice.to_mut();
                    let pair_len = largest_even_below(slice_mut.len());
                    (0..pair_len).step_by(2).for_each(|b| {
                        slice_mut[b >> 1] =
                            slice_mut[b] + (slice_mut[b + 1] - slice_mut[b]) * *point
                    });
                    if pair_len < slice_mut.len() {
                        let lo = slice_mut[pair_len];
                        slice_mut[pair_len >> 1] = lo + (E::ZERO - lo) * *point;
                    }
                }
                FieldType::Unreachable => unreachable!(),
            };
            new_len = new_len.div_ceil(2);
        }
        match &mut self.evaluations {
            FieldType::Base(_) => unreachable!(),
            FieldType::Ext(slice) => {
                slice.truncate_mut(new_len);
            }
            FieldType::Unreachable => unreachable!(),
        }

        self.num_vars = nv - partial_point.len();
    }

    // compute f(r0,r1,b) from block { f(0,0,b), f(1,0,b), f(0,1,b), f(1,1,b) }
    #[inline(always)]
    fn eval_block_2_vars_base(block: &[E::BaseField], r0: E, r1: E) -> E {
        // f(r0,r1,b) = (1 - r1) * f(r0,0,b) + r1 * f(r0,1,b)
        // f(r0,0,b) = (1 - r0) * f(0,0,b) + r0 * f(1,0,b)
        // f(r0,1,b) = (1 - r0) * f(0,1,b) + r0 * f(1,1,b)
        let y0: E = r0 * (block[1] - block[0]) + block[0];
        let y1: E = r0 * (block[3] - block[2]) + block[2];
        y0 + (y1 - y0) * r1
    }

    #[inline(always)]
    fn eval_block_2_vars_ext(block: &[E], r0: E, r1: E) -> E {
        // f(r0,r1,b) = (1 - r1) * f(r0,0,b) + r1 * f(r0,1,b)
        // f(r0,0,b) = (1 - r0) * f(0,0,b) + r0 * f(1,0,b)
        // f(r0,1,b) = (1 - r0) * f(0,1,b) + r0 * f(1,1,b)
        let y0: E = block[0] + (block[1] - block[0]) * r0;
        let y1: E = block[2] + (block[3] - block[2]) * r0;
        y0 + (y1 - y0) * r1
    }

    #[inline(always)]
    fn eval_pair_base(lo: E::BaseField, hi: E::BaseField, point: E) -> E {
        point * (hi - lo) + lo
    }

    #[inline(always)]
    fn eval_pair_base_tail(lo: E::BaseField, point: E) -> E {
        E::from_base(lo) + (E::ZERO - E::from_base(lo)) * point
    }

    #[inline(always)]
    fn eval_pair_ext(lo: E, hi: E, point: E) -> E {
        lo + (hi - lo) * point
    }

    #[inline(always)]
    fn eval_pair_ext_tail(lo: E, point: E) -> E {
        lo + (E::ZERO - lo) * point
    }

    #[inline(always)]
    fn eval_block_2_vars_base_partial(block: &[E::BaseField], r0: E, r1: E) -> E {
        let v0 = block.first().copied().map(E::from).unwrap_or(E::ZERO);
        let v1 = block.get(1).copied().map(E::from).unwrap_or(E::ZERO);
        let v2 = block.get(2).copied().map(E::from).unwrap_or(E::ZERO);
        let v3 = block.get(3).copied().map(E::from).unwrap_or(E::ZERO);
        let y0 = v0 + (v1 - v0) * r0;
        let y1 = v2 + (v3 - v2) * r0;
        y0 + (y1 - y0) * r1
    }

    #[inline(always)]
    fn eval_block_2_vars_ext_partial(block: &[E], r0: E, r1: E) -> E {
        let v0 = block.first().copied().unwrap_or(E::ZERO);
        let v1 = block.get(1).copied().unwrap_or(E::ZERO);
        let v2 = block.get(2).copied().unwrap_or(E::ZERO);
        let v3 = block.get(3).copied().unwrap_or(E::ZERO);
        let y0 = v0 + (v1 - v0) * r0;
        let y1 = v2 + (v3 - v2) * r0;
        y0 + (y1 - y0) * r1
    }

    /// Reduce the number of variables by 2 in one pass.
    ///
    /// This avoids calling `fix_variables` twice and directly computes
    /// `f(r0, r1, ..)` from 4-point blocks.
    pub fn fix_two_variables(&self, r0: E, r1: E) -> Self {
        assert!(self.num_vars() >= 2, "num_vars {} < 2", self.num_vars());
        let nv = self.num_vars();
        match self.evaluations() {
            FieldType::Base(slice) => {
                let quad_len = largest_multiple_of_four_below(slice.len());
                let mut folded = slice[..quad_len]
                    .chunks_exact(4)
                    .map(|buf| Self::eval_block_2_vars_base(buf, r0, r1))
                    .collect::<Vec<_>>();
                if quad_len < slice.len() {
                    folded.push(Self::eval_block_2_vars_base_partial(
                        &slice[quad_len..],
                        r0,
                        r1,
                    ));
                }
                MultilinearExtension::from_evaluations_ext_vec_compact(nv - 2, folded)
            }
            FieldType::Ext(slice) => {
                let quad_len = largest_multiple_of_four_below(slice.len());
                let mut folded = slice[..quad_len]
                    .chunks_exact(4)
                    .map(|buf| Self::eval_block_2_vars_ext(buf, r0, r1))
                    .collect::<Vec<_>>();
                if quad_len < slice.len() {
                    folded.push(Self::eval_block_2_vars_ext_partial(
                        &slice[quad_len..],
                        r0,
                        r1,
                    ));
                }
                MultilinearExtension::from_evaluations_ext_vec_compact(nv - 2, folded)
            }
            FieldType::Unreachable => unreachable!(),
        }
    }

    /// In-place variant of `fix_two_variables`.
    pub fn fix_two_variables_in_place(&mut self, r0: E, r1: E) {
        assert!(self.is_mut());
        assert!(self.num_vars() >= 2, "num_vars {} < 2", self.num_vars());
        let nv = self.num_vars();

        match &mut self.evaluations {
            FieldType::Base(slice) => {
                let quad_len = largest_multiple_of_four_below(slice.len());
                let mut ext_vec = slice[..quad_len]
                    .chunks_exact(4)
                    .map(|buf| Self::eval_block_2_vars_base(buf, r0, r1))
                    .collect::<Vec<_>>();
                if quad_len < slice.len() {
                    ext_vec.push(Self::eval_block_2_vars_base_partial(
                        &slice[quad_len..],
                        r0,
                        r1,
                    ));
                }
                let _ = std::mem::replace(
                    &mut self.evaluations,
                    FieldType::Ext(SmartSlice::Owned(ext_vec)),
                );
            }
            FieldType::Ext(slice) => {
                let slice_mut = slice.to_mut();
                let quad_len = largest_multiple_of_four_below(slice_mut.len());
                let new_len = slice_mut.len().div_ceil(4);
                for i in 0..(quad_len >> 2) {
                    let b = i << 2;
                    slice_mut[i] = Self::eval_block_2_vars_ext(&slice_mut[b..b + 4], r0, r1);
                }
                if quad_len < slice_mut.len() {
                    slice_mut[quad_len >> 2] =
                        Self::eval_block_2_vars_ext_partial(&slice_mut[quad_len..], r0, r1);
                }
                slice.truncate_mut(new_len);
            }
            FieldType::Unreachable => unreachable!(),
        }

        self.num_vars = nv - 2;
    }

    /// Evaluate the MLE at a give point.
    /// Returns an error if the MLE length does not match the point.
    pub fn evaluate(&self, point: &[E]) -> E {
        // TODO: return error.
        assert_eq!(
            self.num_vars(),
            point.len(),
            "MLE size does not match the point"
        );
        let mle = self.fix_variables_parallel(point);
        op_mle!(
            mle,
            |f| {
                assert_eq!(f.len(), 1);
                f[0]
            },
            |v| E::from(v)
        )
    }

    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn occupied_len(&self) -> usize {
        self.evaluations.len()
    }

    pub fn full_evaluation_len(&self) -> usize {
        1 << self.num_vars
    }

    pub fn has_full_occupancy(&self) -> bool {
        self.occupied_len() == self.full_evaluation_len()
    }

    /// Reduce the number of variables of `self` by fixing the
    /// `partial_point.len()` variables at `partial_point`.
    pub fn fix_variables_parallel(&self, partial_point: &[E]) -> Self {
        // TODO: return error.
        assert!(
            partial_point.len() <= self.num_vars(),
            "invalid size of partial point"
        );
        let mut poly = Cow::Borrowed(self);

        // evaluate single variable of partial point from left to right
        // `Cow` type here to skip first evaluation vector copy
        for point in partial_point.iter() {
            match &mut poly {
                poly @ Cow::Borrowed(_) => {
                    *poly = Cow::Owned(match self.evaluations() {
                        FieldType::Base(evaluations) => {
                            let pair_len = largest_even_below(evaluations.len());
                            let mut folded = evaluations[..pair_len]
                                .par_chunks_exact(2)
                                .map(|buf| Self::eval_pair_base(buf[0], buf[1], *point))
                                .collect::<Vec<_>>();
                            if pair_len < evaluations.len() {
                                folded
                                    .push(Self::eval_pair_base_tail(evaluations[pair_len], *point));
                            }
                            MultilinearExtension::from_evaluations_ext_vec_compact(
                                self.num_vars() - 1,
                                folded,
                            )
                        }
                        FieldType::Ext(evaluations) => {
                            let pair_len = largest_even_below(evaluations.len());
                            let mut folded = evaluations[..pair_len]
                                .par_chunks_exact(2)
                                .map(|buf| Self::eval_pair_ext(buf[0], buf[1], *point))
                                .collect::<Vec<_>>();
                            if pair_len < evaluations.len() {
                                folded
                                    .push(Self::eval_pair_ext_tail(evaluations[pair_len], *point));
                            }
                            MultilinearExtension::from_evaluations_ext_vec_compact(
                                self.num_vars() - 1,
                                folded,
                            )
                        }
                        FieldType::Unreachable => unreachable!(),
                    });
                }
                Cow::Owned(poly) => poly.fix_variables_in_place_parallel(&[*point]),
            }
        }
        assert!(poly.num_vars == self.num_vars() - partial_point.len(),);
        poly.into_owned()
    }

    /// Reduce the number of variables of `self` by fixing the
    /// `partial_point.len()` variables at `partial_point` in place
    pub fn fix_variables_in_place_parallel(&mut self, partial_point: &[E]) {
        assert!(self.is_mut());
        assert!(
            partial_point.len() <= self.num_vars(),
            "partial point len {} >= num_vars {}",
            partial_point.len(),
            self.num_vars()
        );
        let nv = self.num_vars();
        let mut new_len = self.occupied_len();
        // evaluate single variable of partial point from left to right
        for point in partial_point.iter() {
            // override buf[b1, b2,..bt, 0] = (1-point) * buf[b1, b2,..bt, 0] + point * buf[b1, b2,..bt, 1] in parallel
            match &mut self.evaluations {
                FieldType::Base(slice) => {
                    let pair_len = largest_even_below(slice.len());
                    let mut slice_ext = slice[..pair_len]
                        .par_chunks_exact(2)
                        .map(|buf| Self::eval_pair_base(buf[0], buf[1], *point))
                        .collect::<Vec<_>>();
                    if pair_len < slice.len() {
                        slice_ext.push(Self::eval_pair_base_tail(slice[pair_len], *point));
                    }
                    let _ = mem::replace(
                        &mut self.evaluations,
                        FieldType::Ext(SmartSlice::Owned(slice_ext)),
                    );
                }
                FieldType::Ext(slice) => {
                    let slice_mut = slice.to_mut();
                    let pair_len = largest_even_below(slice_mut.len());
                    slice_mut[..pair_len]
                        .par_chunks_mut(2)
                        .for_each(|buf| buf[0] = buf[0] + (buf[1] - buf[0]) * *point);

                    // sequentially update buf[b1, b2,..bt] = buf[b1, b2,..bt, 0]
                    for index in 0..(pair_len >> 1) {
                        slice_mut[index] = slice_mut[index << 1];
                    }
                    if pair_len < slice_mut.len() {
                        let lo = slice_mut[pair_len];
                        slice_mut[pair_len >> 1] = lo + (E::ZERO - lo) * *point;
                    }
                }
                FieldType::Unreachable => unreachable!(),
            };
            new_len = new_len.div_ceil(2);
        }
        match &mut self.evaluations {
            FieldType::Base(_) => unreachable!(),
            FieldType::Ext(slice) => {
                slice.truncate_mut(new_len);
            }
            FieldType::Unreachable => unreachable!(),
        }

        self.num_vars = nv - partial_point.len();
    }

    pub fn evaluations(&self) -> &FieldType<'_, E> {
        &self.evaluations
    }

    pub fn as_evaluations_view(&self) -> FieldType<'_, E> {
        self.evaluations.as_borrowed_view()
    }

    pub fn evaluations_to_owned(self) -> FieldType<'a, E> {
        self.evaluations
    }

    pub fn merge(&mut self, rhs: MultilinearExtension<'a, E>) {
        let rhs_num_vars = rhs.num_vars;

        // Take owned version of RHS evaluations
        let rhs_eval = rhs.evaluations_to_owned();

        match (&mut self.evaluations, rhs_eval) {
            (FieldType::Base(e1), FieldType::Base(e2)) => {
                e1.extend(e2.to_vec());
                self.num_vars = ceil_log2(e1.len());
            }
            (FieldType::Ext(e1), FieldType::Ext(e2)) => {
                e1.extend(e2.to_vec());
                self.num_vars = ceil_log2(e1.len());
            }
            (FieldType::Unreachable, b @ FieldType::Base(..)) => {
                self.num_vars = rhs_num_vars;
                self.evaluations = b;
            }
            (FieldType::Unreachable, b @ FieldType::Ext(..)) => {
                self.num_vars = rhs_num_vars;
                self.evaluations = b;
            }
            (a, b) => panic!(
                "do not support merging different field types: a = {:?}, b = {:?}",
                a, b
            ),
        }
    }

    pub fn as_view(&self) -> MultilinearExtension<'_, E> {
        self.as_view_slice(1, 0)
    }

    /// get mle with arbitrary start end
    pub fn as_view_slice(
        &self,
        num_chunks: usize,
        chunk_index: usize,
    ) -> MultilinearExtension<'_, E> {
        let total_len = self.evaluations.len();
        let chunk_size = total_len / num_chunks;
        assert!(
            num_chunks > 0
                && total_len.is_multiple_of(num_chunks)
                && chunk_size > 0
                && chunk_index < num_chunks,
            "invalid num_chunks: {num_chunks} total_len: {total_len}, chunk_index {chunk_index} parameter set"
        );
        let start = chunk_size * chunk_index;

        let sub_evaluations = match &self.evaluations {
            FieldType::Base(slice) => {
                FieldType::Base(SmartSlice::Borrowed(&slice[start..][..chunk_size]))
            }
            FieldType::Ext(slice) => {
                FieldType::Ext(SmartSlice::Borrowed(&slice[start..][..chunk_size]))
            }
            FieldType::Unreachable => FieldType::Unreachable,
        };

        MultilinearExtension {
            evaluations: sub_evaluations,
            num_vars: self.num_vars - num_chunks.trailing_zeros() as usize,
        }
    }

    /// get mutable view
    pub fn as_view_mut(&mut self) -> MultilinearExtension<'_, E> {
        self.as_view_slice_mut(1, 0)
    }

    pub fn as_view_slice_mut(
        &mut self,
        num_chunks: usize,
        chunk_index: usize,
    ) -> MultilinearExtension<'_, E> {
        let total_len = self.evaluations.len();
        let chunk_size = total_len / num_chunks;
        assert!(
            num_chunks > 0
                && total_len.is_multiple_of(num_chunks)
                && chunk_size > 0
                && chunk_index < num_chunks,
            "invalid num_chunks: {num_chunks} total_len: {total_len}, chunk_index {chunk_index} parameter set"
        );
        let start = chunk_size * chunk_index;

        let sub_evaluations = match &mut self.evaluations {
            FieldType::Base(SmartSlice::BorrowedMut(slice)) => {
                let slice = &mut slice[start..][..chunk_size];
                FieldType::Base(SmartSlice::BorrowedMut(slice))
            }
            FieldType::Ext(SmartSlice::BorrowedMut(slice)) => {
                FieldType::Ext(SmartSlice::BorrowedMut(&mut slice[start..][..chunk_size]))
            }
            FieldType::Base(SmartSlice::Owned(slice)) => {
                FieldType::Base(SmartSlice::BorrowedMut(&mut slice[start..][..chunk_size]))
            }
            FieldType::Ext(SmartSlice::Owned(slice)) => {
                FieldType::Ext(SmartSlice::BorrowedMut(&mut slice[start..][..chunk_size]))
            }
            _ => unimplemented!("Unsupported variant"),
        };

        MultilinearExtension {
            evaluations: sub_evaluations,
            num_vars: self.num_vars - num_chunks.trailing_zeros() as usize,
        }
    }

    pub fn get_ext_field_vec(&self) -> &[E] {
        match &self.evaluations() {
            FieldType::Ext(slice) => slice.as_ref(),
            _ => panic!("evaluation not in extension field"),
        }
    }

    pub fn get_base_field_vec(&self) -> &[E::BaseField] {
        match &self.evaluations() {
            FieldType::Base(slice) => slice.as_ref(),
            _ => panic!("evaluation not in base field"),
        }
    }

    /// splits the MLE into `num_chunks` parts, where each part contains disjoint mutable pointers
    /// to the original data (either borrowed mutably or owned).
    pub fn as_view_chunks_mut(&'a mut self, num_chunks: usize) -> Vec<MultilinearExtension<'a, E>> {
        let total_len = self.evaluations.len();
        let chunk_size = total_len / num_chunks;
        assert!(
            num_chunks > 0 && total_len.is_multiple_of(num_chunks) && chunk_size > 0,
            "invalid num_chunks: {num_chunks} total_len: {total_len} parameter set"
        );
        let num_vars_per_chunk = self.num_vars - ceil_log2(num_chunks);

        match &mut self.evaluations {
            FieldType::Base(SmartSlice::BorrowedMut(slice)) => {
                split_eval_chunks!(
                    chunks_mut,
                    BorrowedMut,
                    Base,
                    slice,
                    chunk_size,
                    num_vars_per_chunk
                )
            }
            FieldType::Ext(SmartSlice::BorrowedMut(slice)) => {
                split_eval_chunks!(
                    chunks_mut,
                    BorrowedMut,
                    Ext,
                    slice,
                    chunk_size,
                    num_vars_per_chunk
                )
            }
            FieldType::Base(SmartSlice::Owned(slice)) => {
                split_eval_chunks!(
                    chunks_mut,
                    BorrowedMut,
                    Base,
                    slice,
                    chunk_size,
                    num_vars_per_chunk
                )
            }
            FieldType::Ext(SmartSlice::Owned(slice)) => {
                split_eval_chunks!(
                    chunks_mut,
                    BorrowedMut,
                    Ext,
                    slice,
                    chunk_size,
                    num_vars_per_chunk
                )
            }
            e => {
                panic!(
                    "unsupported {:?}. can only split when evaluations are mutably borrowed",
                    e
                );
            }
        }
    }

    /// immutable counterpart to [`as_view_chunks_mut`]
    pub fn as_view_chunks<'b>(&'a self, num_chunks: usize) -> Vec<MultilinearExtension<'b, E>>
    where
        'a: 'b,
    {
        let total_len = self.evaluations.len();
        let chunk_size = total_len / num_chunks;
        assert!(
            num_chunks > 0 && total_len.is_multiple_of(num_chunks) && chunk_size > 0,
            "invalid num_chunks: {num_chunks} total_len: {total_len} parameter set"
        );
        let num_vars_per_chunk = self.num_vars - ceil_log2(num_chunks);

        match &self.evaluations {
            FieldType::Base(SmartSlice::Borrowed(slice)) => {
                split_eval_chunks!(
                    chunks,
                    Borrowed,
                    Base,
                    slice,
                    chunk_size,
                    num_vars_per_chunk
                )
            }
            FieldType::Ext(SmartSlice::Borrowed(slice)) => {
                split_eval_chunks!(chunks, Borrowed, Ext, slice, chunk_size, num_vars_per_chunk)
            }
            FieldType::Base(SmartSlice::BorrowedMut(slice)) => {
                split_eval_chunks!(
                    chunks,
                    Borrowed,
                    Base,
                    slice,
                    chunk_size,
                    num_vars_per_chunk
                )
            }
            FieldType::Ext(SmartSlice::BorrowedMut(slice)) => {
                split_eval_chunks!(chunks, Borrowed, Ext, slice, chunk_size, num_vars_per_chunk)
            }
            FieldType::Base(SmartSlice::Owned(slice)) => {
                split_eval_chunks!(
                    chunks,
                    Borrowed,
                    Base,
                    slice,
                    chunk_size,
                    num_vars_per_chunk
                )
            }
            FieldType::Ext(SmartSlice::Owned(slice)) => {
                split_eval_chunks!(chunks, Borrowed, Ext, slice, chunk_size, num_vars_per_chunk)
            }
            e => {
                panic!(
                    "unsupported {:?}. can only split when evaluations are mutably borrowed",
                    e
                );
            }
        }
    }

    pub fn as_owned(&self) -> Self {
        let owned_eval = match &self.evaluations {
            FieldType::Base(slice) => FieldType::Base(SmartSlice::Owned(slice.to_vec())),
            FieldType::Ext(slice) => FieldType::Ext(SmartSlice::Owned(slice.to_vec())),
            FieldType::Unreachable => FieldType::Unreachable,
        };
        MultilinearExtension {
            evaluations: owned_eval,
            num_vars: self.num_vars,
        }
    }

    #[inline(always)]
    pub fn index(&self, index: usize) -> Either<E::BaseField, E> {
        self.evaluations.index(index)
    }

    /// encode evaluations vector into a position-sensitive extension field scalar
    pub fn bh_signature(&self) -> E {
        match &self.evaluations() {
            FieldType::Base(slice) => E::from(
                slice
                    .iter()
                    .enumerate()
                    .map(|(i, v)| E::BaseField::from_canonical_u32(i as u32 + 1) + *v)
                    .product::<E::BaseField>(),
            ),
            FieldType::Ext(slice) => slice
                .iter()
                .enumerate()
                .map(|(i, v)| E::from_canonical_u32(i as u32 + 1) + *v)
                .product::<E>(),
            _ => unreachable!(),
        }
    }
}

#[allow(clippy::wrong_self_convention)]
pub trait IntoInstanceIter<'a, T> {
    type Item;
    type IntoIter: Iterator<Item = Self::Item>;
    fn into_instance_iter(&self, n_instances: usize) -> Self::IntoIter;
}

#[allow(clippy::wrong_self_convention)]
pub trait IntoInstanceIterMut<'a, T> {
    type ItemMut;
    type IntoIterMut: Iterator<Item = Self::ItemMut>;
    fn into_instance_iter_mut(&'a mut self, n_instances: usize) -> Self::IntoIterMut;
}

pub struct InstanceIntoIterator<'a, T> {
    pub evaluations: &'a [T],
    pub start: usize,
    pub offset: usize,
}

pub struct InstanceIntoIteratorMut<'a, T> {
    pub evaluations: &'a mut [T],
    pub start: usize,
    pub offset: usize,
    pub origin_len: usize,
}

impl<'a, T> Iterator for InstanceIntoIterator<'a, T> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.start >= self.evaluations.len() {
            None
        } else {
            let next = &self.evaluations[self.start..][..self.offset];
            self.start += self.offset;
            Some(next)
        }
    }
}

impl<'a, T> Iterator for InstanceIntoIteratorMut<'a, T> {
    type Item = &'a mut [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.start >= self.origin_len {
            None
        } else {
            let evaluation = mem::take(&mut self.evaluations);
            let (head, tail) = evaluation.split_at_mut(self.offset);
            self.evaluations = tail;
            self.start += self.offset;
            Some(head)
        }
    }
}

impl<'a, T> IntoInstanceIter<'a, T> for &'a [T] {
    type Item = &'a [T];
    type IntoIter = InstanceIntoIterator<'a, T>;

    fn into_instance_iter(&self, n_instances: usize) -> Self::IntoIter {
        assert!(self.len().is_multiple_of(n_instances));
        let offset = self.len() / n_instances;
        InstanceIntoIterator {
            evaluations: self,
            start: 0,
            offset,
        }
    }
}

impl<'a, T: 'a> IntoInstanceIterMut<'a, T> for Vec<T> {
    type ItemMut = &'a mut [T];
    type IntoIterMut = InstanceIntoIteratorMut<'a, T>;

    fn into_instance_iter_mut<'b>(&'a mut self, n_instances: usize) -> Self::IntoIterMut {
        assert!(self.len().is_multiple_of(n_instances));
        let offset = self.len() / n_instances;
        let origin_len = self.len();
        InstanceIntoIteratorMut {
            evaluations: self,
            start: 0,
            offset,
            origin_len,
        }
    }
}

#[macro_export]
macro_rules! op_mle {
    ($a:ident, |$tmp_a:ident| $op:expr, |$b_out:ident| $op_b_out:expr) => {
        match &$a.evaluations() {
            $crate::mle::FieldType::Base(a) => {
                let $tmp_a = &a[..];
                let $b_out = $op;
                $op_b_out
            }
            $crate::mle::FieldType::Ext(a) => {
                let $tmp_a = &a[..];
                #[allow(clippy::useless_conversion)]
                $op
            }
            _ => unreachable!(),
        }
    };
    ($a:ident, |$tmp_a:ident| $op:expr) => {
        op_mle!($a, |$tmp_a| $op, |out| out)
    };
    (|$a:ident| $op:expr, |$b_out:ident| $op_b_out:expr) => {
        op_mle!($a, |$a| $op, |$b_out| $op_b_out)
    };
    (|$a:ident| $op:expr) => {
        op_mle!(|$a| $op, |out| out)
    };
}

#[macro_export]
macro_rules! field_type_mut_map {
    ($a:ident, |$tmp_a:ident| $op:expr) => {
        match $a {
            $crate::mle::FieldType::Base(a) => {
                let mut $tmp_a = a;
                let out = $op;
                $crate::mle::FieldType::Base(out)
            }
            $crate::mle::FieldType::Ext(a) => {
                let mut $tmp_a = a;
                let out = $op;
                $crate::mle::FieldType::Ext(out)
            }
            _ => unreachable!(),
        }
    };
    ($a:ident, |$tmp_a:ident| $op:expr) => {
        field_type_mut_map!($a, |$tmp_a| $op)
    };
    (|$a:ident| $op:expr) => {
        field_type_mut_map!($a, |$a| $op)
    };
}

#[macro_export]
macro_rules! op_mle3_range {
    ($x:ident, $a:ident, $b:ident, $x_vec:ident, $a_vec:ident, $b_vec:ident, $op:expr, |$bb_out:ident| $op_bb_out:expr) => {{
        let $x = &$x_vec[..];
        let $a = &$a_vec[..];
        let $b = &$b_vec[..];
        let $bb_out = $op;
        $op_bb_out
    }};
}

/// deal with x * a + b or a * x + b
#[macro_export]
macro_rules! op_mle_xa_b {
    (|$x:ident, $a:ident, $b:ident| $op:expr, |$bb_out:ident| $op_bb_out:expr) => {
        match (&$x.evaluations(), &$a.evaluations(), &$b.evaluations()) {
            (
                $crate::mle::FieldType::Base(x_vec),
                $crate::mle::FieldType::Base(a_vec),
                $crate::mle::FieldType::Base(b_vec),
            ) => {
                op_mle3_range!($x, $a, $b, x_vec, a_vec, b_vec, $op, |$bb_out| $op_bb_out)
            }
            (
                $crate::mle::FieldType::Base(x_vec),
                $crate::mle::FieldType::Ext(a_vec),
                $crate::mle::FieldType::Base(b_vec),
            ) => {
                op_mle3_range!($x, $a, $b, x_vec, a_vec, b_vec, $op, |$bb_out| $op_bb_out)
            }
            (
                $crate::mle::FieldType::Base(x_vec),
                $crate::mle::FieldType::Ext(a_vec),
                $crate::mle::FieldType::Ext(b_vec),
            ) => {
                op_mle3_range!($x, $a, $b, x_vec, a_vec, b_vec, $op, |$bb_out| $op_bb_out)
            }
            (
                $crate::mle::FieldType::Ext(x_vec),
                $crate::mle::FieldType::Base(a_vec),
                $crate::mle::FieldType::Base(b_vec),
            ) => {
                op_mle3_range!($a, $x, $b, x_vec, a_vec, b_vec, $op, |$bb_out| $op_bb_out)
            }
            (x, a, b) => unreachable!(
                "unmatched pattern {:?} {:?} {:?}",
                x.variant_name(),
                a.variant_name(),
                b.variant_name()
            ),
        }
    };
    (|$x:ident, $a:ident, $b:ident| $op:expr) => {
        op_mle_xa_b!(|$x, $a, $b| $op, |out| out)
    };
}

/// deal with f1 * f2 * f3
/// applying cumulative rule for f1, f2, f3 to canonical form: Ext field comes first following by Base Field
#[macro_export]
macro_rules! op_mle_product_3 {
    (|$f1:ident, $f2:ident, $f3:ident| $op:expr, |$bb_out:ident| $op_bb_out:expr) => {
        match (&$f1.evaluations(), &$f2.evaluations(), &$f3.evaluations()) {
            // capture non-canonical form
            (
                $crate::mle::FieldType::Ext(_),
                $crate::mle::FieldType::Base(_),
                $crate::mle::FieldType::Ext(_),
            ) => {
                op_mle_product_3!(@internal |$f1, $f3, $f2| {
                    let ($f2, $f3) = ($f3, $f2);
                    $op
                }, |$bb_out| $op_bb_out)
            }
            // ...add more non-canonical form
            // default will go canonical form
            _ => op_mle_product_3!(@internal |$f1, $f2, $f3| $op, |$bb_out| $op_bb_out),
        }
    };
    (|$f1:ident, $f2:ident, $f3:ident| $op:expr) => {
        op_mle_product_3!(|$f1, $f2, $f3| $op, |out| out),
    };
    (@internal |$f1:ident, $f2:ident, $f3:ident| $op:expr, |$bb_out:ident| $op_bb_out:expr) => {
        match (&$f1.evaluations(), &$f2.evaluations(), &$f3.evaluations()) {
            (
                $crate::mle::FieldType::Base(f1_vec),
                $crate::mle::FieldType::Base(f2_vec),
                $crate::mle::FieldType::Base(f3_vec),
            ) => {
                op_mle3_range!($f1, $f2, $f3, f1_vec, f2_vec, f3_vec, $op, |$bb_out| $op_bb_out)
            }
            (
                $crate::mle::FieldType::Ext(f1_vec),
                $crate::mle::FieldType::Base(f2_vec),
                $crate::mle::FieldType::Base(f3_vec),
            ) => {
                op_mle3_range!($f1, $f2, $f3, f1_vec, f2_vec, f3_vec, $op, |out| out)
            }
            (
                $crate::mle::FieldType::Ext(f1_vec),
                $crate::mle::FieldType::Ext(f2_vec),
                $crate::mle::FieldType::Ext(f3_vec),
            ) => {
                op_mle3_range!($f1, $f2, $f3, f1_vec, f2_vec, f3_vec, $op, |out| out)
            }
            (
                $crate::mle::FieldType::Ext(f1_vec),
                $crate::mle::FieldType::Ext(f2_vec),
                $crate::mle::FieldType::Base(f3_vec),
            ) => {
                op_mle3_range!($f1, $f2, $f3, f1_vec, f2_vec, f3_vec, $op, |out| out)
            }
            // ... add more canonial case if missing
            (a, b, c) => unreachable!(
                "unmatched pattern {:?} {:?} {:?}",
                a.variant_name(),
                b.variant_name(),
                c.variant_name()
            ),
        }
    };
    (|$f1:ident, $f2:ident, $f3:ident| $op:expr) => {
        op_mle_product_3!(|$f1, $f2, $f3| $op, |out| out)
    };
}

/// macro support op(a, b) and tackles type matching internally.
/// Please noted that op must satisfy commutative rule w.r.t op(b, a) operand swap.
#[macro_export]
macro_rules! commutative_op_mle_pair {
    (|$first:ident, $second:ident| $op:expr, |$bb_out:ident| $op_bb_out:expr) => {
        match (&$first.evaluations(), &$second.evaluations()) {
            ($crate::mle::FieldType::Base(base1), $crate::mle::FieldType::Base(base2)) => {
                let $first = &base1[..];
                let $second = &base2[..];
                let $bb_out = $op;
                $op_bb_out
            }
            ($crate::mle::FieldType::Ext(ext), $crate::mle::FieldType::Base(base)) => {
                let $first = &ext[..];
                let $second = &base[..];
                $op
            }
            ($crate::mle::FieldType::Base(base), $crate::mle::FieldType::Ext(ext)) => {
                let base = &base[..];
                let ext = &ext[..];
                // swap first and second to make ext field come first before base field.
                // so the same coding template can apply.
                // that's why first and second operand must be commutative
                let $first = ext;
                let $second = base;
                $op
            }
            ($crate::mle::FieldType::Ext(ext), $crate::mle::FieldType::Ext(base)) => {
                let $first = &ext[..];
                let $second = &base[..];
                $op
            }
            _ => unreachable!(),
        }
    };
    (|$a:ident, $b:ident| $op:expr) => {
        commutative_op_mle_pair!(|$a, $b| $op, |out| out)
    };
}

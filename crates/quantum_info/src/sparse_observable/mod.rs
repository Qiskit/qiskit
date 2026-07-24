// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

mod lookup;

use hashbrown::HashSet;
use itertools::Itertools;
use lookup::conjugate_bitterm;
use num_complex::Complex64;
use std::{cmp::Ordering, collections::btree_map};
use thiserror::Error;

/// Named handle to the alphabet of single-qubit terms.
///
/// This is just the Rust-space representation.  We make a separate Python-space `enum.IntEnum` to
/// represent the same information, since we enforce strongly typed interactions in Rust, including
/// not allowing the stored values to be outside the valid `BitTerm`\ s, but doing so in Python would
/// make it very difficult to use the class efficiently with Numpy array views.  We attach this
/// sister class of `BitTerm` to `SparseObservable` as a scoped class.
///
/// # Representation
///
/// The `u8` representation and the exact numerical values of these are part of the public API.  The
/// low two bits are the symplectic Pauli representation of the required measurement basis with Z in
/// the Lsb0 and X in the Lsb1 (e.g. X and its eigenstate projectors all have their two low bits as
/// `0b10`).  The high two bits are `00` for the operator, `10` for the projector to the positive
/// eigenstate, and `01` for the projector to the negative eigenstate.
///
/// The `0b00_00` representation thus ends up being the natural representation of the `I` operator,
/// but this is never stored, and is not named in the enumeration.
///
/// This operator does not store phase terms of $-i$.  `BitTerm::Y` has `(1, 1)` as its `(z, x)`
/// representation, and represents exactly the Pauli Y operator; any additional phase is stored only
/// in a corresponding coefficient.
///
/// # Dev notes
///
/// This type is required to be `u8`, but it's a subtype of `u8` because not all `u8` are valid
/// `BitTerm`\ s.  For interop with Python space, we accept Numpy arrays of `u8` to represent this,
/// which we transmute into slices of `BitTerm`, after checking that all the values are correct (or
/// skipping the check if Python space promises that it upheld the checks).
///
/// We deliberately _don't_ impl `numpy::Element` for `BitTerm` (which would let us accept and
/// return `PyArray1<BitTerm>` at Python-space boundaries) so that it's clear when we're doing
/// the transmute, and we have to be explicit about the safety of that.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum BitTerm {
    /// Pauli X operator.
    X = 0b00_10,
    /// Projector to the positive eigenstate of Pauli X.
    Plus = 0b10_10,
    /// Projector to the negative eigenstate of Pauli X.
    Minus = 0b01_10,
    /// Pauli Y operator.
    Y = 0b00_11,
    /// Projector to the positive eigenstate of Pauli Y.
    Right = 0b10_11,
    /// Projector to the negative eigenstate of Pauli Y.
    Left = 0b01_11,
    /// Pauli Z operator.
    Z = 0b00_01,
    /// Projector to the positive eigenstate of Pauli Z.
    Zero = 0b10_01,
    /// Projector to the negative eigenstate of Pauli Z.
    One = 0b01_01,
}
impl From<BitTerm> for u8 {
    fn from(value: BitTerm) -> u8 {
        value as u8
    }
}
unsafe impl ::bytemuck::CheckedBitPattern for BitTerm {
    type Bits = u8;

    #[inline(always)]
    fn is_valid_bit_pattern(bits: &Self::Bits) -> bool {
        *bits <= 0b11_11 && (*bits & 0b11_00) < 0b11_00 && (*bits & 0b00_11) != 0
    }
}
unsafe impl ::bytemuck::NoUninit for BitTerm {}

impl BitTerm {
    /// Get the label of this `BitTerm` used in Python-space applications.  This is a single-letter
    /// string.
    #[inline]
    pub fn py_label(&self) -> &'static str {
        // Note: these labels are part of the stable Python API and should not be changed.
        match self {
            Self::X => "X",
            Self::Plus => "+",
            Self::Minus => "-",
            Self::Y => "Y",
            Self::Right => "r",
            Self::Left => "l",
            Self::Z => "Z",
            Self::Zero => "0",
            Self::One => "1",
        }
    }

    /// Get the name of this `BitTerm`, which is how Python space refers to the integer constant.
    #[inline]
    pub fn py_name(&self) -> &'static str {
        // Note: these names are part of the stable Python API and should not be changed.
        match self {
            Self::X => "X",
            Self::Plus => "PLUS",
            Self::Minus => "MINUS",
            Self::Y => "Y",
            Self::Right => "RIGHT",
            Self::Left => "LEFT",
            Self::Z => "Z",
            Self::Zero => "ZERO",
            Self::One => "ONE",
        }
    }

    /// Attempt to convert a `u8` into `BitTerm`.
    ///
    /// Unlike the implementation of `TryFrom<u8>`, this allows `b'I'` as an alphabet letter,
    /// returning `Ok(None)` for it.  All other letters outside the alphabet return the complete
    /// error condition.
    #[inline]
    pub(crate) fn try_from_u8(value: u8) -> Result<Option<Self>, BitTermFromU8Error> {
        match value {
            b'+' => Ok(Some(BitTerm::Plus)),
            b'-' => Ok(Some(BitTerm::Minus)),
            b'0' => Ok(Some(BitTerm::Zero)),
            b'1' => Ok(Some(BitTerm::One)),
            b'I' => Ok(None),
            b'X' => Ok(Some(BitTerm::X)),
            b'Y' => Ok(Some(BitTerm::Y)),
            b'Z' => Ok(Some(BitTerm::Z)),
            b'l' => Ok(Some(BitTerm::Left)),
            b'r' => Ok(Some(BitTerm::Right)),
            _ => Err(BitTermFromU8Error(value)),
        }
    }

    /// Does this term include an X component in its ZX representation?
    ///
    /// This is true for the operators and eigenspace projectors associated with X and Y.
    pub fn has_x_component(&self) -> bool {
        ((*self as u8) & (Self::X as u8)) != 0
    }

    /// Does this term include a Z component in its ZX representation?
    ///
    /// This is true for the operators and eigenspace projectors associated with Y and Z.
    pub fn has_z_component(&self) -> bool {
        ((*self as u8) & (Self::Z as u8)) != 0
    }

    pub fn is_projector(&self) -> bool {
        !matches!(self, BitTerm::X | BitTerm::Y | BitTerm::Z)
    }
}

fn bit_term_as_pauli(bit: &BitTerm) -> &'static [(bool, Option<BitTerm>)] {
    match bit {
        BitTerm::X => &[(true, Some(BitTerm::X))],
        BitTerm::Y => &[(true, Some(BitTerm::Y))],
        BitTerm::Z => &[(true, Some(BitTerm::Z))],
        BitTerm::Plus => &[(true, None), (true, Some(BitTerm::X))],
        BitTerm::Minus => &[(true, None), (false, Some(BitTerm::X))],
        BitTerm::Right => &[(true, None), (true, Some(BitTerm::Y))],
        BitTerm::Left => &[(true, None), (false, Some(BitTerm::Y))],
        BitTerm::Zero => &[(true, None), (true, Some(BitTerm::Z))],
        BitTerm::One => &[(true, None), (false, Some(BitTerm::Z))],
    }
}

/// The error type for a failed conversion into `BitTerm`.
#[derive(Error, Debug)]
#[error("{0} is not a valid letter of the single-qubit alphabet")]
pub struct BitTermFromU8Error(u8);

// `BitTerm` allows safe `as` casting into `u8`.  This is the reverse, which is fallible, because
// `BitTerm` is a value-wise subtype of `u8`.
impl ::std::convert::TryFrom<u8> for BitTerm {
    type Error = BitTermFromU8Error;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        ::bytemuck::checked::try_cast(value).map_err(|_| BitTermFromU8Error(value))
    }
}

/// Error cases stemming from data coherence at the point of entry into `SparseObservable` from
/// user-provided arrays.
///
/// These most typically appear during [from_raw_parts], but can also be introduced by various
/// remapping arithmetic functions.
///
/// These are generally associated with the Python-space `ValueError` because all of the
/// `TypeError`-related ones are statically forbidden (within Rust) by the language, and conversion
/// failures on entry to Rust from Python space will automatically raise `TypeError`.
#[derive(Error, Debug)]
pub enum CoherenceError {
    #[error("`boundaries` ({boundaries}) must be one element longer than `coeffs` ({coeffs})")]
    MismatchedTermCount { coeffs: usize, boundaries: usize },
    #[error("`bit_terms` ({bit_terms}) and `indices` ({indices}) must be the same length")]
    MismatchedItemCount { bit_terms: usize, indices: usize },
    #[error("the first item of `boundaries` ({0}) must be 0")]
    BadInitialBoundary(usize),
    #[error(
        "the last item of `boundaries` ({last}) must match the length of `bit_terms` and `indices` ({items})"
    )]
    BadFinalBoundary { last: usize, items: usize },
    #[error("all qubit indices must be less than the number of qubits")]
    BitIndexTooHigh,
    #[error("the values in `boundaries` include backwards slices")]
    DecreasingBoundaries,
    #[error("the values in `indices` are not term-wise increasing")]
    UnsortedIndices,
    #[error("the input contains duplicate qubits")]
    DuplicateIndices,
    #[error("the provided qubit mapping does not account for all contained qubits")]
    IndexMapTooSmall,
    #[error("cannot shrink the qubit count in an observable from {current} to {target}")]
    NotEnoughQubits { current: usize, target: usize },
}

/// An error related to processing of a string label (both dense and sparse).
#[derive(Error, Debug)]
pub enum LabelError {
    #[error("label with length {label} cannot be added to a {num_qubits}-qubit operator")]
    WrongLengthDense { num_qubits: u32, label: usize },
    #[error("label with length {label} does not match indices of length {indices}")]
    WrongLengthIndices { label: usize, indices: usize },
    #[error("index {index} is out of range for a {num_qubits}-qubit operator")]
    BadIndex { index: u32, num_qubits: u32 },
    #[error("index {index} is duplicated in a single specifier")]
    DuplicateIndex { index: u32 },
    #[error("labels must only contain letters from the alphabet 'IXYZ+-rl01'")]
    OutsideAlphabet,
}

#[derive(Error, Debug)]
pub enum ArithmeticError {
    #[error("mismatched numbers of qubits: {left}, {right}")]
    MismatchedQubits { left: u32, right: u32 },

    #[error("invalid operation: {0}")]
    InvalidOperation(String),

    #[error("duplicate indices in qargs")]
    DuplicatedIndex,

    #[error("{0}")]
    OutOfBounds(String),
}

/// One part of the type of the iteration value from [PairwiseOrdered].
///
/// The struct iterates over two sorted lists, and returns values from the left iterator, the right
/// iterator, or both simultaneously, depending on some "ordering" key attached to each.  This
/// `enum` is to pass on the information on which iterator is being returned from.
enum Paired<T> {
    Left(T),
    Right(T),
    Both(T, T),
}

/// An iterator combinator that zip-merges two sorted iterators.
///
/// This is created by [pairwise_ordered]; see that method for the description.
struct PairwiseOrdered<C, T, I1, I2>
where
    C: Ord,
    I1: Iterator<Item = (C, T)>,
    I2: Iterator<Item = (C, T)>,
{
    left: ::std::iter::Peekable<I1>,
    right: ::std::iter::Peekable<I2>,
}
impl<C, T, I1, I2> Iterator for PairwiseOrdered<C, T, I1, I2>
where
    C: Ord,
    I1: Iterator<Item = (C, T)>,
    I2: Iterator<Item = (C, T)>,
{
    type Item = (C, Paired<T>);

    fn next(&mut self) -> Option<Self::Item> {
        let order = match (self.left.peek(), self.right.peek()) {
            (None, None) => return None,
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (Some((left, _)), Some((right, _))) => left.cmp(right),
        };
        match order {
            Ordering::Less => self.left.next().map(|(i, value)| (i, Paired::Left(value))),
            Ordering::Greater => self
                .right
                .next()
                .map(|(i, value)| (i, Paired::Right(value))),
            Ordering::Equal => {
                let (index, left) = self.left.next().unwrap();
                let (_, right) = self.right.next().unwrap();
                Some((index, Paired::Both(left, right)))
            }
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = self.left.size_hint();
        let right = self.right.size_hint();
        (
            left.0.max(right.0),
            left.1.and_then(|left| right.1.map(|right| left.max(right))),
        )
    }
}
/// An iterator combinator that zip-merges two sorted iterators.
///
/// The two iterators must yield the same items, where each item comprises some totally ordered
/// index, and an associated value.  Both input iterators must be sorted in terms of the index.  The
/// output iteration is over 2-tuples, also in sorted order, of the seen ordered index values, and a
/// [Paired] object indicating which iterator (or both) the values were drawn from.
fn pairwise_ordered<C, T, I1, I2>(
    left: I1,
    right: I2,
) -> PairwiseOrdered<C, T, <I1 as IntoIterator>::IntoIter, <I2 as IntoIterator>::IntoIter>
where
    C: Ord,
    I1: IntoIterator<Item = (C, T)>,
    I2: IntoIterator<Item = (C, T)>,
{
    PairwiseOrdered {
        left: left.into_iter().peekable(),
        right: right.into_iter().peekable(),
    }
}

/// An observable over Pauli bases that stores its data in a qubit-sparse format.
///
/// See [PySparseObservable] for detailed docs.
#[derive(Clone, Debug, PartialEq)]
pub struct SparseObservable {
    /// The number of qubits the operator acts on.  This is not inferable from any other shape or
    /// values, since identities are not stored explicitly.
    num_qubits: u32,
    /// The coefficients of each abstract term in the sum.  This has as many elements as terms in
    /// the sum.
    coeffs: Vec<Complex64>,
    /// A flat list of single-qubit terms.  This is more naturally a list of lists, but is stored
    /// flat for memory usage and locality reasons, with the sublists denoted by `boundaries.`
    bit_terms: Vec<BitTerm>,
    /// A flat list of the qubit indices that the corresponding entries in `bit_terms` act on.  This
    /// list must always be term-wise sorted, where a term is a sublist as denoted by `boundaries`.
    indices: Vec<u32>,
    /// Indices that partition `bit_terms` and `indices` into sublists for each individual term in
    /// the sum.  `boundaries[0]..boundaries[1]` is the range of indices into `bit_terms` and
    /// `indices` that correspond to the first term of the sum.  All unspecified qubit indices are
    /// implicitly the identity.  This is one item longer than `coeffs`, since `boundaries[0]` is
    /// always an explicit zero (for algorithmic ease).
    boundaries: Vec<usize>,
}

impl SparseObservable {
    /// Create a new observable from the raw components that make it up.
    ///
    /// This checks the input values for data coherence on entry.  If you are certain you have the
    /// correct values, you can call `new_unchecked` instead.
    pub fn new(
        num_qubits: u32,
        coeffs: Vec<Complex64>,
        bit_terms: Vec<BitTerm>,
        indices: Vec<u32>,
        boundaries: Vec<usize>,
    ) -> Result<Self, CoherenceError> {
        if coeffs.len() + 1 != boundaries.len() {
            return Err(CoherenceError::MismatchedTermCount {
                coeffs: coeffs.len(),
                boundaries: boundaries.len(),
            });
        }
        if bit_terms.len() != indices.len() {
            return Err(CoherenceError::MismatchedItemCount {
                bit_terms: bit_terms.len(),
                indices: indices.len(),
            });
        }
        // We already checked that `boundaries` is at least length 1.
        if *boundaries.first().unwrap() != 0 {
            return Err(CoherenceError::BadInitialBoundary(boundaries[0]));
        }
        if *boundaries.last().unwrap() != indices.len() {
            return Err(CoherenceError::BadFinalBoundary {
                last: *boundaries.last().unwrap(),
                items: indices.len(),
            });
        }
        for (&left, &right) in boundaries[..].iter().zip(&boundaries[1..]) {
            if right < left {
                return Err(CoherenceError::DecreasingBoundaries);
            }
            let indices = &indices[left..right];
            if !indices.is_empty() {
                for (index_left, index_right) in indices[..].iter().zip(&indices[1..]) {
                    if index_left == index_right {
                        return Err(CoherenceError::DuplicateIndices);
                    } else if index_left > index_right {
                        return Err(CoherenceError::UnsortedIndices);
                    }
                }
            }
            if indices.last().map(|&ix| ix >= num_qubits).unwrap_or(false) {
                return Err(CoherenceError::BitIndexTooHigh);
            }
        }
        // SAFETY: we've just done the coherence checks.
        Ok(unsafe { Self::new_unchecked(num_qubits, coeffs, bit_terms, indices, boundaries) })
    }

    /// Create a new observable from the raw components without checking data coherence.
    ///
    /// # Safety
    ///
    /// It is up to the caller to ensure that the data-coherence requirements, as enumerated in the
    /// struct-level documentation, have been upheld.
    #[inline(always)]
    pub unsafe fn new_unchecked(
        num_qubits: u32,
        coeffs: Vec<Complex64>,
        bit_terms: Vec<BitTerm>,
        indices: Vec<u32>,
        boundaries: Vec<usize>,
    ) -> Self {
        Self {
            num_qubits,
            coeffs,
            bit_terms,
            indices,
            boundaries,
        }
    }

    /// Create a zero operator with pre-allocated space for the given number of summands and
    /// single-qubit bit terms.
    #[inline]
    pub fn with_capacity(num_qubits: u32, num_terms: usize, num_bit_terms: usize) -> Self {
        Self {
            num_qubits,
            coeffs: Vec::with_capacity(num_terms),
            bit_terms: Vec::with_capacity(num_bit_terms),
            indices: Vec::with_capacity(num_bit_terms),
            boundaries: {
                let mut boundaries = Vec::with_capacity(num_terms + 1);
                boundaries.push(0);
                boundaries
            },
        }
    }

    /// Get an iterator over the individual terms of the operator.
    ///
    /// Recall that two [SparseObservable]s that have different term orders can still represent the
    /// same object.  Use [canonicalize] to apply a canonical ordering to the terms.
    pub fn iter(&'_ self) -> impl ExactSizeIterator<Item = SparseTermView<'_>> + '_ {
        self.coeffs.iter().enumerate().map(|(i, coeff)| {
            let start = self.boundaries[i];
            let end = self.boundaries[i + 1];
            SparseTermView {
                num_qubits: self.num_qubits,
                coeff: *coeff,
                bit_terms: &self.bit_terms[start..end],
                indices: &self.indices[start..end],
            }
        })
    }

    /// Get an iterator over the individual terms of the operator that allows in-place mutation.
    ///
    /// The length and indices of these views cannot be mutated, since both would allow breaking
    /// data coherence.
    pub fn iter_mut(&mut self) -> IterMut<'_> {
        self.into()
    }

    /// Get the number of qubits the observable is defined on.
    #[inline]
    pub fn num_qubits(&self) -> u32 {
        self.num_qubits
    }

    /// Get the number of terms in the observable.
    #[inline]
    pub fn num_terms(&self) -> usize {
        self.coeffs.len()
    }

    /// Get the coefficients of the terms.
    #[inline]
    pub fn coeffs(&self) -> &[Complex64] {
        &self.coeffs
    }

    /// Get a mutable slice of the coefficients.
    #[inline]
    pub fn coeffs_mut(&mut self) -> &mut [Complex64] {
        &mut self.coeffs
    }

    /// Get the indices of each [BitTerm].
    #[inline]
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// Get a mutable slice of the indices.
    ///
    /// # Safety
    ///
    /// Modifying the indices can cause an incoherent state of the [SparseObservable].
    /// It should be ensured that the indices are consistent with the coeffs, bit_terms, and
    /// boundaries.
    #[inline]
    pub unsafe fn indices_mut(&mut self) -> &mut [u32] {
        &mut self.indices
    }

    /// Get the boundaries of each term.
    #[inline]
    pub fn boundaries(&self) -> &[usize] {
        &self.boundaries
    }

    /// Get a mutable slice of the boundaries.
    ///
    /// # Safety
    ///
    /// Modifying the boundaries can cause an incoherent state of the [SparseObservable].
    /// It should be ensured that the boundaries are sorted and the length/elements are consistent
    /// with the coeffs, bit_terms, and indices.
    #[inline]
    pub unsafe fn boundaries_mut(&mut self) -> &mut [usize] {
        &mut self.boundaries
    }

    /// Get the [BitTerm]s in the observable.
    #[inline]
    pub fn bit_terms(&self) -> &[BitTerm] {
        &self.bit_terms
    }

    /// Get a muitable slice of the bit terms.
    #[inline]
    pub fn bit_terms_mut(&mut self) -> &mut [BitTerm] {
        &mut self.bit_terms
    }

    /// Create a zero operator on ``num_qubits`` qubits.
    pub fn zero(num_qubits: u32) -> Self {
        Self::with_capacity(num_qubits, 0, 0)
    }

    /// Create an identity operator on ``num_qubits`` qubits.
    pub fn identity(num_qubits: u32) -> Self {
        Self {
            num_qubits,
            coeffs: vec![Complex64::new(1.0, 0.0)],
            bit_terms: vec![],
            indices: vec![],
            boundaries: vec![0, 0],
        }
    }

    /// Clear all the terms from this operator, making it equal to the zero operator again.
    ///
    /// This does not change the capacity of the internal allocations, so subsequent addition or
    /// subtraction operations may not need to reallocate.
    pub fn clear(&mut self) {
        self.coeffs.clear();
        self.bit_terms.clear();
        self.indices.clear();
        self.boundaries.truncate(1);
    }

    /// Reduce the observable to its canonical form.
    ///
    /// This sums like terms, removing them if the final complex coefficient's absolute value is
    /// less than or equal to the tolerance.  The terms are reordered to some canonical ordering.
    ///
    /// This function is idempotent.
    pub fn canonicalize(&self, tol: f64) -> SparseObservable {
        let mut terms = btree_map::BTreeMap::new();
        for term in self.iter() {
            terms
                .entry((term.indices, term.bit_terms))
                .and_modify(|c| *c += term.coeff)
                .or_insert(term.coeff);
        }
        let mut out = SparseObservable::zero(self.num_qubits);
        for ((indices, bit_terms), coeff) in terms {
            if coeff.norm_sqr() <= tol * tol {
                continue;
            }
            out.coeffs.push(coeff);
            out.bit_terms.extend_from_slice(bit_terms);
            out.indices.extend_from_slice(indices);
            out.boundaries.push(out.indices.len());
        }
        out
    }

    /// Tensor product of `self` with `other`.
    ///
    /// The bit ordering is defined such that the qubit indices of `other` will remain the same, and
    /// the indices of `self` will be offset by the number of qubits in `other`.  This is the same
    /// convention as used by the rest of Qiskit's `quantum_info` operators.
    ///
    /// Put another way, in the simplest case of two observables formed of dense labels:
    ///
    /// ```
    /// let mut left = SparseObservable::zero(5);
    /// left.add_dense_label("IXY+Z", Complex64::new(1.0, 0.0));
    /// let mut right = SparseObservable::zero(6);
    /// right.add_dense_label("IIrl01", Complex64::new(1.0, 0.0));
    ///
    /// // The result is the concatenation of the two labels.
    /// let mut out = SparseObservable::zero(11);
    /// out.add_dense_label("IXY+ZIIrl01", Complex64::new(1.0, 0.0));
    ///
    /// assert_eq!(left.tensor(right), out);
    /// ```
    pub fn tensor(&self, other: &SparseObservable) -> SparseObservable {
        let mut out = SparseObservable::with_capacity(
            self.num_qubits + other.num_qubits,
            self.coeffs.len() * other.coeffs.len(),
            other.coeffs.len() * self.bit_terms.len() + self.coeffs.len() * other.bit_terms.len(),
        );
        let mut self_indices = Vec::new();
        for self_term in self.iter() {
            self_indices.clear();
            self_indices.extend(self_term.indices.iter().map(|i| i + other.num_qubits));
            for other_term in other.iter() {
                out.coeffs.push(self_term.coeff * other_term.coeff);
                out.indices.extend_from_slice(other_term.indices);
                out.indices.extend_from_slice(&self_indices);
                out.bit_terms.extend_from_slice(other_term.bit_terms);
                out.bit_terms.extend_from_slice(self_term.bit_terms);
                out.boundaries.push(out.bit_terms.len());
            }
        }
        out
    }

    /// Calculate the adjoint of this observable.
    ///
    /// This is well defined in the abstract mathematical sense.  All the terms of the single-qubit
    /// alphabet are self-adjoint, so the result of this operation is the same observable, except
    /// its coefficients are all their complex conjugates.
    pub fn adjoint(&self) -> SparseObservable {
        SparseObservable {
            num_qubits: self.num_qubits,
            coeffs: self.coeffs.iter().map(|c| c.conj()).collect(),
            bit_terms: self.bit_terms.clone(),
            indices: self.indices.clone(),
            boundaries: self.boundaries.clone(),
        }
    }

    /// Calculate the transpose.
    ///
    /// This operation transposes the individual bit terms but does not directly act
    /// on the coefficients.
    pub fn transpose(&self) -> SparseObservable {
        let mut out = self.clone();
        for term in out.iter_mut() {
            for bit_term in term.bit_terms {
                match bit_term {
                    BitTerm::Y => {
                        *term.coeff = -*term.coeff;
                    }
                    BitTerm::Right => {
                        *bit_term = BitTerm::Left;
                    }
                    BitTerm::Left => {
                        *bit_term = BitTerm::Right;
                    }
                    _ => (),
                }
            }
        }
        out
    }

    /// Calculate the complex conjugate.
    ///
    /// This operation equals transposing the observable and complex conjugating the coefficients.
    pub fn conjugate(&self) -> SparseObservable {
        let mut out = self.transpose();
        for coeff in out.coeffs.iter_mut() {
            *coeff = coeff.conj();
        }
        out
    }

    /// Compose another [SparseObservable] onto this one.
    ///
    /// In terms of operator algebras, composition corresponds to left-multiplication:
    /// ``let c = a.compose(b);`` corresponds to $C = B A$.
    ///
    /// Beware that this function can cause exponential explosion of the memory usage of the
    /// observable, as the alphabet of [SparseObservable] is not closed under composition; the
    /// composition of two single-bit terms can be a sum, which multiplies the total number of
    /// terms.  This memory usage is not _necessarily_ inherent to the resultant observable, but
    /// finding an efficient re-factorization of the sum is generally equally computationally hard.
    /// It's better to use domain knowledge of your observables to minimize the number of terms that
    /// ever exist, rather than trying to simplify them after the fact.
    ///
    /// # Panics
    ///
    /// If `self` and `other` have different numbers of qubits.
    pub fn compose(&self, other: &SparseObservable) -> SparseObservable {
        if other.num_qubits != self.num_qubits {
            panic!(
                "operand ({}) has a different number of qubits to the base ({})",
                other.num_qubits, self.num_qubits
            );
        }
        let mut out = SparseObservable::zero(self.num_qubits);
        let mut term_state = compose::Iter::new(self.num_qubits);

        for left in other.iter() {
            for right in self.iter() {
                term_state.load_from(
                    left.coeff * right.coeff,
                    left.indices
                        .iter()
                        .copied()
                        .zip(left.bit_terms.iter().copied()),
                    right
                        .indices
                        .iter()
                        .copied()
                        .zip(right.bit_terms.iter().copied()),
                );
                out.boundaries.reserve(term_state.num_terms());
                out.coeffs.reserve(term_state.num_terms());
                out.indices
                    .reserve(term_state.num_terms() * term_state.term_len());
                out.bit_terms
                    .reserve(term_state.num_terms() * term_state.term_len());
                while let Some(term) = term_state.next() {
                    out.add_term(term)
                        .expect("qubit counts were checked during initialisation");
                }
            }
        }
        out
    }

    /// Compose another [SparseObservable] onto this one, remapping the qubits.
    ///
    /// The `qubit_fn` is called for each qubit in each term in `other` to determine which qubit in
    /// `self` it corresponds to; this should typically be a very cheap function to call, like a
    /// getter from a slice.
    ///
    /// See [compose] for further information.
    ///
    /// # Panics
    ///
    /// If `other` has more qubits than `self`, if the `qubit_fn` returns a qubit index greater
    /// or equal to the number of qubits in `self`, or if `qubit_fn` returns a duplicate index (this
    /// is only guaranteed to be detected if an individual term contains duplicates).
    pub fn compose_map(
        &self,
        other: &SparseObservable,
        mut qubit_fn: impl FnMut(u32) -> u32,
    ) -> SparseObservable {
        if other.num_qubits > self.num_qubits {
            panic!(
                "operand has more qubits ({}) than the base ({})",
                other.num_qubits, self.num_qubits
            );
        }
        let mut out = SparseObservable::zero(self.num_qubits);
        let mut mapped_term = btree_map::BTreeMap::<u32, BitTerm>::new();
        let mut term_state = compose::Iter::new(self.num_qubits);

        // This choice of loop ordering is because we already know that `self`'s `indices` are
        // sorted, but there's no reason that that the output of `qubit_fn` should maintain order,
        // and this way round, we sort as few times as possible.
        for left in other.iter() {
            mapped_term.clear();
            for (index, term) in left.indices.iter().zip(left.bit_terms) {
                let qubit = qubit_fn(*index);
                if qubit >= self.num_qubits {
                    panic!("remapped {index} to {qubit}, which is out of bounds");
                }
                if mapped_term.insert(qubit, *term).is_some() {
                    panic!("duplicate qubit {qubit} in remapped term");
                };
            }
            for right in self.iter() {
                term_state.load_from(
                    left.coeff * right.coeff,
                    mapped_term.iter().map(|(k, v)| (*k, *v)),
                    right
                        .indices
                        .iter()
                        .copied()
                        .zip(right.bit_terms.iter().copied()),
                );
                out.boundaries.reserve(term_state.num_terms());
                out.coeffs.reserve(term_state.num_terms());
                out.indices
                    .reserve(term_state.num_terms() * term_state.term_len());
                out.bit_terms
                    .reserve(term_state.num_terms() * term_state.term_len());
                while let Some(term) = term_state.next() {
                    out.add_term(term)
                        .expect("qubit counts were checked during initialisation");
                }
            }
        }
        out
    }

    /// Evolve this [SparseObservable] by another one.
    ///
    /// In terms of operator algebra, evolution corresponds to conjugation:
    /// ``let out = q.evolve(p);`` corresponds to $P^\dagger Q P$.
    ///
    /// This implements Heisenberg-picture evolution of the observable.  Unlike a
    /// literal implementation via two full compositions, this method performs the
    /// conjugation directly at the single-qubit level using a fixed lookup table
    /// for $P^\dagger Q P$.  This avoids materializing any intermediate
    /// [SparseObservable] and computes the evolved observable in a single pass.
    ///
    /// Currently, this method supports evolution only by a *single-term* [SparseObservable].
    pub fn evolve(
        &self,
        op: &SparseObservable,
        qargs: Option<&[u32]>,
    ) -> Result<SparseObservable, ArithmeticError> {
        if op.num_terms() != 1 {
            return Err(ArithmeticError::InvalidOperation(
                "evolve only supports single-term operators".to_string(),
            ));
        }

        let t = op.iter().next().unwrap();
        let op_coeff = t.coeff;
        let mut layout = vec![None; self.num_qubits as usize];

        if let Some(qargs) = qargs {
            if op.num_qubits > self.num_qubits {
                return Err(ArithmeticError::OutOfBounds(format!(
                    "operator has more qubits ({}) than the base ({})",
                    op.num_qubits, self.num_qubits
                )));
            }
            // Handling the zero-qubit scalar edge case (Identity Operator evolution).
            if op.num_qubits == 0 {
                let scalar = op.coeffs()[0];
                return Ok(self * (scalar.conj() * scalar));
            }

            if qargs.len() != op.num_qubits as usize {
                return Err(ArithmeticError::OutOfBounds(format!(
                    "qargs has length {}, but operator has {} qubit(s)",
                    qargs.len(),
                    op.num_qubits
                )));
            }

            let qargs_set = HashSet::<&u32>::from_iter(qargs.iter());
            if qargs_set.len() != qargs.len() {
                return Err(ArithmeticError::DuplicatedIndex);
            }

            if let Some(&max_q) = qargs.iter().max()
                && max_q >= self.num_qubits
            {
                return Err(ArithmeticError::OutOfBounds(
                    "qargs contains out-of-range qubits".to_string(),
                ));
            }

            // This maps operator bit terms to observable qubits via qargs, considering
            // qargs[i] specifies which observable qubit (at index i), the next operator qubit
            // in consideration acts on.  Operator qubits are numbered 0 to (num_qubits - 1),
            // where qubit 0 is considered, the rightmost (least significant) qubit.
            for (op_qubit, &self_qubit) in qargs.iter().enumerate() {
                if let Some(bit_term_idx) = t.indices.iter().position(|&q| q as usize == op_qubit) {
                    layout[self_qubit as usize] = Some(t.bit_terms[bit_term_idx]);
                }
            }
        } else {
            if self.num_qubits != op.num_qubits {
                return Err(ArithmeticError::MismatchedQubits {
                    left: self.num_qubits,
                    right: op.num_qubits,
                });
            }

            for (q, bt) in t.indices.iter().zip(t.bit_terms.iter()) {
                layout[*q as usize] = Some(*bt);
            }
        }

        let mut out = SparseObservable::zero(self.num_qubits);

        for term in self.iter() {
            let mut frontier = vec![(term.coeff, Vec::<u32>::new(), Vec::<BitTerm>::new())];
            let mut term_map = vec![None; self.num_qubits as usize];
            for (i, &q) in term.indices.iter().enumerate() {
                term_map[q as usize] = Some(term.bit_terms[i]);
            }

            for (q, &op_bt) in layout.iter().enumerate() {
                let term_bt = term_map[q];

                let mut next_frontier = Vec::new();

                for (coeff, indices, bit_terms) in frontier {
                    match (op_bt, term_bt) {
                        (None, None) => {
                            next_frontier.push((coeff, indices, bit_terms));
                        }
                        (None, Some(bt)) => {
                            let mut indices = indices;
                            let mut bit_terms = bit_terms;
                            indices.push(q as u32);
                            bit_terms.push(bt);
                            next_frontier.push((coeff, indices, bit_terms));
                        }
                        (Some(_), None) => {
                            next_frontier.push((coeff, indices, bit_terms));
                        }
                        (Some(p), Some(qbt)) => {
                            let outputs = conjugate_bitterm(p, qbt);
                            for &(c, new_bt) in outputs {
                                let mut new_indices = indices.clone();
                                let mut new_bit_terms = bit_terms.clone();
                                new_indices.push(q as u32);
                                new_bit_terms.push(new_bt);
                                next_frontier.push((coeff * c, new_indices, new_bit_terms));
                            }
                        }
                    }
                }

                frontier = next_frontier;
                if frontier.is_empty() {
                    break;
                }
            }

            for (coeff, indices, bit_terms) in frontier {
                if coeff == Complex64::new(0.0, 0.0) {
                    continue;
                }
                out.coeffs.push(op_coeff.conj() * coeff * op_coeff);
                out.indices.extend(indices);
                out.bit_terms.extend(bit_terms);
                out.boundaries.push(out.indices.len());
            }
        }

        Ok(out)
    }

    /// Add another [SparseObservable] onto this one, while scaling its coefficients.
    ///
    /// # Panics
    ///
    /// If the number of qubits of `rhs` and `self` differ.
    pub fn scaled_add_inplace(&mut self, rhs: &SparseObservable, factor: Complex64) {
        if rhs.num_qubits != self.num_qubits {
            panic!(
                "operand ({}) has a different number of qubits to the base ({})",
                rhs.num_qubits, self.num_qubits
            );
        }
        self.coeffs.extend(rhs.coeffs.iter().map(|c| c * factor));
        self.bit_terms.extend_from_slice(&rhs.bit_terms);
        self.indices.extend_from_slice(&rhs.indices);
        // We only need to write out the new endpoints, not the initial zero.
        let offset = self.boundaries[self.boundaries.len() - 1];
        self.boundaries
            .extend(rhs.boundaries[1..].iter().map(|boundary| offset + boundary));
    }

    /// Add two [SparseObservable] instances while scaling the coefficients of `rhs`
    /// with `factor`.
    ///
    /// # Panics
    ///
    /// If the number of qubits of `other` and `self` differ.
    pub fn scaled_add(&self, rhs: &SparseObservable, factor: Complex64) -> SparseObservable {
        let mut out = SparseObservable::with_capacity(
            self.num_qubits,
            self.coeffs.len() + rhs.coeffs.len(),
            self.bit_terms.len() + rhs.bit_terms.len(),
        );
        out += self;
        out.scaled_add_inplace(rhs, factor);
        out
    }

    /// Get a view onto a representation of a single sparse term.
    ///
    /// This is effectively an indexing operation into the [SparseObservable].  Recall that two
    /// [SparseObservable]s that have different term orders can still represent the same object.
    /// Use [canonicalize] to apply a canonical ordering to the terms.
    ///
    /// # Panics
    ///
    /// If the index is out of bounds.
    pub fn term(&self, index: usize) -> SparseTermView<'_> {
        debug_assert!(index < self.num_terms(), "index {index} out of bounds");
        let start = self.boundaries[index];
        let end = self.boundaries[index + 1];
        SparseTermView {
            num_qubits: self.num_qubits,
            coeff: self.coeffs[index],
            bit_terms: &self.bit_terms[start..end],
            indices: &self.indices[start..end],
        }
    }

    /// Expand all projectors into Pauli representation.
    ///
    /// # Warning
    ///
    /// This representation is highly inefficient for projectors. For example, a term with
    /// :math:`n` projectors :math:`|+\rangle\langle +|` will use :math:`2^n` Pauli terms.
    pub fn as_paulis(&self) -> Self {
        let mut paulis: Vec<BitTerm> = Vec::new(); // maybe get capacity here
        let mut indices: Vec<u32> = Vec::new();
        let mut coeffs: Vec<Complex64> = Vec::new();
        let mut boundaries: Vec<usize> = vec![0];

        for view in self.iter() {
            let num_projectors = view
                .bit_terms
                .iter()
                .filter(|&bit| bit.is_projector())
                .count();
            let div = 2_f64.powi(num_projectors as i32);

            let combinations = view
                .bit_terms
                .iter()
                .map(bit_term_as_pauli)
                .multi_cartesian_product();

            for combination in combinations {
                let mut positive = true; // keep track of the global sign

                for (index, (sign, bit)) in combination.iter().enumerate() {
                    positive ^= !sign; // accumulate the sign; global_sign *= local_sign
                    if let Some(bit) = bit {
                        paulis.push(*bit);
                        indices.push(view.indices[index]);
                    }
                }
                boundaries.push(paulis.len());

                let coeff = if positive { view.coeff } else { -view.coeff };
                coeffs.push(coeff / div)
            }
        }

        Self {
            num_qubits: self.num_qubits,
            coeffs,
            bit_terms: paulis,
            indices,
            boundaries,
        }
    }

    /// Add the term implied by a dense string label onto this observable.
    pub fn add_dense_label<L: AsRef<[u8]>>(
        &mut self,
        label: L,
        coeff: Complex64,
    ) -> Result<(), LabelError> {
        let label = label.as_ref();
        if label.len() != self.num_qubits() as usize {
            return Err(LabelError::WrongLengthDense {
                num_qubits: self.num_qubits(),
                label: label.len(),
            });
        }
        // The only valid characters in the alphabet are ASCII, so if we see something other than
        // ASCII, we're already in the failure path.
        for (i, letter) in label.iter().rev().enumerate() {
            match BitTerm::try_from_u8(*letter) {
                Ok(Some(term)) => {
                    self.bit_terms.push(term);
                    self.indices.push(i as u32);
                }
                Ok(None) => (),
                Err(_) => {
                    // Undo any modifications to ourselves so we stay in a consistent state.
                    let num_single_terms = self.boundaries[self.boundaries.len() - 1];
                    self.bit_terms.truncate(num_single_terms);
                    self.indices.truncate(num_single_terms);
                    return Err(LabelError::OutsideAlphabet);
                }
            }
        }
        self.coeffs.push(coeff);
        self.boundaries.push(self.bit_terms.len());
        Ok(())
    }

    /// Relabel the `indices` in the operator to new values.
    ///
    /// This fails if any of the new indices are too large, or if any mapping would cause a term to
    /// contain duplicates of the same index.  It may not detect if multiple qubits are mapped to
    /// the same index, if those qubits never appear together in the same term.  Such a mapping
    /// would not cause data-coherence problems (the output observable will be valid), but is
    /// unlikely to be what you intended.
    ///
    /// *Panics* if `new_qubits` is not long enough to map every index used in the operator.
    pub fn relabel_qubits_from_slice(&mut self, new_qubits: &[u32]) -> Result<(), CoherenceError> {
        for qubit in new_qubits {
            if *qubit >= self.num_qubits {
                return Err(CoherenceError::BitIndexTooHigh);
            }
        }
        let mut order = btree_map::BTreeMap::new();
        for i in 0..self.num_terms() {
            let start = self.boundaries[i];
            let end = self.boundaries[i + 1];
            for j in start..end {
                order.insert(new_qubits[self.indices[j] as usize], self.bit_terms[j]);
            }
            if order.len() != end - start {
                return Err(CoherenceError::DuplicateIndices);
            }
            for (index, dest) in order.keys().zip(&mut self.indices[start..end]) {
                *dest = *index;
            }
            for (bit_term, dest) in order.values().zip(&mut self.bit_terms[start..end]) {
                *dest = *bit_term;
            }
            order.clear();
        }
        Ok(())
    }

    /// Apply a transpiler layout.
    pub fn apply_layout(
        &self,
        layout: Option<&[u32]>,
        num_qubits: u32,
    ) -> Result<Self, CoherenceError> {
        match layout {
            None => {
                let mut out = self.clone();
                if num_qubits < self.num_qubits {
                    return Err(CoherenceError::NotEnoughQubits {
                        current: self.num_qubits as usize,
                        target: num_qubits as usize,
                    });
                }
                out.num_qubits = num_qubits;
                Ok(out)
            }
            Some(layout) => {
                if layout.len() < self.num_qubits as usize {
                    return Err(CoherenceError::IndexMapTooSmall);
                }
                if layout.iter().any(|qubit| *qubit >= num_qubits) {
                    return Err(CoherenceError::BitIndexTooHigh);
                }
                if layout.iter().collect::<HashSet<_>>().len() != layout.len() {
                    return Err(CoherenceError::DuplicateIndices);
                }
                let mut out = self.clone();
                out.num_qubits = num_qubits;
                out.relabel_qubits_from_slice(layout)?;
                Ok(out)
            }
        }
    }

    /// Add a single term to this operator.
    pub fn add_term(&mut self, term: SparseTermView) -> Result<(), ArithmeticError> {
        if self.num_qubits != term.num_qubits {
            return Err(ArithmeticError::MismatchedQubits {
                left: self.num_qubits,
                right: term.num_qubits,
            });
        }
        self.coeffs.push(term.coeff);
        self.bit_terms.extend_from_slice(term.bit_terms);
        self.indices.extend_from_slice(term.indices);
        self.boundaries.push(self.bit_terms.len());
        Ok(())
    }

    /// Return a suitable Python error if two observables do not have equal numbers of qubits.
    pub fn check_equal_qubits(&self, other: &SparseObservable) -> Result<(), ArithmeticError> {
        if self.num_qubits != other.num_qubits {
            Err(ArithmeticError::MismatchedQubits {
                left: self.num_qubits,
                right: other.num_qubits,
            })
        } else {
            Ok(())
        }
    }

    /// Check whether the observable commutes with another one.
    ///
    /// # Arguments
    ///
    /// * `tol` - If coefficients in the product of `self` and `other` are below the tolerance
    ///   (in magnitude), the terms are ignored.
    ///
    /// # Returns
    ///
    /// `true` if the observables commute, `false` otherwise.
    pub fn commutes(&self, other: &SparseObservable, tol: f64) -> bool {
        let ab = self.compose(other).canonicalize(tol);
        let ba = other.compose(self).canonicalize(tol);
        ab == ba
    }
}

impl ::std::ops::Add<&SparseObservable> for SparseObservable {
    type Output = SparseObservable;

    fn add(mut self, rhs: &SparseObservable) -> SparseObservable {
        self += rhs;
        self
    }
}
impl ::std::ops::Add for &SparseObservable {
    type Output = SparseObservable;

    fn add(self, rhs: &SparseObservable) -> SparseObservable {
        let mut out = SparseObservable::with_capacity(
            self.num_qubits,
            self.coeffs.len() + rhs.coeffs.len(),
            self.bit_terms.len() + rhs.bit_terms.len(),
        );
        out += self;
        out += rhs;
        out
    }
}
impl ::std::ops::AddAssign<&SparseObservable> for SparseObservable {
    fn add_assign(&mut self, rhs: &SparseObservable) {
        if self.num_qubits != rhs.num_qubits {
            panic!("attempt to add two operators with incompatible qubit counts");
        }
        self.coeffs.extend_from_slice(&rhs.coeffs);
        self.bit_terms.extend_from_slice(&rhs.bit_terms);
        self.indices.extend_from_slice(&rhs.indices);
        // We only need to write out the new endpoints, not the initial zero.
        let offset = self.boundaries[self.boundaries.len() - 1];
        self.boundaries
            .extend(rhs.boundaries[1..].iter().map(|boundary| offset + boundary));
    }
}

impl ::std::ops::Sub<&SparseObservable> for SparseObservable {
    type Output = SparseObservable;

    fn sub(mut self, rhs: &SparseObservable) -> SparseObservable {
        self -= rhs;
        self
    }
}
impl ::std::ops::Sub for &SparseObservable {
    type Output = SparseObservable;

    fn sub(self, rhs: &SparseObservable) -> SparseObservable {
        let mut out = SparseObservable::with_capacity(
            self.num_qubits,
            self.coeffs.len() + rhs.coeffs.len(),
            self.bit_terms.len() + rhs.bit_terms.len(),
        );
        out += self;
        out -= rhs;
        out
    }
}
impl ::std::ops::SubAssign<&SparseObservable> for SparseObservable {
    fn sub_assign(&mut self, rhs: &SparseObservable) {
        if self.num_qubits != rhs.num_qubits {
            panic!("attempt to subtract two operators with incompatible qubit counts");
        }
        self.coeffs.extend(rhs.coeffs.iter().map(|coeff| -coeff));
        self.bit_terms.extend_from_slice(&rhs.bit_terms);
        self.indices.extend_from_slice(&rhs.indices);
        // We only need to write out the new endpoints, not the initial zero.
        let offset = self.boundaries[self.boundaries.len() - 1];
        self.boundaries
            .extend(rhs.boundaries[1..].iter().map(|boundary| offset + boundary));
    }
}

impl ::std::ops::Mul<Complex64> for SparseObservable {
    type Output = SparseObservable;

    fn mul(mut self, rhs: Complex64) -> SparseObservable {
        self *= rhs;
        self
    }
}
impl ::std::ops::Mul<Complex64> for &SparseObservable {
    type Output = SparseObservable;

    fn mul(self, rhs: Complex64) -> SparseObservable {
        if rhs == Complex64::new(0.0, 0.0) {
            SparseObservable::zero(self.num_qubits)
        } else {
            SparseObservable {
                num_qubits: self.num_qubits,
                coeffs: self.coeffs.iter().map(|c| c * rhs).collect(),
                bit_terms: self.bit_terms.clone(),
                indices: self.indices.clone(),
                boundaries: self.boundaries.clone(),
            }
        }
    }
}
impl ::std::ops::Mul<SparseObservable> for Complex64 {
    type Output = SparseObservable;

    fn mul(self, mut rhs: SparseObservable) -> SparseObservable {
        rhs *= self;
        rhs
    }
}
impl ::std::ops::Mul<&SparseObservable> for Complex64 {
    type Output = SparseObservable;

    fn mul(self, rhs: &SparseObservable) -> SparseObservable {
        rhs * self
    }
}
impl ::std::ops::MulAssign<Complex64> for SparseObservable {
    fn mul_assign(&mut self, rhs: Complex64) {
        if rhs == Complex64::new(0.0, 0.0) {
            self.coeffs.clear();
            self.bit_terms.clear();
            self.indices.clear();
            self.boundaries.clear();
            self.boundaries.push(0);
        } else {
            self.coeffs.iter_mut().for_each(|c| *c *= rhs)
        }
    }
}

impl ::std::ops::Div<Complex64> for SparseObservable {
    type Output = SparseObservable;

    fn div(mut self, rhs: Complex64) -> SparseObservable {
        self /= rhs;
        self
    }
}
impl ::std::ops::Div<Complex64> for &SparseObservable {
    type Output = SparseObservable;

    fn div(self, rhs: Complex64) -> SparseObservable {
        SparseObservable {
            num_qubits: self.num_qubits,
            coeffs: self.coeffs.iter().map(|c| c / rhs).collect(),
            bit_terms: self.bit_terms.clone(),
            indices: self.indices.clone(),
            boundaries: self.boundaries.clone(),
        }
    }
}
impl ::std::ops::DivAssign<Complex64> for SparseObservable {
    fn div_assign(&mut self, rhs: Complex64) {
        self.coeffs.iter_mut().for_each(|c| *c /= rhs)
    }
}

impl ::std::ops::Neg for &SparseObservable {
    type Output = SparseObservable;

    fn neg(self) -> SparseObservable {
        SparseObservable {
            num_qubits: self.num_qubits,
            coeffs: self.coeffs.iter().map(|c| -c).collect(),
            bit_terms: self.bit_terms.clone(),
            indices: self.indices.clone(),
            boundaries: self.boundaries.clone(),
        }
    }
}
impl ::std::ops::Neg for SparseObservable {
    type Output = SparseObservable;

    fn neg(mut self) -> SparseObservable {
        self.coeffs.iter_mut().for_each(|c| *c = -*c);
        self
    }
}

/// Worker objects for the [SparseObservable::compose]-alike functions.
mod compose {
    use super::*;

    /// An non-scalar entry in the multi-Cartesian product iteration.
    ///
    /// Each [MultipleItem] corresponds to a bit index that becomes a summation as part of the
    /// composition, so the multi-Cartesian product has to keep iterating through it.
    struct MultipleItem {
        /// The index into the full `bit_terms` [Vec] that this item refers to.
        loc: usize,
        /// The next item in the slice that should be written out.  If this equal to the length
        /// of the slice, the implication is that it needs to be reset to 0 by a lower-index
        /// item getting incremented and flowing forwards.
        cur: usize,
        /// The underlying slice of the multiple iteration, from the compose lookup table.
        slice: &'static [(Complex64, BitTerm)],
    }

    /// An implementation of the multiple Cartesian-product iterator that produces the sum terms
    /// that make up the composition of two individual [SparseObservable] terms.
    ///
    /// This mutates itself in-place to keep track of the state, which allows us to re-use
    /// partial results and to avoid allocations per item (since we have to copy out of the
    /// buffers to the output [SparseObservable] each time anyway).
    pub struct Iter {
        num_qubits: u32,
        /// Tracking data for the places in the output that have multiple branches to take.
        multiples: Vec<MultipleItem>,
        /// Stack of the coefficients to this point.  We could recalculate by a full
        /// multiplication on each go, but most steps will be in the low indices, where we can
        /// re-use all the multiplications that came before.  This is one longer than the length
        /// of `multiples`, because it starts off populated with the product of the
        /// non-multiple coefficients (or 1).
        coeffs: Vec<Complex64>,
        /// The full set of indices (including ones that don't correspond to multiples).  Within
        /// a given iteration, this never changes; it's just stored to make it easier to memcpy
        /// out of.
        indices: Vec<u32>,
        /// The full set of [BitTerm]s (including ones that don't correspond to multiples).  The
        /// multiple ones get updated inplace during iteration.
        bit_terms: Vec<BitTerm>,
        /// Whether iteration has started.
        started: bool,
        /// Whether iteration will yield any further items.
        exhausted: bool,
    }
    impl Iter {
        pub fn new(num_qubits: u32) -> Self {
            Self {
                num_qubits,
                multiples: Vec::new(),
                coeffs: vec![Complex64::new(1., 0.)],
                indices: Vec::new(),
                bit_terms: Vec::new(),
                started: false,
                exhausted: false,
            }
        }
        /// Load up the next terms an iterator over the `(index, term)` pairs from the left-hand
        /// side and the right-hand side.  The iterators must return indices in strictly increasing
        /// order.
        pub fn load_from(
            &mut self,
            coeff: Complex64,
            left: impl IntoIterator<Item = (u32, BitTerm)>,
            right: impl IntoIterator<Item = (u32, BitTerm)>,
        ) {
            self.multiples.clear();
            self.coeffs.clear();
            self.coeffs.push(coeff);
            self.indices.clear();
            self.bit_terms.clear();
            self.started = false;
            self.exhausted = false;

            for (index, values) in pairwise_ordered(left, right) {
                match values {
                    Paired::Left(term) | Paired::Right(term) => {
                        self.indices.push(index);
                        self.bit_terms.push(term);
                    }
                    Paired::Both(left, right) => {
                        let Some(slice) = lookup::matmul(left, right) else {
                            self.exhausted = true;
                            return;
                        };
                        match slice {
                            &[] => (),
                            &[(coeff, term)] => {
                                self.coeffs[0] *= coeff;
                                self.indices.push(index);
                                self.bit_terms.push(term);
                            }
                            slice => {
                                self.multiples.push(MultipleItem {
                                    loc: self.bit_terms.len(),
                                    cur: 0,
                                    slice,
                                });
                                self.indices.push(index);
                                self.coeffs.push(Default::default());
                                self.bit_terms.push(slice[0].1);
                            }
                        }
                    }
                }
            }
        }
        /// Expose the current iteration item, assuming the state has been updated.
        fn iter_item(&self) -> SparseTermView<'_> {
            SparseTermView {
                num_qubits: self.num_qubits,
                coeff: *self.coeffs.last().expect("coeffs is never empty"),
                indices: &self.indices,
                bit_terms: &self.bit_terms,
            }
        }
        // Not actually the iterator method, because we're borrowing from `self`.
        /// Get the next term in the iteration.
        pub fn next(&mut self) -> Option<SparseTermView<'_>> {
            if self.exhausted {
                return None;
            }
            if !self.started {
                self.started = true;
                // Initialising the struct has to put us in a place where the indices and bit terms
                // are ready, but we still need to initialise the coefficient stack.
                for (i, item) in self.multiples.iter().enumerate() {
                    // `item.cur` should always be 0 at this point.
                    self.coeffs[i + 1] = self.coeffs[i] * item.slice[item.cur].0;
                }
                return Some(self.iter_item());
            }
            // The lowest index into `multiples` that didn't overflow as we were updating the
            // bit-terms state.
            let non_overflow_index = 'inc_index: {
                for (i, item) in self.multiples.iter_mut().enumerate().rev() {
                    // The slices are always non-empty.
                    if item.cur == item.slice.len() - 1 {
                        item.cur = 0;
                    } else {
                        item.cur += 1;
                    }
                    self.bit_terms[item.loc] = item.slice[item.cur].1;
                    if item.cur > 0 {
                        break 'inc_index i;
                    }
                }
                self.exhausted = true;
                return None;
            };
            // Now run forwards updating the coefficient tree from the point it changes.
            for (offset, item) in self.multiples[non_overflow_index..].iter_mut().enumerate() {
                let base = non_overflow_index + offset;
                self.coeffs[base + 1] = self.coeffs[base] * item.slice[item.cur].0;
            }
            Some(self.iter_item())
        }
        /// The total number of items in the iteration (this ignores the iteration state; only
        /// [load_from] changes it).  If a 0 multiplier is encountered during the load, the
        /// iterator is empty.
        pub fn num_terms(&self) -> usize {
            if self.exhausted {
                0
            } else {
                self.multiples.iter().map(|item| item.slice.len()).product()
            }
        }
        /// The length of each individual term in the iteration.
        pub fn term_len(&self) -> usize {
            self.bit_terms.len()
        }
    }
}

/// A view object onto a single term of a `SparseObservable`.
///
/// The lengths of `bit_terms` and `indices` are guaranteed to be created equal, but might be zero
/// (in the case that the term is proportional to the identity).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SparseTermView<'a> {
    pub num_qubits: u32,
    pub coeff: Complex64,
    pub bit_terms: &'a [BitTerm],
    pub indices: &'a [u32],
}
impl SparseTermView<'_> {
    /// Convert this `SparseTermView` into an owning [SparseTerm] of the same data.
    pub fn to_term(&self) -> SparseTerm {
        SparseTerm {
            num_qubits: self.num_qubits,
            coeff: self.coeff,
            bit_terms: self.bit_terms.into(),
            indices: self.indices.into(),
        }
    }

    pub fn to_sparse_str(self) -> String {
        let coeff = format!("{}", self.coeff).replace('i', "j");
        let paulis = self
            .indices
            .iter()
            .zip(self.bit_terms)
            .rev()
            .map(|(i, op)| format!("{}_{}", op.py_label(), i))
            .collect::<Vec<String>>()
            .join(" ");
        format!("({coeff})({paulis})")
    }
}

/// A mutable view object onto a single term of a [SparseObservable].
///
/// The lengths of [bit_terms] and [indices] are guaranteed to be created equal, but might be zero
/// (in the case that the term is proportional to the identity).  [indices] is not mutable because
/// this would allow data coherence to be broken.
#[derive(Debug)]
pub struct SparseTermViewMut<'a> {
    pub num_qubits: u32,
    pub coeff: &'a mut Complex64,
    pub bit_terms: &'a mut [BitTerm],
    pub indices: &'a [u32],
}

/// Iterator type allowing in-place mutation of the [SparseObservable].
///
/// Created by [SparseObservable::iter_mut].
#[derive(Debug)]
pub struct IterMut<'a> {
    num_qubits: u32,
    coeffs: &'a mut [Complex64],
    bit_terms: &'a mut [BitTerm],
    indices: &'a [u32],
    boundaries: &'a [usize],
    i: usize,
}
impl<'a> From<&'a mut SparseObservable> for IterMut<'a> {
    fn from(value: &mut SparseObservable) -> IterMut<'_> {
        IterMut {
            num_qubits: value.num_qubits,
            coeffs: &mut value.coeffs,
            bit_terms: &mut value.bit_terms,
            indices: &value.indices,
            boundaries: &value.boundaries,
            i: 0,
        }
    }
}
impl<'a> Iterator for IterMut<'a> {
    type Item = SparseTermViewMut<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        // The trick here is that the lifetime of the 'self' borrow is shorter than the lifetime of
        // the inner borrows.  We can't give out mutable references to our inner borrow, because
        // after the lifetime on 'self' expired, there'd be nothing preventing somebody using the
        // 'self' borrow to see _another_ mutable borrow of the inner data, which would be an
        // aliasing violation.  Instead, we keep splitting the inner views we took out so the
        // mutable references we return don't overlap with the ones we continue to hold.
        let coeffs = ::std::mem::take(&mut self.coeffs);
        let (coeff, other_coeffs) = coeffs.split_first_mut()?;
        self.coeffs = other_coeffs;

        let len = self.boundaries[self.i + 1] - self.boundaries[self.i];
        self.i += 1;

        let all_bit_terms = ::std::mem::take(&mut self.bit_terms);
        let all_indices = ::std::mem::take(&mut self.indices);
        let (bit_terms, rest_bit_terms) = all_bit_terms.split_at_mut(len);
        let (indices, rest_indices) = all_indices.split_at(len);
        self.bit_terms = rest_bit_terms;
        self.indices = rest_indices;

        Some(SparseTermViewMut {
            num_qubits: self.num_qubits,
            coeff,
            bit_terms,
            indices,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.coeffs.len(), Some(self.coeffs.len()))
    }
}
impl ExactSizeIterator for IterMut<'_> {}
impl ::std::iter::FusedIterator for IterMut<'_> {}

/// A single term from a complete :class:`SparseObservable`.
///
/// These are typically created by indexing into or iterating through a :class:`SparseObservable`.
#[derive(Clone, Debug, PartialEq)]
pub struct SparseTerm {
    /// Number of qubits the entire term applies to.
    num_qubits: u32,
    /// The complex coefficient of the term.
    coeff: Complex64,
    bit_terms: Box<[BitTerm]>,
    indices: Box<[u32]>,
}
impl SparseTerm {
    pub fn new(
        num_qubits: u32,
        coeff: Complex64,
        bit_terms: Box<[BitTerm]>,
        indices: Box<[u32]>,
    ) -> Result<Self, CoherenceError> {
        if bit_terms.len() != indices.len() {
            return Err(CoherenceError::MismatchedItemCount {
                bit_terms: bit_terms.len(),
                indices: indices.len(),
            });
        }

        if indices.iter().any(|index| *index >= num_qubits) {
            return Err(CoherenceError::BitIndexTooHigh);
        }

        Ok(Self {
            num_qubits,
            coeff,
            bit_terms,
            indices,
        })
    }

    pub fn num_qubits(&self) -> u32 {
        self.num_qubits
    }

    pub fn coeff(&self) -> Complex64 {
        self.coeff
    }

    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    pub fn bit_terms(&self) -> &[BitTerm] {
        &self.bit_terms
    }

    pub fn view(&self) -> SparseTermView<'_> {
        SparseTermView {
            num_qubits: self.num_qubits,
            coeff: self.coeff,
            bit_terms: &self.bit_terms,
            indices: &self.indices,
        }
    }

    /// Convert this term to a complete :class:`SparseObservable`.
    pub fn to_observable(&self) -> SparseObservable {
        SparseObservable {
            num_qubits: self.num_qubits,
            coeffs: vec![self.coeff],
            bit_terms: self.bit_terms.to_vec(),
            indices: self.indices.to_vec(),
            boundaries: vec![0, self.bit_terms.len()],
        }
    }
}

#[derive(Error, Debug)]
pub struct InnerReadError;

impl ::std::fmt::Display for InnerReadError {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        write!(f, "Failed acquiring lock for reading.")
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_error_safe_add_dense_label() {
        let base = SparseObservable::identity(5);
        let mut modified = base.clone();
        assert!(matches!(
            modified.add_dense_label("IIZ$X", Complex64::new(1.0, 0.0)),
            Err(LabelError::OutsideAlphabet)
        ));
        // `modified` should have been left in a valid state.
        assert_eq!(base, modified);
    }
}

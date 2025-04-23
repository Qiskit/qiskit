// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use hashbrown::HashSet;
use indexmap::IndexSet;
use itertools::Itertools;
use ndarray::Array2;
use num_traits::Zero;
use numpy::{
    PyArray1, PyArray2, PyArrayDescr, PyArrayDescrMethods, PyArrayLike1, PyArrayMethods,
    PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::{
    exceptions::{PyRuntimeError, PyTypeError, PyValueError, PyZeroDivisionError},
    intern,
    prelude::*,
    sync::GILOnceCell,
    types::{IntoPyDict, PyList, PyString, PyTuple, PyType},
    IntoPyObjectExt, PyErr,
};
use std::{
    cmp::Ordering,
    collections::btree_map,
    ops::{AddAssign, DivAssign, MulAssign, SubAssign},
    sync::{Arc, RwLock},
};
use thiserror::Error;

use qiskit_circuit::{
    imports::{ImportOnceCell, NUMPY_COPY_ONLY_IF_NEEDED},
    slice::{PySequenceIndex, SequenceIndex},
};

static PAULI_TYPE: ImportOnceCell = ImportOnceCell::new("qiskit.quantum_info", "Pauli");
static PAULI_LIST_TYPE: ImportOnceCell = ImportOnceCell::new("qiskit.quantum_info", "PauliList");
static SPARSE_PAULI_OP_TYPE: ImportOnceCell =
    ImportOnceCell::new("qiskit.quantum_info", "SparsePauliOp");
static BIT_TERM_PY_ENUM: GILOnceCell<Py<PyType>> = GILOnceCell::new();
static BIT_TERM_INTO_PY: GILOnceCell<[Option<Py<PyAny>>; 16]> = GILOnceCell::new();

/// Named handle to the alphabet of single-qubit terms.
///
/// This is just the Rust-space representation.  We make a separate Python-space `enum.IntEnum` to
/// represent the same information, since we enforce strongly typed interactions in Rust, including
/// not allowing the stored values to be outside the valid `BitTerm`s, but doing so in Python would
/// make it very difficult to use the class efficiently with Numpy array views.  We attach this
/// sister class of `BitTerm` to `PauliLindbladMap` as a scoped class.
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
/// `BitTerm`s.  For interop with Python space, we accept Numpy arrays of `u8` to represent this,
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
    X = 0b10,
    /// Pauli Y operator.
    Y = 0b11,
    /// Pauli Z operator.
    Z = 0b01,
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
        *bits <= 0b11 && *bits != 0
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
            Self::Y => "Y",
            Self::Z => "Z",
        }
    }

    /// Get the name of this `BitTerm`, which is how Python space refers to the integer constant.
    #[inline]
    pub fn py_name(&self) -> &'static str {
        // Note: these names are part of the stable Python API and should not be changed.
        match self {
            Self::X => "X",
            Self::Y => "Y",
            Self::Z => "Z",
        }
    }

    /// Attempt to convert a `u8` into `BitTerm`.
    ///
    /// Unlike the implementation of `TryFrom<u8>`, this allows `b'I'` as an alphabet letter,
    /// returning `Ok(None)` for it.  All other letters outside the alphabet return the complete
    /// error condition.
    #[inline]
    fn try_from_u8(value: u8) -> Result<Option<Self>, BitTermFromU8Error> {
        match value {
            b'I' => Ok(None),
            b'X' => Ok(Some(BitTerm::X)),
            b'Y' => Ok(Some(BitTerm::Y)),
            b'Z' => Ok(Some(BitTerm::Z)),
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

/// Error cases stemming from data coherence at the point of entry into `PauliLindbladMap` from
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
    #[error("the last item of `boundaries` ({last}) must match the length of `bit_terms` and `indices` ({items})")]
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
    #[error("labels must only contain letters from the alphabet 'IXYZ'")]
    OutsideAlphabet,
}

#[derive(Error, Debug)]
pub enum ArithmeticError {
    #[error("mismatched numbers of qubits: {left}, {right}")]
    MismatchedQubits { left: u32, right: u32 },
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
/// See [PyPauliLindbladMap] for detailed docs.
#[derive(Clone, Debug, PartialEq)]
pub struct PauliLindbladMap {
    /// The number of qubits the operator acts on.  This is not inferable from any other shape or
    /// values, since identities are not stored explicitly.
    num_qubits: u32,
    /// The coefficients of each abstract term in in the sum.  This has as many elements as terms in
    /// the sum.
    coeffs: Vec<f64>,
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

impl PauliLindbladMap {
    /// Create a new observable from the raw components that make it up.
    ///
    /// This checks the input values for data coherence on entry.  If you are certain you have the
    /// correct values, you can call `new_unchecked` instead.
    pub fn new(
        num_qubits: u32,
        coeffs: Vec<f64>,
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
        coeffs: Vec<f64>,
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
    /// Recall that two [PauliLindbladMap]s that have different term orders can still represent the
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
    pub fn coeffs(&self) -> &[f64] {
        &self.coeffs
    }

    /// Get a mutable slice of the coefficients.
    #[inline]
    pub fn coeffs_mut(&mut self) -> &mut [f64] {
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
    /// Modifying the indices can cause an incoherent state of the [PauliLindbladMap].
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
    /// Modifying the boundaries can cause an incoherent state of the [PauliLindbladMap].
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
            coeffs: vec![1.0],
            bit_terms: vec![],
            indices: vec![],
            boundaries: vec![0, 0],
        }
    }

    /// Clear all the terms from this operator, making it equal to the zero operator again.
    ///
    /// This does not change the capacity of the internal allocations, so subsequent addition or
    /// substraction operations may not need to reallocate.
    pub fn clear(&mut self) {
        self.coeffs.clear();
        self.bit_terms.clear();
        self.indices.clear();
        self.boundaries.truncate(1);
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
    pub fn term(&self, index: usize) -> SparseTermView {
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

    /// Add the term implied by a dense string label onto this observable.
    pub fn add_dense_label<L: AsRef<[u8]>>(
        &mut self,
        label: L,
        coeff: f64,
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

    /// Reduce the observable to its canonical form.
    ///
    /// This sums like terms, removing them if the final real coefficient's absolute value is
    /// less than or equal to the tolerance.  The terms are reordered to some canonical ordering.
    ///
    /// This function is idempotent.
    pub fn canonicalize(&self, tol: f64) -> PauliLindbladMap {
        let mut terms = btree_map::BTreeMap::new();
        for term in self.iter() {
            terms
                .entry((term.indices, term.bit_terms))
                .and_modify(|c| *c += term.coeff)
                .or_insert(term.coeff);
        }
        let mut out = PauliLindbladMap::zero(self.num_qubits);
        for ((indices, bit_terms), coeff) in terms {
            if coeff * coeff <= tol * tol {
                continue;
            }
            out.coeffs.push(coeff);
            out.bit_terms.extend_from_slice(bit_terms);
            out.indices.extend_from_slice(indices);
            out.boundaries.push(out.indices.len());
        }
        out
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
}

/// A view object onto a single term of a `PauliLindbladMap`.
///
/// The lengths of `bit_terms` and `indices` are guaranteed to be created equal, but might be zero
/// (in the case that the term is proportional to the identity).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SparseTermView<'a> {
    pub num_qubits: u32,
    pub coeff: f64,
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
        format!("({})({})", coeff, paulis)
    }
}

/// A mutable view object onto a single term of a [PauliLindbladMap].
///
/// The lengths of [bit_terms] and [indices] are guaranteed to be created equal, but might be zero
/// (in the case that the term is proportional to the identity).  [indices] is not mutable because
/// this would allow data coherence to be broken.
#[derive(Debug)]
pub struct SparseTermViewMut<'a> {
    pub num_qubits: u32,
    pub coeff: &'a mut f64,
    pub bit_terms: &'a mut [BitTerm],
    pub indices: &'a [u32],
}

/// Iterator type allowing in-place mutation of the [PauliLindbladMap].
///
/// Created by [PauliLindbladMap::iter_mut].
#[derive(Debug)]
pub struct IterMut<'a> {
    num_qubits: u32,
    coeffs: &'a mut [f64],
    bit_terms: &'a mut [BitTerm],
    indices: &'a [u32],
    boundaries: &'a [usize],
    i: usize,
}
impl<'a> From<&'a mut PauliLindbladMap> for IterMut<'a> {
    fn from(value: &mut PauliLindbladMap) -> IterMut {
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

/// A single term from a complete :class:`PauliLindbladMap`.
///
/// These are typically created by indexing into or iterating through a :class:`PauliLindbladMap`.
#[derive(Clone, Debug, PartialEq)]
pub struct SparseTerm {
    /// Number of qubits the entire term applies to.
    num_qubits: u32,
    /// The real coefficient of the term.
    coeff: f64,
    bit_terms: Box<[BitTerm]>,
    indices: Box<[u32]>,
}
impl SparseTerm {
    pub fn new(
        num_qubits: u32,
        coeff: f64,
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

    pub fn coeff(&self) -> f64 {
        self.coeff
    }

    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    pub fn bit_terms(&self) -> &[BitTerm] {
        &self.bit_terms
    }

    pub fn view(&self) -> SparseTermView {
        SparseTermView {
            num_qubits: self.num_qubits,
            coeff: self.coeff,
            bit_terms: &self.bit_terms,
            indices: &self.indices,
        }
    }

    /// Convert this term to a complete :class:`PauliLindbladMap`.
    pub fn to_observable(&self) -> PauliLindbladMap {
        PauliLindbladMap {
            num_qubits: self.num_qubits,
            coeffs: vec![self.coeff],
            bit_terms: self.bit_terms.to_vec(),
            indices: self.indices.to_vec(),
            boundaries: vec![0, self.bit_terms.len()],
        }
    }
}

#[derive(Error, Debug)]
struct InnerReadError;

#[derive(Error, Debug)]
struct InnerWriteError;

impl ::std::fmt::Display for InnerReadError {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        write!(f, "Failed acquiring lock for reading.")
    }
}

impl ::std::fmt::Display for InnerWriteError {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        write!(f, "Failed acquiring lock for writing.")
    }
}

impl From<InnerReadError> for PyErr {
    fn from(value: InnerReadError) -> PyErr {
        PyRuntimeError::new_err(value.to_string())
    }
}
impl From<InnerWriteError> for PyErr {
    fn from(value: InnerWriteError) -> PyErr {
        PyRuntimeError::new_err(value.to_string())
    }
}

impl From<BitTermFromU8Error> for PyErr {
    fn from(value: BitTermFromU8Error) -> PyErr {
        PyValueError::new_err(value.to_string())
    }
}
impl From<CoherenceError> for PyErr {
    fn from(value: CoherenceError) -> PyErr {
        PyValueError::new_err(value.to_string())
    }
}
impl From<LabelError> for PyErr {
    fn from(value: LabelError) -> PyErr {
        PyValueError::new_err(value.to_string())
    }
}
impl From<ArithmeticError> for PyErr {
    fn from(value: ArithmeticError) -> PyErr {
        PyValueError::new_err(value.to_string())
    }
}

/// The single-character string label used to represent this term in the :class:`PauliLindbladMap`
/// alphabet.
#[pyfunction]
#[pyo3(name = "label")]
fn bit_term_label(py: Python, slf: BitTerm) -> &Bound<PyString> {
    // This doesn't use `py_label` so we can use `intern!`.
    match slf {
        BitTerm::X => intern!(py, "X"),
        BitTerm::Y => intern!(py, "Y"),
        BitTerm::Z => intern!(py, "Z"),
    }
}
/// Construct the Python-space `IntEnum` that represents the same values as the Rust-spce `BitTerm`.
///
/// We don't make `BitTerm` a direct `pyclass` because we want the behaviour of `IntEnum`, which
/// specifically also makes its variants subclasses of the Python `int` type; we use a type-safe
/// enum in Rust, but from Python space we expect people to (carefully) deal with the raw ints in
/// Numpy arrays for efficiency.
///
/// The resulting class is attached to `PauliLindbladMap` as a class attribute, and its
/// `__qualname__` is set to reflect this.
fn make_py_bit_term(py: Python) -> PyResult<Py<PyType>> {
    let terms = [
        BitTerm::X,
        BitTerm::Y,
        BitTerm::Z,
    ]
    .into_iter()
    .flat_map(|term| {
        let mut out = vec![(term.py_name(), term as u8)];
        if term.py_name() != term.py_label() {
            // Also ensure that the labels are created as aliases.  These can't be (easily) accessed
            // by attribute-getter (dot) syntax, but will work with the item-getter (square-bracket)
            // syntax, or programmatically with `getattr`.
            out.push((term.py_label(), term as u8));
        }
        out
    })
    .collect::<Vec<_>>();
    let obj = py.import("enum")?.getattr("IntEnum")?.call(
        ("BitTerm", terms),
        Some(
            &[
                ("module", "qiskit.quantum_info"),
                ("qualname", "PauliLindbladMap.BitTerm"),
            ]
            .into_py_dict(py)?,
        ),
    )?;
    let label_property = py
        .import("builtins")?
        .getattr("property")?
        .call1((wrap_pyfunction!(bit_term_label, py)?,))?;
    obj.setattr("label", label_property)?;
    Ok(obj.downcast_into::<PyType>()?.unbind())
}

// Return the relevant value from the Python-space sister enumeration.  These are Python-space
// singletons and subclasses of Python `int`.  We only use this for interaction with "high level"
// Python space; the efficient Numpy-like array paths use `u8` directly so Numpy can act on it
// efficiently.
impl<'py> IntoPyObject<'py> for BitTerm {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        let terms = BIT_TERM_INTO_PY.get_or_init(py, || {
            let py_enum = BIT_TERM_PY_ENUM
                .get_or_try_init(py, || make_py_bit_term(py))
                .expect("creating a simple Python enum class should be infallible")
                .bind(py);
            ::std::array::from_fn(|val| {
                ::bytemuck::checked::try_cast(val as u8)
                    .ok()
                    .map(|term: BitTerm| {
                        py_enum
                            .getattr(term.py_name())
                            .expect("the created `BitTerm` enum should have matching attribute names to the terms")
                            .unbind()
                    })
            })
        });
        Ok(terms[self as usize]
            .as_ref()
            .expect("the lookup table initializer populated a 'Some' in all valid locations")
            .bind(py)
            .clone())
    }
}

impl<'py> FromPyObject<'py> for BitTerm {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let value = ob
            .extract::<isize>()
            .map_err(|_| match ob.get_type().repr() {
                Ok(repr) => PyTypeError::new_err(format!("bad type for 'BitTerm': {}", repr)),
                Err(err) => err,
            })?;
        let value_error = || {
            PyValueError::new_err(format!(
                "value {} is not a valid letter of the single-qubit alphabet for 'BitTerm'",
                value
            ))
        };
        let value: u8 = value.try_into().map_err(|_| value_error())?;
        value.try_into().map_err(|_| value_error())
    }
}

/// A single term from a complete :class:`PauliLindbladMap`.
///
/// These are typically created by indexing into or iterating through a :class:`PauliLindbladMap`.
#[pyclass(name = "Term", frozen, module = "qiskit.quantum_info")]
#[derive(Clone, Debug)]
struct PySparseTerm {
    inner: SparseTerm,
}
#[pymethods]
impl PySparseTerm {
    // Mark the Python class as being defined "within" the `PauliLindbladMap` class namespace.
    #[classattr]
    #[pyo3(name = "__qualname__")]
    fn type_qualname() -> &'static str {
        "PauliLindbladMap.Term"
    }

    #[new]
    #[pyo3(signature = (/, num_qubits, coeff, bit_terms, indices))]
    fn py_new(
        num_qubits: u32,
        coeff: f64,
        bit_terms: Vec<BitTerm>,
        indices: Vec<u32>,
    ) -> PyResult<Self> {
        if bit_terms.len() != indices.len() {
            return Err(CoherenceError::MismatchedItemCount {
                bit_terms: bit_terms.len(),
                indices: indices.len(),
            }
            .into());
        }
        let mut order = (0..bit_terms.len()).collect::<Vec<_>>();
        order.sort_unstable_by_key(|a| indices[*a]);
        let bit_terms = order.iter().map(|i| bit_terms[*i]).collect();
        let mut sorted_indices = Vec::<u32>::with_capacity(order.len());
        for i in order {
            let index = indices[i];
            if sorted_indices
                .last()
                .map(|prev| *prev >= index)
                .unwrap_or(false)
            {
                return Err(CoherenceError::UnsortedIndices.into());
            }
            sorted_indices.push(index)
        }
        let inner = SparseTerm::new(
            num_qubits,
            coeff,
            bit_terms,
            sorted_indices.into_boxed_slice(),
        )?;
        Ok(PySparseTerm { inner })
    }

    /// Convert this term to a complete :class:`PauliLindbladMap`.
    fn to_observable(&self) -> PyResult<PyPauliLindbladMap> {
        let obs = PauliLindbladMap::new(
            self.inner.num_qubits(),
            vec![self.inner.coeff()],
            self.inner.bit_terms().to_vec(),
            self.inner.indices().to_vec(),
            vec![0, self.inner.bit_terms().len()],
        )?;
        Ok(obs.into())
    }

    fn to_label(&self) -> PyResult<String> {
        Ok(self.inner.view().to_sparse_str())
    }

    fn __eq__(slf: Bound<Self>, other: Bound<PyAny>) -> PyResult<bool> {
        if slf.is(&other) {
            return Ok(true);
        }
        let Ok(other) = other.downcast_into::<Self>() else {
            return Ok(false);
        };
        let slf = slf.borrow();
        let other = other.borrow();
        Ok(slf.inner.eq(&other.inner))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "<{} on {} qubit{}: {}>",
            Self::type_qualname(),
            self.inner.num_qubits(),
            if self.inner.num_qubits() == 1 {
                ""
            } else {
                "s"
            },
            self.inner.view().to_sparse_str(),
        ))
    }

    fn __getnewargs__(slf_: Bound<Self>) -> PyResult<Bound<PyTuple>> {
        let py = slf_.py();
        let borrowed = slf_.borrow();
        (
            borrowed.inner.num_qubits(),
            borrowed.inner.coeff(),
            Self::get_bit_terms(slf_.clone()),
            Self::get_indices(slf_),
        )
            .into_pyobject(py)
    }

    /// Get a copy of this term.
    fn copy(&self) -> Self {
        self.clone()
    }

    /// Read-only view onto the individual single-qubit terms.
    ///
    /// The only valid values in the array are those with a corresponding
    /// :class:`~PauliLindbladMap.BitTerm`.
    #[getter]
    fn get_bit_terms(slf_: Bound<Self>) -> Bound<PyArray1<u8>> {
        let borrowed = slf_.borrow();
        let bit_terms = borrowed.inner.bit_terms();
        let arr = ::ndarray::aview1(::bytemuck::cast_slice::<_, u8>(bit_terms));
        // SAFETY: in order to call this function, the lifetime of `self` must be managed by Python.
        // We tie the lifetime of the array to `slf_`, and there are no public ways to modify the
        // `Box<[BitTerm]>` allocation (including dropping or reallocating it) other than the entire
        // object getting dropped, which Python will keep safe.
        let out = unsafe { PyArray1::borrow_from_array(&arr, slf_.into_any()) };
        out.readwrite().make_nonwriteable();
        out
    }

    /// The number of qubits the term is defined on.
    #[getter]
    fn get_num_qubits(&self) -> u32 {
        self.inner.num_qubits()
    }

    /// The term's coefficient.
    #[getter]
    fn get_coeff(&self) -> f64 {
        self.inner.coeff()
    }

    /// Read-only view onto the indices of each non-identity single-qubit term.
    ///
    /// The indices will always be in sorted order.
    #[getter]
    fn get_indices(slf_: Bound<Self>) -> Bound<PyArray1<u32>> {
        let borrowed = slf_.borrow();
        let indices = borrowed.inner.indices();
        let arr = ::ndarray::aview1(indices);
        // SAFETY: in order to call this function, the lifetime of `self` must be managed by Python.
        // We tie the lifetime of the array to `slf_`, and there are no public ways to modify the
        // `Box<[u32]>` allocation (including dropping or reallocating it) other than the entire
        // object getting dropped, which Python will keep safe.
        let out = unsafe { PyArray1::borrow_from_array(&arr, slf_.into_any()) };
        out.readwrite().make_nonwriteable();
        out
    }

    /// Get a :class:`.Pauli` object that represents the measurement basis needed for this term.
    ///
    /// For example, the projector ``0l+`` will return a Pauli ``ZYX``.  The resulting
    /// :class:`.Pauli` is dense, in the sense that explicit identities are stored.  An identity in
    /// the Pauli output does not require a concrete measurement.
    ///
    /// Returns:
    ///     :class:`.Pauli`: the Pauli operator representing the necessary measurement basis.
    ///
    /// See also:
    ///     :meth:`PauliLindbladMap.pauli_bases`
    ///         A similar method for an entire observable at once.
    fn pauli_base<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let mut x = vec![false; self.inner.num_qubits() as usize];
        let mut z = vec![false; self.inner.num_qubits() as usize];
        for (bit_term, index) in self
            .inner
            .bit_terms()
            .iter()
            .zip(self.inner.indices().iter())
        {
            x[*index as usize] = bit_term.has_x_component();
            z[*index as usize] = bit_term.has_z_component();
        }
        PAULI_TYPE
            .get_bound(py)
            .call1(((PyArray1::from_vec(py, z), PyArray1::from_vec(py, x)),))
    }

    /// Return the bit labels of the term as string.
    ///
    /// The bit labels will match the order of :attr:`.SparseTerm.indices`, such that the
    /// i-th character in the string is applied to the qubit index at ``term.indices[i]``.
    ///
    /// Returns:
    ///     The non-identity bit terms as concatenated string.
    fn bit_labels<'py>(&self, py: Python<'py>) -> Bound<'py, PyString> {
        let string: String = self
            .inner
            .bit_terms()
            .iter()
            .map(|bit| bit.py_label())
            .collect();
        PyString::new(py, string.as_str())
    }
}

#[pyclass(name = "PauliLindbladMap", module = "qiskit.quantum_info", sequence)]
#[derive(Debug)]
pub struct PyPauliLindbladMap {
    // This class keeps a pointer to a pure Rust-SparseTerm and serves as interface from Python.
    inner: Arc<RwLock<PauliLindbladMap>>,
}
#[pymethods]
impl PyPauliLindbladMap {
    #[pyo3(signature = (data, /, num_qubits=None))]
    #[new]
    fn py_new(data: &Bound<PyAny>, num_qubits: Option<u32>) -> PyResult<Self> {
        let py = data.py();
        let check_num_qubits = |data: &Bound<PyAny>| -> PyResult<()> {
            let Some(num_qubits) = num_qubits else {
                return Ok(());
            };
            let other_qubits = data.getattr(intern!(py, "num_qubits"))?.extract::<u32>()?;
            if num_qubits == other_qubits {
                return Ok(());
            }
            Err(PyValueError::new_err(format!(
                "explicitly given 'num_qubits' ({num_qubits}) does not match operator ({other_qubits})"
            )))
        };

        if data.is_instance(PAULI_TYPE.get_bound(py))? {
            check_num_qubits(data)?;
            return Self::from_pauli(data);
        }
        if data.is_instance(SPARSE_PAULI_OP_TYPE.get_bound(py))? {
            check_num_qubits(data)?;
            return Self::from_sparse_pauli_op(data);
        }
        if let Ok(label) = data.extract::<String>() {
            let num_qubits = num_qubits.unwrap_or(label.len() as u32);
            if num_qubits as usize != label.len() {
                return Err(PyValueError::new_err(format!(
                    "explicitly given 'num_qubits' ({}) does not match label ({})",
                    num_qubits,
                    label.len(),
                )));
            }
            return Self::from_label(&label).map_err(PyErr::from);
        }
        if let Ok(observable) = data.downcast_exact::<Self>() {
            check_num_qubits(data)?;
            let borrowed = observable.borrow();
            let inner = borrowed.inner.read().map_err(|_| InnerReadError)?;
            return Ok(inner.clone().into());
        }
        // The type of `vec` is inferred from the subsequent calls to `Self::from_list` or
        // `Self::from_sparse_list` to be either the two-tuple or the three-tuple form during the
        // `extract`.  The empty list will pass either, but it means the same to both functions.
        if let Ok(vec) = data.extract() {
            return Self::from_list(vec, num_qubits);
        }
        if let Ok(vec) = data.extract() {
            let Some(num_qubits) = num_qubits else {
                return Err(PyValueError::new_err(
                    "if using the sparse-list form, 'num_qubits' must be provided",
                ));
            };
            return Self::from_sparse_list(vec, num_qubits);
        }
        if let Ok(term) = data.downcast_exact::<PySparseTerm>() {
            return term.borrow().to_observable();
        };
        if let Ok(observable) = Self::from_terms(data, num_qubits) {
            return Ok(observable);
        }
        Err(PyTypeError::new_err(format!(
            "unknown input format for 'PauliLindbladMap': {}",
            data.get_type().repr()?,
        )))
    }

    /// Get a copy of this observable.
    ///
    /// Examples:
    ///
    ///     .. code-block:: python
    ///
    ///         >>> obs = PauliLindbladMap.from_list([("IXZ+lr01", 2.5), ("ZXI-rl10", 0.5j)])
    ///         >>> assert obs == obs.copy()
    ///         >>> assert obs is not obs.copy()
    fn copy(&self) -> PyResult<Self> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        Ok(inner.clone().into())
    }

    /// The number of qubits the operator acts on.
    ///
    /// This is not inferable from any other shape or values, since identities are not stored
    /// explicitly.
    #[getter]
    #[inline]
    pub fn num_qubits(&self) -> PyResult<u32> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        Ok(inner.num_qubits())
    }

    /// The number of terms in the sum this operator is tracking.
    #[getter]
    #[inline]
    pub fn num_terms(&self) -> PyResult<usize> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        Ok(inner.num_terms())
    }

    /// The coefficients of each abstract term in in the sum.  This has as many elements as terms in
    /// the sum.
    #[getter]
    fn get_coeffs(slf_: &Bound<Self>) -> ArrayView {
        let borrowed = slf_.borrow();
        ArrayView {
            base: borrowed.inner.clone(),
            slot: ArraySlot::Coeffs,
        }
    }

    /// A flat list of single-qubit terms.  This is more naturally a list of lists, but is stored
    /// flat for memory usage and locality reasons, with the sublists denoted by `boundaries.`
    #[getter]
    fn get_bit_terms(slf_: &Bound<Self>) -> ArrayView {
        let borrowed = slf_.borrow();
        ArrayView {
            base: borrowed.inner.clone(),
            slot: ArraySlot::BitTerms,
        }
    }

    /// A flat list of the qubit indices that the corresponding entries in :attr:`bit_terms` act on.
    /// This list must always be term-wise sorted, where a term is a sublist as denoted by
    /// :attr:`boundaries`.
    ///
    /// .. warning::
    ///
    ///     If writing to this attribute from Python space, you *must* ensure that you only write in
    ///     indices that are term-wise sorted.
    #[getter]
    fn get_indices(slf_: &Bound<Self>) -> ArrayView {
        let borrowed = slf_.borrow();
        ArrayView {
            base: borrowed.inner.clone(),
            slot: ArraySlot::Indices,
        }
    }

    /// Indices that partition :attr:`bit_terms` and :attr:`indices` into sublists for each
    /// individual term in the sum.  ``boundaries[0] : boundaries[1]`` is the range of indices into
    /// :attr:`bit_terms` and :attr:`indices` that correspond to the first term of the sum.  All
    /// unspecified qubit indices are implicitly the identity.  This is one item longer than
    /// :attr:`coeffs`, since ``boundaries[0]`` is always an explicit zero (for algorithmic ease).
    #[getter]
    fn get_boundaries(slf_: &Bound<Self>) -> ArrayView {
        let borrowed = slf_.borrow();
        ArrayView {
            base: borrowed.inner.clone(),
            slot: ArraySlot::Boundaries,
        }
    }

    /// Get the zero operator over the given number of qubits.
    ///
    /// The zero operator is the operator whose expectation value is zero for all quantum states.
    /// It has no terms.  It is the identity element for addition of two :class:`PauliLindbladMap`
    /// instances; anything added to the zero operator is equal to itself.
    ///
    /// If you want the projector onto the all zeros state, use::
    ///
    ///     >>> num_qubits = 10
    ///     >>> all_zeros = PauliLindbladMap.from_label("0" * num_qubits)
    ///
    /// Examples:
    ///
    ///     Get the zero operator for 100 qubits::
    ///
    ///         >>> PauliLindbladMap.zero(100)
    ///         <PauliLindbladMap with 0 terms on 100 qubits: 0.0>
    #[pyo3(signature = (/, num_qubits))]
    #[staticmethod]
    pub fn zero(num_qubits: u32) -> Self {
        PauliLindbladMap::zero(num_qubits).into()
    }

    /// Get the identity operator over the given number of qubits.
    ///
    /// Examples:
    ///
    ///     Get the identity operator for 100 qubits::
    ///
    ///         >>> PauliLindbladMap.identity(100)
    ///         <PauliLindbladMap with 1 term on 100 qubits: (1+0j)()>
    #[pyo3(signature = (/, num_qubits))]
    #[staticmethod]
    pub fn identity(num_qubits: u32) -> Self {
        PauliLindbladMap::identity(num_qubits).into()
    }

        /// Construct a :class:`.SparseObservable` from a single :class:`.Pauli` instance.
    ///
    /// The output observable will have a single term, with a unitary coefficient dependent on the
    /// phase.
    ///
    /// Args:
    ///     pauli (:class:`.Pauli`): the single Pauli to convert.
    ///
    /// Examples:
    ///
    ///     .. code-block:: python
    ///
    ///         >>> label = "IYXZI"
    ///         >>> pauli = Pauli(label)
    ///         >>> SparseObservable.from_pauli(pauli)
    ///         <SparseObservable with 1 term on 5 qubits: (1+0j)(Y_3 X_2 Z_1)>
    ///         >>> assert SparseObservable.from_label(label) == SparseObservable.from_pauli(pauli)
    #[staticmethod]
    #[pyo3(signature = (pauli, /))]
    fn from_pauli(pauli: &Bound<PyAny>) -> PyResult<Self> {
        let py = pauli.py();
        let num_qubits = pauli.getattr(intern!(py, "num_qubits"))?.extract::<u32>()?;
        let z = pauli
            .getattr(intern!(py, "z"))?
            .extract::<PyReadonlyArray1<bool>>()?;
        let x = pauli
            .getattr(intern!(py, "x"))?
            .extract::<PyReadonlyArray1<bool>>()?;
        let mut bit_terms = Vec::new();
        let mut indices = Vec::new();
        let mut num_ys = 0;
        for (i, (x, z)) in x.as_array().iter().zip(z.as_array().iter()).enumerate() {
            // The only failure case possible here is the identity, because of how we're
            // constructing the value to convert.
            let Ok(term) = ::bytemuck::checked::try_cast(((*x as u8) << 1) | (*z as u8)) else {
                continue;
            };
            num_ys += (term == BitTerm::Y) as isize;
            indices.push(i as u32);
            bit_terms.push(term);
        }
        let boundaries = vec![0, indices.len()];
        // The "empty" state of a `Pauli` represents the identity, which isn't our empty state
        // (that's zero), so we're always going to have a coefficient.
        let group_phase = pauli
            // `Pauli`'s `_phase` is a Numpy array ...
            .getattr(intern!(py, "_phase"))?
            // ... that should have exactly 1 element ...
            .call_method0(intern!(py, "item"))?
            // ... which is some integral type.
            .extract::<isize>()?;

        let phase_idx = (group_phase - num_ys).rem_euclid(4);

        // If the phase is j or -j, raise an error
        if phase_idx == 1 || phase_idx == 3 {
            return Err(PyValueError::new_err("only 'Pauli' with real phases can be converted to 'PauliLindbladMap'"));
        }

        let phase = match phase_idx {
            0 => 1.0,
            2 => -1.0,
            _ => unreachable!("`x % 4` has only four values"),
        };
        
        let coeffs = vec![phase];
        let inner = PauliLindbladMap::new(num_qubits, coeffs, bit_terms, indices, boundaries)?;
        Ok(inner.into())
    }

    /// Construct a single-term observable from a dense string label.
    ///
    /// The resulting operator will have a coefficient of 1.  The label must be a sequence of the
    /// alphabet ``'IXYZ+-rl01'``.  The label is interpreted analogously to a bitstring.  In other
    /// words, the right-most letter is associated with qubit 0, and so on.  This is the same as the
    /// labels for :class:`.Pauli` and :class:`.SparsePauliOp`.
    ///
    /// Args:
    ///     label (str): the dense label.
    ///
    /// Examples:
    ///
    ///     .. code-block:: python
    ///
    ///         >>> SparseObservable.from_label("IIII+ZI")
    ///         <SparseObservable with 1 term on 7 qubits: (1+0j)(+_2 Z_1)>
    ///         >>> label = "IYXZI"
    ///         >>> pauli = Pauli(label)
    ///         >>> assert SparseObservable.from_label(label) == SparseObservable.from_pauli(pauli)
    ///
    /// See also:
    ///     :meth:`from_list`
    ///         A generalization of this method that constructs a sum operator from multiple labels
    ///         and their corresponding coefficients.
    #[staticmethod]
    #[pyo3(signature = (label, /))]
    fn from_label(label: &str) -> Result<Self, LabelError> {
        let mut inner = PauliLindbladMap::zero(label.len() as u32);
        inner.add_dense_label(label, 1.0)?;
        Ok(inner.into())
    }

    /// Construct an observable from a list of dense labels and coefficients.
    ///
    /// This is analogous to :meth:`.SparsePauliOp.from_list`, except it uses
    /// :ref:`the extended alphabet <sparse-observable-alphabet>` of :class:`.SparseObservable`.  In
    /// this dense form, you must supply all identities explicitly in each label.
    ///
    /// The label must be a sequence of the alphabet ``'IXYZ+-rl01'``.  The label is interpreted
    /// analogously to a bitstring.  In other words, the right-most letter is associated with qubit
    /// 0, and so on.  This is the same as the labels for :class:`.Pauli` and
    /// :class:`.SparsePauliOp`.
    ///
    /// Args:
    ///     iter (list[tuple[str, float]]): Pairs of labels and their associated coefficients to
    ///         sum. The labels are interpreted the same way as in :meth:`from_label`.
    ///     num_qubits (int | None): It is not necessary to specify this if you are sure that
    ///         ``iter`` is not an empty sequence, since it can be inferred from the label lengths.
    ///         If ``iter`` may be empty, you must specify this argument to disambiguate how many
    ///         qubits the observable is for.  If this is given and ``iter`` is not empty, the value
    ///         must match the label lengths.
    ///
    /// Examples:
    ///
    ///     Construct an observable from a list of labels of the same length::
    ///
    ///         >>> SparseObservable.from_list([
    ///         ...     ("III++", 1.0),
    ///         ...     ("II--I", 1.0j),
    ///         ...     ("I++II", -0.5),
    ///         ...     ("--III", -0.25j),
    ///         ... ])
    ///         <SparseObservable with 4 terms on 5 qubits:
    ///             (1+0j)(+_1 +_0) + (0+1j)(-_2 -_1) + (-0.5+0j)(+_3 +_2) + (-0-0.25j)(-_4 -_3)>
    ///
    ///     Use ``num_qubits`` to disambiguate potentially empty inputs::
    ///
    ///         >>> SparseObservable.from_list([], num_qubits=10)
    ///         <SparseObservable with 0 terms on 10 qubits: 0.0>
    ///
    ///     This method is equivalent to calls to :meth:`from_sparse_list` with the explicit
    ///     qubit-arguments field set to decreasing integers::
    ///
    ///         >>> labels = ["XY+Z", "rl01", "-lXZ"]
    ///         >>> coeffs = [1.5j, 2.0, -0.5]
    ///         >>> from_list = SparseObservable.from_list(list(zip(labels, coeffs)))
    ///         >>> from_sparse_list = SparseObservable.from_sparse_list([
    ///         ...     (label, (3, 2, 1, 0), coeff)
    ///         ...     for label, coeff in zip(labels, coeffs)
    ///         ... ])
    ///         >>> assert from_list == from_sparse_list
    ///
    /// See also:
    ///     :meth:`from_label`
    ///         A similar constructor, but takes only a single label and always has its coefficient
    ///         set to ``1.0``.
    ///
    ///     :meth:`from_sparse_list`
    ///         Construct the observable from a list of labels without explicit identities, but with
    ///         the qubits each single-qubit term applies to listed explicitly.
    #[staticmethod]
    #[pyo3(signature = (iter, /, *, num_qubits=None))]
    fn from_list(iter: Vec<(String, f64)>, num_qubits: Option<u32>) -> PyResult<Self> {
        if iter.is_empty() && num_qubits.is_none() {
            return Err(PyValueError::new_err(
                "cannot construct an observable from an empty list without knowing `num_qubits`",
            ));
        }
        let num_qubits = match num_qubits {
            Some(num_qubits) => num_qubits,
            None => iter[0].0.len() as u32,
        };
        let mut inner = PauliLindbladMap::with_capacity(num_qubits, iter.len(), 0);
        for (label, coeff) in iter {
            inner.add_dense_label(&label, coeff)?;
        }
        Ok(inner.into())
    }

    /// Construct a :class:`.SparseObservable` from a :class:`.SparsePauliOp` instance.
    ///
    /// This will be a largely direct translation of the :class:`.SparsePauliOp`; in particular,
    /// there is no on-the-fly summing of like terms, nor any attempt to refactorize sums of Pauli
    /// terms into equivalent projection operators.
    ///
    /// Args:
    ///     op (:class:`.SparsePauliOp`): the operator to convert.
    ///
    /// Examples:
    ///
    ///     .. code-block:: python
    ///
    ///         >>> spo = SparsePauliOp.from_list([("III", 1.0), ("IIZ", 0.5), ("IZI", 0.5)])
    ///         >>> SparseObservable.from_sparse_pauli_op(spo)
    ///         <SparseObservable with 3 terms on 3 qubits: (1+0j)() + (0.5+0j)(Z_0) + (0.5+0j)(Z_1)>
    #[staticmethod]
    #[pyo3(signature = (op, /))]
    fn from_sparse_pauli_op(op: &Bound<PyAny>) -> PyResult<Self> {
        let py = op.py();
        let pauli_list_ob = op.getattr(intern!(py, "paulis"))?;
        let coeffs = op
            .getattr(intern!(py, "coeffs"))?
            .extract::<PyReadonlyArray1<f64>>()
            .map_err(|_| PyTypeError::new_err("only 'SparsePauliOp' with real-typed coefficients can be converted to 'PauliLindbladMap'"))?
            .as_array()
            .to_vec();
        let op_z = pauli_list_ob
            .getattr(intern!(py, "z"))?
            .extract::<PyReadonlyArray2<bool>>()?;
        let op_x = pauli_list_ob
            .getattr(intern!(py, "x"))?
            .extract::<PyReadonlyArray2<bool>>()?;
        // We don't extract the `phase`, because that's supposed to be 0 for all `SparsePauliOp`
        // instances - they use the symplectic convention in the representation with any phase term
        // absorbed into the coefficients (like us).
        let [num_terms, num_qubits] = *op_z.shape() else {
            unreachable!("shape is statically known to be 2D")
        };
        if op_x.shape() != [num_terms, num_qubits] {
            return Err(PyValueError::new_err(format!(
                "'x' and 'z' have different shapes ({:?} and {:?})",
                op_x.shape(),
                op_z.shape()
            )));
        }
        if num_terms != coeffs.len() {
            return Err(PyValueError::new_err(format!(
                "'x' and 'z' have a different number of operators to 'coeffs' ({} and {})",
                num_terms,
                coeffs.len(),
            )));
        }

        let mut bit_terms = Vec::new();
        let mut indices = Vec::new();
        let mut boundaries = Vec::with_capacity(num_terms + 1);
        boundaries.push(0);
        for (term_x, term_z) in op_x
            .as_array()
            .rows()
            .into_iter()
            .zip(op_z.as_array().rows())
        {
            for (i, (x, z)) in term_x.iter().zip(term_z.iter()).enumerate() {
                // The only failure case possible here is the identity, because of how we're
                // constructing the value to convert.
                let Ok(term) = ::bytemuck::checked::try_cast(((*x as u8) << 1) | (*z as u8)) else {
                    continue;
                };
                indices.push(i as u32);
                bit_terms.push(term);
            }
            boundaries.push(indices.len());
        }

        let inner =
            PauliLindbladMap::new(num_qubits as u32, coeffs, bit_terms, indices, boundaries)?;
        Ok(inner.into())
    }

    /// Construct a :class:`SparseObservable` out of individual terms.
    ///
    /// All the terms must have the same number of qubits.  If supplied, the ``num_qubits`` argument
    /// must match the terms.
    ///
    /// No simplification is done as part of the observable creation.
    ///
    /// Args:
    ///     obj (Iterable[Term]): Iterable of individual terms to build the observable from.
    ///     num_qubits (int | None): The number of qubits the observable should act on.  This is
    ///         usually inferred from the input, but can be explicitly given to handle the case
    ///         of an empty iterable.
    ///
    /// Returns:
    ///     The corresponding observable.
    #[staticmethod]
    #[pyo3(signature = (obj, /, num_qubits=None))]
    fn from_terms(obj: &Bound<PyAny>, num_qubits: Option<u32>) -> PyResult<Self> {
        let mut iter = obj.try_iter()?;
        let mut inner = match num_qubits {
            Some(num_qubits) => PauliLindbladMap::zero(num_qubits),
            None => {
                let Some(first) = iter.next() else {
                    return Err(PyValueError::new_err(
                        "cannot construct an observable from an empty list without knowing `num_qubits`",
                    ));
                };
                let py_term = first?.downcast::<PySparseTerm>()?.borrow();
                py_term.inner.to_observable()
            }
        };
        for bound_py_term in iter {
            let py_term = bound_py_term?.downcast::<PySparseTerm>()?.borrow();
            inner.add_term(py_term.inner.view())?;
        }
        Ok(inner.into())
    }

    // SAFETY: this cannot invoke undefined behaviour if `check = true`, but if `check = false` then
    // the `bit_terms` must all be valid `BitTerm` representations.
    /// Construct a :class:`.SparseObservable` from raw Numpy arrays that match :ref:`the required
    /// data representation described in the class-level documentation <sparse-observable-arrays>`.
    ///
    /// The data from each array is copied into fresh, growable Rust-space allocations.
    ///
    /// Args:
    ///     num_qubits: number of qubits in the observable.
    ///     coeffs: float coefficients of each term of the observable.  This should be a Numpy
    ///         array with dtype :attr:`~numpy.float64`.
    ///     bit_terms: flattened list of the single-qubit terms comprising all complete terms.  This
    ///         should be a Numpy array with dtype :attr:`~numpy.uint8` (which is compatible with
    ///         :class:`.BitTerm`).
    ///     indices: flattened term-wise sorted list of the qubits each single-qubit term corresponds
    ///         to.  This should be a Numpy array with dtype :attr:`~numpy.uint32`.
    ///     boundaries: the indices that partition ``bit_terms`` and ``indices`` into terms.  This
    ///         should be a Numpy array with dtype :attr:`~numpy.uintp`.
    ///     check: if ``True`` (the default), validate that the data satisfies all coherence
    ///         guarantees.  If ``False``, no checks are done.
    ///
    ///         .. warning::
    ///
    ///             If ``check=False``, the ``bit_terms`` absolutely *must* be all be valid values
    ///             of :class:`.SparseObservable.BitTerm`.  If they are not, Rust-space undefined
    ///             behavior may occur, entirely invalidating the program execution.
    ///
    /// Examples:
    ///
    ///     Construct a sum of :math:`Z` on each individual qubit::
    ///
    ///         >>> num_qubits = 100
    ///         >>> terms = np.full((num_qubits,), SparseObservable.BitTerm.Z, dtype=np.uint8)
    ///         >>> indices = np.arange(num_qubits, dtype=np.uint32)
    ///         >>> coeffs = np.ones((num_qubits,), dtype=float)
    ///         >>> boundaries = np.arange(num_qubits + 1, dtype=np.uintp)
    ///         >>> SparseObservable.from_raw_parts(num_qubits, coeffs, terms, indices, boundaries)
    ///         <SparseObservable with 100 terms on 100 qubits: (1+0j)(Z_0) + ... + (1+0j)(Z_99)>
    #[staticmethod]
    #[pyo3(
        signature = (/, num_qubits, coeffs, bit_terms, indices, boundaries, check=true),
    )]
    unsafe fn from_raw_parts<'py>(
        num_qubits: u32,
        coeffs: PyArrayLike1<'py, f64>,
        bit_terms: PyArrayLike1<'py, u8>,
        indices: PyArrayLike1<'py, u32>,
        boundaries: PyArrayLike1<'py, usize>,
        check: bool,
    ) -> PyResult<Self> {
        let coeffs = coeffs.as_array().to_vec();
        let bit_terms = if check {
            bit_terms
                .as_array()
                .into_iter()
                .copied()
                .map(BitTerm::try_from)
                .collect::<Result<_, _>>()?
        } else {
            let bit_terms_as_u8 = bit_terms.as_array().to_vec();
            // SAFETY: the caller enforced that each `u8` is a valid `BitTerm`, and `BitTerm` is be
            // represented by a `u8`.  We can't use `bytemuck` because we're casting a `Vec`.
            unsafe { ::std::mem::transmute::<Vec<u8>, Vec<BitTerm>>(bit_terms_as_u8) }
        };
        let indices = indices.as_array().to_vec();
        let boundaries = boundaries.as_array().to_vec();

        let inner = if check {
            PauliLindbladMap::new(num_qubits, coeffs, bit_terms, indices, boundaries)
                .map_err(PyErr::from)
        } else {
            // SAFETY: the caller promised they have upheld the coherence guarantees.
            Ok(unsafe {
                PauliLindbladMap::new_unchecked(num_qubits, coeffs, bit_terms, indices, boundaries)
            })
        }?;
        Ok(inner.into())
    }

    /// Clear all the terms from this operator, making it equal to the zero operator again.
    ///
    /// This does not change the capacity of the internal allocations, so subsequent addition or
    /// substraction operations may not need to reallocate.
    ///
    /// Examples:
    ///
    ///     .. code-block:: python
    ///
    ///         >>> obs = SparseObservable.from_list([("IX+-rl", 2.0), ("01YZII", -1j)])
    ///         >>> obs.clear()
    ///         >>> assert obs == SparseObservable.zero(obs.py_num_qubits())
    pub fn clear(&mut self) -> PyResult<()> {
        let mut inner = self.inner.write().map_err(|_| InnerWriteError)?;
        inner.clear();
        Ok(())
    }

    /// Construct an observable from a list of labels, the qubits each item applies to, and the
    /// coefficient of the whole term.
    ///
    /// This is analogous to :meth:`.SparsePauliOp.from_sparse_list`, except it uses
    /// :ref:`the extended alphabet <sparse-observable-alphabet>` of :class:`.PauliLindbladMap`.
    ///
    /// The "labels" and "indices" fields of the triples are associated by zipping them together.
    /// For example, this means that a call to :meth:`from_list` can be converted to the form used
    /// by this method by setting the "indices" field of each triple to ``(num_qubits-1, ..., 1,
    /// 0)``.
    ///
    /// Args:
    ///     iter (list[tuple[str, Sequence[int], float]]): triples of labels, the qubits
    ///         each single-qubit term applies to, and the coefficient of the entire term.
    ///
    ///     num_qubits (int): the number of qubits in the operator.
    ///
    /// Examples:
    ///
    ///     Construct a simple operator::
    ///
    ///         >>> PauliLindbladMap.from_sparse_list(
    ///         ...     [("ZX", (1, 4), 1.0), ("YY", (0, 3), 2j)],
    ///         ...     num_qubits=5,
    ///         ... )
    ///         <PauliLindbladMap with 2 terms on 5 qubits: (1+0j)(X_4 Z_1) + (0+2j)(Y_3 Y_0)>
    ///
    ///     Construct the identity observable (though really, just use :meth:`identity`)::
    ///
    ///         >>> PauliLindbladMap.from_sparse_list([("", (), 1.0)], num_qubits=100)
    ///         <PauliLindbladMap with 1 term on 100 qubits: (1+0j)()>
    ///
    ///     This method can replicate the behavior of :meth:`from_list`, if the qubit-arguments
    ///     field of the triple is set to decreasing integers::
    ///
    ///         >>> labels = ["XY+Z", "rl01", "-lXZ"]
    ///         >>> coeffs = [1.5j, 2.0, -0.5]
    ///         >>> from_list = PauliLindbladMap.from_list(list(zip(labels, coeffs)))
    ///         >>> from_sparse_list = PauliLindbladMap.from_sparse_list([
    ///         ...     (label, (3, 2, 1, 0), coeff)
    ///         ...     for label, coeff in zip(labels, coeffs)
    ///         ... ])
    ///         >>> assert from_list == from_sparse_list
    ///
    /// See also:
    ///     :meth:`to_sparse_list`
    ///         The reverse of this method.
    #[staticmethod]
    #[pyo3(signature = (iter, /, num_qubits))]
    fn from_sparse_list(
        iter: Vec<(String, Vec<u32>, f64)>,
        num_qubits: u32,
    ) -> PyResult<Self> {
        let coeffs = iter.iter().map(|(_, _, coeff)| *coeff).collect();
        let mut boundaries = Vec::with_capacity(iter.len() + 1);
        boundaries.push(0);
        let mut indices = Vec::new();
        let mut bit_terms = Vec::new();
        // Insertions to the `BTreeMap` keep it sorted by keys, so we use this to do the termwise
        // sorting on-the-fly.
        let mut sorted = btree_map::BTreeMap::new();
        for (label, qubits, _) in iter {
            sorted.clear();
            let label: &[u8] = label.as_ref();
            if label.len() != qubits.len() {
                return Err(LabelError::WrongLengthIndices {
                    label: label.len(),
                    indices: indices.len(),
                }
                .into());
            }
            for (letter, index) in label.iter().zip(qubits) {
                if index >= num_qubits {
                    return Err(LabelError::BadIndex { index, num_qubits }.into());
                }
                let btree_map::Entry::Vacant(entry) = sorted.entry(index) else {
                    return Err(LabelError::DuplicateIndex { index }.into());
                };
                entry.insert(
                    BitTerm::try_from_u8(*letter).map_err(|_| LabelError::OutsideAlphabet)?,
                );
            }
            for (index, term) in sorted.iter() {
                let Some(term) = term else {
                    continue;
                };
                indices.push(*index);
                bit_terms.push(*term);
            }
            boundaries.push(bit_terms.len());
        }
        let inner = PauliLindbladMap::new(num_qubits, coeffs, bit_terms, indices, boundaries)?;
        Ok(inner.into())
    }

    /// Express the observable in terms of a sparse list format.
    ///
    /// This can be seen as counter-operation of :meth:`.PauliLindbladMap.from_sparse_list`, however
    /// the order of terms is not guaranteed to be the same at after a roundtrip to a sparse
    /// list and back.
    ///
    /// Examples:
    ///
    ///     >>> obs = PauliLindbladMap.from_list([("IIXIZ", 2j), ("IIZIX", 2j)])
    ///     >>> reconstructed = PauliLindbladMap.from_sparse_list(obs.to_sparse_list(), obs.num_qubits)
    ///
    /// See also:
    ///     :meth:`from_sparse_list`
    ///         The constructor that can interpret these lists.
    #[pyo3(signature = ())]
    fn to_sparse_list(&self, py: Python) -> PyResult<Py<PyList>> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;

        // turn a SparseView into a Python tuple of (bit terms, indices, coeff)
        let to_py_tuple = |view: SparseTermView| {
            let mut pauli_string = String::with_capacity(view.bit_terms.len());

            for bit in view.bit_terms.iter() {
                pauli_string.push_str(bit.py_label());
            }
            let py_string = PyString::new(py, &pauli_string).unbind();
            let py_indices = PyList::new(py, view.indices.iter())?.unbind();
            let py_coeff = view.coeff.into_py_any(py)?;

            PyTuple::new(py, vec![py_string.as_any(), py_indices.as_any(), &py_coeff])
        };

        let out = PyList::empty(py);
        for view in inner.iter() {
            out.append(to_py_tuple(view)?)?;
        }
        Ok(out.unbind())
    }

    /// Sum any like terms in this operator, removing them if the resulting real coefficient has
    /// an absolute value within tolerance of zero.
    ///
    /// As a side effect, this sorts the operator into :ref:`canonical order
    /// <sparse-observable-canonical-order>`.
    ///
    /// .. note::
    ///
    ///     When using this for equality comparisons, note that floating-point rounding and the
    ///     non-associativity fo floating-point addition may cause non-zero coefficients of summed
    ///     terms to compare unequal.  To compare two observables up to a tolerance, it is safest to
    ///     compare the canonicalized difference of the two observables to zero.
    ///
    /// Args:
    ///     tol (float): after summing like terms, any coefficients whose absolute value is less
    ///         than the given absolute tolerance will be suppressed from the output.
    ///
    /// Examples:
    ///
    ///     Using :meth:`simplify` to compare two operators that represent the same observable, but
    ///     would compare unequal due to the structural tests by default::
    ///
    ///         >>> base = PauliLindbladMap.from_sparse_list([
    ///         ...     ("XZ", (2, 1), 1e-10),  # value too small
    ///         ...     ("+-", (3, 1), 2j),
    ///         ...     ("+-", (3, 1), 2j),     # can be combined with the above
    ///         ...     ("01", (3, 1), 0.5),    # out of order compared to `expected`
    ///         ... ], num_qubits=5)
    ///         >>> expected = PauliLindbladMap.from_list([("I0I1I", 0.5), ("I+I-I", 4j)])
    ///         >>> assert base != expected  # non-canonical comparison
    ///         >>> assert base.simplify() == expected.simplify()
    ///
    ///     Note that in the above example, the coefficients are chosen such that all floating-point
    ///     calculations are exact, and there are no intermediate rounding or associativity
    ///     concerns.  If this cannot be guaranteed to be the case, the safer form is::
    ///
    ///         >>> left = PauliLindbladMap.from_list([("XYZ", 1.0/3.0)] * 3)   # sums to 1.0
    ///         >>> right = PauliLindbladMap.from_list([("XYZ", 1.0/7.0)] * 7)  # doesn't sum to 1.0
    ///         >>> assert left.simplify() != right.simplify()
    ///         >>> assert (left - right).simplify() == PauliLindbladMap.zero(left.num_qubits)
    #[pyo3(
        signature = (/, tol=1e-8),
    )]
    fn simplify(&self, tol: f64) -> PyResult<Self> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        let simplified = inner.canonicalize(tol);
        Ok(simplified.into())
    }

    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        index: PySequenceIndex<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        let indices = match index.with_len(inner.num_terms())? {
            SequenceIndex::Int(index) => {
                return PySparseTerm {
                    inner: inner.term(index).to_term(),
                }
                .into_bound_py_any(py)
            }
            indices => indices,
        };
        let mut out = PauliLindbladMap::zero(inner.num_qubits());
        for index in indices.iter() {
            out.add_term(inner.term(index))?;
        }
        out.into_bound_py_any(py)
    }

    fn __eq__(slf: Bound<Self>, other: Bound<PyAny>) -> PyResult<bool> {
        // this is also important to check before trying to read both slf and other
        if slf.is(&other) {
            return Ok(true);
        }
        let Ok(other) = other.downcast_into::<Self>() else {
            return Ok(false);
        };
        let slf_borrowed = slf.borrow();
        let other_borrowed = other.borrow();
        let slf_inner = slf_borrowed.inner.read().map_err(|_| InnerReadError)?;
        let other_inner = other_borrowed.inner.read().map_err(|_| InnerReadError)?;
        Ok(slf_inner.eq(&other_inner))
    }

    fn __repr__(&self) -> PyResult<String> {
        let num_terms = self.num_terms()?;
        let num_qubits = self.num_qubits()?;

        let str_num_terms = format!(
            "{} term{}",
            num_terms,
            if num_terms == 1 { "" } else { "s" }
        );
        let str_num_qubits = format!(
            "{} qubit{}",
            num_qubits,
            if num_qubits == 1 { "" } else { "s" }
        );

        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        let str_terms = if num_terms == 0 {
            "0.0".to_owned()
        } else {
            inner
                .iter()
                .map(SparseTermView::to_sparse_str)
                .collect::<Vec<_>>()
                .join(" + ")
        };
        Ok(format!(
            "<SparseObservable with {} on {}: {}>",
            str_num_terms, str_num_qubits, str_terms
        ))
    }
}

impl From<PauliLindbladMap> for PyPauliLindbladMap {
    fn from(val: PauliLindbladMap) -> PyPauliLindbladMap {
        PyPauliLindbladMap {
            inner: Arc::new(RwLock::new(val)),
        }
    }
}
impl<'py> IntoPyObject<'py> for PauliLindbladMap {
    type Target = PyPauliLindbladMap;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        PyPauliLindbladMap::from(self).into_pyobject(py)
    }
}

/// Helper class of `ArrayView` that denotes the slot of the `PauliLindbladMap` we're looking at.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ArraySlot {
    Coeffs,
    BitTerms,
    Indices,
    Boundaries,
}

/// Custom wrapper sequence class to get safe views onto the Rust-space data.  We can't directly
/// expose Python-managed wrapped pointers without introducing some form of runtime exclusion on the
/// ability of `PauliLindbladMap` to re-allocate in place; we can't leave dangling pointers for
/// Python space.
#[pyclass(frozen, sequence)]
struct ArrayView {
    base: Arc<RwLock<PauliLindbladMap>>,
    slot: ArraySlot,
}
#[pymethods]
impl ArrayView {
    fn __repr__(&self, py: Python) -> PyResult<String> {
        let obs = self.base.read().map_err(|_| InnerReadError)?;
        let data = match self.slot {
            // Simple integers look the same in Rust-space debug as Python.
            ArraySlot::Indices => format!("{:?}", obs.indices()),
            ArraySlot::Boundaries => format!("{:?}", obs.boundaries()),
            // Complexes don't have a nice repr in Rust, so just delegate the whole load to Python
            // and convert back.
            ArraySlot::Coeffs => PyList::new(py, obs.coeffs())?.repr()?.to_string(),
            ArraySlot::BitTerms => format!(
                "[{}]",
                obs.bit_terms()
                    .iter()
                    .map(BitTerm::py_label)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        };
        Ok(format!(
            "<observable {} view: {}>",
            match self.slot {
                ArraySlot::Coeffs => "coeffs",
                ArraySlot::BitTerms => "bit_terms",
                ArraySlot::Indices => "indices",
                ArraySlot::Boundaries => "boundaries",
            },
            data,
        ))
    }

    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        index: PySequenceIndex,
    ) -> PyResult<Bound<'py, PyAny>> {
        // The slightly verbose generic setup here is to allow the type of a scalar return to be
        // different to the type that gets put into the Numpy array, since the `BitTerm` enum can be
        // a direct scalar, but for Numpy, we need it to be a raw `u8`.
        fn get_from_slice<'py, T, S>(
            py: Python<'py>,
            slice: &[T],
            index: PySequenceIndex,
        ) -> PyResult<Bound<'py, PyAny>>
        where
            T: IntoPyObject<'py> + Copy + Into<S>,
            S: ::numpy::Element,
        {
            match index.with_len(slice.len())? {
                SequenceIndex::Int(index) => slice[index].into_bound_py_any(py),
                indices => PyArray1::from_iter(py, indices.iter().map(|index| slice[index].into()))
                    .into_bound_py_any(py),
            }
        }

        let obs = self.base.read().map_err(|_| InnerReadError)?;
        match self.slot {
            ArraySlot::Coeffs => get_from_slice::<_, f64>(py, obs.coeffs(), index),
            ArraySlot::BitTerms => get_from_slice::<_, u8>(py, obs.bit_terms(), index),
            ArraySlot::Indices => get_from_slice::<_, u32>(py, obs.indices(), index),
            ArraySlot::Boundaries => get_from_slice::<_, usize>(py, obs.boundaries(), index),
        }
    }

    fn __setitem__(&self, index: PySequenceIndex, values: &Bound<PyAny>) -> PyResult<()> {
        /// Set values of a slice according to the indexer, using `extract` to retrieve the
        /// Rust-space object from the collection of Python-space values.
        ///
        /// This indirects the Python extraction through an intermediate type to marginally improve
        /// the error messages for things like `BitTerm`, where Python-space extraction might fail
        /// because the user supplied an invalid alphabet letter.
        ///
        /// This allows broadcasting a single item into many locations in a slice (like Numpy), but
        /// otherwise requires that the index and values are the same length (unlike Python's
        /// `list`) because that would change the length.
        fn set_in_slice<'py, T, S>(
            slice: &mut [T],
            index: PySequenceIndex<'py>,
            values: &Bound<'py, PyAny>,
        ) -> PyResult<()>
        where
            T: Copy + TryFrom<S>,
            S: FromPyObject<'py>,
            PyErr: From<<T as TryFrom<S>>::Error>,
        {
            match index.with_len(slice.len())? {
                SequenceIndex::Int(index) => {
                    slice[index] = values.extract::<S>()?.try_into()?;
                    Ok(())
                }
                indices => {
                    if let Ok(value) = values.extract::<S>() {
                        let value = value.try_into()?;
                        for index in indices {
                            slice[index] = value;
                        }
                    } else {
                        let values = values
                            .try_iter()?
                            .map(|value| value?.extract::<S>()?.try_into().map_err(PyErr::from))
                            .collect::<PyResult<Vec<_>>>()?;
                        if indices.len() != values.len() {
                            return Err(PyValueError::new_err(format!(
                                "tried to set a slice of length {} with a sequence of length {}",
                                indices.len(),
                                values.len(),
                            )));
                        }
                        for (index, value) in indices.into_iter().zip(values) {
                            slice[index] = value;
                        }
                    }
                    Ok(())
                }
            }
        }

        let mut obs = self.base.write().map_err(|_| InnerWriteError)?;
        match self.slot {
            ArraySlot::Coeffs => set_in_slice::<_, f64>(obs.coeffs_mut(), index, values),
            ArraySlot::BitTerms => set_in_slice::<BitTerm, u8>(obs.bit_terms_mut(), index, values),
            ArraySlot::Indices => unsafe {
                set_in_slice::<_, u32>(obs.indices_mut(), index, values)
            },
            ArraySlot::Boundaries => unsafe {
                set_in_slice::<_, usize>(obs.boundaries_mut(), index, values)
            },
        }
    }

    fn __len__(&self, _py: Python) -> PyResult<usize> {
        let obs = self.base.read().map_err(|_| InnerReadError)?;
        let len = match self.slot {
            ArraySlot::Coeffs => obs.coeffs().len(),
            ArraySlot::BitTerms => obs.bit_terms().len(),
            ArraySlot::Indices => obs.indices().len(),
            ArraySlot::Boundaries => obs.boundaries().len(),
        };
        Ok(len)
    }

    #[pyo3(signature = (/, dtype=None, copy=None))]
    fn __array__<'py>(
        &self,
        py: Python<'py>,
        dtype: Option<&Bound<'py, PyAny>>,
        copy: Option<bool>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // This method always copies, so we don't leave dangling pointers lying around in Numpy
        // arrays; it's not enough just to set the `base` of the Numpy array to the
        // `PauliLindbladMap`, since the `Vec` we're referring to might re-allocate and invalidate
        // the pointer the Numpy array is wrapping.
        if !copy.unwrap_or(true) {
            return Err(PyValueError::new_err(
                "cannot produce a safe view onto movable memory",
            ));
        }
        let obs = self.base.read().map_err(|_| InnerReadError)?;
        match self.slot {
            ArraySlot::Coeffs => cast_array_type(py, PyArray1::from_slice(py, obs.coeffs()), dtype),
            ArraySlot::Indices => {
                cast_array_type(py, PyArray1::from_slice(py, obs.indices()), dtype)
            }
            ArraySlot::Boundaries => {
                cast_array_type(py, PyArray1::from_slice(py, obs.boundaries()), dtype)
            }
            ArraySlot::BitTerms => {
                let bit_terms: &[u8] = ::bytemuck::cast_slice(obs.bit_terms());
                cast_array_type(py, PyArray1::from_slice(py, bit_terms), dtype)
            }
        }
    }
}

/// Use the Numpy Python API to convert a `PyArray` into a dynamically chosen `dtype`, copying only
/// if required.
fn cast_array_type<'py, T>(
    py: Python<'py>,
    array: Bound<'py, PyArray1<T>>,
    dtype: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let base_dtype = array.dtype();
    let dtype = dtype
        .map(|dtype| PyArrayDescr::new(py, dtype))
        .unwrap_or_else(|| Ok(base_dtype.clone()))?;
    if dtype.is_equiv_to(&base_dtype) {
        return Ok(array.into_any());
    }
    PyModule::import(py, intern!(py, "numpy"))?
        .getattr(intern!(py, "array"))?
        .call(
            (array,),
            Some(
                &[
                    (intern!(py, "copy"), NUMPY_COPY_ONLY_IF_NEEDED.get_bound(py)),
                    (intern!(py, "dtype"), dtype.as_any()),
                ]
                .into_py_dict(py)?,
            ),
        )
}

pub fn pauli_lindblad_map(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyPauliLindbladMap>()?;
    Ok(())
}
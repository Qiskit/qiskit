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

use numpy::{
    PyArray1, PyArrayDescr, PyArrayDescrMethods, PyArrayLike1, PyArrayMethods,
    PyUntypedArrayMethods,
};
use pyo3::{
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    intern,
    prelude::*,
    sync::GILOnceCell,
    types::{IntoPyDict, PyList, PyString, PyTuple, PyType},
    IntoPyObjectExt, PyErr,
};
use std::{
    collections::btree_map,
    sync::{Arc, RwLock},
};
use thiserror::Error;

use qiskit_circuit::{
    imports::NUMPY_COPY_ONLY_IF_NEEDED,
    slice::{PySequenceIndex, SequenceIndex},
};

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
/// two bits are the symplectic Pauli representation of the Pauli operator, with the associations
/// `0b10` <-> `X`, `0b01` <-> `Z`, `0b11` <-> `Y`. The `0b00` representation thus ends up being the
/// natural representation of the `I` operator, but this is never stored, and is not named in the
/// enumeration.
///
/// This operator does not store phase terms of $-i$.  `BitTerm::Y` has `(1, 1)` as its `(z, x)`
/// representation, and represents exactly the Pauli Y operator. Additional phases, if needed, must
/// be stored elsewhere.
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
    #[error("cannot shrink the qubit count in a Pauli Lindblad map from {current} to {target}")]
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

/// A list of Pauli operators stored in a qubit-sparse format.
///
/// See [PyQubitSparsePauliList] for detailed docs.
#[derive(Clone, Debug, PartialEq)]
pub struct QubitSparsePauliList {
    /// The number of qubits the map acts on.  This is not inferable from any other shape or
    /// values, since identities are not stored explicitly.
    num_qubits: u32,
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


impl QubitSparsePauliList {
    /// Create a new qubit-sparse Pauli list from the raw components that make it up.
    ///
    /// This checks the input values for data coherence on entry.  If you are certain you have the
    /// correct values, you can call `new_unchecked` instead.
    pub fn new(
        num_qubits: u32,
        bit_terms: Vec<BitTerm>,
        indices: Vec<u32>,
        boundaries: Vec<usize>,
    ) -> Result<Self, CoherenceError> {
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
        Ok(unsafe { Self::new_unchecked(num_qubits, bit_terms, indices, boundaries) })
    }

    /// Create a new [QubitSparsePauliList] from the raw components without checking data coherence.
    ///
    /// # Safety
    ///
    /// It is up to the caller to ensure that the data-coherence requirements, as enumerated in the
    /// struct-level documentation, have been upheld.
    #[inline(always)]
    pub unsafe fn new_unchecked(
        num_qubits: u32,
        bit_terms: Vec<BitTerm>,
        indices: Vec<u32>,
        boundaries: Vec<usize>,
    ) -> Self {
        Self {
            num_qubits,
            bit_terms,
            indices,
            boundaries,
        }
    }

    /// Create a new empty list with pre-allocated space for the given number of Paulis and
    /// single-qubit bit terms.
    #[inline]
    pub fn with_capacity(num_qubits: u32, num_terms: usize, num_bit_terms: usize) -> Self {
        Self {
            num_qubits,
            bit_terms: Vec::with_capacity(num_bit_terms),
            indices: Vec::with_capacity(num_bit_terms),
            boundaries: {
                let mut boundaries = Vec::with_capacity(num_terms + 1);
                boundaries.push(0);
                boundaries
            },
        }
    }

    /// Get an iterator over the individual elements of the list.
    pub fn iter(&'_ self) -> impl ExactSizeIterator<Item = SparseTermView<'_>> + '_ {
        std::ops::Range {start: 0, end: self.boundaries.len() - 1}.map(|i| {
            let start = self.boundaries[i];
            let end = self.boundaries[i + 1];
            SparseTermView {
                num_qubits: self.num_qubits,
                bit_terms: &self.bit_terms[start..end],
                indices: &self.indices[start..end],
            }
        })
    }

    /// Get an iterator over the individual list terms that allows in-place mutation.
    ///
    /// The length and indices of these views cannot be mutated, since both would allow breaking
    /// data coherence.
    pub fn iter_mut(&mut self) -> IterMut<'_> {
        self.into()
    }

    /// Get the number of qubits the map is defined on.
    #[inline]
    pub fn num_qubits(&self) -> u32 {
        self.num_qubits
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
    /// Modifying the indices can cause an incoherent state of the [QubitSparsePauliList].
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
    /// Modifying the boundaries can cause an incoherent state of the [QubitSparsePauliList].
    /// It should be ensured that the boundaries are sorted and the length/elements are consistent
    /// with the bit_terms and indices.
    #[inline]
    pub unsafe fn boundaries_mut(&mut self) -> &mut [usize] {
        &mut self.boundaries
    }

    /// Get the [BitTerm]s in the list.
    #[inline]
    pub fn bit_terms(&self) -> &[BitTerm] {
        &self.bit_terms
    }

    /// Get a mutable slice of the bit terms.
    #[inline]
    pub fn bit_terms_mut(&mut self) -> &mut [BitTerm] {
        &mut self.bit_terms
    }

    /// Clear all the elements of the list.
    ///
    /// This does not change the capacity of the internal allocations, so subsequent addition or
    /// substraction of elements in the list may not need to reallocate.
    pub fn clear(&mut self) {
        self.bit_terms.clear();
        self.indices.clear();
        self.boundaries.truncate(1);
    }

    /// Get a view onto a representation of a single sparse Pauli.
    ///
    /// This is effectively an indexing operation into the [QubitSparsePauliList].
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
            bit_terms: &self.bit_terms[start..end],
            indices: &self.indices[start..end],
        }
    }

    /// Add an element to the list implied by a dense string label.
    pub fn add_dense_label<L: AsRef<[u8]>>(
        &mut self,
        label: L,
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
        self.boundaries.push(self.bit_terms.len());
        Ok(())
    }

    /// Add a single generator term to this map.
    pub fn add_term(&mut self, term: SparseTermView) -> Result<(), ArithmeticError> {
        if self.num_qubits != term.num_qubits {
            return Err(ArithmeticError::MismatchedQubits {
                left: self.num_qubits,
                right: term.num_qubits,
            });
        }
        self.bit_terms.extend_from_slice(term.bit_terms);
        self.indices.extend_from_slice(term.indices);
        self.boundaries.push(self.bit_terms.len());
        Ok(())
    }
}

/// A view object onto a single term of a `QubitSparsePauliList`.
///
/// The lengths of `bit_terms` and `indices` are guaranteed to be created equal, but might be zero
/// (in the case that the term is proportional to the identity).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SparseTermView<'a> {
    pub num_qubits: u32,
    pub bit_terms: &'a [BitTerm],
    pub indices: &'a [u32],
}
impl SparseTermView<'_> {
    /// Convert this `SparseTermView` into an owning [SparseTerm] of the same data.
    pub fn to_term(&self) -> SparseTerm {
        SparseTerm {
            num_qubits: self.num_qubits,
            bit_terms: self.bit_terms.into(),
            indices: self.indices.into(),
        }
    }

    pub fn to_sparse_str(self) -> String {
        let paulis = self
            .indices
            .iter()
            .zip(self.bit_terms)
            .rev()
            .map(|(i, op)| format!("{}_{}", op.py_label(), i))
            .collect::<Vec<String>>()
            .join(" ");
        format!("{}", paulis)
    }
}

/// A mutable view object onto a single term of a [QubitSparsePauliList].
///
/// The lengths of [bit_terms] and [indices] are guaranteed to be created equal, but might be zero
/// (in the case that the generator term is proportional to the identity).  [indices] is not mutable
/// because this would allow data coherence to be broken.
#[derive(Debug)]
pub struct SparseTermViewMut<'a> {
    pub num_qubits: u32,
    pub bit_terms: &'a mut [BitTerm],
    pub indices: &'a [u32],
}

/// Iterator type allowing in-place mutation of the [QubitSparsePauliList].
///
/// Created by [QubitSparsePauliList::iter_mut].
#[derive(Debug)]
pub struct IterMut<'a> {
    num_qubits: u32,
    bit_terms: &'a mut [BitTerm],
    indices: &'a [u32],
    boundaries: &'a [usize],
    i: usize,
}
impl<'a> From<&'a mut QubitSparsePauliList> for IterMut<'a> {
    fn from(value: &mut QubitSparsePauliList) -> IterMut {
        IterMut {
            num_qubits: value.num_qubits,
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
            bit_terms,
            indices,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.boundaries.len() - 1, Some(self.boundaries.len() - 1))
    }
}
impl ExactSizeIterator for IterMut<'_> {}
impl ::std::iter::FusedIterator for IterMut<'_> {}


/// A single term from a complete :class:`QubitSparsePauliList`.
///
/// These are typically created by indexing into or iterating through a :class:`QubitSparsePauliList`.
#[derive(Clone, Debug, PartialEq)]
pub struct SparseTerm {
    /// Number of qubits the entire term applies to.
    num_qubits: u32,
    bit_terms: Box<[BitTerm]>,
    indices: Box<[u32]>,
}
impl SparseTerm {
    pub fn new(
        num_qubits: u32,
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
            bit_terms,
            indices,
        })
    }

    pub fn num_qubits(&self) -> u32 {
        self.num_qubits
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
            bit_terms: &self.bit_terms,
            indices: &self.indices,
        }
    }

    /// Convert this term to a complete :class:`QubitSparsePauliList`.
    pub fn to_qubit_sparse_pauli_list(&self) -> QubitSparsePauliList {
        QubitSparsePauliList {
            num_qubits: self.num_qubits,
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

/// The single-character string label used to represent this term in the :class:`QubitSparsePauliList`
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
/// The resulting class is attached to `QubitSparsePauliList` as a class attribute, and its
/// `__qualname__` is set to reflect this.
fn make_py_bit_term(py: Python) -> PyResult<Py<PyType>> {
    let terms = [BitTerm::X, BitTerm::Y, BitTerm::Z]
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
                ("qualname", "QubitSparsePauliList.BitTerm"),
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

/// A single term from a complete :class:`QubitSparsePauliList`.
///
/// These are typically created by indexing into or iterating through a :class:`QubitSparsePauliList`.
#[pyclass(name = "Term", frozen, module = "qiskit.quantum_info")]
#[derive(Clone, Debug)]
struct PySparseTerm {
    inner: SparseTerm,
}
#[pymethods]
impl PySparseTerm {
    // Mark the Python class as being defined "within" the `QubitSparsePauliList` class namespace.
    #[classattr]
    #[pyo3(name = "__qualname__")]
    fn type_qualname() -> &'static str {
        "QubitSparsePauliList.Term"
    }

    #[new]
    #[pyo3(signature = (/, num_qubits, bit_terms, indices))]
    fn py_new(
        num_qubits: u32,
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
            bit_terms,
            sorted_indices.into_boxed_slice(),
        )?;
        Ok(PySparseTerm { inner })
    }

    /// Convert this term to a complete :class:`QubitSparsePauliList`.
    fn to_pauli_lindblad_map(&self) -> PyResult<PyQubitSparsePauliList> {
        let qubit_sparse_pauli_list = QubitSparsePauliList::new(
            self.inner.num_qubits(),
            self.inner.bit_terms().to_vec(),
            self.inner.indices().to_vec(),
            vec![0, self.inner.bit_terms().len()],
        )?;
        Ok(qubit_sparse_pauli_list.into())
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

#[pyclass(name = "QubitSparsePauliList", module = "qiskit.quantum_info", sequence)]
#[derive(Debug)]
pub struct PyQubitSparsePauliList {
    // This class keeps a pointer to a pure Rust-SparseTerm and serves as interface from Python.
    inner: Arc<RwLock<QubitSparsePauliList>>,
}
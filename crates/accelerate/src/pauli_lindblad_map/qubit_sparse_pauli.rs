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

use num_complex::Complex64;
use numpy::{
    PyArray1, PyArrayDescr, PyArrayDescrMethods, PyArrayLike1, PyArrayMethods,
    PyReadonlyArray1, PyUntypedArrayMethods,
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
    imports::{ImportOnceCell, NUMPY_COPY_ONLY_IF_NEEDED},
    slice::{PySequenceIndex, SequenceIndex},
};

static PAULI_TYPE: ImportOnceCell = ImportOnceCell::new("qiskit.quantum_info", "Pauli");
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
    pub fn iter(&'_ self) -> impl ExactSizeIterator<Item = QubitSparsePauliView<'_>> + '_ {
        std::ops::Range {start: 0, end: self.boundaries.len() - 1}.map(|i| {
            let start = self.boundaries[i];
            let end = self.boundaries[i + 1];
            QubitSparsePauliView {
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

    /// Get the number of elements in the list.
    #[inline]
    pub fn num_terms(&self) -> usize {
        self.boundaries.len() - 1
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

    /// Create a [QubitSparsePauliList] representing the empty list on ``num_qubits`` qubits.
    pub fn empty(num_qubits: u32) -> Self {
        Self::with_capacity(num_qubits, 0, 0)
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
    pub fn term(&self, index: usize) -> QubitSparsePauliView {
        debug_assert!(index < self.num_terms(), "index {index} out of bounds");
        let start = self.boundaries[index];
        let end = self.boundaries[index + 1];
        QubitSparsePauliView {
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
    pub fn add_qubit_sparse_pauli(&mut self, term: QubitSparsePauliView) -> Result<(), ArithmeticError> {
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
pub struct QubitSparsePauliView<'a> {
    pub num_qubits: u32,
    pub bit_terms: &'a [BitTerm],
    pub indices: &'a [u32],
}
impl QubitSparsePauliView<'_> {
    /// Convert this `QubitSparsePauliView` into an owning [QubitSparsePauli] of the same data.
    pub fn to_term(&self) -> QubitSparsePauli {
        QubitSparsePauli {
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
pub struct QubitSparsePauliViewMut<'a> {
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
    type Item = QubitSparsePauliViewMut<'a>;

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

        Some(QubitSparsePauliViewMut {
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
pub struct QubitSparsePauli {
    /// Number of qubits the entire term applies to.
    num_qubits: u32,
    bit_terms: Box<[BitTerm]>,
    indices: Box<[u32]>,
}
impl QubitSparsePauli {
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

    pub fn view(&self) -> QubitSparsePauliView {
        QubitSparsePauliView {
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
#[pyclass(name = "QubitSparsePauli", frozen, module = "qiskit.quantum_info")]
#[derive(Clone, Debug)]
pub struct PyQubitSparsePauli {
    inner: QubitSparsePauli,
}
#[pymethods]
impl PyQubitSparsePauli {

    #[new]
    #[pyo3(signature = (data, /, num_qubits=None))]
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
        if let Ok(sparse_label) = data.extract() {
            let Some(num_qubits) = num_qubits else {
                return Err(PyValueError::new_err(
                    "if using the sparse-label form, 'num_qubits' must be provided",
                ));
            };
            return Self::from_sparse_label(sparse_label, num_qubits);
        }
        Err(PyTypeError::new_err(format!(
            "unknown input format for 'PauliLindbladMap': {}",
            data.get_type().repr()?,
        )))
    }

    // SAFETY: this cannot invoke undefined behaviour if `check = true`, but if `check = false` then
    // the `bit_terms` must all be valid `BitTerm` representations.
    /// Construct a :class:`.PauliLindbladMap` from raw Numpy arrays that match :ref:`the required
    /// data representation described in the class-level documentation
    /// <pauli-lindblad-map-arrays>`.
    ///
    /// The data from each array is copied into fresh, growable Rust-space allocations.
    ///
    /// Args:
    ///     num_qubits: number of qubits the map acts on.
    ///     coeffs: float coefficients of each generator term of the map.  This should be a Numpy
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
    ///             of :class:`.PauliLindbladMap.BitTerm`.  If they are not, Rust-space undefined
    ///             behavior may occur, entirely invalidating the program execution.
    ///
    /// Examples:
    ///
    ///     Construct a sum of :math:`Z` on each individual qubit::
    ///
    ///         >>> num_qubits = 100
    ///         >>> terms = np.full((num_qubits,), PauliLindbladMap.BitTerm.Z, dtype=np.uint8)
    ///         >>> indices = np.arange(num_qubits, dtype=np.uint32)
    ///         >>> coeffs = np.ones((num_qubits,), dtype=float)
    ///         >>> boundaries = np.arange(num_qubits + 1, dtype=np.uintp)
    ///         >>> PauliLindbladMap.from_raw_parts(num_qubits, coeffs, terms, indices, boundaries)
    ///         <PauliLindbladMap with 100 terms on 100 qubits: (1)L(Z_0) + ... + (1)L(Z_99)>
    #[staticmethod]
    #[pyo3(signature = (/, num_qubits, bit_terms, indices))]
    fn from_raw_parts(
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
        let inner = QubitSparsePauli::new(
            num_qubits,
            bit_terms,
            sorted_indices.into_boxed_slice(),
        )?;
        Ok(PyQubitSparsePauli { inner })
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
    fn from_label(label: &str) -> PyResult<Self> {
        let label: &[u8] = label.as_ref();
        let num_qubits = label.len() as u32;
        let mut bit_terms = Vec::new();
        let mut indices = Vec::new();
        // The only valid characters in the alphabet are ASCII, so if we see something other than
        // ASCII, we're already in the failure path.
        for (i, letter) in label.iter().rev().enumerate() {
            match BitTerm::try_from_u8(*letter) {
                Ok(Some(term)) => {
                    bit_terms.push(term);
                    indices.push(i as u32);
                }
                Ok(None) => (),
                Err(_) => {
                    return Err(PyErr::from(LabelError::OutsideAlphabet));
                }
            }
        }
        let inner = QubitSparsePauli::new(num_qubits, bit_terms.into_boxed_slice(), indices.into_boxed_slice())?;
        Ok(inner.into())
    }

    /// NOTE DAN: The phase is dropped, document this ************************************************
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
        let inner = QubitSparsePauli::new(num_qubits, bit_terms.into_boxed_slice(), indices.into_boxed_slice())?;
        Ok(inner.into())
    }

    /// Construct a Pauli Lindblad map from a list of labels, the qubits each item applies to, and
    /// the coefficient of the whole term.
    ///
    /// This is analogous to :meth:`.SparsePauliOp.from_sparse_list`.
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
    ///     num_qubits (int): the number of qubits the map acts on.
    ///
    /// Examples:
    ///
    ///     Construct a simple map::
    ///
    ///         >>> PauliLindbladMap.from_sparse_list(
    ///         ...     [("ZX", (1, 4), 1.0), ("YY", (0, 3), 2)],
    ///         ...     num_qubits=5,
    ///         ... )
    ///         <PauliLindbladMap with 2 terms on 5 qubits: (1)L(X_4 Z_1) + (2)L(Y_3 Y_0)>
    ///
    ///     This method can replicate the behavior of :meth:`from_list`, if the qubit-arguments
    ///     field of the triple is set to decreasing integers::
    ///
    ///         >>> labels = ["XYXZ", "YYZZ", "XYXZ"]
    ///         >>> coeffs = [1.5, 2.0, -0.5]
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
    #[pyo3(signature = (/, sparse_label, num_qubits))]
    fn from_sparse_label(sparse_label: (String, Vec<u32>), num_qubits: u32) -> PyResult<Self> {
        
        let label = sparse_label.0;
        let indices = sparse_label.1;
        let mut bit_terms = Vec::new();
        let mut sorted_indices = Vec::new();

        let label: &[u8] = label.as_ref();
        let mut sorted = btree_map::BTreeMap::new();
        if label.len() != indices.len() {
            return Err(LabelError::WrongLengthIndices {
                label: label.len(),
                indices: indices.len(),
            }
            .into());
        }
        for (letter, index) in label.iter().zip(indices) {
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
            sorted_indices.push(*index);
            bit_terms.push(*term);
        }

        let inner = QubitSparsePauli::new(num_qubits, bit_terms.into_boxed_slice(), sorted_indices.into_boxed_slice())?;
        Ok(inner.into())
    }

    /// Convert this term to a complete :class:`QubitSparsePauliList`.
    fn to_qubit_sparse_pauli_list(&self) -> PyResult<PyQubitSparsePauliList> {
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
            "QubitSparsePauli",
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

    // The documentation for this is inlined into the class-level documentation of
    // `PauliLindbladMap`.
    #[allow(non_snake_case)]
    #[classattr]
    fn BitTerm(py: Python) -> PyResult<Py<PyType>> {
        BIT_TERM_PY_ENUM
            .get_or_try_init(py, || make_py_bit_term(py))
            .map(|obj| obj.clone_ref(py))
    }
}


/// A Pauli Lindblad map stored in a qubit-sparse format.
///
/// Mathematics
/// ===========
///
/// A Pauli-Lindblad map is a linear map acting on density matrices on :math:`n`-qubits of the form
///
/// .. math::
///
///     \Lamdba = \exp\left(\sum_{P \in K} \lambda_P P \cdot P - \cdot\right)
///
/// where :math:`K` is a subset of :math:`n`-qubit Pauli operators, and the coefficients
/// :math:`\lambda_P` are real numbers. When all the coefficients :math:`\lambda_P` are
/// non-negative, this corresponds to a completely positive and trace preserving map. The sum in the
/// exponential is called the generator, and each individual term the generators. To simplify
/// notation in the rest of the documention, we denote :math:`L(P) = P \cdot P - \cdot`.
///
/// Representation
/// ==============
///
/// Each individual Pauli operator in the generator is a tensor product of single-qubit Pauli
/// operators of the form :math:`P = \bigotimes_n A^{(n)}_i`, for :math:`A^{(n)}_i \in \{I, X, Y,
/// Z\}`. The internal representation of a :class:`PauliLindbladMap` stores only the non-identity
/// single-qubit Pauli operators.  This makes it significantly more efficient to represent
/// generators such as :math:`\sum_{n\in \text{qubits}} c_n L(Z^{(n)})`; for which
/// :class:`PauliLindbladMap` requires an amount of memory linear in the total number of qubits.
///
/// Internally, each single-qubit Pauli operator is stored with a numeric value, explicitly:
///
/// .. _pauli-lindblad-map-alphabet:
/// .. table:: Alphabet of single-qubit Pauli operators used in :class:`PauliLindbladMap`
///
///   =======  =======================================  ===============  ===========================
///   Label    Operator                                 Numeric value    :class:`.BitTerm` attribute
///   =======  =======================================  ===============  ===========================
///   ``"I"``  :math:`I` (identity)                     Not stored.      Not stored.
///
///   ``"X"``  :math:`X` (Pauli X)                      ``0b10`` (2)     :attr:`~.BitTerm.X`
///
///   ``"Y"``  :math:`Y` (Pauli Y)                      ``0b11`` (3)     :attr:`~.BitTerm.Y`
///
///   ``"Z"``  :math:`Z` (Pauli Z)                      ``0b01`` (1)     :attr:`~.BitTerm.Z`
///
///   =======  =======================================  ===============  ===========================
///
/// Each generator term is stored as a compression of the corresponding Pauli operator, similar in
/// spirit to the compressed sparse row format of sparse matrices.  In this analogy, the terms of
/// the sum are the "rows", and the qubit terms are the "columns", where an absent entry represents
/// the identity rather than a zero. More explicitly, the representation is made up of four
/// contiguous arrays:
///
/// .. _pauli-lindblad-map-arrays:
/// .. table:: Data arrays used to represent :class:`.PauliLindbladMap`
///
///   ==================  ===========  =============================================================
///   Attribute           Length       Description
///   ==================  ===========  =============================================================
///   :attr:`coeffs`      :math:`t`    The real scalar coefficient for each term.
///
///   :attr:`bit_terms`   :math:`s`    Each of the non-identity single-qubit Pauli operators for all
///                                    of the generator terms, in order.  These correspond to the
///                                    non-identity :math:`A^{(n)}_i` in the sum description, where
///                                    the entries are stored in order of increasing :math:`i`
///                                    first, and in order of increasing :math:`n` within each term.
///
///   :attr:`indices`     :math:`s`    The corresponding qubit (:math:`n`) for each of the operators
///                                    in :attr:`bit_terms`.  :class:`PauliLindbladMap` requires
///                                    that this list is term-wise sorted, and algorithms can rely
///                                    on this invariant being upheld.
///
///   :attr:`boundaries`  :math:`t+1`  The indices that partition :attr:`bit_terms` and
///                                    :attr:`indices` into complete terms.  For term number
///                                    :math:`i`, its complex coefficient is ``coeffs[i]``, and its
///                                    non-identity single-qubit operators and their corresponding
///                                    qubits are the slice ``boundaries[i] : boundaries[i+1]`` into
///                                    :attr:`bit_terms` and :attr:`indices` respectively.
///                                    :attr:`boundaries` always has an explicit 0 as its first
///                                    element.
///   ==================  ===========  =============================================================
///
/// The length parameter :math:`t` is the number of generator terms in the sum, and the parameter
/// :math:`s` is the total number of non-identity single-qubit terms.
///
/// As illustrative examples:
///
/// * in the case of the identity map, which contains no generator terms, :attr:`boundaries` is
///   length 1 (a single 0) and all other vectors are empty.
/// * for the map :math:`\exp\left(2 Z_2 Z_0 - 3 X_3 Y_1`, :attr:`boundaries` is ``[0, 2, 4]``,
///   :attr:`coeffs` is ``[2.0, -3.0]``, :attr:`bit_terms` is ``[BitTerm.Z, BitTerm.Z, BitTerm.Y,
///   BitTerm.X]`` and :attr:`indices` is ``[0, 2, 1, 3]``.  The map might act on more than
///   four qubits, depending on the :attr:`num_qubits` parameter.  The :attr:`bit_terms` are integer
///   values, whose magic numbers can be accessed via the :class:`BitTerm` attribute class.  Note
///   that the single-bit terms and indices are sorted into termwise sorted order.  This is a
///   requirement of the class.
///
/// These cases are not special, they're fully consistent with the rules and should not need special
/// handling.
///
/// The scalar item of the :attr:`bit_terms` array is stored as a numeric byte.  The numeric values
/// are related to the symplectic Pauli representation that :class:`.SparsePauliOp` uses, and are
/// accessible with named access by an enumeration:
///
/// ..
///     This is documented manually here because the Python-space `Enum` is generated
///     programmatically from Rust - it'd be _more_ confusing to try and write a docstring somewhere
///     else in this source file. The use of `autoattribute` is because it pulls in the numeric
///     value.
///
/// .. py:class:: PauliLindbladMap.BitTerm
///
///     An :class:`~enum.IntEnum` that provides named access to the numerical values used to
///     represent each of the single-qubit alphabet terms enumerated in
///     :ref:`pauli-lindblad-map-alphabet`.
///
///     This class is attached to :class:`.PauliLindbladMap`.  Access it as
///     :class:`.PauliLindbladMap.BitTerm`.  If this is too much typing, and you are solely dealing
///     with :class:¬PauliLindbladMap` objects and the :class:`BitTerm` name is not ambiguous, you
///     might want to shorten it as::
///
///         >>> ops = PauliLindbladMap.BitTerm
///         >>> assert ops.X is PauliLindbladMap.BitTerm.X
///
///     You can access all the values of the enumeration by either their full all-capitals name, or
///     by their single-letter label.  The single-letter labels are not generally valid Python
///     identifiers, so you must use indexing notation to access them::
///
///         >>> assert PauliLindbladMap.BitTerm.X is PauliLindbladMap.BitTerm["X"]
///
///     The bits representing each single-qubit Pauli are the (phase-less) symplectic representation
///     of the Pauli operator.
///
///     Values
///     ------
///
///     .. autoattribute:: qiskit.quantum_info::PauliLindbladMap.BitTerm.X
///
///         The Pauli :math:`X` operator.  Uses the single-letter label ``"X"``.
///
///     .. autoattribute:: qiskit.quantum_info::PauliLindbladMap.BitTerm.Y
///
///         The Pauli :math:`Y` operator.  Uses the single-letter label ``"Y"``.
///
///     .. autoattribute:: qiskit.quantum_info::PauliLindbladMap.BitTerm.Z
///
///         The Pauli :math:`Z` operator.  Uses the single-letter label ``"Z"``.
///
///     Attributes
///     ----------
///
///     .. autoproperty:: qiskit.quantum_info::PauliLindbladMap.BitTerm.label
///
///
/// Each of the array-like attributes behaves like a Python sequence.  You can index and slice these
/// with standard :class:`list`-like semantics.  Slicing an attribute returns a Numpy
/// :class:`~numpy.ndarray` containing a copy of the relevant data with the natural ``dtype`` of the
/// field; this lets you easily do mathematics on the results, like bitwise operations on
/// :attr:`bit_terms`.  You can assign to indices or slices of each of the attributes, but beware
/// that you must uphold :ref:`the data coherence rules <pauli-lindblad-map-arrays>` while doing
/// this.  For example::
///
///     >>> pauli_lindblad_map = PauliLindbladMap.from_list([("XZY", 1.5), ("YXZ", -0.5)])
///     >>> assert isinstance(pauli_lindblad_map.coeffs[:], np.ndarray)
///
/// Indexing
/// --------
///
/// :class:`PauliLindbladMap` behaves as `a Python sequence
/// <https://docs.python.org/3/glossary.html#term-sequence>`__ (the standard form, not the expanded
/// :class:`collections.abc.Sequence`).  The generators of the map can be indexed by integers, and
/// iterated through to yield individual generator terms.
///
/// Each generator term appears as an instance a self-contained class.  The individual terms are
/// copied out of the base map; mutations to them will not affect the original map from which they
/// are indexed.
///
/// .. autoclass:: qiskit.quantum_info::PauliLindbladMap.Term
///     :members:
///
/// Construction
/// ============
///
/// :class:`PauliLindbladMap` defines several constructors.  The default constructor will attempt to
/// delegate to one of the more specific constructors, based on the type of the input.  You can
/// always use the specific constructors to have more control over the construction.
///
/// .. _pauli-lindblad-map-convert-constructors:
/// .. table:: Construction from other objects
///
///   ============================  ================================================================
///   Method                        Summary
///   ============================  ================================================================
///   :meth:`from_list`             Generators given as a list of tuples of dense string labels and
///                                 the associated coefficients.
///
///   :meth:`from_sparse_list`      Generators given as a list of tuples of sparse string labels,
///                                 the qubits they apply to, and their coefficients.
///
///   :meth:`from_terms`            Sum explicit single :class:`Term` instances.
///
///   :meth:`from_raw_parts`        Build the observable from :ref:`the raw data arrays
///                                 <pauli-lindblad-map-arrays>`.
///   ============================  ================================================================
///
/// .. py:function:: PauliLindbladMap.__new__(data, /, num_qubits=None)
///
///     The default constructor of :class:`PauliLindbladMap`.
///
///     This delegates to one of :ref:`the explicit conversion-constructor methods
///     <pauli-lindblad-map-convert-constructors>`, based on the type of the ``data`` argument.  If
///     ``num_qubits`` is supplied and constructor implied by the type of ``data`` does not accept a
///     number, the given integer must match the input.
///
///     :param data: The data type of the input.  This can be another :class:`PauliLindbladMap`, in
///         which case the input is copied, or it can be a list in a valid format for either
///         :meth:`from_list` or :meth:`from_sparse_list`.
///     :param int|None num_qubits: Optional number of qubits for the map.  For most data
///         inputs, this can be inferred and need not be passed.  It is only necessary for empty
///         lists or the sparse-list format.  If given unnecessarily, it must match the data input.
///
/// In addition to the conversion-based constructors, there are also helper methods that construct
/// special forms of maps.
///
/// .. table:: Construction of special maps
///
///   ============================  ================================================================
///   Method                        Summary
///   ============================  ================================================================
///   :meth:`identity`              The identity map on a given number of qubits.
///   ============================  ================================================================
///
/// Conversions
/// ===========
///
/// An existing :class:`PauliLindbladMap` can be converted into other formats.
///
/// .. table:: Conversion methods to other observable forms.
///
///   ===========================  =================================================================
///   Method                       Summary
///   ===========================  =================================================================
///   :meth:`to_sparse_list`       Express the observable in a sparse list format with elements
///                                ``(bit_terms, indices, coeff)``.
///   ===========================  =================================================================
#[pyclass(name = "QubitSparsePauliList", module = "qiskit.quantum_info", sequence)]
#[derive(Debug)]
pub struct PyQubitSparsePauliList {
    // This class keeps a pointer to a pure Rust-SparseTerm and serves as interface from Python.
    inner: Arc<RwLock<QubitSparsePauliList>>,
}
#[pymethods]
impl PyQubitSparsePauliList {
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
        if let Ok(pauli_list) = data.downcast_exact::<Self>() {
            check_num_qubits(data)?;
            let borrowed = pauli_list.borrow();
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
        if let Ok(term) = data.downcast_exact::<PyQubitSparsePauli>() {
            return term.borrow().to_qubit_sparse_pauli_list();
        };
        if let Ok(pauli_list) = Self::from_qubit_sparse_paulis(data, num_qubits) {
            return Ok(pauli_list);
        }
        Err(PyTypeError::new_err(format!(
            "unknown input format for 'PauliLindbladMap': {}",
            data.get_type().repr()?,
        )))
    }

    /// Get a copy of this Pauli Lindblad map.
    ///
    /// Examples:
    ///
    ///     .. code-block:: python
    ///
    ///         >>> pauli_lindblad_map = PauliLindbladMap.from_list([("IXZXYYZZ", 2.5), ("ZXIXYYZZ", 0.5)])
    ///         >>> assert pauli_lindblad_map == pauli_lindblad_map.copy()
    ///         >>> assert pauli_lindblad_map is not pauli_lindblad_map.copy()
    fn copy(&self) -> PyResult<Self> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        Ok(inner.clone().into())
    }

    /// The number of qubits the map acts on.
    ///
    /// This is not inferable from any other shape or values, since identities are not stored
    /// explicitly.
    #[getter]
    #[inline]
    pub fn num_qubits(&self) -> PyResult<u32> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        Ok(inner.num_qubits())
    }

    /// The number of generator terms in the exponent for this map.
    #[getter]
    #[inline]
    pub fn num_terms(&self) -> PyResult<usize> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        Ok(inner.num_terms())
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

    /// Get the empty list for a given number of qubits.
    ///
    /// The identity map contains no generator terms, and is the identity element for composition of
    /// two :class:`PauliLindbladMap` instances; anything composed with the identity map is equal to
    /// itself.
    ///
    /// Examples:
    ///
    ///     Get the identity map on 100 qubits::
    ///
    ///         >>> PauliLindbladMap.identity(100)
    ///         <PauliLindbladMap with 0 terms on 100 qubits: 0.0>
    #[pyo3(signature = (/, num_qubits))]
    #[staticmethod]
    pub fn empty(num_qubits: u32) -> Self {
        QubitSparsePauliList::empty(num_qubits).into()
    }

    /// NOTE DAN: The phase is dropped, document this ************************************************
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
        let inner = QubitSparsePauliList::new(num_qubits, bit_terms, indices, boundaries)?;
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
        let mut inner = QubitSparsePauliList::empty(label.len() as u32);
        inner.add_dense_label(label)?;
        Ok(inner.into())
    }

    /// Construct a Pauli Lindblad map from a list of dense generator labels and coefficients.
    ///
    /// This is analogous to :meth:`.SparsePauliOp.from_list`. In this dense form, you must supply
    /// all identities explicitly in each label.
    ///
    /// The label must be a sequence of the alphabet ``'IXYZ'``.  The label is interpreted
    /// analogously to a bitstring.  In other words, the right-most letter is associated with qubit
    /// 0, and so on.  This is the same as the labels for :class:`.Pauli` and
    /// :class:`.SparsePauliOp`.
    ///
    /// Args:
    ///     iter (list[tuple[str, float]]): Pairs of labels and their associated coefficients in the
    ///         generator sum.
    ///     num_qubits (int | None): It is not necessary to specify this if you are sure that
    ///         ``iter`` is not an empty sequence, since it can be inferred from the label lengths.
    ///         If ``iter`` may be empty, you must specify this argument to disambiguate how many
    ///         qubits the map acts on.  If this is given and ``iter`` is not empty, the value
    ///         must match the label lengths.
    ///
    /// Examples:
    ///
    ///     Construct a Pauli Lindblad map from a list of labels::
    ///
    ///         >>> PauliLindbladMap.from_list([
    ///         ...     ("IIIXX", 1.0),
    ///         ...     ("IIYYI", 1.0),
    ///         ...     ("IXXII", -0.5),
    ///         ...     ("ZZIII", -0.25),
    ///         ... ])
    ///         <PauliLindbladMap with 4 terms on 5 qubits:
    ///             (1)L(X_1 X_0) + (1)L(Y_2 Y_1) + (-0.5)L(X_3 X_2) + (-0.25)L(Z_4 Z_3)>
    ///
    ///     Use ``num_qubits`` to disambiguate potentially empty inputs::
    ///
    ///         >>> PauliLindbladMap.from_list([], num_qubits=10)
    ///         <PauliLindbladMap with 0 terms on 10 qubits: 0.0>
    ///
    ///     This method is equivalent to calls to :meth:`from_sparse_list` with the explicit
    ///     qubit-arguments field set to decreasing integers::
    ///
    ///         >>> labels = ["XYXZ", "YYZZ", "XYXZ"]
    ///         >>> coeffs = [1.5, 2.0, -0.5]
    ///         >>> from_list = PauliLindbladMap.from_list(list(zip(labels, coeffs)))
    ///         >>> from_sparse_list = PauliLindbladMap.from_sparse_list([
    ///         ...     (label, (3, 2, 1, 0), coeff)
    ///         ...     for label, coeff in zip(labels, coeffs)
    ///         ... ])
    ///         >>> assert from_list == from_sparse_list
    ///
    /// See also:
    ///     :meth:`from_sparse_list`
    ///         Construct the map from a list of labels without explicit identities, but with
    ///         the qubits each single-qubit generator term applies to listed explicitly.
    #[staticmethod]
    #[pyo3(signature = (iter, /, *, num_qubits=None))]
    fn from_list(iter: Vec<String>, num_qubits: Option<u32>) -> PyResult<Self> {
        if iter.is_empty() && num_qubits.is_none() {
            return Err(PyValueError::new_err(
                "cannot construct a PauliLindbladMap from an empty list without knowing `num_qubits`",
            ));
        }
        let num_qubits = match num_qubits {
            Some(num_qubits) => num_qubits,
            None => iter[0].len() as u32,
        };
        let mut inner = QubitSparsePauliList::with_capacity(num_qubits, iter.len(), 0);
        for label in iter {
            inner.add_dense_label(&label)?;
        }
        Ok(inner.into())
    }

    /// Construct a :class:`PauliLindbladMap` out of individual terms.
    ///
    /// All the terms must have the same number of qubits.  If supplied, the ``num_qubits`` argument
    /// must match the terms.
    ///
    /// No simplification is done as part of the map creation.
    ///
    /// Args:
    ///     obj (Iterable[Term]): Iterable of individual terms to build the map generator from.
    ///     num_qubits (int | None): The number of qubits the map should act on.  This is
    ///         usually inferred from the input, but can be explicitly given to handle the case
    ///         of an empty iterable.
    ///
    /// Returns:
    ///     The corresponding map.
    #[staticmethod]
    #[pyo3(signature = (obj, /, num_qubits=None))]
    fn from_qubit_sparse_paulis(obj: &Bound<PyAny>, num_qubits: Option<u32>) -> PyResult<Self> {
        let mut iter = obj.try_iter()?;
        let mut inner = match num_qubits {
            Some(num_qubits) => QubitSparsePauliList::empty(num_qubits),
            None => {
                let Some(first) = iter.next() else {
                    return Err(PyValueError::new_err(
                        "cannot construct an empty QubitSparsePauliList without knowing `num_qubits`",
                    ));
                };
                let py_term = first?.downcast::<PyQubitSparsePauli>()?.borrow();
                py_term.inner.to_qubit_sparse_pauli_list()
            }
        };
        for bound_py_term in iter {
            let py_term = bound_py_term?.downcast::<PyQubitSparsePauli>()?.borrow();
            inner.add_qubit_sparse_pauli(py_term.inner.view())?;
        }
        Ok(inner.into())
    }

    // SAFETY: this cannot invoke undefined behaviour if `check = true`, but if `check = false` then
    // the `bit_terms` must all be valid `BitTerm` representations.
    /// Construct a :class:`.PauliLindbladMap` from raw Numpy arrays that match :ref:`the required
    /// data representation described in the class-level documentation
    /// <pauli-lindblad-map-arrays>`.
    ///
    /// The data from each array is copied into fresh, growable Rust-space allocations.
    ///
    /// Args:
    ///     num_qubits: number of qubits the map acts on.
    ///     coeffs: float coefficients of each generator term of the map.  This should be a Numpy
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
    ///             of :class:`.PauliLindbladMap.BitTerm`.  If they are not, Rust-space undefined
    ///             behavior may occur, entirely invalidating the program execution.
    ///
    /// Examples:
    ///
    ///     Construct a sum of :math:`Z` on each individual qubit::
    ///
    ///         >>> num_qubits = 100
    ///         >>> terms = np.full((num_qubits,), PauliLindbladMap.BitTerm.Z, dtype=np.uint8)
    ///         >>> indices = np.arange(num_qubits, dtype=np.uint32)
    ///         >>> coeffs = np.ones((num_qubits,), dtype=float)
    ///         >>> boundaries = np.arange(num_qubits + 1, dtype=np.uintp)
    ///         >>> PauliLindbladMap.from_raw_parts(num_qubits, coeffs, terms, indices, boundaries)
    ///         <PauliLindbladMap with 100 terms on 100 qubits: (1)L(Z_0) + ... + (1)L(Z_99)>
    #[staticmethod]
    #[pyo3(
        signature = (/, num_qubits, bit_terms, indices, boundaries, check=true),
    )]
    unsafe fn from_raw_parts<'py>(
        num_qubits: u32,
        bit_terms: PyArrayLike1<'py, u8>,
        indices: PyArrayLike1<'py, u32>,
        boundaries: PyArrayLike1<'py, usize>,
        check: bool,
    ) -> PyResult<Self> {
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
            QubitSparsePauliList::new(num_qubits, bit_terms, indices, boundaries)
                .map_err(PyErr::from)
        } else {
            // SAFETY: the caller promised they have upheld the coherence guarantees.
            Ok(unsafe {
                QubitSparsePauliList::new_unchecked(num_qubits, bit_terms, indices, boundaries)
            })
        }?;
        Ok(inner.into())
    }

    /// Clear all the generator terms from this map, making it equal to the identity map again.
    ///
    /// This does not change the capacity of the internal allocations, so subsequent addition or
    /// substraction operations resulting from composition may not need to reallocate.
    ///
    /// Examples:
    ///
    ///     .. code-block:: python
    ///
    ///         >>> pauli_lindblad_map = PauliLindbladMap.from_list([("IXXXYY", 2.0), ("ZZYZII", -1)])
    ///         >>> pauli_lindblad_map.clear()
    ///         >>> assert pauli_lindblad_map == PauliLindbladMap.identity(pauli_lindblad_map.py_num_qubits())
    pub fn clear(&mut self) -> PyResult<()> {
        let mut inner = self.inner.write().map_err(|_| InnerWriteError)?;
        inner.clear();
        Ok(())
    }

    /// Construct a Pauli Lindblad map from a list of labels, the qubits each item applies to, and
    /// the coefficient of the whole term.
    ///
    /// This is analogous to :meth:`.SparsePauliOp.from_sparse_list`.
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
    ///     num_qubits (int): the number of qubits the map acts on.
    ///
    /// Examples:
    ///
    ///     Construct a simple map::
    ///
    ///         >>> PauliLindbladMap.from_sparse_list(
    ///         ...     [("ZX", (1, 4), 1.0), ("YY", (0, 3), 2)],
    ///         ...     num_qubits=5,
    ///         ... )
    ///         <PauliLindbladMap with 2 terms on 5 qubits: (1)L(X_4 Z_1) + (2)L(Y_3 Y_0)>
    ///
    ///     This method can replicate the behavior of :meth:`from_list`, if the qubit-arguments
    ///     field of the triple is set to decreasing integers::
    ///
    ///         >>> labels = ["XYXZ", "YYZZ", "XYXZ"]
    ///         >>> coeffs = [1.5, 2.0, -0.5]
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
    fn from_sparse_list(iter: Vec<(String, Vec<u32>)>, num_qubits: u32) -> PyResult<Self> {
        let mut boundaries = Vec::with_capacity(iter.len() + 1);
        boundaries.push(0);
        let mut indices = Vec::new();
        let mut bit_terms = Vec::new();
        // Insertions to the `BTreeMap` keep it sorted by keys, so we use this to do the termwise
        // sorting on-the-fly.
        let mut sorted = btree_map::BTreeMap::new();
        for (label, qubits) in iter {
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
        let inner = QubitSparsePauliList::new(num_qubits, bit_terms, indices, boundaries)?;
        Ok(inner.into())
    }

    /// Express the map in terms of a sparse list format.
    ///
    /// This can be seen as counter-operation of :meth:`.PauliLindbladMap.from_sparse_list`, however
    /// the order of terms is not guaranteed to be the same at after a roundtrip to a sparse
    /// list and back.
    ///
    /// Examples:
    ///
    ///     >>> pauli_lindblad_map = PauliLindbladMap.from_list([("IIXIZ", 2), ("IIZIX", 3)])
    ///     >>> reconstructed = PauliLindbladMap.from_sparse_list(pauli_lindblad_map.to_sparse_list(), pauli_lindblad_map.num_qubits)
    ///
    /// See also:
    ///     :meth:`from_sparse_list`
    ///         The constructor that can interpret these lists.
    #[pyo3(signature = ())]
    fn to_sparse_list(&self, py: Python) -> PyResult<Py<PyList>> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;

        // turn a SparseView into a Python tuple of (bit terms, indices, coeff)
        let to_py_tuple = |view: QubitSparsePauliView| {
            let mut pauli_string = String::with_capacity(view.bit_terms.len());

            for bit in view.bit_terms.iter() {
                pauli_string.push_str(bit.py_label());
            }
            let py_string = PyString::new(py, &pauli_string).unbind();
            let py_indices = PyList::new(py, view.indices.iter())?.unbind();

            PyTuple::new(py, vec![py_string.as_any(), py_indices.as_any()])
        };

        let out = PyList::empty(py);
        for view in inner.iter() {
            out.append(to_py_tuple(view)?)?;
        }
        Ok(out.unbind())
    }

    fn __len__(&self) -> PyResult<usize> {
        self.num_terms()
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        let bit_terms: &[u8] = ::bytemuck::cast_slice(inner.bit_terms());
        (
            py.get_type::<Self>().getattr("from_raw_parts")?,
            (
                inner.num_qubits(),
                PyArray1::from_slice(py, bit_terms),
                PyArray1::from_slice(py, inner.indices()),
                PyArray1::from_slice(py, inner.boundaries()),
                false,
            ),
        )
            .into_pyobject(py)
    }

    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        index: PySequenceIndex<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        let indices = match index.with_len(inner.num_terms())? {
            SequenceIndex::Int(index) => {
                return PyQubitSparsePauli {
                    inner: inner.term(index).to_term(),
                }
                .into_bound_py_any(py)
            }
            indices => indices,
        };
        let mut out = QubitSparsePauliList::empty(inner.num_qubits());
        for index in indices.iter() {
            out.add_qubit_sparse_pauli(inner.term(index))?;
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
            "{} element{}",
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
            "".to_owned()
        } else {
            inner
                .iter()
                .map(QubitSparsePauliView::to_sparse_str)
                .collect::<Vec<_>>()
                .join(", ")
        };
        Ok(format!(
            "<QubitSparsePauliList with {} on {}: [{}]>",
            str_num_terms, str_num_qubits, str_terms
        ))
    }

    // The documentation for this is inlined into the class-level documentation of
    // `PauliLindbladMap`.
    #[allow(non_snake_case)]
    #[classattr]
    fn BitTerm(py: Python) -> PyResult<Py<PyType>> {
        BIT_TERM_PY_ENUM
            .get_or_try_init(py, || make_py_bit_term(py))
            .map(|obj| obj.clone_ref(py))
    }
}

impl From<QubitSparsePauli> for PyQubitSparsePauli {
    fn from(val: QubitSparsePauli) -> PyQubitSparsePauli {
        PyQubitSparsePauli {
            inner: val,
        }
    }
}
impl<'py> IntoPyObject<'py> for QubitSparsePauli {
    type Target = PyQubitSparsePauli;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        PyQubitSparsePauli::from(self).into_pyobject(py)
    }
}
impl From<QubitSparsePauliList> for PyQubitSparsePauliList {
    fn from(val: QubitSparsePauliList) -> PyQubitSparsePauliList {
        PyQubitSparsePauliList {
            inner: Arc::new(RwLock::new(val)),
        }
    }
}
impl<'py> IntoPyObject<'py> for QubitSparsePauliList {
    type Target = PyQubitSparsePauliList;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        PyQubitSparsePauliList::from(self).into_pyobject(py)
    }
}

/// Helper class of `ArrayView` that denotes the slot of the `QubitSparsePauliList` we're looking at.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ArraySlot {
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
    base: Arc<RwLock<QubitSparsePauliList>>,
    slot: ArraySlot,
}
#[pymethods]
impl ArrayView {
    fn __repr__(&self, py: Python) -> PyResult<String> {
        let qubit_sparse_pauli_list = self.base.read().map_err(|_| InnerReadError)?;
        let data = match self.slot {
            // Simple integers look the same in Rust-space debug as Python.
            ArraySlot::Indices => format!("{:?}", qubit_sparse_pauli_list.indices()),
            ArraySlot::Boundaries => format!("{:?}", qubit_sparse_pauli_list.boundaries()),
            ArraySlot::BitTerms => format!(
                "[{}]",
                qubit_sparse_pauli_list
                    .bit_terms()
                    .iter()
                    .map(BitTerm::py_label)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        };
        Ok(format!(
            "<qubit sparse pauli list {} view: {}>",
            match self.slot {
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

        let qubit_sparse_pauli_list = self.base.read().map_err(|_| InnerReadError)?;
        match self.slot {
            ArraySlot::BitTerms => {
                get_from_slice::<_, u8>(py, qubit_sparse_pauli_list.bit_terms(), index)
            }
            ArraySlot::Indices => get_from_slice::<_, u32>(py, qubit_sparse_pauli_list.indices(), index),
            ArraySlot::Boundaries => {
                get_from_slice::<_, usize>(py, qubit_sparse_pauli_list.boundaries(), index)
            }
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

        let mut qubit_sparse_pauli_list = self.base.write().map_err(|_| InnerWriteError)?;
        match self.slot {
            ArraySlot::BitTerms => {
                set_in_slice::<BitTerm, u8>(qubit_sparse_pauli_list.bit_terms_mut(), index, values)
            }
            ArraySlot::Indices => unsafe {
                set_in_slice::<_, u32>(qubit_sparse_pauli_list.indices_mut(), index, values)
            },
            ArraySlot::Boundaries => unsafe {
                set_in_slice::<_, usize>(qubit_sparse_pauli_list.boundaries_mut(), index, values)
            },
        }
    }

    fn __len__(&self, _py: Python) -> PyResult<usize> {
        let qubit_sparse_pauli_list = self.base.read().map_err(|_| InnerReadError)?;
        let len = match self.slot {
            ArraySlot::BitTerms => qubit_sparse_pauli_list.bit_terms().len(),
            ArraySlot::Indices => qubit_sparse_pauli_list.indices().len(),
            ArraySlot::Boundaries => qubit_sparse_pauli_list.boundaries().len(),
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
        let qubit_sparse_pauli_list = self.base.read().map_err(|_| InnerReadError)?;
        match self.slot {
            ArraySlot::Indices => cast_array_type(
                py,
                PyArray1::from_slice(py, qubit_sparse_pauli_list.indices()),
                dtype,
            ),
            ArraySlot::Boundaries => cast_array_type(
                py,
                PyArray1::from_slice(py, qubit_sparse_pauli_list.boundaries()),
                dtype,
            ),
            ArraySlot::BitTerms => {
                let bit_terms: &[u8] = ::bytemuck::cast_slice(qubit_sparse_pauli_list.bit_terms());
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
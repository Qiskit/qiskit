// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use hashbrown::HashSet;

use ndarray::Array2;
use std::collections::btree_map;

use thiserror::Error;

/// Named handle to the alphabet of single-qubit terms.
///
/// This is just the Rust-space representation.  We make a separate Python-space `enum.IntEnum` to
/// represent the same information, since we enforce strongly typed interactions in Rust, including
/// not allowing the stored values to be outside the valid `Pauli`\ s, but doing so in Python would
/// make it very difficult to use the class efficiently with Numpy array views.  We attach this
/// sister class of `Pauli` to `QubitSparsePauli` and `QubitSparsePauliList` as a scoped class.
///
/// # Representation
///
/// The `u8` representation and the exact numerical values of these are part of the public API.  The
/// two bits are the symplectic Pauli representation of the Pauli operator, with the associations
/// `0b10` <-> `X`, `0b01` <-> `Z`, `0b11` <-> `Y`. The `0b00` representation thus ends up being the
/// natural representation of the `I` operator, but this is never stored, and is not named in the
/// enumeration.
///
/// This operator does not store phase terms of $-i$.  `Pauli::Y` has `(1, 1)` as its `(z, x)`
/// representation, and represents exactly the Pauli Y operator. Additional phases, if needed, must
/// be stored elsewhere.
///
/// # Dev notes
///
/// This type is required to be `u8`, but it's a subtype of `u8` because not all `u8` are valid
/// `Pauli`\ s.  For interop with Python space, we accept Numpy arrays of `u8` to represent this,
/// which we transmute into slices of `Pauli`, after checking that all the values are correct (or
/// skipping the check if Python space promises that it upheld the checks).
///
/// We deliberately _don't_ impl `numpy::Element` for `Pauli` (which would let us accept and
/// return `PyArray1<Pauli>` at Python-space boundaries) so that it's clear when we're doing
/// the transmute, and we have to be explicit about the safety of that.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum Pauli {
    /// Pauli X operator.
    X = 0b10,
    /// Pauli Y operator.
    Y = 0b11,
    /// Pauli Z operator.
    Z = 0b01,
}
impl From<Pauli> for u8 {
    fn from(value: Pauli) -> u8 {
        value as u8
    }
}
unsafe impl ::bytemuck::CheckedBitPattern for Pauli {
    type Bits = u8;

    #[inline(always)]
    fn is_valid_bit_pattern(bits: &Self::Bits) -> bool {
        *bits <= 0b11 && *bits != 0
    }
}
unsafe impl ::bytemuck::NoUninit for Pauli {}

impl Pauli {
    /// Get the label of this `Pauli` used in Python-space applications.  This is a single-letter
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

    /// Attempt to convert a `u8` into `Pauli`.
    ///
    /// Unlike the implementation of `TryFrom<u8>`, this allows `b'I'` as an alphabet letter,
    /// returning `Ok(None)` for it.  All other letters outside the alphabet return the complete
    /// error condition.
    #[inline]
    pub fn try_from_u8(value: u8) -> Result<Option<Self>, PauliFromU8Error> {
        match value {
            b'I' => Ok(None),
            b'X' => Ok(Some(Pauli::X)),
            b'Y' => Ok(Some(Pauli::Y)),
            b'Z' => Ok(Some(Pauli::Z)),
            _ => Err(PauliFromU8Error(value)),
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

/// The error type for a failed conversion into `Pauli`.
#[derive(Error, Debug)]
#[error("{0} is not a valid letter of the single-qubit alphabet")]
pub struct PauliFromU8Error(u8);

// `Pauli` allows safe `as` casting into `u8`.  This is the reverse, which is fallible, because
// `Pauli` is a value-wise subtype of `u8`.
impl ::std::convert::TryFrom<u8> for Pauli {
    type Error = PauliFromU8Error;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        ::bytemuck::checked::try_cast(value).map_err(|_| PauliFromU8Error(value))
    }
}

/// Error cases stemming from data coherence at the point of entry into `QubitSparsePauli` or
/// `QubitSparsePauliList` from user-provided arrays.
///
/// These most typically appear during [from_raw_parts], but can also be introduced by various
/// remapping arithmetic functions.
///
/// These are generally associated with the Python-space `ValueError` because all of the
/// `TypeError`-related ones are statically forbidden (within Rust) by the language, and conversion
/// failures on entry to Rust from Python space will automatically raise `TypeError`.
#[derive(Error, Debug)]
pub enum CoherenceError {
    #[error("`phases` ({phases}) must be the same length as `qubit_sparse_pauli_list` ({qspl})")]
    MismatchedPhaseCount { phases: usize, qspl: usize },
    #[error("`rates` ({rates}) must be the same length as `qubit_sparse_pauli_list` ({qspl})")]
    MismatchedTermCount { rates: usize, qspl: usize },
    #[error("`paulis` ({paulis}) and `indices` ({indices}) must be the same length")]
    MismatchedItemCount { paulis: usize, indices: usize },
    #[error("the first item of `boundaries` ({0}) must be 0")]
    BadInitialBoundary(usize),
    #[error(
        "the last item of `boundaries` ({last}) must match the length of `paulis` and `indices` ({items})"
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
    #[error("cannot shrink the qubit count in a QubitSparsePauliList from {current} to {target}")]
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
    #[error("multiplying single qubit paulis resulted in bit value: {b}")]
    PauliMultiplication { b: u8 },
}

/// A list of Pauli operators stored in a qubit-sparse format.
///
/// See [PyQubitSparsePauliList] for detailed docs.
#[derive(Clone, Debug, PartialEq)]
pub struct QubitSparsePauliList {
    /// The number of qubits the Paulis are defined on.  This is not inferable from any other shape
    /// or values, since identities are not stored explicitly.
    num_qubits: u32,
    /// A flat list of single-qubit paulis.  This is more naturally a list of lists, but is stored
    /// flat for memory usage and locality reasons, with the sublists denoted by `boundaries.`
    paulis: Vec<Pauli>,
    /// A flat list of the qubit indices that the corresponding entries in `paulis` act on.  This
    /// list must always be term-wise sorted, where a term is a sublist as denoted by `boundaries`.
    indices: Vec<u32>,
    /// Indices that partition `paulis` and `indices` into sublists for each individual sparse
    /// pauli.  `boundaries[0]..boundaries[1]` is the range of indices into `paulis` and
    /// `indices` that correspond to the first term of the sum.  All unspecified qubit indices are
    /// implicitly the identity.
    boundaries: Vec<usize>,
}

impl QubitSparsePauliList {
    /// Create a new qubit-sparse Pauli list from the raw components that make it up.
    ///
    /// This checks the input values for data coherence on entry.  If you are certain you have the
    /// correct values, you can call `new_unchecked` instead.
    pub fn new(
        num_qubits: u32,
        paulis: Vec<Pauli>,
        indices: Vec<u32>,
        boundaries: Vec<usize>,
    ) -> Result<Self, CoherenceError> {
        if paulis.len() != indices.len() {
            return Err(CoherenceError::MismatchedItemCount {
                paulis: paulis.len(),
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
        Ok(unsafe { Self::new_unchecked(num_qubits, paulis, indices, boundaries) })
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
        paulis: Vec<Pauli>,
        indices: Vec<u32>,
        boundaries: Vec<usize>,
    ) -> Self {
        Self {
            num_qubits,
            paulis,
            indices,
            boundaries,
        }
    }

    /// Create a new empty list with pre-allocated space for the given number of Paulis and
    /// single-qubit pauli terms.
    #[inline]
    pub fn with_capacity(num_qubits: u32, num_terms: usize, num_paulis: usize) -> Self {
        Self {
            num_qubits,
            paulis: Vec::with_capacity(num_paulis),
            indices: Vec::with_capacity(num_paulis),
            boundaries: {
                let mut boundaries = Vec::with_capacity(num_terms + 1);
                boundaries.push(0);
                boundaries
            },
        }
    }

    /// Get an iterator over the individual elements of the list.
    pub fn iter(&'_ self) -> impl ExactSizeIterator<Item = QubitSparsePauliView<'_>> + '_ {
        std::ops::Range {
            start: 0,
            end: self.boundaries.len() - 1,
        }
        .map(|i| {
            let start = self.boundaries[i];
            let end = self.boundaries[i + 1];
            QubitSparsePauliView {
                num_qubits: self.num_qubits,
                paulis: &self.paulis[start..end],
                indices: &self.indices[start..end],
            }
        })
    }

    /// Get the number of qubits the paulis are defined on.
    #[inline]
    pub fn num_qubits(&self) -> u32 {
        self.num_qubits
    }

    /// Get the number of elements in the list.
    #[inline]
    pub fn num_terms(&self) -> usize {
        self.boundaries.len() - 1
    }

    /// Get the indices of each [Pauli].
    #[inline]
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// Get the boundaries of each term.
    #[inline]
    pub fn boundaries(&self) -> &[usize] {
        &self.boundaries
    }

    /// Get the [Pauli]s in the list.
    #[inline]
    pub fn paulis(&self) -> &[Pauli] {
        &self.paulis
    }

    /// Create a [QubitSparsePauliList] representing the empty list on ``num_qubits`` qubits.
    pub fn empty(num_qubits: u32) -> Self {
        Self::with_capacity(num_qubits, 0, 0)
    }

    /// Clear all the elements of the list.
    ///
    /// This does not change the capacity of the internal allocations, so subsequent addition or
    /// subtraction of elements in the list may not need to reallocate.
    pub fn clear(&mut self) {
        self.paulis.clear();
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
    pub fn term(&self, index: usize) -> QubitSparsePauliView<'_> {
        debug_assert!(index < self.num_terms(), "index {index} out of bounds");
        let start = self.boundaries[index];
        let end = self.boundaries[index + 1];
        QubitSparsePauliView {
            num_qubits: self.num_qubits,
            paulis: &self.paulis[start..end],
            indices: &self.indices[start..end],
        }
    }

    /// Add an element to the list implied by a dense string label.
    pub fn add_dense_label<L: AsRef<[u8]>>(&mut self, label: L) -> Result<(), LabelError> {
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
            match Pauli::try_from_u8(*letter) {
                Ok(Some(term)) => {
                    self.paulis.push(term);
                    self.indices.push(i as u32);
                }
                Ok(None) => (),
                Err(_) => {
                    // Undo any modifications to ourselves so we stay in a consistent state.
                    let num_single_terms = self.boundaries[self.boundaries.len() - 1];
                    self.paulis.truncate(num_single_terms);
                    self.indices.truncate(num_single_terms);
                    return Err(LabelError::OutsideAlphabet);
                }
            }
        }
        self.boundaries.push(self.paulis.len());
        Ok(())
    }

    /// Add a single sparse Pauli term to the list.
    pub fn add_qubit_sparse_pauli(
        &mut self,
        term: QubitSparsePauliView,
    ) -> Result<(), ArithmeticError> {
        if self.num_qubits != term.num_qubits {
            return Err(ArithmeticError::MismatchedQubits {
                left: self.num_qubits,
                right: term.num_qubits,
            });
        }
        self.paulis.extend_from_slice(term.paulis);
        self.indices.extend_from_slice(term.indices);
        self.boundaries.push(self.paulis.len());
        Ok(())
    }

    /// Relabel the `indices` in the map to new values.
    ///
    /// This fails if any of the new indices are too large, or if any mapping would cause a term to
    /// contain duplicates of the same index.  It may not detect if multiple qubits are mapped to
    /// the same index, if those qubits never appear together in the same term.  Such a mapping
    /// would not cause data-coherence problems (the output map will be valid), but is
    /// unlikely to be what you intended.
    ///
    /// *Panics* if `new_qubits` is not long enough to map every index used in the map.
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
                order.insert(new_qubits[self.indices[j] as usize], self.paulis[j]);
            }
            if order.len() != end - start {
                return Err(CoherenceError::DuplicateIndices);
            }
            for (index, dest) in order.keys().zip(&mut self.indices[start..end]) {
                *dest = *index;
            }
            for (pauli, dest) in order.values().zip(&mut self.paulis[start..end]) {
                *dest = *pauli;
            }
            order.clear();
        }
        Ok(())
    }

    // Return a Vec of dense labels representing this Pauli list
    pub fn to_dense_label_list(&self) -> Vec<String> {
        let mut dense_label_list = Vec::with_capacity(self.num_terms());

        for qubit_sparse_pauli in self.iter() {
            dense_label_list.push(qubit_sparse_pauli.to_term().to_dense_label());
        }
        dense_label_list
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

    // Check if the elements of `self` commute with the elements of `other`.
    pub fn commutes(&self, other: &QubitSparsePauliList) -> Result<Array2<bool>, ArithmeticError> {
        if self.num_qubits != other.num_qubits {
            return Err(ArithmeticError::MismatchedQubits {
                left: self.num_qubits,
                right: other.num_qubits,
            });
        }

        Ok(Array2::from_shape_fn(
            (self.num_terms(), other.num_terms()),
            |(i, j)| {
                self.term(i)
                    .to_term()
                    .commutes(&other.term(j).to_term())
                    .unwrap()
            },
        ))
    }
}

type RawParts = (Vec<Pauli>, Vec<u32>, Vec<usize>);

pub fn raw_parts_from_sparse_list(
    iter: Vec<(String, Vec<u32>)>,
    num_qubits: u32,
) -> Result<RawParts, LabelError> {
    let mut boundaries = Vec::with_capacity(iter.len() + 1);
    boundaries.push(0);
    let mut indices = Vec::new();
    let mut paulis = Vec::new();
    // Insertions to the `BTreeMap` keep it sorted by keys, so we use this to do the termwise
    // sorting on-the-fly.
    let mut sorted = btree_map::BTreeMap::new();
    for (label, qubits) in iter {
        sorted.clear();
        let label: &[u8] = label.as_ref();
        if label.len() != qubits.len() {
            return Err(LabelError::WrongLengthIndices {
                label: label.len(),
                indices: qubits.len(),
            });
        }
        for (letter, index) in label.iter().zip(qubits) {
            if index >= num_qubits {
                return Err(LabelError::BadIndex { index, num_qubits });
            }
            let btree_map::Entry::Vacant(entry) = sorted.entry(index) else {
                return Err(LabelError::DuplicateIndex { index });
            };
            entry.insert(Pauli::try_from_u8(*letter).map_err(|_| LabelError::OutsideAlphabet)?);
        }
        for (index, term) in sorted.iter() {
            let Some(term) = term else {
                continue;
            };
            indices.push(*index);
            paulis.push(*term);
        }
        boundaries.push(paulis.len());
    }
    Ok((paulis, indices, boundaries))
}

/// A view object onto a single term of a `QubitSparsePauliList`.
///
/// The lengths of `paulis` and `indices` are guaranteed to be created equal, but might be zero
/// (in the case that the term is the identity).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct QubitSparsePauliView<'a> {
    pub num_qubits: u32,
    pub paulis: &'a [Pauli],
    pub indices: &'a [u32],
}
impl QubitSparsePauliView<'_> {
    /// Convert this `QubitSparsePauliView` into an owning [QubitSparsePauli] of the same data.
    pub fn to_term(&self) -> QubitSparsePauli {
        QubitSparsePauli {
            num_qubits: self.num_qubits,
            paulis: self.paulis.into(),
            indices: self.indices.into(),
        }
    }

    pub fn num_ys(self) -> isize {
        let mut num_ys = 0;
        for pauli in self.paulis {
            num_ys += (*pauli == Pauli::Y) as isize;
        }
        num_ys
    }

    pub fn to_sparse_str(self) -> String {
        let paulis = self
            .indices
            .iter()
            .zip(self.paulis)
            .rev()
            .map(|(i, op)| format!("{}_{}", op.py_label(), i))
            .collect::<Vec<String>>()
            .join(" ");
        paulis.to_string()
    }
}

/// A single qubit-spare Pauli operator.
#[derive(Clone, Debug, PartialEq)]
pub struct QubitSparsePauli {
    /// Number of qubits the Pauli operator is defined on.
    num_qubits: u32,
    /// A list of the non-identity single-qubit Paulis in the operator.
    paulis: Box<[Pauli]>,
    /// A flat list of the qubit indices that the corresponding entries in `paulis` act on.
    indices: Box<[u32]>,
}

impl QubitSparsePauli {
    /// Create a new qubit-sparse Pauli from the raw components that make it up.
    pub fn new(
        num_qubits: u32,
        paulis: Box<[Pauli]>,
        indices: Box<[u32]>,
    ) -> Result<Self, CoherenceError> {
        if paulis.len() != indices.len() {
            return Err(CoherenceError::MismatchedItemCount {
                paulis: paulis.len(),
                indices: indices.len(),
            });
        }

        if indices.iter().any(|index| *index >= num_qubits) {
            return Err(CoherenceError::BitIndexTooHigh);
        }

        Ok(Self {
            num_qubits,
            paulis,
            indices,
        })
    }

    pub fn from_dense_label(label: &str) -> Result<QubitSparsePauli, LabelError> {
        let label: &[u8] = label.as_ref();
        let num_qubits = label.len() as u32;
        let mut paulis = Vec::new();
        let mut indices = Vec::new();
        // The only valid characters in the alphabet are ASCII, so if we see something other than
        // ASCII, we're already in the failure path.
        for (i, letter) in label.iter().rev().enumerate() {
            match Pauli::try_from_u8(*letter) {
                Ok(Some(term)) => {
                    paulis.push(term);
                    indices.push(i as u32);
                }
                Ok(None) => (),
                Err(_) => {
                    return Err(LabelError::OutsideAlphabet);
                }
            }
        }
        Ok(unsafe {
            QubitSparsePauli::new_unchecked(
                num_qubits,
                paulis.into_boxed_slice(),
                indices.into_boxed_slice(),
            )
        })
    }

    // Return a dense label representing this Pauli
    pub fn to_dense_label(&self) -> String {
        let mut pauli_str = "".to_string();

        let mut current_idx = 0;

        for (index, pauli) in self.indices().iter().zip(self.paulis().iter()) {
            if *index > current_idx {
                pauli_str =
                    (0..(index - current_idx)).map(|_| "I").collect::<String>() + &pauli_str;
                current_idx = *index;
            }
            pauli_str = pauli.py_label().to_string() + &pauli_str;
            current_idx += 1;
        }

        if current_idx < self.num_qubits() {
            pauli_str = (0..(self.num_qubits() - current_idx))
                .map(|_| "I")
                .collect::<String>()
                + &pauli_str;
        }
        pauli_str
    }

    /// Create a new [QubitSparsePauli] from the raw components without checking data coherence.
    ///
    /// # Safety
    ///
    /// It is up to the caller to ensure that the data-coherence requirements, as enumerated in the
    /// struct-level documentation, have been upheld.
    #[inline(always)]
    pub unsafe fn new_unchecked(
        num_qubits: u32,
        paulis: Box<[Pauli]>,
        indices: Box<[u32]>,
    ) -> Self {
        Self {
            num_qubits,
            paulis,
            indices,
        }
    }

    /// Get the number of qubits the paulis are defined on.
    #[inline]
    pub fn num_qubits(&self) -> u32 {
        self.num_qubits
    }

    /// Get the indices of each [Pauli].
    #[inline]
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// Get the [Pauli]s in the list.
    #[inline]
    pub fn paulis(&self) -> &[Pauli] {
        &self.paulis
    }

    /// Create the identity [QubitSparsePauli] on ``num_qubits`` qubits.
    pub fn identity(num_qubits: u32) -> Self {
        Self {
            num_qubits,
            paulis: Box::new([]),
            indices: Box::new([]),
        }
    }

    // Phaseless composition of two pauli operators.
    pub fn compose(&self, other: &QubitSparsePauli) -> Result<QubitSparsePauli, ArithmeticError> {
        if self.num_qubits != other.num_qubits {
            return Err(ArithmeticError::MismatchedQubits {
                left: self.num_qubits,
                right: other.num_qubits,
            });
        }

        // if either are the identity, return a clone of the other
        if self.indices.is_empty() {
            return Ok(other.clone());
        }

        if other.indices.is_empty() {
            return Ok(self.clone());
        }

        let mut paulis = Vec::new();
        let mut indices = Vec::new();

        let mut self_idx = 0;
        let mut other_idx = 0;

        // iterate through each entry of self and other one time, incrementing based on the ordering
        // or equality of self_idx and other_idx, until one of them runs out of entries
        while self_idx < self.indices.len() && other_idx < other.indices.len() {
            if self.indices[self_idx] < other.indices[other_idx] {
                // if the current qubit index of self is strictly less than other, append the pauli
                paulis.push(self.paulis[self_idx]);
                indices.push(self.indices[self_idx]);
                self_idx += 1;
            } else if self.indices[self_idx] == other.indices[other_idx] {
                // if the indices are the same, perform multiplication and append if non-identity
                let new_pauli = (self.paulis[self_idx] as u8) ^ (other.paulis[other_idx] as u8);
                if new_pauli != 0 {
                    paulis.push(match new_pauli {
                        0b01 => Ok(Pauli::Z),
                        0b10 => Ok(Pauli::X),
                        0b11 => Ok(Pauli::Y),
                        _ => Err(ArithmeticError::PauliMultiplication { b: new_pauli }),
                    }?);
                    indices.push(self.indices[self_idx])
                }
                self_idx += 1;
                other_idx += 1;
            } else {
                // same as the first if block but with roles of self and other reversed
                paulis.push(other.paulis[other_idx]);
                indices.push(other.indices[other_idx]);
                other_idx += 1;
            }
        }

        // if any entries remain in either pauli, append them
        if other_idx != other.indices.len() {
            paulis.append(&mut other.paulis[other_idx..].to_vec());
            indices.append(&mut other.indices[other_idx..].to_vec());
        } else if self_idx != self.indices.len() {
            paulis.append(&mut self.paulis[self_idx..].to_vec());
            indices.append(&mut self.indices[self_idx..].to_vec());
        }

        Ok(QubitSparsePauli {
            num_qubits: self.num_qubits,
            paulis: paulis.into_boxed_slice(),
            indices: indices.into_boxed_slice(),
        })
    }

    /// Get a view version of this object.
    pub fn view(&self) -> QubitSparsePauliView<'_> {
        QubitSparsePauliView {
            num_qubits: self.num_qubits,
            paulis: &self.paulis,
            indices: &self.indices,
        }
    }

    /// Convert this single Pauli into a :class:`QubitSparsePauliList`.
    pub fn to_qubit_sparse_pauli_list(&self) -> QubitSparsePauliList {
        QubitSparsePauliList {
            num_qubits: self.num_qubits,
            paulis: self.paulis.to_vec(),
            indices: self.indices.to_vec(),
            boundaries: vec![0, self.paulis.len()],
        }
    }

    // Check if `self` commutes with `other`.
    pub fn commutes(&self, other: &QubitSparsePauli) -> Result<bool, ArithmeticError> {
        if self.num_qubits != other.num_qubits {
            return Err(ArithmeticError::MismatchedQubits {
                left: self.num_qubits,
                right: other.num_qubits,
            });
        }

        let mut commutes = true;
        let mut self_idx = 0;
        let mut other_idx = 0;

        // iterate through each entry of self and other one time, incrementing based on the ordering
        // or equality of self_idx and other_idx, until one of them runs out of entries
        while self_idx < self.indices.len() && other_idx < other.indices.len() {
            if self.indices[self_idx] < other.indices[other_idx] {
                self_idx += 1;
            } else if self.indices[self_idx] == other.indices[other_idx] {
                // if the indices are the same, check commutation
                commutes = commutes == (self.paulis[self_idx] == other.paulis[other_idx]);
                self_idx += 1;
                other_idx += 1;
            } else {
                other_idx += 1;
            }
        }

        Ok(commutes)
    }
}

#[derive(Error, Debug)]
pub struct InnerReadError;

#[derive(Error, Debug)]
pub struct InnerWriteError;

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

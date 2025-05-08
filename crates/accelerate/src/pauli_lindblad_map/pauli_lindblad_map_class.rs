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

use numpy::{PyArray1, PyArrayLike1, PyArrayMethods};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    intern,
    prelude::*,
    sync::GILOnceCell,
    types::{PyList, PyString, PyTuple, PyType},
    IntoPyObjectExt, PyErr,
};
use std::sync::{Arc, RwLock};

use qiskit_circuit::slice::{PySequenceIndex, SequenceIndex};

use super::qubit_sparse_pauli::{
    cast_array_type, make_py_pauli, raw_parts_from_sparse_list, ArithmeticError, CoherenceError,
    InnerReadError, InnerWriteError, LabelError, Pauli, QubitSparsePauliList,
};

static PAULI_PY_ENUM: GILOnceCell<Py<PyType>> = GILOnceCell::new();

/// A Pauli Lindblad map that stores its data in a qubit-sparse format.
///
/// See [PyPauliLindbladMap] for detailed docs.
#[derive(Clone, Debug, PartialEq)]
pub struct PauliLindbladMap {
    /// The rates of each abstract term in the generator sum.  This has as many elements as
    /// terms in the sum.
    rates: Vec<f64>,
    /// A list of qubit sparse Paulis corresponding to the rates
    qubit_sparse_pauli_list: QubitSparsePauliList,
}

impl PauliLindbladMap {
    pub fn new(
        rates: Vec<f64>,
        qubit_sparse_pauli_list: QubitSparsePauliList,
    ) -> Result<Self, CoherenceError> {
        if rates.len() + 1 != qubit_sparse_pauli_list.boundaries().len() {
            return Err(CoherenceError::MismatchedTermCount {
                rates: rates.len(),
                boundaries: qubit_sparse_pauli_list.boundaries().len(),
            });
        }

        Ok(Self {
            rates,
            qubit_sparse_pauli_list,
        })
    }

    /// Create a new Pauli Lindblad map from the raw components that make it up.
    ///
    /// This checks the input values for data coherence on entry.  If you are certain you have the
    /// correct values, you can call `new_unchecked` instead.
    pub fn new_from_raw_parts(
        num_qubits: u32,
        rates: Vec<f64>,
        paulis: Vec<Pauli>,
        indices: Vec<u32>,
        boundaries: Vec<usize>,
    ) -> Result<Self, CoherenceError> {
        if rates.len() + 1 != boundaries.len() {
            return Err(CoherenceError::MismatchedTermCount {
                rates: rates.len(),
                boundaries: boundaries.len(),
            });
        }
        let qubit_sparse_pauli_list: QubitSparsePauliList =
            QubitSparsePauliList::new(num_qubits, paulis, indices, boundaries)?;
        Ok(Self {
            rates,
            qubit_sparse_pauli_list,
        })
    }

    /// Create a new [PauliLindbladMap] from the raw components without checking data coherence.
    ///
    /// # Safety
    ///
    /// It is up to the caller to ensure that the data-coherence requirements, as enumerated in the
    /// struct-level documentation, have been upheld.
    #[inline(always)]
    pub unsafe fn new_unchecked(
        num_qubits: u32,
        rates: Vec<f64>,
        paulis: Vec<Pauli>,
        indices: Vec<u32>,
        boundaries: Vec<usize>,
    ) -> Self {
        unsafe {
            let qubit_sparse_pauli_list: QubitSparsePauliList =
                QubitSparsePauliList::new_unchecked(num_qubits, paulis, indices, boundaries);
            Self {
                rates,
                qubit_sparse_pauli_list,
            }
        }
    }

    /// Get an iterator over the individual generator terms of the map that allows in-place
    /// mutation.
    ///
    /// The length and indices of these views cannot be mutated, since both would allow breaking
    /// data coherence.
    pub fn iter_mut(&mut self) -> IterMut<'_> {
        self.into()
    }

    /// Get an iterator over the individual generator terms of the map.
    ///
    /// Recall that two [PauliLindbladMap]s that have different term orders can still represent the
    /// same object.  Use [canonicalize] to apply a canonical ordering to the terms.
    pub fn iter(&'_ self) -> impl ExactSizeIterator<Item = SparseTermView<'_>> + '_ {
        self.rates.iter().enumerate().map(|(i, rate)| {
            let start = self.qubit_sparse_pauli_list.boundaries[i];
            let end = self.qubit_sparse_pauli_list.boundaries[i + 1];
            SparseTermView {
                num_qubits: self.qubit_sparse_pauli_list.num_qubits,
                rate: *rate,
                paulis: &self.qubit_sparse_pauli_list.paulis[start..end],
                indices: &self.qubit_sparse_pauli_list.indices[start..end],
            }
        })
    }

    /// Get the number of qubits the map is defined on.
    #[inline]
    pub fn num_qubits(&self) -> u32 {
        self.qubit_sparse_pauli_list.num_qubits()
    }

    /// Get the number of generator terms in the map.
    #[inline]
    pub fn num_terms(&self) -> usize {
        self.rates.len()
    }

    /// Get the rates of the generator terms.
    #[inline]
    pub fn rates(&self) -> &[f64] {
        &self.rates
    }

    /// Get a mutable slice of the rates.
    #[inline]
    pub fn rates_mut(&mut self) -> &mut [f64] {
        &mut self.rates
    }

    /// Get the indices of each [Pauli].
    #[inline]
    pub fn indices(&self) -> &[u32] {
        self.qubit_sparse_pauli_list.indices()
    }

    /// Get a mutable slice of the indices.
    ///
    /// # Safety
    ///
    /// Modifying the indices can cause an incoherent state of the [PauliLindbladMap].
    /// It should be ensured that the indices are consistent with the rates, paulis, and
    /// boundaries.
    #[inline]
    pub unsafe fn indices_mut(&mut self) -> &mut [u32] {
        unsafe { self.qubit_sparse_pauli_list.indices_mut() }
    }

    /// Get the boundaries of each term.
    #[inline]
    pub fn boundaries(&self) -> &[usize] {
        self.qubit_sparse_pauli_list.boundaries()
    }

    /// Get a mutable slice of the boundaries.
    ///
    /// # Safety
    ///
    /// Modifying the boundaries can cause an incoherent state of the [PauliLindbladMap].
    /// It should be ensured that the boundaries are sorted and the length/elements are consistent
    /// with the rates, paulis, and indices.
    #[inline]
    pub unsafe fn boundaries_mut(&mut self) -> &mut [usize] {
        unsafe { self.qubit_sparse_pauli_list.boundaries_mut() }
    }

    /// Get the [Pauli]s in the map.
    #[inline]
    pub fn paulis(&self) -> &[Pauli] {
        self.qubit_sparse_pauli_list.paulis()
    }

    /// Get a mutable slice of the bit terms.
    #[inline]
    pub fn paulis_mut(&mut self) -> &mut [Pauli] {
        self.qubit_sparse_pauli_list.paulis_mut()
    }

    /// Create a [PauliLindbladMap] representing the identity map on ``num_qubits`` qubits.
    pub fn identity(num_qubits: u32) -> Self {
        Self::with_capacity(num_qubits, 0, 0)
    }

    /// Add the generator term implied by a dense string label onto this map.
    pub fn add_dense_label<L: AsRef<[u8]>>(
        &mut self,
        label: L,
        rate: f64,
    ) -> Result<(), LabelError> {
        self.qubit_sparse_pauli_list.add_dense_label(label)?;
        self.rates.push(rate);
        Ok(())
    }

    /// Create a new identity map (with zero generator) with pre-allocated space for the given
    /// number of summands and single-qubit bit terms.
    #[inline]
    pub fn with_capacity(num_qubits: u32, num_terms: usize, num_paulis: usize) -> Self {
        let qubit_sparse_pauli_list =
            QubitSparsePauliList::with_capacity(num_qubits, num_terms, num_paulis);
        Self {
            rates: Vec::with_capacity(num_terms),
            qubit_sparse_pauli_list,
        }
    }

    /// Clear all the generator terms from this map, making it equal to the identity map again.
    ///
    /// This does not change the capacity of the internal allocations, so subsequent addition or
    /// substraction of generator terms may not need to reallocate.
    pub fn clear(&mut self) {
        self.rates.clear();
        self.qubit_sparse_pauli_list.clear();
    }

    /// Add a single generator term to this map.
    pub fn add_term(&mut self, term: SparseTermView) -> Result<(), ArithmeticError> {
        if self.num_qubits() != term.num_qubits {
            return Err(ArithmeticError::MismatchedQubits {
                left: self.num_qubits(),
                right: term.num_qubits,
            });
        }
        self.rates.push(term.rate);
        self.qubit_sparse_pauli_list
            .paulis
            .extend_from_slice(term.paulis);
        self.qubit_sparse_pauli_list
            .indices
            .extend_from_slice(term.indices);
        self.qubit_sparse_pauli_list
            .boundaries
            .push(self.qubit_sparse_pauli_list.paulis.len());
        Ok(())
    }

    /// Get a view onto a representation of a single sparse term.
    ///
    /// This is effectively an indexing operation into the [PauliLindbladMap].  Recall that two
    /// [PauliLindbladMap]s that have different generator term orders can still represent the same
    /// object. Use [canonicalize] to apply a canonical ordering to the terms.
    ///
    /// # Panics
    ///
    /// If the index is out of bounds.
    pub fn term(&self, index: usize) -> SparseTermView {
        debug_assert!(index < self.num_terms(), "index {index} out of bounds");
        let start = self.qubit_sparse_pauli_list.boundaries[index];
        let end = self.qubit_sparse_pauli_list.boundaries[index + 1];
        SparseTermView {
            num_qubits: self.qubit_sparse_pauli_list.num_qubits,
            rate: self.rates[index],
            paulis: &self.qubit_sparse_pauli_list.paulis[start..end],
            indices: &self.qubit_sparse_pauli_list.indices[start..end],
        }
    }
}

/// A view object onto a single term of a `PauliLindbladMap`.
///
/// The lengths of `paulis` and `indices` are guaranteed to be created equal, but might be zero
/// (in the case that the term is proportional to the identity).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SparseTermView<'a> {
    pub num_qubits: u32,
    pub rate: f64,
    pub paulis: &'a [Pauli],
    pub indices: &'a [u32],
}
impl SparseTermView<'_> {
    /// Convert this `SparseTermView` into an owning [SparseTerm] of the same data.
    pub fn to_term(&self) -> SparseTerm {
        SparseTerm {
            num_qubits: self.num_qubits,
            rate: self.rate,
            paulis: self.paulis.into(),
            indices: self.indices.into(),
        }
    }

    pub fn to_sparse_str(self) -> String {
        let rate = format!("{}", self.rate).replace('i', "j");
        let paulis = self
            .indices
            .iter()
            .zip(self.paulis)
            .rev()
            .map(|(i, op)| format!("{}_{}", op.py_label(), i))
            .collect::<Vec<String>>()
            .join(" ");
        format!("({})L({})", rate, paulis)
    }
}

/// A mutable view object onto a single term of a [PauliLindbladMap].
///
/// The lengths of [paulis] and [indices] are guaranteed to be created equal, but might be zero
/// (in the case that the generator term is proportional to the identity).  [indices] is not mutable
/// because this would allow data coherence to be broken.
#[derive(Debug)]
pub struct SparseTermViewMut<'a> {
    pub num_qubits: u32,
    pub rate: &'a mut f64,
    pub paulis: &'a mut [Pauli],
    pub indices: &'a [u32],
}

/// Iterator type allowing in-place mutation of the [PauliLindbladMap].
///
/// Created by [PauliLindbladMap::iter_mut].
#[derive(Debug)]
pub struct IterMut<'a> {
    num_qubits: u32,
    rates: &'a mut [f64],
    paulis: &'a mut [Pauli],
    indices: &'a [u32],
    boundaries: &'a [usize],
    i: usize,
}
impl<'a> From<&'a mut PauliLindbladMap> for IterMut<'a> {
    fn from(value: &mut PauliLindbladMap) -> IterMut {
        IterMut {
            num_qubits: value.qubit_sparse_pauli_list.num_qubits,
            rates: &mut value.rates,
            paulis: &mut value.qubit_sparse_pauli_list.paulis,
            indices: &value.qubit_sparse_pauli_list.indices,
            boundaries: &value.qubit_sparse_pauli_list.boundaries,
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
        let rates = ::std::mem::take(&mut self.rates);
        let (rate, other_rates) = rates.split_first_mut()?;
        self.rates = other_rates;

        let len = self.boundaries[self.i + 1] - self.boundaries[self.i];
        self.i += 1;

        let all_paulis = ::std::mem::take(&mut self.paulis);
        let all_indices = ::std::mem::take(&mut self.indices);
        let (paulis, rest_paulis) = all_paulis.split_at_mut(len);
        let (indices, rest_indices) = all_indices.split_at(len);
        self.paulis = rest_paulis;
        self.indices = rest_indices;

        Some(SparseTermViewMut {
            num_qubits: self.num_qubits,
            rate,
            paulis,
            indices,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.rates.len(), Some(self.rates.len()))
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
    /// The real rate of the term.
    rate: f64,
    paulis: Box<[Pauli]>,
    indices: Box<[u32]>,
}
impl SparseTerm {
    pub fn new(
        num_qubits: u32,
        rate: f64,
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
            rate,
            paulis,
            indices,
        })
    }

    pub fn num_qubits(&self) -> u32 {
        self.num_qubits
    }

    pub fn rate(&self) -> f64 {
        self.rate
    }

    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    pub fn paulis(&self) -> &[Pauli] {
        &self.paulis
    }

    pub fn view(&self) -> SparseTermView {
        SparseTermView {
            num_qubits: self.num_qubits,
            rate: self.rate,
            paulis: &self.paulis,
            indices: &self.indices,
        }
    }

    /// Convert this term to a complete :class:`PauliLindbladMap`.
    pub fn to_pauli_lindblad_map(&self) -> PauliLindbladMap {
        let qubit_sparse_pauli_list = QubitSparsePauliList {
            num_qubits: self.num_qubits(),
            paulis: self.paulis.to_vec(),
            indices: self.indices.to_vec(),
            boundaries: vec![0, self.paulis.len()],
        };
        PauliLindbladMap {
            rates: vec![self.rate],
            qubit_sparse_pauli_list,
        }
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
    #[pyo3(signature = (/, num_qubits, rate, paulis, indices))]
    fn py_new(num_qubits: u32, rate: f64, paulis: Vec<Pauli>, indices: Vec<u32>) -> PyResult<Self> {
        if paulis.len() != indices.len() {
            return Err(CoherenceError::MismatchedItemCount {
                paulis: paulis.len(),
                indices: indices.len(),
            }
            .into());
        }
        let mut order = (0..paulis.len()).collect::<Vec<_>>();
        order.sort_unstable_by_key(|a| indices[*a]);
        let paulis = order.iter().map(|i| paulis[*i]).collect();
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
        let inner = SparseTerm::new(num_qubits, rate, paulis, sorted_indices.into_boxed_slice())?;
        Ok(PySparseTerm { inner })
    }

    /// Convert this term to a complete :class:`PauliLindbladMap`.
    fn to_pauli_lindblad_map(&self) -> PyResult<PyPauliLindbladMap> {
        let pauli_lindblad_map = PauliLindbladMap::new_from_raw_parts(
            self.inner.num_qubits(),
            vec![self.inner.rate()],
            self.inner.paulis().to_vec(),
            self.inner.indices().to_vec(),
            vec![0, self.inner.paulis().len()],
        )?;
        Ok(pauli_lindblad_map.into())
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
            borrowed.inner.rate(),
            Self::get_paulis(slf_.clone()),
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
    /// :class:`~PauliLindbladMap.Pauli`.
    #[getter]
    fn get_paulis(slf_: Bound<Self>) -> Bound<PyArray1<u8>> {
        let borrowed = slf_.borrow();
        let paulis = borrowed.inner.paulis();
        let arr = ::ndarray::aview1(::bytemuck::cast_slice::<_, u8>(paulis));
        // SAFETY: in order to call this function, the lifetime of `self` must be managed by Python.
        // We tie the lifetime of the array to `slf_`, and there are no public ways to modify the
        // `Box<[Pauli]>` allocation (including dropping or reallocating it) other than the entire
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

    /// The term's rate.
    #[getter]
    fn get_rate(&self) -> f64 {
        self.inner.rate()
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
            .paulis()
            .iter()
            .map(|bit| bit.py_label())
            .collect();
        PyString::new(py, string.as_str())
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
/// where :math:`K` is a subset of :math:`n`-qubit Pauli operators, and the rates, or coefficients,
/// :math:`\lambda_P` are real numbers. When all the rates :math:`\lambda_P` are
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
///   Label    Operator                                 Numeric value    :class:`.Pauli` attribute
///   =======  =======================================  ===============  ===========================
///   ``"I"``  :math:`I` (identity)                     Not stored.      Not stored.
///
///   ``"X"``  :math:`X` (Pauli X)                      ``0b10`` (2)     :attr:`~.Pauli.X`
///
///   ``"Y"``  :math:`Y` (Pauli Y)                      ``0b11`` (3)     :attr:`~.Pauli.Y`
///
///   ``"Z"``  :math:`Z` (Pauli Z)                      ``0b01`` (1)     :attr:`~.Pauli.Z`
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
///   :attr:`rates`       :math:`t`    The real scalar coefficient for each term.
///
///   :attr:`paulis`      :math:`s`    Each of the non-identity single-qubit Pauli operators for all
///                                    of the generator terms, in order.  These correspond to the
///                                    non-identity :math:`A^{(n)}_i` in the sum description, where
///                                    the entries are stored in order of increasing :math:`i`
///                                    first, and in order of increasing :math:`n` within each term.
///
///   :attr:`indices`     :math:`s`    The corresponding qubit (:math:`n`) for each of the operators
///                                    in :attr:`paulis`.  :class:`PauliLindbladMap` requires
///                                    that this list is term-wise sorted, and algorithms can rely
///                                    on this invariant being upheld.
///
///   :attr:`boundaries`  :math:`t+1`  The indices that partition :attr:`paulis` and :attr:`indices`
///                                    into complete terms.  For term number :math:`i`, its real
///                                    coefficient is ``rates[i]``, and its non-identity
///                                    single-qubit operators and their corresponding qubits are the
///                                    slice ``boundaries[i] : boundaries[i+1]`` into :attr:`paulis`
///                                    and :attr:`indices` respectively. :attr:`boundaries` always
///                                    has an explicit 0 as its first element.
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
///   :attr:`rates` is ``[2.0, -3.0]``, :attr:`paulis` is ``[Pauli.Z, Pauli.Z, Pauli.Y,
///   Pauli.X]`` and :attr:`indices` is ``[0, 2, 1, 3]``.  The map might act on more than
///   four qubits, depending on the :attr:`num_qubits` parameter.  The :attr:`paulis` are integer
///   values, whose magic numbers can be accessed via the :class:`Pauli` attribute class.  Note
///   that the single-bit terms and indices are sorted into termwise sorted order.  This is a
///   requirement of the class.
///
/// These cases are not special, they're fully consistent with the rules and should not need special
/// handling.
///
/// The scalar item of the :attr:`paulis` array is stored as a numeric byte.  The numeric values
/// are related to the symplectic Pauli representation that :class:`.SparsePauliOp` uses, and are
/// accessible with named access by an enumeration:
///
/// ..
///     This is documented manually here because the Python-space `Enum` is generated
///     programmatically from Rust - it'd be _more_ confusing to try and write a docstring somewhere
///     else in this source file. The use of `autoattribute` is because it pulls in the numeric
///     value.
///
/// .. py:class:: PauliLindbladMap.Pauli
///
///     An :class:`~enum.IntEnum` that provides named access to the numerical values used to
///     represent each of the single-qubit alphabet terms enumerated in
///     :ref:`pauli-lindblad-map-alphabet`.
///
///     This class is attached to :class:`.PauliLindbladMap`.  Access it as
///     :class:`.PauliLindbladMap.Pauli`.  If this is too much typing, and you are solely dealing
///     with :class:¬PauliLindbladMap` objects and the :class:`Pauli` name is not ambiguous, you
///     might want to shorten it as::
///
///         >>> ops = PauliLindbladMap.Pauli
///         >>> assert ops.X is PauliLindbladMap.Pauli.X
///
///     You can access all the values of the enumeration by either their full all-capitals name, or
///     by their single-letter label.  The single-letter labels are not generally valid Python
///     identifiers, so you must use indexing notation to access them::
///
///         >>> assert PauliLindbladMap.Pauli.X is PauliLindbladMap.Pauli["X"]
///
///     The bits representing each single-qubit Pauli are the (phase-less) symplectic representation
///     of the Pauli operator.
///
///     Values
///     ------
///
///     .. autoattribute:: qiskit.quantum_info::PauliLindbladMap.Pauli.X
///
///         The Pauli :math:`X` operator.  Uses the single-letter label ``"X"``.
///
///     .. autoattribute:: qiskit.quantum_info::PauliLindbladMap.Pauli.Y
///
///         The Pauli :math:`Y` operator.  Uses the single-letter label ``"Y"``.
///
///     .. autoattribute:: qiskit.quantum_info::PauliLindbladMap.Pauli.Z
///
///         The Pauli :math:`Z` operator.  Uses the single-letter label ``"Z"``.
///
///     Attributes
///     ----------
///
///     .. autoproperty:: qiskit.quantum_info::PauliLindbladMap.Pauli.label
///
///
/// Each of the array-like attributes behaves like a Python sequence.  You can index and slice these
/// with standard :class:`list`-like semantics.  Slicing an attribute returns a Numpy
/// :class:`~numpy.ndarray` containing a copy of the relevant data with the natural ``dtype`` of the
/// field; this lets you easily do mathematics on the results, like bitwise operations on
/// :attr:`paulis`.  You can assign to indices or slices of each of the attributes, but beware
/// that you must uphold :ref:`the data coherence rules <pauli-lindblad-map-arrays>` while doing
/// this.  For example::
///
///     >>> pauli_lindblad_map = PauliLindbladMap.from_list([("XZY", 1.5), ("YXZ", -0.5)])
///     >>> assert isinstance(pauli_lindblad_map.rates[:], np.ndarray)
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
///                                 the associated rates.
///
///   :meth:`from_sparse_list`      Generators given as a list of tuples of sparse string labels,
///                                 the qubits they apply to, and their rates.
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
///                                ``(paulis, indices, rate)``.
///   ===========================  =================================================================
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
        if let Ok(pauli_lindblad_map) = data.downcast_exact::<Self>() {
            check_num_qubits(data)?;
            let borrowed = pauli_lindblad_map.borrow();
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
            return term.borrow().to_pauli_lindblad_map();
        };
        if let Ok(pauli_lindblad_map) = Self::from_terms(data, num_qubits) {
            return Ok(pauli_lindblad_map);
        }
        Err(PyTypeError::new_err(format!(
            "unknown input format for 'PauliLindbladMap': {}",
            data.get_type().repr()?,
        )))
    }

    /// Construct a Pauli Lindblad map from a list of dense generator labels and rates.
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
    ///     iter (list[tuple[str, float]]): Pairs of labels and their associated rates in the
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
    ///         >>> rates = [1.5, 2.0, -0.5]
    ///         >>> from_list = PauliLindbladMap.from_list(list(zip(labels, rates)))
    ///         >>> from_sparse_list = PauliLindbladMap.from_sparse_list([
    ///         ...     (label, (3, 2, 1, 0), rate)
    ///         ...     for label, rate in zip(labels, rates)
    ///         ... ])
    ///         >>> assert from_list == from_sparse_list
    ///
    /// See also:
    ///     :meth:`from_sparse_list`
    ///         Construct the map from a list of labels without explicit identities, but with
    ///         the qubits each single-qubit generator term applies to listed explicitly.
    #[staticmethod]
    #[pyo3(signature = (iter, /, *, num_qubits=None))]
    fn from_list(iter: Vec<(String, f64)>, num_qubits: Option<u32>) -> PyResult<Self> {
        if iter.is_empty() && num_qubits.is_none() {
            return Err(PyValueError::new_err(
                "cannot construct a PauliLindbladMap from an empty list without knowing `num_qubits`",
            ));
        }
        let num_qubits = match num_qubits {
            Some(num_qubits) => num_qubits,
            None => iter[0].0.len() as u32,
        };
        let mut inner = PauliLindbladMap::with_capacity(num_qubits, iter.len(), 0);
        for (label, rate) in iter {
            inner.add_dense_label(&label, rate)?;
        }
        Ok(inner.into())
    }

    /// Get the identity map on the given number of qubits.
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
    pub fn identity(num_qubits: u32) -> Self {
        PauliLindbladMap::identity(num_qubits).into()
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
    fn from_terms(obj: &Bound<PyAny>, num_qubits: Option<u32>) -> PyResult<Self> {
        let mut iter = obj.try_iter()?;
        let mut inner = match num_qubits {
            Some(num_qubits) => PauliLindbladMap::identity(num_qubits),
            None => {
                let Some(first) = iter.next() else {
                    return Err(PyValueError::new_err(
                        "cannot construct a PauliLindbladMap from an empty list without knowing `num_qubits`",
                    ));
                };
                let py_term = first?.downcast::<PySparseTerm>()?.borrow();
                py_term.inner.to_pauli_lindblad_map()
            }
        };
        for bound_py_term in iter {
            let py_term = bound_py_term?.downcast::<PySparseTerm>()?.borrow();
            inner.add_term(py_term.inner.view())?;
        }
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
    /// the rate of the whole term.
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
    ///         each single-qubit term applies to, and the rate of the entire term.
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
    ///         >>> rates = [1.5, 2.0, -0.5]
    ///         >>> from_list = PauliLindbladMap.from_list(list(zip(labels, rates)))
    ///         >>> from_sparse_list = PauliLindbladMap.from_sparse_list([
    ///         ...     (label, (3, 2, 1, 0), rate)
    ///         ...     for label, rate in zip(labels, rates)
    ///         ... ])
    ///         >>> assert from_list == from_sparse_list
    ///
    /// See also:
    ///     :meth:`to_sparse_list`
    ///         The reverse of this method.
    #[staticmethod]
    #[pyo3(signature = (iter, /, num_qubits))]
    fn from_sparse_list(iter: Vec<(String, Vec<u32>, f64)>, num_qubits: u32) -> PyResult<Self> {
        let rates = iter.iter().map(|(_, _, rate)| *rate).collect();
        let op_iter = iter
            .iter()
            .map(|(label, indices, _)| (label.clone(), indices.clone()))
            .collect();
        let (paulis, indices, boundaries) = raw_parts_from_sparse_list(op_iter, num_qubits)?;
        let qubit_sparse_pauli_list =
            QubitSparsePauliList::new(num_qubits, paulis, indices, boundaries)?;
        let inner: PauliLindbladMap = PauliLindbladMap::new(rates, qubit_sparse_pauli_list)?;
        Ok(inner.into())
    }

    // SAFETY: this cannot invoke undefined behaviour if `check = true`, but if `check = false` then
    // the `paulis` must all be valid `Pauli` representations.
    /// Construct a :class:`.PauliLindbladMap` from raw Numpy arrays that match :ref:`the required
    /// data representation described in the class-level documentation
    /// <pauli-lindblad-map-arrays>`.
    ///
    /// The data from each array is copied into fresh, growable Rust-space allocations.
    ///
    /// Args:
    ///     num_qubits: number of qubits the map acts on.
    ///     rates: float coefficients of each generator term of the map.  This should be a Numpy
    ///         array with dtype :attr:`~numpy.float64`.
    ///     paulis: flattened list of the single-qubit terms comprising all complete terms.  This
    ///         should be a Numpy array with dtype :attr:`~numpy.uint8` (which is compatible with
    ///         :class:`.Pauli`).
    ///     indices: flattened term-wise sorted list of the qubits each single-qubit term corresponds
    ///         to.  This should be a Numpy array with dtype :attr:`~numpy.uint32`.
    ///     boundaries: the indices that partition ``paulis`` and ``indices`` into terms.  This
    ///         should be a Numpy array with dtype :attr:`~numpy.uintp`.
    ///     check: if ``True`` (the default), validate that the data satisfies all coherence
    ///         guarantees.  If ``False``, no checks are done.
    ///
    ///         .. warning::
    ///
    ///             If ``check=False``, the ``paulis`` absolutely *must* be all be valid values
    ///             of :class:`.PauliLindbladMap.Pauli`.  If they are not, Rust-space undefined
    ///             behavior may occur, entirely invalidating the program execution.
    ///
    /// Examples:
    ///
    ///     Construct a sum of :math:`Z` on each individual qubit::
    ///
    ///         >>> num_qubits = 100
    ///         >>> terms = np.full((num_qubits,), PauliLindbladMap.Pauli.Z, dtype=np.uint8)
    ///         >>> indices = np.arange(num_qubits, dtype=np.uint32)
    ///         >>> rates = np.ones((num_qubits,), dtype=float)
    ///         >>> boundaries = np.arange(num_qubits + 1, dtype=np.uintp)
    ///         >>> PauliLindbladMap.from_raw_parts(num_qubits, rates, terms, indices, boundaries)
    ///         <PauliLindbladMap with 100 terms on 100 qubits: (1)L(Z_0) + ... + (1)L(Z_99)>
    #[staticmethod]
    #[pyo3(
        signature = (/, num_qubits, rates, paulis, indices, boundaries, check=true),
    )]
    unsafe fn from_raw_parts<'py>(
        num_qubits: u32,
        rates: PyArrayLike1<'py, f64>,
        paulis: PyArrayLike1<'py, u8>,
        indices: PyArrayLike1<'py, u32>,
        boundaries: PyArrayLike1<'py, usize>,
        check: bool,
    ) -> PyResult<Self> {
        let rates = rates.as_array().to_vec();
        let paulis = if check {
            paulis
                .as_array()
                .into_iter()
                .copied()
                .map(Pauli::try_from)
                .collect::<Result<_, _>>()?
        } else {
            let paulis_as_u8 = paulis.as_array().to_vec();
            // SAFETY: the caller enforced that each `u8` is a valid `Pauli`, and `Pauli` is be
            // represented by a `u8`.  We can't use `bytemuck` because we're casting a `Vec`.
            unsafe { ::std::mem::transmute::<Vec<u8>, Vec<Pauli>>(paulis_as_u8) }
        };
        let indices = indices.as_array().to_vec();
        let boundaries = boundaries.as_array().to_vec();

        let inner = if check {
            let qubit_sparse_pauli_list: QubitSparsePauliList =
                QubitSparsePauliList::new(num_qubits, paulis, indices, boundaries)?;
            PauliLindbladMap::new(rates, qubit_sparse_pauli_list).map_err(PyErr::from)
        } else {
            // SAFETY: the caller promised they have upheld the coherence guarantees.
            Ok(unsafe {
                PauliLindbladMap::new_unchecked(num_qubits, rates, paulis, indices, boundaries)
            })
        }?;
        Ok(inner.into())
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

    /// The coefficients of each abstract term in the generator sum.  This has as many elements as
    /// terms in the sum.
    #[getter]
    fn get_rates(slf_: &Bound<Self>) -> ArrayView {
        let borrowed = slf_.borrow();
        ArrayView {
            base: borrowed.inner.clone(),
            slot: ArraySlot::Rates,
        }
    }

    /// A flat list of single-qubit terms.  This is more naturally a list of lists, but is stored
    /// flat for memory usage and locality reasons, with the sublists denoted by `boundaries.`
    #[getter]
    fn get_paulis(slf_: &Bound<Self>) -> ArrayView {
        let borrowed = slf_.borrow();
        ArrayView {
            base: borrowed.inner.clone(),
            slot: ArraySlot::Paulis,
        }
    }

    /// A flat list of the qubit indices that the corresponding entries in :attr:`paulis` act on.
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

    /// Indices that partition :attr:`paulis` and :attr:`indices` into sublists for each
    /// individual term in the sum.  ``boundaries[0] : boundaries[1]`` is the range of indices into
    /// :attr:`paulis` and :attr:`indices` that correspond to the first term of the sum.  All
    /// unspecified qubit indices are implicitly the identity.  This is one item longer than
    /// :attr:`rates`, since ``boundaries[0]`` is always an explicit zero (for algorithmic ease).
    #[getter]
    fn get_boundaries(slf_: &Bound<Self>) -> ArrayView {
        let borrowed = slf_.borrow();
        ArrayView {
            base: borrowed.inner.clone(),
            slot: ArraySlot::Boundaries,
        }
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

        // turn a SparseView into a Python tuple of (bit terms, indices, rate)
        let to_py_tuple = |view: SparseTermView| {
            let mut pauli_string = String::with_capacity(view.paulis.len());

            for bit in view.paulis.iter() {
                pauli_string.push_str(bit.py_label());
            }
            let py_string = PyString::new(py, &pauli_string).unbind();
            let py_indices = PyList::new(py, view.indices.iter())?.unbind();
            let py_rate = view.rate.into_py_any(py)?;

            PyTuple::new(py, vec![py_string.as_any(), py_indices.as_any(), &py_rate])
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
        let paulis: &[u8] = ::bytemuck::cast_slice(inner.paulis());
        (
            py.get_type::<Self>().getattr("from_raw_parts")?,
            (
                inner.num_qubits(),
                PyArray1::from_slice(py, inner.rates()),
                PyArray1::from_slice(py, paulis),
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
                return PySparseTerm {
                    inner: inner.term(index).to_term(),
                }
                .into_bound_py_any(py)
            }
            indices => indices,
        };
        let mut out = PauliLindbladMap::identity(inner.num_qubits());
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

    // The documentation for this is inlined into the class-level documentation of
    // `PauliLindbladMap`.
    #[allow(non_snake_case)]
    #[classattr]
    fn Pauli(py: Python) -> PyResult<Py<PyType>> {
        PAULI_PY_ENUM
            .get_or_try_init(py, || make_py_pauli(py))
            .map(|obj| obj.clone_ref(py))
    }

    // The documentation for this is inlined into the class-level documentation of
    // `PauliLindbladMap`.
    #[allow(non_snake_case)]
    #[classattr]
    fn Term(py: Python) -> Bound<PyType> {
        py.get_type::<PySparseTerm>()
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
            "<PauliLindbladMap with {} on {}: {}>",
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
    Rates,
    Paulis,
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
        let pauli_lindblad_map = self.base.read().map_err(|_| InnerReadError)?;
        let data = match self.slot {
            // Simple integers look the same in Rust-space debug as Python.
            ArraySlot::Indices => format!("{:?}", pauli_lindblad_map.indices()),
            ArraySlot::Boundaries => format!("{:?}", pauli_lindblad_map.boundaries()),
            // Complexes don't have a nice repr in Rust, so just delegate the whole load to Python
            // and convert back.
            ArraySlot::Rates => PyList::new(py, pauli_lindblad_map.rates())?
                .repr()?
                .to_string(),
            ArraySlot::Paulis => format!(
                "[{}]",
                pauli_lindblad_map
                    .paulis()
                    .iter()
                    .map(Pauli::py_label)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        };
        Ok(format!(
            "<pauli lindblad map {} view: {}>",
            match self.slot {
                ArraySlot::Rates => "rates",
                ArraySlot::Paulis => "paulis",
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
        // different to the type that gets put into the Numpy array, since the `Pauli` enum can be
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

        let pauli_lindblad_map = self.base.read().map_err(|_| InnerReadError)?;
        match self.slot {
            ArraySlot::Rates => get_from_slice::<_, f64>(py, pauli_lindblad_map.rates(), index),
            ArraySlot::Paulis => get_from_slice::<_, u8>(py, pauli_lindblad_map.paulis(), index),
            ArraySlot::Indices => get_from_slice::<_, u32>(py, pauli_lindblad_map.indices(), index),
            ArraySlot::Boundaries => {
                get_from_slice::<_, usize>(py, pauli_lindblad_map.boundaries(), index)
            }
        }
    }

    fn __setitem__(&self, index: PySequenceIndex, values: &Bound<PyAny>) -> PyResult<()> {
        /// Set values of a slice according to the indexer, using `extract` to retrieve the
        /// Rust-space object from the collection of Python-space values.
        ///
        /// This indirects the Python extraction through an intermediate type to marginally improve
        /// the error messages for things like `Pauli`, where Python-space extraction might fail
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

        let mut pauli_lindblad_map = self.base.write().map_err(|_| InnerWriteError)?;
        match self.slot {
            ArraySlot::Rates => {
                set_in_slice::<_, f64>(pauli_lindblad_map.rates_mut(), index, values)
            }
            ArraySlot::Paulis => {
                set_in_slice::<Pauli, u8>(pauli_lindblad_map.paulis_mut(), index, values)
            }
            ArraySlot::Indices => unsafe {
                set_in_slice::<_, u32>(pauli_lindblad_map.indices_mut(), index, values)
            },
            ArraySlot::Boundaries => unsafe {
                set_in_slice::<_, usize>(pauli_lindblad_map.boundaries_mut(), index, values)
            },
        }
    }

    fn __len__(&self, _py: Python) -> PyResult<usize> {
        let pauli_lindblad_map = self.base.read().map_err(|_| InnerReadError)?;
        let len = match self.slot {
            ArraySlot::Rates => pauli_lindblad_map.rates().len(),
            ArraySlot::Paulis => pauli_lindblad_map.paulis().len(),
            ArraySlot::Indices => pauli_lindblad_map.indices().len(),
            ArraySlot::Boundaries => pauli_lindblad_map.boundaries().len(),
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
        let pauli_lindblad_map = self.base.read().map_err(|_| InnerReadError)?;
        match self.slot {
            ArraySlot::Rates => cast_array_type(
                py,
                PyArray1::from_slice(py, pauli_lindblad_map.rates()),
                dtype,
            ),
            ArraySlot::Indices => cast_array_type(
                py,
                PyArray1::from_slice(py, pauli_lindblad_map.indices()),
                dtype,
            ),
            ArraySlot::Boundaries => cast_array_type(
                py,
                PyArray1::from_slice(py, pauli_lindblad_map.boundaries()),
                dtype,
            ),
            ArraySlot::Paulis => {
                let paulis: &[u8] = ::bytemuck::cast_slice(pauli_lindblad_map.paulis());
                cast_array_type(py, PyArray1::from_slice(py, paulis), dtype)
            }
        }
    }
}

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

use numpy::{PyArray1, PyArrayMethods, ToPyArray};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    intern,
    prelude::*,
    types::{PyList, PyString, PyTuple, PyType},
    IntoPyObjectExt, PyErr,
};
use std::sync::{Arc, RwLock};

use qiskit_circuit::slice::{PySequenceIndex, SequenceIndex};

use super::qubit_sparse_pauli::{
    raw_parts_from_sparse_list, ArithmeticError, CoherenceError, InnerReadError, InnerWriteError,
    LabelError, Pauli, PyQubitSparsePauli, PyQubitSparsePauliList, QubitSparsePauli,
    QubitSparsePauliList, QubitSparsePauliView,
};

/// A Pauli Lindblad map that stores its data in a qubit-sparse format. Note that gamma,
/// probabilities, and non_negative_rates are quantities derived from rates.
///
/// See [PyPauliLindbladMap] for detailed docs.
#[derive(Clone, Debug, PartialEq)]
pub struct PauliLindbladMap {
    /// The rates of each abstract term in the generator sum.  This has as many elements as
    /// terms in the sum.
    rates: Vec<f64>,
    /// A list of qubit sparse Paulis corresponding to the rates
    qubit_sparse_pauli_list: QubitSparsePauliList,
    /// The gamma parameter.
    gamma: f64,
    /// Probability of application of a given Pauli operator in product form. Note that if the
    /// corresponding rate is less than 0, this is a quasi-probability.
    probabilities: Vec<f64>,
    /// List of boolean values for the statement rate >= 0 for each rate in rates.
    non_negative_rates: Vec<bool>,
}

impl PauliLindbladMap {
    pub fn new(
        rates: Vec<f64>,
        qubit_sparse_pauli_list: QubitSparsePauliList,
    ) -> Result<Self, CoherenceError> {
        if rates.len() + 1 != qubit_sparse_pauli_list.boundaries().len() {
            return Err(CoherenceError::MismatchedTermCount {
                rates: rates.len(),
                qspl: qubit_sparse_pauli_list.boundaries().len() - 1,
            });
        }

        let (gamma, probabilities, non_negative_rates) = derived_values_from_rates(&rates);

        Ok(Self {
            rates,
            qubit_sparse_pauli_list,
            gamma,
            probabilities,
            non_negative_rates,
        })
    }

    /// Get an iterator over the individual generator terms of the map.
    ///
    /// Recall that two [PauliLindbladMap]s that have different term orders can still represent the
    /// same object.  Use [canonicalize] to apply a canonical ordering to the terms.
    pub fn iter(&'_ self) -> impl ExactSizeIterator<Item = GeneratorTermView<'_>> + '_ {
        self.rates
            .iter()
            .zip(self.qubit_sparse_pauli_list.iter())
            .map(|(&rate, qubit_sparse_pauli)| GeneratorTermView {
                rate,
                qubit_sparse_pauli,
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

    /// Get the probabilities associated with each generator term.
    #[inline]
    pub fn probabilities(&self) -> &[f64] {
        &self.probabilities
    }

    /// Get the list of booleans for which rates are non-negative.
    #[inline]
    pub fn non_negative_rates(&self) -> &[bool] {
        &self.non_negative_rates
    }

    /// Get the indices of each [Pauli].
    #[inline]
    pub fn indices(&self) -> &[u32] {
        self.qubit_sparse_pauli_list.indices()
    }

    /// Get the boundaries of each term.
    #[inline]
    pub fn boundaries(&self) -> &[usize] {
        self.qubit_sparse_pauli_list.boundaries()
    }

    /// Get the [Pauli]s in the map.
    #[inline]
    pub fn paulis(&self) -> &[Pauli] {
        self.qubit_sparse_pauli_list.paulis()
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

        let (g, p, pr) = derived_values_from_rate(rate);
        self.rates.push(rate);
        self.gamma *= g;
        self.probabilities.push(p);
        self.non_negative_rates.push(pr);
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
            gamma: 1.0,
            probabilities: Vec::with_capacity(num_terms),
            non_negative_rates: Vec::with_capacity(num_terms),
        }
    }

    /// Clear all the generator terms from this map, making it equal to the identity map again.
    ///
    /// This does not change the capacity of the internal allocations, so subsequent addition or
    /// substraction of generator terms may not need to reallocate.
    pub fn clear(&mut self) {
        self.rates.clear();
        self.gamma = 1.0;
        self.probabilities.clear();
        self.non_negative_rates.clear();
        self.qubit_sparse_pauli_list.clear();
    }

    /// Add a single generator term to this map.
    pub fn add_term(&mut self, term: GeneratorTermView) -> Result<(), ArithmeticError> {
        let term = term.to_term();
        if self.num_qubits() != term.num_qubits() {
            return Err(ArithmeticError::MismatchedQubits {
                left: self.num_qubits(),
                right: term.num_qubits(),
            });
        }
        let (g, p, pr) = derived_values_from_rate(term.rate);
        self.rates.push(term.rate);
        self.gamma *= g;
        self.probabilities.push(p);
        self.non_negative_rates.push(pr);
        // SAFETY: at this point we already know the term needs to be valid.
        let new_pauli = unsafe {
            QubitSparsePauli::new_unchecked(
                self.num_qubits(),
                term.qubit_sparse_pauli.paulis().to_vec().into_boxed_slice(),
                term.indices().to_vec().into_boxed_slice(),
            )
        };
        self.qubit_sparse_pauli_list
            .add_qubit_sparse_pauli(new_pauli.view())?;
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
    pub fn term(&self, index: usize) -> GeneratorTermView {
        debug_assert!(index < self.num_terms(), "index {index} out of bounds");
        GeneratorTermView {
            rate: self.rates[index],
            qubit_sparse_pauli: self.qubit_sparse_pauli_list.term(index),
        }
    }
}

/// Given a rate, return the corresponding gamma, probability, and boolean for whether the rate is
/// non-negative.
fn derived_values_from_rate(rate: f64) -> (f64, f64, bool) {
    let w: f64 = 0.5 * (1.0 + (-2.0 * rate).exp());
    let g: f64 = w.abs() + (1.0 - w).abs();
    let p: f64 = w / g;
    let nnr: bool = rate >= 0.0;
    (g, p, nnr)
}

/// Return the gamma, probabilities, and non-negative rates bools for a vector of rates.
fn derived_values_from_rates(rates: &[f64]) -> (f64, Vec<f64>, Vec<bool>) {
    let mut gamma = 1.0;
    let mut probabilities = Vec::with_capacity(rates.len());
    let mut non_negative_rates = Vec::with_capacity(rates.len());

    for rate in rates {
        let (g, p, nnr) = derived_values_from_rate(*rate);
        gamma *= g;
        probabilities.push(p);
        non_negative_rates.push(nnr);
    }
    (gamma, probabilities, non_negative_rates)
}

/// A view object onto a single generator term of a `PauliLindbladMap`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct GeneratorTermView<'a> {
    pub rate: f64,
    pub qubit_sparse_pauli: QubitSparsePauliView<'a>,
}
impl GeneratorTermView<'_> {
    /// Convert this `GeneratorTermView` into an owning [GeneratorTerm] of the same data.
    pub fn to_term(&self) -> GeneratorTerm {
        GeneratorTerm {
            rate: self.rate,
            qubit_sparse_pauli: self.qubit_sparse_pauli.to_term(),
        }
    }

    pub fn to_sparse_str(self) -> String {
        let rate = format!("{}", self.rate).replace('i', "j");
        let paulis = self
            .qubit_sparse_pauli
            .indices
            .iter()
            .zip(self.qubit_sparse_pauli.paulis)
            .rev()
            .map(|(i, op)| format!("{}_{}", op.py_label(), i))
            .collect::<Vec<String>>()
            .join(" ");
        format!("({})L({})", rate, paulis)
    }
}

/// A single term from a complete `PauliLindbladMap`.
///
/// These are typically created by indexing into or iterating through a `PauliLindbladMap`.
#[derive(Clone, Debug, PartialEq)]
pub struct GeneratorTerm {
    /// The real rate of the term.
    rate: f64,
    qubit_sparse_pauli: QubitSparsePauli,
}
impl GeneratorTerm {
    pub fn num_qubits(&self) -> u32 {
        self.qubit_sparse_pauli.num_qubits()
    }

    pub fn rate(&self) -> f64 {
        self.rate
    }

    pub fn indices(&self) -> &[u32] {
        self.qubit_sparse_pauli.indices()
    }

    pub fn paulis(&self) -> &[Pauli] {
        self.qubit_sparse_pauli.paulis()
    }

    pub fn view(&self) -> GeneratorTermView {
        GeneratorTermView {
            rate: self.rate,
            qubit_sparse_pauli: self.qubit_sparse_pauli.view(),
        }
    }

    /// Convert this term to a complete :class:`PauliLindbladMap`.
    pub fn to_pauli_lindblad_map(&self) -> Result<PauliLindbladMap, CoherenceError> {
        PauliLindbladMap::new(
            vec![self.rate],
            self.qubit_sparse_pauli.to_qubit_sparse_pauli_list(),
        )
    }
}

/// A single term from a complete :class:`PauliLindbladMap`.
///
/// These are typically created by indexing into or iterating through a :class:`PauliLindbladMap`.
#[pyclass(name = "GeneratorTerm", frozen, module = "qiskit.quantum_info")]
#[derive(Clone, Debug)]
struct PyGeneratorTerm {
    inner: GeneratorTerm,
}
#[pymethods]
impl PyGeneratorTerm {
    // Mark the Python class as being defined "within" the `PauliLindbladMap` class namespace.
    #[classattr]
    #[pyo3(name = "__qualname__")]
    fn type_qualname() -> &'static str {
        "PauliLindbladMap.GeneratorTerm"
    }

    #[new]
    #[pyo3(signature = (/, rate, qubit_sparse_pauli))]
    fn py_new(rate: f64, qubit_sparse_pauli: &PyQubitSparsePauli) -> PyResult<Self> {
        let inner = GeneratorTerm {
            rate,
            qubit_sparse_pauli: qubit_sparse_pauli.inner().clone(),
        };
        Ok(PyGeneratorTerm { inner })
    }

    /// Convert this term to a complete :class:`PauliLindbladMap`.
    fn to_pauli_lindblad_map(&self) -> PyResult<PyPauliLindbladMap> {
        let pauli_lindblad_map = PauliLindbladMap::new(
            vec![self.inner.rate()],
            self.inner.qubit_sparse_pauli.to_qubit_sparse_pauli_list(),
        )?;
        Ok(pauli_lindblad_map.into())
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

    /// The term's qubit_sparse_pauli.
    #[getter]
    fn get_qubit_sparse_pauli(&self) -> PyQubitSparsePauli {
        self.inner.qubit_sparse_pauli.clone().into()
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

    fn __getnewargs__(slf_: Bound<Self>) -> PyResult<Bound<PyTuple>> {
        let py = slf_.py();
        let borrowed = slf_.borrow();
        (
            borrowed.inner.rate,
            borrowed.inner.qubit_sparse_pauli.clone(),
        )
            .into_pyobject(py)
    }

    /// Return the pauli labels of the term as a string.
    ///
    /// The pauli labels will match the order of :attr:`.GeneratorTerm.indices`, such that the
    /// i-th character in the string is applied to the qubit index at ``term.indices[i]``. E.g. the
    /// term with operator ``X`` acting on qubit 0 and ``Y`` acting on qubit ``3`` will have
    /// ``term.indices == np.array([0, 3])`` and ``term.pauli_labels == "XY"``.
    ///
    /// Returns:
    ///     The non-identity bit terms as a concatenated string.
    fn pauli_labels<'py>(&self, py: Python<'py>) -> Bound<'py, PyString> {
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
/// A Pauli-Lindblad map is a linear map acting on density matrices on :math:`n`-qubits of the form:
///
/// .. math::
///
///     \Lambda\bigl[\circ\bigr] = \exp\left(\sum_{P \in K} \lambda_P (P \circ P - \circ)\right)
///
/// where :math:`K` is a subset of :math:`n`-qubit Pauli operators, and the rates, or coefficients,
/// :math:`\lambda_P` are real numbers. When all the rates :math:`\lambda_P` are non-negative, this
/// corresponds to a completely positive and trace preserving map. The sum in the exponential is
/// called the generator, and each individual term the generators. To simplify notation in the rest
/// of the documention, we denote :math:`L(P)\bigl[\circ\bigr] = P \circ P - \circ`.
///
/// Quasi-probability representation
/// ================================
///
/// The map :math:`\Lambda` can be written as a product:
///
/// .. math::
///
///     \Lambda\bigl[\circ\bigr] = \prod_{P \in K}\exp\left(\lambda_P(P \circ P - \circ)\right).
///
/// For each :math:`P`, it holds that
///
/// .. math::
///
///     \exp\left(\lambda_P(P \circ P - \circ)\right) = \omega(\lambda_P) \circ + (1 - \omega(\lambda_P)) P \circ P,
///
/// where :math:`\omega(x) = \frac{1}{2}(1 + e^{-2 x})`. Observe that if :math:`\lambda_P \geq 0`,
/// then :math:`\omega(\lambda_P) \in (\frac{1}{2}, 1]`, and this term is a completely-positive and
/// trace-preserving map. However, if :math:`\lambda_P < 0`, then :math:`\omega(\lambda_P) > 1` and
/// the map is not completely positive or trace preserving. Letting
/// :math:`\gamma_P = \omega(\lambda_P) + |1 - \omega(\lambda_P)|`,
/// :math:`p_P = \omega(\lambda_P) / \gamma_P` and :math:`b_P \in \{0, 1\}` be :math:`1` if
/// :math:`\lambda_P < 0` and :math:`0` otherwise, we rewrite the map as:
///
/// .. math::
///
///     \omega(\lambda_P) \circ + (1 - \omega(\lambda_P)) P \circ P = \gamma_P \left(p_P \circ + (-1)^{b_P}(1 - p_P) P \circ P\right).
///
/// If :math:`\lambda_P \geq 0`, :math:`\gamma_P = 1` and the expression reduces to the standard
/// mixture of the identity map and conjugation by :math:`P`. If :math:`\lambda_P < 0`,
/// :math:`\gamma_P > 1`, and the map is a scaled difference of the identity map and conjugation by
/// :math:`P`, with probability weights (hence "quasi-probability"). Note that this is a slightly
/// different presentation than in the literature, but this notation allows us to handle both
/// non-negative and negative rates simultaneously. The overall :math:`\gamma` of the channel is the
/// product :math:`\gamma = \prod_{P \in K} \gamma_P`.
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
/// Internally, :class:`.PauliLindbladMap` stores an array of rates and a
/// :class:`.QubitSparsePauliList` containing the corresponding sparse Pauli operators.
/// Additionally, :class:`.PauliLindbladMap` can compute the overall channel :math:`\gamma` in the
/// :meth:`get_gamma` method, as well as the corresponding probabilities (or quasi-probabilities)
/// via the :meth:`get_probabilities` method.
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
/// .. autoclass:: qiskit.quantum_info::PauliLindbladMap.GeneratorTerm
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
///   :meth:`from_terms`            Sum explicit single :class:`GeneratorTerm` instances.
///
///   :meth:`from_components`       Build from an array of rates and a
///                                 :class:`.QubitSparsePauliList` instance.
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
/// .. table:: Conversion methods to other forms.
///
///   ===========================  =================================================================
///   Method                       Summary
///   ===========================  =================================================================
///   :meth:`to_sparse_list`       Express the map in a sparse list format with elements
///                                ``(paulis, indices, rate)``.
///   ===========================  =================================================================
#[pyclass(name = "PauliLindbladMap", module = "qiskit.quantum_info", sequence)]
#[derive(Debug)]
pub struct PyPauliLindbladMap {
    // This class keeps a pointer to a pure Rust-GeneratorTerm and serves as interface from Python.
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
        if let Ok(term) = data.downcast_exact::<PyGeneratorTerm>() {
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

    #[staticmethod]
    fn from_components(
        rates: Vec<f64>,
        qubit_sparse_pauli_list: &PyQubitSparsePauliList,
    ) -> PyResult<Self> {
        let qubit_sparse_pauli_list = qubit_sparse_pauli_list
            .inner
            .read()
            .map_err(|_| InnerReadError)?;
        let inner = PauliLindbladMap::new(rates, qubit_sparse_pauli_list.clone())?;

        Ok(inner.into())
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
                let py_term = first?.downcast::<PyGeneratorTerm>()?.borrow();
                py_term.inner.to_pauli_lindblad_map()?
            }
        };
        for bound_py_term in iter {
            let py_term = bound_py_term?.downcast::<PyGeneratorTerm>()?.borrow();
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

    /// Calculate the :math:`\gamma` for the map.
    #[inline]
    fn gamma(&self) -> PyResult<f64> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        Ok(inner.gamma)
    }

    /// The rates for the map.
    #[getter]
    fn get_rates(slf_: Bound<Self>) -> Bound<PyArray1<f64>> {
        let borrowed = &slf_.borrow();
        let inner = borrowed.inner.read().unwrap();
        let rates = inner.rates();
        let arr = ::ndarray::aview1(rates);
        // SAFETY: in order to call this function, the lifetime of `self` must be managed by Python.
        // We tie the lifetime of the array to `slf_`, and there are no public ways to modify the
        // `Box<[u32]>` allocation (including dropping or reallocating it) other than the entire
        // object getting dropped, which Python will keep safe.
        let out = unsafe { PyArray1::borrow_from_array(&arr, slf_.into_any()) };
        out.readwrite().make_nonwriteable();
        out
    }

    /// Calculate the probabilities for the map.
    fn probabilities<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let inner = self.inner.read().unwrap();
        inner.probabilities().to_pyarray(py)
    }

    /// Get a copy of the map's qubit sparse pauli list.
    fn get_qubit_sparse_pauli_list_copy(&self) -> PyQubitSparsePauliList {
        let inner = self.inner.read().unwrap();
        inner.qubit_sparse_pauli_list.clone().into()
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
        let to_py_tuple = |view: GeneratorTermView| {
            let mut pauli_string = String::with_capacity(view.qubit_sparse_pauli.paulis.len());

            for bit in view.qubit_sparse_pauli.paulis.iter() {
                pauli_string.push_str(bit.py_label());
            }
            let py_string = PyString::new(py, &pauli_string).unbind();
            let py_indices = PyList::new(py, view.qubit_sparse_pauli.indices.iter())?.unbind();
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
        (
            py.get_type::<Self>().getattr("from_sparse_list")?,
            (self.to_sparse_list(py)?, inner.num_qubits()),
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
                return PyGeneratorTerm {
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
    fn GeneratorTerm(py: Python) -> Bound<PyType> {
        py.get_type::<PyGeneratorTerm>()
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
                .map(GeneratorTermView::to_sparse_str)
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

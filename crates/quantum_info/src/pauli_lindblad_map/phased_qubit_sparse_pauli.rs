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

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
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
    imports::ImportOnceCell,
    slice::{PySequenceIndex, SequenceIndex},
};

use super::qubit_sparse_pauli::{
    raw_parts_from_sparse_list, ArithmeticError, CoherenceError, InnerReadError, InnerWriteError,
    LabelError, Pauli, PyQubitSparsePauli, PyQubitSparsePauliList, QubitSparsePauli,
    QubitSparsePauliList, QubitSparsePauliView,
};

static PAULI_TYPE: ImportOnceCell = ImportOnceCell::new("qiskit.quantum_info", "Pauli");

/// A list of Pauli operators stored in a qubit-sparse format.
///
/// See [PyQubitSparsePauliList] for detailed docs.
#[derive(Clone, Debug, PartialEq)]
pub struct PhasedQubitSparsePauliList {
    /// The paulis.
    qubit_sparse_pauli_list: QubitSparsePauliList,
    /// Phases.
    phases: Vec<isize>,
}

impl PhasedQubitSparsePauliList {
    /// Create a new phased qubit-sparse Pauli list from the raw components that make it up.
    ///
    /// This checks the input values for data coherence on entry.  If you are certain you have the
    /// correct values, you can call `new_unchecked` instead.
    pub fn new(
        qubit_sparse_pauli_list: QubitSparsePauliList,
        phases: Vec<isize>,
    ) -> Result<Self, CoherenceError> {
        if phases.len() != qubit_sparse_pauli_list.num_terms() {
            return Err(CoherenceError::MismatchedPhaseCount {
                phases: phases.len(),
                qspl: qubit_sparse_pauli_list.num_terms(),
            });
        }
        // SAFETY: we've just done the coherence checks.
        Ok(unsafe { Self::new_unchecked(qubit_sparse_pauli_list, phases) })
    }

    /// Create a new [QubitSparsePauliList] from the raw components without checking data coherence.
    ///
    /// # Safety
    ///
    /// It is up to the caller to ensure that the data-coherence requirements, as enumerated in the
    /// struct-level documentation, have been upheld.
    #[inline(always)]
    pub unsafe fn new_unchecked(
        qubit_sparse_pauli_list: QubitSparsePauliList,
        phases: Vec<isize>,
    ) -> Self {
        Self {
            qubit_sparse_pauli_list,
            phases,
        }
    }

    
    /// Get the number of qubits the paulis are defined on.
    #[inline]
    pub fn num_qubits(&self) -> u32 {
        self.qubit_sparse_pauli_list.num_qubits()
    }

    /// Get the number of elements in the list.
    #[inline]
    pub fn num_terms(&self) -> usize {
        self.phases.len()
    }

    /// Clear all the elements of the list.
    ///
    /// This does not change the capacity of the internal allocations, so subsequent addition or
    /// substraction of elements in the list may not need to reallocate.
    pub fn clear(&mut self) {
        self.qubit_sparse_pauli_list.clear();
        self.phases.clear();
    }
}

/// A view object onto a single term of a `QubitSparsePauliList`.
///
/// The lengths of `paulis` and `indices` are guaranteed to be created equal, but might be zero
/// (in the case that the term is the identity).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct PhasedQubitSparsePauliView<'a> {
    pub qubit_sparse_pauli_view: QubitSparsePauliView<'a>,
    pub phase: &'a isize,
}
impl PhasedQubitSparsePauliView<'_> {
    /// Convert this `PhasedQubitSparsePauliView` into an owning [PhasedQubitSparsePauli] of the same data.
    pub fn to_term(&self) -> PhasedQubitSparsePauli {
        PhasedQubitSparsePauli {
            qubit_sparse_pauli: self.qubit_sparse_pauli_view.to_term(),
            phase: *self.phase
        }
    }
    
    pub fn to_sparse_str(self) -> String {
        let num_ys = self.qubit_sparse_pauli_view.num_ys();
        let phase_str = match (self.phase - num_ys).rem_euclid(4) {
            0 => "",
            1 => "-i",
            2 => "-",
            3 => "i",
            _ => unreachable!("`x % 4` has only four values"),
        };

        phase_str.to_owned() + &self.qubit_sparse_pauli_view.to_sparse_str()
    }
}

/// A single phased qubit-sparse Pauli operator.
#[derive(Clone, Debug, PartialEq)]
pub struct PhasedQubitSparsePauli {
    /// The qubit sparse Pauli.
    qubit_sparse_pauli: QubitSparsePauli,
    /// phase.
    phase: isize,
}

impl PhasedQubitSparsePauli {
    /// Create a new phased qubit-sparse Pauli from the raw components that make it up.
    pub fn new(
        qubit_sparse_pauli: QubitSparsePauli,
        phase: isize,
    ) -> Self {

        Self {
            qubit_sparse_pauli,
            phase,
        }
    }

    /// Get the number of qubits the paulis are defined on.
    #[inline]
    pub fn num_qubits(&self) -> u32 {
        self.qubit_sparse_pauli.num_qubits()
    }

    /// Create the identity [QubitSparsePauli] on ``num_qubits`` qubits.
    pub fn identity(num_qubits: u32) -> Self {
        Self {
            qubit_sparse_pauli: QubitSparsePauli::identity(num_qubits),
            phase: 0
        }
    }

    /// Get a view version of this object.
    pub fn view(&self) -> PhasedQubitSparsePauliView<'_> {
        PhasedQubitSparsePauliView {
            qubit_sparse_pauli_view: self.qubit_sparse_pauli.view(),
            phase: &self.phase
        }
    }

    /// Convert this single Pauli into a :class:`PhasedQubitSparsePauliList`.
    pub fn to_phased_qubit_sparse_pauli_list(&self) -> PhasedQubitSparsePauliList {
        PhasedQubitSparsePauliList {
            qubit_sparse_pauli_list: self.qubit_sparse_pauli.to_qubit_sparse_pauli_list(),
            phases: vec![self.phase],
        }
    }

    // Check if `self` commutes with `other`
    pub fn commutes(&self, other: &PhasedQubitSparsePauli) -> Result<bool, ArithmeticError> {
        if self.num_qubits() != other.num_qubits() {
            return Err(ArithmeticError::MismatchedQubits {
                left: self.num_qubits(),
                right: other.num_qubits(),
            });
        }

        return self.qubit_sparse_pauli.commutes(&other.qubit_sparse_pauli)
    }
}


#[pyclass(name = "PhasedQubitSparsePauli", frozen, module = "qiskit.quantum_info")]
#[derive(Clone, Debug)]
pub struct PyPhasedQubitSparsePauli {
    inner: PhasedQubitSparsePauli,
}

impl PyPhasedQubitSparsePauli {
    pub fn inner(&self) -> &PhasedQubitSparsePauli {
        &self.inner
    }
}

#[pymethods]
impl PyPhasedQubitSparsePauli {
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
        //if let Ok(label) = data.extract::<String>() {
        //    let num_qubits = num_qubits.unwrap_or(label.len() as u32);
        //    if num_qubits as usize != label.len() {
        //        return Err(PyValueError::new_err(format!(
        //            "explicitly given 'num_qubits' ({}) does not match label ({})",
        //            num_qubits,
        //            label.len(),
        //        )));
        //    }
        //    return Self::from_label(&label);
        //}
        //if let Ok(sparse_label) = data.extract() {
        //    let Some(num_qubits) = num_qubits else {
        //        return Err(PyValueError::new_err(
        //            "if using the sparse-label form, 'num_qubits' must be provided",
        //        ));
        //    };
        //    return Self::from_sparse_label(sparse_label, num_qubits);
        //}
        Err(PyTypeError::new_err(format!(
            "unknown input format for 'PhasedQubitSparsePauli': {}",
            data.get_type().repr()?,
        )))
    }

    /// Construct a :class:`.QubitSparsePauli` from a single :class:`~.quantum_info.Pauli` instance.
    ///
    /// Note that the phase of the Pauli is dropped.
    ///
    /// Args:
    ///     pauli (:class:`~.quantum_info.Pauli`): the single Pauli to convert.
    ///
    /// Examples:
    ///
    ///     .. code-block:: python
    ///
    ///         >>> label = "IYXZI"
    ///         >>> pauli = Pauli(label)
    ///         >>> QubitSparsePauli.from_pauli(pauli)
    ///         <QubitSparsePauli on 5 qubits: Y_3 X_2 Z_1>
    ///         >>> assert QubitSparsePauli.from_label(label) == QubitSparsePauli.from_pauli(pauli)
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
        let mut paulis = Vec::new();
        let mut indices = Vec::new();
        let mut num_ys = 0;
        for (i, (x, z)) in x.as_array().iter().zip(z.as_array().iter()).enumerate() {
            // The only failure case possible here is the identity, because of how we're
            // constructing the value to convert.
            let Ok(term) = ::bytemuck::checked::try_cast(((*x as u8) << 1) | (*z as u8)) else {
                continue;
            };
            num_ys += (term == Pauli::Y) as isize;
            indices.push(i as u32);
            paulis.push(term);
        }
        let group_phase = pauli
            // `Pauli`'s `_phase` is a Numpy array ...
            .getattr(intern!(py, "_phase"))?
            // ... that should have exactly 1 element ...
            .call_method0(intern!(py, "item"))?
            // ... which is some integral type.
            .extract::<isize>()?;

        let inner = PhasedQubitSparsePauli::new(
            QubitSparsePauli::new(
                num_qubits,
                paulis.into_boxed_slice(),
                indices.into_boxed_slice(),
            )?,
            group_phase
        );
        Ok(inner.into())
    }


    /// Get the identity operator for a given number of qubits.
    ///
    /// Examples:
    ///
    ///     Get the identity on 100 qubits::
    ///
    ///         >>> QubitSparsePauli.identity(100)
    ///         <QubitSparsePauli on 100 qubits: >
    #[pyo3(signature = (/, num_qubits))]
    #[staticmethod]
    pub fn identity(num_qubits: u32) -> Self {
        PhasedQubitSparsePauli::identity(num_qubits).into()
    }

    /// Convert this Pauli into a single element :class:`PhaseddQubitSparsePauliList`.
    fn to_phased_qubit_sparse_pauli_list(&self) -> PyResult<PyPhasedQubitSparsePauliList> {
        Ok(self.inner.to_phased_qubit_sparse_pauli_list().into())
    }

    /// Check if `self`` commutes with another qubit sparse pauli.
    ///
    /// Args:
    ///     other (PhasedQubitSparsePauli): the qubit sparse Pauli to check for commutation with.
    fn commutes(&self, other: PyPhasedQubitSparsePauli) -> PyResult<bool> {
        Ok(self.inner.commutes(&other.inner)?)
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
            "PhasedQubitSparsePauli",
            self.inner.num_qubits(),
            if self.inner.num_qubits() == 1 {
                ""
            } else {
                "s"
            },
            self.inner.view().to_sparse_str(),
        ))
    }

    //fn __getnewargs__(slf_: Bound<Self>) -> PyResult<Bound<PyTuple>> {
    //    let py = slf_.py();
    //    let borrowed = slf_.borrow();
    //    (
    //        borrowed.inner.num_qubits(),
    //        Self::get_paulis(slf_.clone()),
    //        Self::get_indices(slf_),
    //    )
    //        .into_pyobject(py)
    //}

    //fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
    //    let paulis: &[u8] = ::bytemuck::cast_slice(self.inner.paulis());
    //    (
    //        py.get_type::<Self>().getattr("from_raw_parts")?,
    //        (
    //            self.inner.num_qubits(),
    //            PyArray1::from_slice(py, paulis),
    //            PyArray1::from_slice(py, self.inner.indices()),
    //        ),
    //    )
    //        .into_pyobject(py)
    //}

    /// Get a copy of this term.
    fn copy(&self) -> Self {
        self.clone()
    }

    /// The number of qubits the term is defined on.
    #[getter]
    fn get_num_qubits(&self) -> u32 {
        self.inner.num_qubits()
    }
}

#[pyclass(
    name = "PhasedQubitSparsePauliList",
    module = "qiskit.quantum_info",
    sequence
)]
#[derive(Debug)]
pub struct PyPhasedQubitSparsePauliList {
    // This class keeps a pointer to a pure Rust-SparseTerm and serves as interface from Python.
    pub inner: Arc<RwLock<PhasedQubitSparsePauliList>>,
}
#[pymethods]
impl PyPhasedQubitSparsePauliList {
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
        //if let Ok(label) = data.extract::<String>() {
        //    let num_qubits = num_qubits.unwrap_or(label.len() as u32);
        //    if num_qubits as usize != label.len() {
        //        return Err(PyValueError::new_err(format!(
        //            "explicitly given 'num_qubits' ({}) does not match label ({})",
        //            num_qubits,
        //            label.len(),
        //        )));
        //    }
        //    return Self::from_label(&label).map_err(PyErr::from);
        //}
        //if let Ok(pauli_list) = data.downcast_exact::<Self>() {
        //    check_num_qubits(data)?;
        //    let borrowed = pauli_list.borrow();
        //    let inner = borrowed.inner.read().map_err(|_| InnerReadError)?;
        //    return Ok(inner.clone().into());
        //}
        // The type of `vec` is inferred from the subsequent calls to `Self::from_list` or
        // `Self::from_sparse_list` to be either the two-tuple or the three-tuple form during the
        // `extract`.  The empty list will pass either, but it means the same to both functions.
        //if let Ok(vec) = data.extract() {
        //    return Self::from_list(vec, num_qubits);
        //}
        //if let Ok(vec) = data.extract() {
        //    let Some(num_qubits) = num_qubits else {
        //        return Err(PyValueError::new_err(
        //            "if using the sparse-list form, 'num_qubits' must be provided",
        //        ));
        //    };
        //    return Self::from_sparse_list(vec, num_qubits);
        //}
        if let Ok(term) = data.downcast_exact::<PyPhasedQubitSparsePauli>() {
            return term.borrow().to_phased_qubit_sparse_pauli_list();
        };
        //if let Ok(pauli_list) = Self::from_qubit_sparse_paulis(data, num_qubits) {
        //    return Ok(pauli_list);
        //}
        Err(PyTypeError::new_err(format!(
            "unknown input format for 'PhasedQubitSparsePauliList': {}",
            data.get_type().repr()?,
        )))
    }

    /// Get a copy of this qubit sparse Pauli list.
    ///
    /// Examples:
    ///
    ///     .. code-block:: python
    ///
    ///         >>> qubit_sparse_pauli_list = QubitSparsePauliList.from_list(["IXZXYYZZ", "ZXIXYYZZ"])
    ///         >>> assert qubit_sparse_pauli_list == qubit_sparse_pauli_list.copy()
    ///         >>> assert qubit_sparse_pauli_list is not qubit_sparse_pauli_list.copy()
    fn copy(&self) -> PyResult<Self> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        Ok(inner.clone().into())
    }

    /// The number of qubits the operators in the list act on.
    ///
    /// This is not inferable from any other shape or values, since identities are not stored
    /// explicitly.
    #[getter]
    #[inline]
    pub fn num_qubits(&self) -> PyResult<u32> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        Ok(inner.num_qubits())
    }

    /// The number of elements in the list.
    #[getter]
    #[inline]
    pub fn num_terms(&self) -> PyResult<usize> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        Ok(inner.num_terms())
    }

    /// Get the empty list for a given number of qubits.
    ///
    /// The empty list contains no elements, and is the identity element for joining two
    /// :class:`QubitSparsePauliList` instances.
    ///
    /// Examples:
    ///
    ///     Get the empty list on 100 qubits::
    ///
    ///         >>> QubitSparsePauliList.empty(100)
    ///         <QubitSparsePauliList with 0 elements on 100 qubits: []>
    //#[pyo3(signature = (/, num_qubits))]
    //#[staticmethod]
    //pub fn empty(num_qubits: u32) -> Self {
    //    PhasedQubitSparsePauliList::empty(num_qubits).into()
    //}

    /// Construct a :class:`.QubitSparsePauliList` from a single :class:`~.quantum_info.Pauli`
    /// instance.
    ///
    /// The output list will have a single term. Note that the phase is dropped.
    ///
    /// Args:
    ///     pauli (:class:`~.quantum_info.Pauli`): the single Pauli to convert.
    ///
    /// Examples:
    ///
    ///     .. code-block:: python
    ///
    ///         >>> label = "IYXZI"
    ///         >>> pauli = Pauli(label)
    ///         >>> QubitSparsePauliList.from_pauli(pauli)
    ///         <QubitSparsePauliList with 1 element on 5 qubits: [Y_3 X_2 Z_1]>
    ///         >>> assert QubitSparsePauliList.from_label(label) == QubitSparsePauliList.from_pauli(pauli)
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
        let mut paulis = Vec::new();
        let mut indices = Vec::new();
        for (i, (x, z)) in x.as_array().iter().zip(z.as_array().iter()).enumerate() {
            // The only failure case possible here is the identity, because of how we're
            // constructing the value to convert.
            let Ok(term) = ::bytemuck::checked::try_cast(((*x as u8) << 1) | (*z as u8)) else {
                continue;
            };
            indices.push(i as u32);
            paulis.push(term);
        }
        let boundaries = vec![0, indices.len()];
        let qspl = QubitSparsePauliList::new(num_qubits, paulis, indices, boundaries)?;
        let inner = PhasedQubitSparsePauliList{
            qubit_sparse_pauli_list: qspl,
            phases: vec![0] //needs to be corrected ***************************************************
        };
        Ok(inner.into())
    }

    /// Clear all the elements from the list, making it equal to the empty list again.
    ///
    /// This does not change the capacity of the internal allocations, so subsequent addition or
    /// substraction operations resulting from composition may not need to reallocate.
    ///
    /// Examples:
    ///
    ///     .. code-block:: python
    ///
    ///         >>> pauli_list = QubitSparsePauliList.from_list(["IXXXYY", "ZZYZII"])
    ///         >>> pauli_list.clear()
    ///         >>> assert pauli_list == QubitSparsePauliList.empty(pauli_list.num_qubits)
    pub fn clear(&mut self) -> PyResult<()> {
        let mut inner = self.inner.write().map_err(|_| InnerWriteError)?;
        inner.clear();
        Ok(())
    }

    fn __len__(&self) -> PyResult<usize> {
        self.num_terms()
    }

    //fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
    //    let inner = self.inner.read().map_err(|_| InnerReadError)?;
    //    (
    //        py.get_type::<Self>().getattr("from_sparse_list")?,
    //        (self.to_sparse_list(py)?, inner.num_qubits()),
    //    )
    //        .into_pyobject(py)
    //}

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
}

impl From<PhasedQubitSparsePauli> for PyPhasedQubitSparsePauli {
    fn from(val: PhasedQubitSparsePauli) -> PyPhasedQubitSparsePauli {
        PyPhasedQubitSparsePauli { inner: val }
    }
}
impl<'py> IntoPyObject<'py> for PhasedQubitSparsePauli {
    type Target = PyPhasedQubitSparsePauli;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        PyPhasedQubitSparsePauli::from(self).into_pyobject(py)
    }
}
impl From<PhasedQubitSparsePauliList> for PyPhasedQubitSparsePauliList {
    fn from(val: PhasedQubitSparsePauliList) -> PyPhasedQubitSparsePauliList {
        PyPhasedQubitSparsePauliList {
            inner: Arc::new(RwLock::new(val)),
        }
    }
}
impl<'py> IntoPyObject<'py> for PhasedQubitSparsePauliList {
    type Target = PyPhasedQubitSparsePauliList;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        PyPhasedQubitSparsePauliList::from(self).into_pyobject(py)
    }
}

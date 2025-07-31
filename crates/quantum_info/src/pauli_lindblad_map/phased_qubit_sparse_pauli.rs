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
    bit::PyQubit, imports::ImportOnceCell, slice::{PySequenceIndex, SequenceIndex}
};

use crate::pauli_lindblad_map::qubit_sparse_pauli;

use super::qubit_sparse_pauli::{
    raw_parts_from_sparse_list, ArithmeticError, CoherenceError, InnerReadError, InnerWriteError,
    LabelError, Pauli, PyQubitSparsePauli, PyQubitSparsePauliList, QubitSparsePauli,
    QubitSparsePauliList, QubitSparsePauliView
};

static PAULI_TYPE: ImportOnceCell = ImportOnceCell::new("qiskit.quantum_info", "Pauli");
static PAULI_PY_ENUM: GILOnceCell<Py<PyType>> = GILOnceCell::new();

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
            1 => "(-i)",
            2 => "(-1)",
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
    
    // Composition of two pauli operators self @ other.
    pub fn compose(&self, other: &PhasedQubitSparsePauli) -> Result<PhasedQubitSparsePauli, ArithmeticError> {
        if self.num_qubits() != other.num_qubits() {
            return Err(ArithmeticError::MismatchedQubits {
                left: self.num_qubits(),
                right: other.num_qubits(),
            });
        }

        // if either are proportional to the identity, return a clone of the other with corrected
        // phase
        if self.qubit_sparse_pauli.indices().is_empty() {
            return Ok(PhasedQubitSparsePauli { 
                qubit_sparse_pauli: other.qubit_sparse_pauli.clone(), 
                phase: self.phase + other.phase 
            });
        }

        if other.qubit_sparse_pauli.indices().is_empty() {
            return Ok(PhasedQubitSparsePauli { 
                qubit_sparse_pauli: self.qubit_sparse_pauli.clone(), 
                phase: self.phase + other.phase 
            });
        }

        let mut paulis = Vec::new();
        let mut indices = Vec::new();
        let mut new_phase = self.phase + other.phase;

        let mut self_idx = 0;
        let mut other_idx = 0;

        let self_indices = self.qubit_sparse_pauli.indices();
        let self_paulis = self.qubit_sparse_pauli.paulis();
        let other_indices = other.qubit_sparse_pauli.indices();
        let other_paulis = other.qubit_sparse_pauli.paulis();
        
        // iterate through each entry of self and other one time, incrementing based on the ordering
        // or equality of self_idx and other_idx, until one of them runs out of entries
        while self_idx < self_indices.len() && other_idx < other_indices.len() {
            if self_indices[self_idx] < other_indices[other_idx] {
                // if the current qubit index of self is strictly less than other, append the pauli
                paulis.push(self_paulis[self_idx]);
                indices.push(self_indices[self_idx]);
                self_idx += 1;
            } else if self_indices[self_idx] == other_indices[other_idx] {
                // if the indices are the same, perform multiplication and append if non-identity
                let new_pauli = (self_paulis[self_idx] as u8) ^ (other_paulis[other_idx] as u8);
                if new_pauli != 0 {
                    paulis.push(match new_pauli {
                        0b01 => Ok(Pauli::Z),
                        0b10 => Ok(Pauli::X),
                        0b11 => Ok(Pauli::Y),
                        _ => Err(ArithmeticError::PauliMultiplication { b: new_pauli }),
                    }?);
                    indices.push(self_indices[self_idx])
                }
                if (self_paulis[self_idx] == Pauli::X || self_paulis[self_idx] == Pauli::Y) && (other_paulis[other_idx] == Pauli::Z || other_paulis[other_idx] == Pauli::Y) {
                    new_phase += 2;
                }
                self_idx += 1;
                other_idx += 1;
            } else {
                // same as the first if block but with roles of self and other reversed
                paulis.push(other_paulis[other_idx]);
                indices.push(other_indices[other_idx]);
                other_idx += 1;
            }
        }

        // if any entries remain in either pauli, append them
        if other_idx != other_indices.len() {
            paulis.append(&mut other_paulis[other_idx..].to_vec());
            indices.append(&mut other_indices[other_idx..].to_vec());
        } else if self_idx != self_indices.len() {
            paulis.append(&mut self_paulis[self_idx..].to_vec());
            indices.append(&mut self_indices[self_idx..].to_vec());
        }

        Ok(PhasedQubitSparsePauli { 
            qubit_sparse_pauli: unsafe {QubitSparsePauli::new_unchecked(
                self.num_qubits(),
                paulis.into_boxed_slice(),
                indices.into_boxed_slice(),
            )}, 
            phase: new_phase
        })
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

    // Check equality of operators
    fn eq(&self, other: &PhasedQubitSparsePauli) -> bool {
        ((self.phase - other.phase).rem_euclid(4) == 0) && self.qubit_sparse_pauli == other.qubit_sparse_pauli
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
        if let Ok(label) = data.extract::<String>() {
            let num_qubits = num_qubits.unwrap_or(label.len() as u32);
            if num_qubits as usize != label.len() {
                return Err(PyValueError::new_err(format!(
                    "explicitly given 'num_qubits' ({}) does not match label ({})",
                    num_qubits,
                    label.len(),
                )));
            }
            return Self::from_label(&label);
        }
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

    /// Construct a :class:`.QubitSparsePauli` from raw Numpy arrays that match :ref:`the required
    /// data representation described in the class-level documentation
    /// <qubit-sparse-pauli-arrays>`.
    ///
    /// The data from each array is copied into fresh, growable Rust-space allocations.
    ///
    /// Args:
    ///     num_qubits: number of qubits the operator acts on.
    ///     paulis: list of the single-qubit terms.  This should be a Numpy array with dtype
    ///         :attr:`~numpy.uint8` (which is compatible with :class:`.Pauli`).
    ///     indices: sorted list of the qubits each single-qubit term corresponds to.  This should
    ///         be a Numpy array with dtype :attr:`~numpy.uint32`.
    ///
    /// Examples:
    ///
    ///     Construct a :math:`Z` operator acting on qubit 50 of 100 qubits.
    ///
    ///         >>> num_qubits = 100
    ///         >>> terms = np.array([QubitSparsePauli.Pauli.Z], dtype=np.uint8)
    ///         >>> indices = np.array([50], dtype=np.uint32)
    ///         >>> QubitSparsePauli.from_raw_parts(num_qubits, terms, indices)
    ///         <QubitSparsePauli on 100 qubits: Z_50>
    /// NOTE: *********************************************************************************************
    /// The phase here is the internally stored phase, not the "group phase". Need to think about this
    #[staticmethod]
    #[pyo3(signature = (/, num_qubits, paulis, indices, phase))]
    fn from_raw_parts(num_qubits: u32, paulis: Vec<Pauli>, indices: Vec<u32>, phase: isize) -> PyResult<Self> {
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
        let qubit_sparse_pauli = QubitSparsePauli::new(num_qubits, paulis, sorted_indices.into_boxed_slice())?;
        let inner = PhasedQubitSparsePauli::new(qubit_sparse_pauli, phase);
        Ok(PyPhasedQubitSparsePauli { inner })
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
        for (i, (x, z)) in x.as_array().iter().zip(z.as_array().iter()).enumerate() {
            // The only failure case possible here is the identity, because of how we're
            // constructing the value to convert.
            let Ok(term) = ::bytemuck::checked::try_cast(((*x as u8) << 1) | (*z as u8)) else {
                continue;
            };
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

    /// Construct a qubit sparse Pauli from a sparse label, given as a tuple of a string of Paulis,
    /// and the indices of the corresponding qubits.
    ///
    /// This is analogous to :meth:`.SparsePauliOp.from_sparse_list`.
    ///
    /// Args:
    ///     sparse_label (tuple[str, Sequence[int]]): labels and the qubits each single-qubit term
    ///         applies to.
    ///
    ///     num_qubits (int): the number of qubits the operator acts on.
    ///
    /// Examples:
    ///
    ///     Construct a simple Pauli::
    ///
    ///         >>> QubitSparsePauli.from_sparse_label(
    ///         ...     ("ZX", (1, 4)),
    ///         ...     num_qubits=5,
    ///         ... )
    ///         <QubitSparsePauli on 5 qubits: X_4 Z_1>
    ///
    ///     This method can replicate the behavior of :meth:`from_label`, if the qubit-arguments
    ///     field of the tuple is set to decreasing integers::
    ///
    ///         >>> label = "XYXZ"
    ///         >>> from_label = QubitSparsePauli.from_label(label)
    ///         >>> from_sparse_label = QubitSparsePauli.from_sparse_label(
    ///         ...     (label, (3, 2, 1, 0)),
    ///         ...     num_qubits=4
    ///         ... )
    ///         >>> assert from_label == from_sparse_label
    #[staticmethod]
    #[pyo3(signature = (/, sparse_label, num_qubits))]
    fn from_sparse_label(sparse_label: (isize, String, Vec<u32>), num_qubits: u32) -> PyResult<Self> {
        let phase = sparse_label.0;
        let label = sparse_label.1;
        let indices = sparse_label.2;
        let mut paulis = Vec::new();
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
            entry.insert(Pauli::try_from_u8(*letter).map_err(|_| LabelError::OutsideAlphabet)?);
        }
        for (index, term) in sorted.iter() {
            let Some(term) = term else {
                continue;
            };
            sorted_indices.push(*index);
            paulis.push(*term);
        }

        let num_ys = isize::try_from(paulis.iter().filter(|&p| *p == Pauli::Y).count())?;

        let qubit_sparse_pauli = QubitSparsePauli::new(
            num_qubits,
            paulis.into_boxed_slice(),
            sorted_indices.into_boxed_slice(),
        )?;
        let inner = PhasedQubitSparsePauli::new(qubit_sparse_pauli, phase + num_ys);
        Ok(inner.into())
    }

    /// Construct from a dense string label.
    ///
    /// The label must be a sequence of the alphabet ``'IXYZ'``.  The label is interpreted
    /// analogously to a bitstring.  In other words, the right-most letter is associated with qubit
    /// 0, and so on.  This is the same as the labels for :class:`~.quantum_info.Pauli` and
    /// :class:`.SparsePauliOp`.
    ///
    /// Args:
    ///     label (str): the dense label.
    ///
    /// Examples:
    ///
    ///     .. code-block:: python
    ///
    ///         >>> QubitSparsePauli.from_label("IIIIXZI")
    ///         <QubitSparsePauli on 7 qubits: X_2 Z_1>
    ///         >>> label = "IYXZI"
    ///         >>> pauli = Pauli(label)
    ///         >>> assert QubitSparsePauli.from_label(label) == QubitSparsePauli.from_pauli(pauli)
    #[staticmethod]
    #[pyo3(signature = (label, /))]
    fn from_label(label: &str) -> PyResult<Self> {
        
        let qubit_sparse_pauli: QubitSparsePauli = QubitSparsePauli::from_dense_label(label)?;
        let num_ys = qubit_sparse_pauli.view().num_ys();
        let inner = PhasedQubitSparsePauli {
            qubit_sparse_pauli: qubit_sparse_pauli,
            phase: num_ys.rem_euclid(4)
        };
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

    /// Read-only view onto the phase.
    #[getter]
    fn get_phase(slf_: Bound<Self>) -> isize {
        let borrowed = slf_.borrow();
        let phase = borrowed.inner.phase;

        let num_ys = borrowed.inner.qubit_sparse_pauli.view().num_ys();
        return (phase - num_ys).rem_euclid(4)
    }

    /// Convert this Pauli into a single element :class:`PhaseddQubitSparsePauliList`.
    fn to_phased_qubit_sparse_pauli_list(&self) -> PyResult<PyPhasedQubitSparsePauliList> {
        Ok(self.inner.to_phased_qubit_sparse_pauli_list().into())
    }

    /// Composition with another :class:`PhasedQubitSparsePauli`.
    ///
    /// Args:
    ///     other (PhasedQubitSparsePauli): the qubit sparse Pauli to compose with.
    fn compose(&self, other: PyPhasedQubitSparsePauli) -> PyResult<Self> {
        Ok(PyPhasedQubitSparsePauli {
            inner: self.inner.compose(&other.inner)?,
        })
    }

    fn __matmul__(&self, other: PyPhasedQubitSparsePauli) -> PyResult<Self> {
        self.compose(other)
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

    /// Read-only view onto the individual single-qubit terms.
    ///
    /// The only valid values in the array are those with a corresponding
    /// :class:`~QubitSparsePauli.Pauli`.
    #[getter]
    fn get_paulis(slf_: Bound<Self>) -> Bound<PyArray1<u8>> {
        let borrowed = slf_.borrow();
        let paulis = borrowed.inner.qubit_sparse_pauli.paulis();
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

    /// Read-only view onto the indices of each non-identity single-qubit term.
    ///
    /// The indices will always be in sorted order.
    #[getter]
    fn get_indices(slf_: Bound<Self>) -> Bound<PyArray1<u32>> {
        let borrowed = slf_.borrow();
        let indices = borrowed.inner.qubit_sparse_pauli.indices();
        let arr = ::ndarray::aview1(indices);
        // SAFETY: in order to call this function, the lifetime of `self` must be managed by Python.
        // We tie the lifetime of the array to `slf_`, and there are no public ways to modify the
        // `Box<[u32]>` allocation (including dropping or reallocating it) other than the entire
        // object getting dropped, which Python will keep safe.
        let out = unsafe { PyArray1::borrow_from_array(&arr, slf_.into_any()) };
        out.readwrite().make_nonwriteable();
        out
    }

    // The documentation for this is inlined into the class-level documentation of
    // :class:`QubitSparsePauliList`.
    #[allow(non_snake_case)]
    #[classattr]
    fn Pauli(py: Python) -> PyResult<Py<PyType>> {
        PyQubitSparsePauli::Pauli(py)
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

// This code is part of Qiskit.
//
// (C) Copyright IBM 2023
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::quantum_circuit::py_ext;
use pyo3::basic::CompareOp;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use pyo3::{PyObject, PyResult};

/// A single instruction in a :class:`.QuantumCircuit`, comprised of the :attr:`operation` and
/// various operands.
///
/// .. note::
///
///     There is some possible confusion in the names of this class, :class:`~.circuit.Instruction`,
///     and :class:`~.circuit.Operation`, and this class's attribute :attr:`operation`.  Our
///     preferred terminology is by analogy to assembly languages, where an "instruction" is made up
///     of an "operation" and its "operands".
///
///     Historically, :class:`~.circuit.Instruction` came first, and originally contained the qubits
///     it operated on and any parameters, so it was a true "instruction".  Over time,
///     :class:`.QuantumCircuit` became responsible for tracking qubits and clbits, and the class
///     became better described as an "operation".  Changing the name of such a core object would be
///     a very unpleasant API break for users, and so we have stuck with it.
///
///     This class was created to provide a formal "instruction" context object in
///     :class:`.QuantumCircuit.data`, which had long been made of ad-hoc tuples.  With this, and
///     the advent of the :class:`~.circuit.Operation` interface for adding more complex objects to
///     circuits, we took the opportunity to correct the historical naming.  For the time being,
///     this leads to an awkward case where :attr:`.CircuitInstruction.operation` is often an
///     :class:`~.circuit.Instruction` instance (:class:`~.circuit.Instruction` implements the
///     :class:`.Operation` interface), but as the :class:`.Operation` interface gains more use,
///     this confusion will hopefully abate.
///
/// .. warning::
///
///     This is a lightweight internal class and there is minimal error checking; you must respect
///     the type hints when using it.  It is the user's responsibility to ensure that direct
///     mutations of the object do not invalidate the types, nor the restrictions placed on it by
///     its context.  Typically this will mean, for example, that :attr:`qubits` must be a sequence
///     of distinct items, with no duplicates.
#[pyclass(
    freelist = 20,
    sequence,
    get_all,
    module = "qiskit._accelerate.quantum_circuit"
)]
#[derive(Clone, Debug)]
pub struct CircuitInstruction {
    /// The logical operation that this instruction represents an execution of.
    pub operation: PyObject,
    /// A sequence of the qubits that the operation is applied to.
    pub qubits: Py<PyTuple>,
    /// A sequence of the classical bits that this operation reads from or writes to.
    pub clbits: Py<PyTuple>,
}

#[pymethods]
impl CircuitInstruction {
    #[new]
    pub fn new(
        py: Python<'_>,
        operation: PyObject,
        qubits: Option<&PyAny>,
        clbits: Option<&PyAny>,
    ) -> PyResult<Self> {
        fn as_tuple(py: Python<'_>, seq: Option<&PyAny>) -> PyResult<Py<PyTuple>> {
            match seq {
                None => Ok(py_ext::tuple_new_empty(py)),
                Some(seq) => {
                    if seq.is_instance_of::<PyTuple>() {
                        Ok(seq.downcast_exact::<PyTuple>()?.into_py(py))
                    } else if seq.is_instance_of::<PyList>() {
                        let seq = seq.downcast_exact::<PyList>()?;
                        Ok(py_ext::tuple_from_list(seq))
                    } else {
                        // New tuple from iterable.
                        Ok(py_ext::tuple_new(
                            py,
                            seq.iter()?
                                .map(|o| Ok(o?.into_py(py)))
                                .collect::<PyResult<Vec<PyObject>>>()?,
                        ))
                    }
                }
            }
        }

        Ok(CircuitInstruction {
            operation,
            qubits: as_tuple(py, qubits)?,
            clbits: as_tuple(py, clbits)?,
        })
    }

    /// Returns a shallow copy.
    ///
    /// Returns:
    ///     CircuitInstruction: The shallow copy.
    pub fn copy(&self) -> Self {
        self.clone()
    }

    /// Creates a shallow copy with the given fields replaced.
    ///
    /// Returns:
    ///     CircuitInstruction: A new instance with the given fields replaced.
    pub fn replace(
        &self,
        py: Python<'_>,
        operation: Option<PyObject>,
        qubits: Option<&PyAny>,
        clbits: Option<&PyAny>,
    ) -> PyResult<Self> {
        CircuitInstruction::new(
            py,
            operation.unwrap_or_else(|| self.operation.clone_ref(py)),
            Some(qubits.unwrap_or_else(|| self.qubits.as_ref(py))),
            Some(clbits.unwrap_or_else(|| self.clbits.as_ref(py))),
        )
    }

    fn __getstate__(&self, py: Python<'_>) -> PyObject {
        (
            self.operation.as_ref(py),
            self.qubits.as_ref(py),
            self.clbits.as_ref(py),
        )
            .into_py(py)
    }

    fn __setstate__(&mut self, _py: Python<'_>, state: &PyTuple) -> PyResult<()> {
        self.operation = state.get_item(0)?.extract()?;
        self.qubits = state.get_item(1)?.extract()?;
        self.clbits = state.get_item(2)?.extract()?;
        Ok(())
    }

    pub fn __getnewargs__(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok((
            self.operation.as_ref(py),
            self.qubits.as_ref(py),
            self.clbits.as_ref(py),
        )
            .into_py(py))
    }

    pub fn __repr__(self_: &PyCell<Self>, py: Python<'_>) -> PyResult<String> {
        let type_name = self_.get_type().name()?;
        let r = self_.try_borrow()?;
        Ok(format!(
            "{}(\
            operation={}\
            , qubits={}\
            , clbits={}\
            )",
            type_name,
            r.operation.as_ref(py).repr()?,
            r.qubits.as_ref(py).repr()?,
            r.clbits.as_ref(py).repr()?
        ))
    }

    // Legacy tuple-like interface support.
    //
    // For a best attempt at API compatibility during the transition to using this new class, we need
    // the interface to behave exactly like the old 3-tuple `(inst, qargs, cargs)` if it's treated
    // like that via unpacking or similar.  That means that the `parameters` field is completely
    // absent, and the qubits and clbits must be converted to lists.
    pub fn _legacy_format(&self, py: Python<'_>) -> PyObject {
        PyTuple::new(
            py,
            [
                self.operation.as_ref(py),
                self.qubits.as_ref(py).to_list(),
                self.clbits.as_ref(py).to_list(),
            ],
        )
        .into_py(py)
    }

    pub fn __getitem__(&self, py: Python<'_>, key: &PyAny) -> PyResult<PyObject> {
        Ok(self
            ._legacy_format(py)
            .as_ref(py)
            .get_item(key)?
            .into_py(py))
    }

    pub fn __iter__(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(self._legacy_format(py).as_ref(py).iter()?.into_py(py))
    }

    pub fn __len__(&self) -> usize {
        3
    }

    pub fn __richcmp__(
        self_: &PyCell<Self>,
        other: &PyAny,
        op: CompareOp,
        py: Python<'_>,
    ) -> PyResult<PyObject> {
        fn eq(
            py: Python<'_>,
            self_: &PyCell<CircuitInstruction>,
            other: &PyAny,
        ) -> PyResult<Option<bool>> {
            if self_.is(other) {
                return Ok(Some(true));
            }

            let self_ = self_.try_borrow()?;
            if other.is_instance_of::<CircuitInstruction>() {
                let other: PyResult<&PyCell<CircuitInstruction>> = other.extract();
                return other.map_or(Ok(Some(false)), |v| {
                    let v = v.try_borrow()?;
                    Ok(Some(
                        self_.clbits.as_ref(py).eq(v.clbits.as_ref(py))?
                            && self_.qubits.as_ref(py).eq(v.qubits.as_ref(py))?
                            && self_.operation.as_ref(py).eq(v.operation.as_ref(py))?,
                    ))
                });
            }

            if other.is_instance_of::<PyTuple>() {
                return Ok(Some(self_._legacy_format(py).as_ref(py).eq(other)?));
            }

            Ok(None)
        }

        match op {
            CompareOp::Eq => eq(py, self_, other).map(|r| {
                r.map(|b| b.into_py(py))
                    .unwrap_or_else(|| py.NotImplemented())
            }),
            CompareOp::Ne => eq(py, self_, other).map(|r| {
                r.map(|b| (!b).into_py(py))
                    .unwrap_or_else(|| py.NotImplemented())
            }),
            _ => Ok(py.NotImplemented()),
        }
    }
}

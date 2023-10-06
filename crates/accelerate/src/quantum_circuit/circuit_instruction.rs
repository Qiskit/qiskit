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

use pyo3::basic::CompareOp;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::{PyObject, PyResult};

#[pyclass(sequence, get_all, module = "qiskit._accelerate.quantum_circuit")]
#[derive(Clone, Debug)]
pub struct CircuitInstruction {
    pub operation: PyObject,
    pub qubits: Py<PyTuple>,
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
                None => Ok(PyTuple::empty(py).into_py(py)),
                Some(seq) => {
                    if seq.is_instance_of::<PyTuple>() {
                        Ok(seq.extract::<Py<PyTuple>>()?)
                    } else {
                        Ok(
                            PyTuple::new(py, seq.iter()?.collect::<PyResult<Vec<&PyAny>>>()?)
                                .into_py(py),
                        )
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

    pub fn copy(&self) -> Self {
        self.clone()
    }

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

    pub fn _legacy_format(&self, py: Python<'_>) -> PyObject {
        PyTuple::new(
            py,
            [
                self.operation.as_ref(py),
                self.qubits.as_ref(py),
                self.clbits.as_ref(py),
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

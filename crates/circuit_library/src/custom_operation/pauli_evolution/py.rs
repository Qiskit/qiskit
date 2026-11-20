// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use std::error;

use numpy::{Complex64, PyArray2};
use pyo3::{
    Bound, PyErr, PyResult, Python,
    exceptions::{PyRuntimeError, PyTypeError},
    pyclass, pymethods,
};
use qiskit_circuit::operations::Param;
use qiskit_quantum_info::sparse_observable::PySparseObservable;

use super::model::{PauliEvolution, PauliEvolutionError};

#[derive(Debug, PartialEq)]
#[pyclass(
    name = "PauliEvolution",
    module = "qiskit._accelerate.circuit.library",
    frozen,
    eq
)]
pub struct PyPauliEvolution {
    inner: PauliEvolution,
}

#[pymethods]
impl PyPauliEvolution {
    #[new]
    pub fn new(operator: &Bound<'_, PySparseObservable>, time: Param) -> PyResult<Self> {
        let operator = operator.borrow().as_inner()?.clone();
        let inner = PauliEvolution::new(operator, time)?;
        Ok(Self { inner })
    }

    #[getter]
    pub fn operator(&self) -> PySparseObservable {
        let operator = self.inner.operator().clone();
        PySparseObservable::from(operator)
    }

    #[getter]
    pub fn time(&self) -> Param {
        self.inner.time().clone()
    }

    pub fn inverse(&self) -> Self {
        let mut inner = self.inner.clone();
        inner.inverse();
        Self { inner }
    }

    pub fn to_matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
        let matrix = self.inner.to_matrix()?;
        Ok(PyArray2::from_owned_array(py, matrix))
    }
}

impl From<PauliEvolutionError> for PyErr {
    fn from(error: PauliEvolutionError) -> Self {
        let message = compose_error(&error);

        match error {
            PauliEvolutionError::TimeInvalid => PyTypeError::new_err(message),
            PauliEvolutionError::TimeNotEvaluated | PauliEvolutionError::Matrix(_) => {
                PyRuntimeError::new_err(message)
            }
        }
    }
}

fn compose_error(error: &dyn error::Error) -> String {
    use std::fmt::Write;

    let mut message = error.to_string();

    let next = error;
    while let Some(error) = next.source() {
        write!(message, ": {error}").expect("should have memory");
    }

    message
}

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

use ndarray::Array2;
use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray2};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyRuntimeError;
use pyo3::intern;
use pyo3::prelude::*;
use std::error;
use std::f64::consts::PI;
use std::hash::{DefaultHasher, Hash, Hasher};

use crate::imports;
use crate::operations::{CustomOperation, Operation, Param};

/// The Quantum Fourier Transform Gate.
///
/// On `n` qubits this is the operation
///
/// ```text
/// |j> -> 1/sqrt(2^n) * sum_k exp(2 pi i j k / 2^n) |k>
/// ```
#[pyclass(
    name = "QFTGate",
    module = "qiskit._accelerate.circuit",
    frozen,
    from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QFTGate {
    num_qubits: u32,
}

#[pymethods]
impl QFTGate {
    #[new]
    fn py_new(num_qubits: u32) -> Self {
        Self { num_qubits }
    }

    #[getter]
    #[pyo3(name = "num_qubits")]
    fn py_num_qubits(&self) -> u32 {
        self.num_qubits
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.num_qubits.hash(&mut hasher);
        hasher.finish()
    }

    fn __repr__(&self) -> String {
        format!("QFTGate({})", self.num_qubits)
    }

    /// Support `copy.copy`. The gate is immutable (`frozen`), so returning an
    /// equal value rather than sharing the instance is safe either way.
    fn __copy__(&self) -> Self {
        *self
    }

    /// Support `copy.deepcopy`. There is no interior mutability or nested
    /// Python state, so a bitwise copy is a complete deep copy.
    #[pyo3(signature = (_memo=None))]
    fn __deepcopy__(&self, _memo: Option<&Bound<PyAny>>) -> Self {
        *self
    }

    /// Support `pickle`. Circuits are pickled when shipped to worker processes
    /// (e.g. parallel transpilation), so this is required, not optional.
    fn __reduce__(&self, py: Python) -> PyResult<Py<PyAny>> {
        (py.get_type::<Self>(), (self.num_qubits,)).into_py_any(py)
    }

    /// The dense unitary matrix of this QFT, as a NumPy array.
    ///
    /// This is the single matrix implementation shared with the Python `QFTGate`,
    /// whose `__array__` delegates here.
    #[pyo3(name = "matrix")]
    fn py_matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
        let matrix = CustomOperation::matrix(self, &[])
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .ok_or_else(|| PyRuntimeError::new_err("QFTGate has no matrix representation"))?;
        Ok(matrix.into_pyarray(py))
    }
}

impl QFTGate {
    pub fn new(num_qubits: u32) -> Self {
        Self { num_qubits }
    }

    /// The number of qubits the QFT acts on.
    pub fn num_qubits(&self) -> u32 {
        self.num_qubits
    }
}

impl Operation for QFTGate {
    fn name(&self) -> &str {
        "qft"
    }

    fn num_qubits(&self) -> u32 {
        self.num_qubits
    }

    fn num_clbits(&self) -> u32 {
        0
    }

    fn num_params(&self) -> u32 {
        0
    }

    fn directive(&self) -> bool {
        false
    }
}

pub fn create_py_op_for_qft(py: Python, qft: &QFTGate) -> PyResult<Py<PyAny>> {
    // Hand the Rust gate itself to Python.
    Ok(imports::QFT_GATE
        .get_bound(py)
        .call_method1(intern!(py, "_from_inner"), (*qft,))?
        .unbind())
}

impl CustomOperation for QFTGate {
    fn is_unitary(&self) -> bool {
        true
    }

    fn matrix(
        &self,
        _params: &[Param],
    ) -> Result<Option<Array2<Complex64>>, Box<dyn error::Error>> {
        // ToDo: should we return `None` if the number of qubits is too large?
        // This would also prevent overflow errors when computing 1 << num_qubits.
        let size = 1usize << self.num_qubits;
        let norm = (size as f64).sqrt().recip();
        Ok(Some(Array2::from_shape_fn((size, size), |(i, j)| {
            let phase = 2.0 * PI * (i * j) as f64 / (size as f64);
            Complex64::from_polar(norm, phase)
        })))
    }

    // ToDo:
    // Due to dependency between rust packages, we cannot take the definition from the synthesis
    // crate. Should we implement the textbook synthesis method here or leave it as None?
}

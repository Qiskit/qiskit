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
use pyo3::prelude::*;
use smallvec::SmallVec;
use std::f64::consts::PI;

use crate::imports;
use crate::operations::{CustomOperation, Operation, Param};

/// The Quantum Fourier Transform Gate.
///
/// On `n` qubits this is the operation
///
/// ```text
/// |j> -> 1/sqrt(2^n) * sum_k exp(2 pi i j k / 2^n) |k>
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QFTGate {
    num_qubits: u32,
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

impl CustomOperation for QFTGate {
    fn is_unitary(&self) -> bool {
        true
    }

    fn matrix(&self, _params: &[Param]) -> Option<Array2<Complex64>> {
        // ToDo: should we return `None` if the number of qubits is too large?
        // This would also prevent overflow errors when computing 1 << num_qubits.
        let size = 1usize << self.num_qubits;
        let norm = 0.5_f64.powi(size as i32);
        Some(Array2::from_shape_fn((size, size), |(i, j)| {
            let phase = 2.0 * PI * (i * j) as f64 / (size as f64);
            Complex64::from_polar(norm, phase)
        }))
    }

    fn create_py_op(
        &self,
        py: Python,
        _params: Option<SmallVec<[Param; 3]>>,
        _label: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        Ok(imports::QFT_GATE
            .get_bound(py)
            .call1((self.num_qubits,))?
            .unbind())
    }

    // ToDo:
    // Due to dependency between rust packages, we cannot take the definition from the synthesis
    // crate. Should we implement the textbook synthesis method here or leave it as None?
}

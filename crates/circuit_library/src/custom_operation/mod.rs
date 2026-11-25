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

use pyo3::prelude::*;

use crate::custom_operation::rv_gate::PyRVGate;

pub mod pauli_evolution;
pub mod rv_gate;

#[pymodule(submodule, module = "qiskit._accelerate")]
pub mod library {
    #[pymodule_export]
    use super::PyRVGate;
}

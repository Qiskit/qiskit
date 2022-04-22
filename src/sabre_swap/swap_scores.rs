// This code is part of Qiskit.
//
// (C) Copyright IBM 2022
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use pyo3::prelude::*;

use hashbrown::HashMap;

/// An unsigned integer Vector based layout class
///
/// This class tracks the layout (or mapping between virtual qubits in the the
/// circuit and physical qubits on the physical device) efficiently
///
/// Args:
///     qubit_indices (dict): A dictionary mapping the virtual qubit index in the circuit to the
///         physical qubit index on the coupling graph.
///     logical_qubits (int): The number of logical qubits in the layout
///     physical_qubits (int): The number of physical qubits in the layout
#[pyclass(module = "qiskit._accelerate.sabre_swap")]
#[pyo3(text_signature = "(qubit_indices, logical_qubits, physical_qubits, /)")]
#[derive(Clone, Debug)]
pub struct SwapScores {
    pub scores: HashMap<[usize; 2], f64>,
}

#[pymethods]
impl SwapScores {
    #[new]
    pub fn new(swap_candidates: Vec<[usize; 2]>) -> Self {
        SwapScores {
            scores: swap_candidates
                .into_iter()
                .map(|candiate| (candiate, std::f64::INFINITY))
                .collect(),
        }
    }

    // Mapping Protocol
    pub fn __len__(&self) -> usize {
        self.scores.len()
    }

    pub fn __contains__(&self, object: [usize; 2]) -> bool {
        self.scores.contains_key(&object)
    }

    pub fn __getitem__(&self, object: [usize; 2]) -> f64 {
        self.scores[&object]
    }

    pub fn __str__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.scores))
    }
}

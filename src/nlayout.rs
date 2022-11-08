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
#[pyclass(module = "qiskit._accelerate.stochastic_swap")]
#[pyo3(text_signature = "(qubit_indices, logical_qubits, physical_qubits, /)")]
#[derive(Clone, Debug)]
pub struct NLayout {
    pub logic_to_phys: Vec<usize>,
    pub phys_to_logic: Vec<usize>,
}

impl NLayout {
    pub fn swap(&mut self, idx1: usize, idx2: usize) {
        self.phys_to_logic.swap(idx1, idx2);
        self.logic_to_phys[self.phys_to_logic[idx1]] = idx1;
        self.logic_to_phys[self.phys_to_logic[idx2]] = idx2;
    }
}

#[pymethods]
impl NLayout {
    #[new]
    fn new(
        qubit_indices: HashMap<usize, usize>,
        logical_qubits: usize,
        physical_qubits: usize,
    ) -> Self {
        let mut res = NLayout {
            logic_to_phys: vec![std::usize::MAX; logical_qubits],
            phys_to_logic: vec![std::usize::MAX; physical_qubits],
        };
        for (key, value) in qubit_indices {
            res.logic_to_phys[key] = value;
            res.phys_to_logic[value] = key;
        }
        res
    }

    fn __getstate__(&self) -> [Vec<usize>; 2] {
        [self.logic_to_phys.clone(), self.phys_to_logic.clone()]
    }

    fn __setstate__(&mut self, state: [Vec<usize>; 2]) {
        self.logic_to_phys = state[0].clone();
        self.phys_to_logic = state[1].clone();
    }

    /// Return the layout mapping
    ///
    /// .. note::
    ///
    ///     this copies the data from Rust to Python and has linear
    ///     overhead based on the number of qubits.
    ///
    /// Returns:
    ///     list: A list of 2 element lists in the form:
    ///     ``[[logical_qubit, physical_qubit], ...]``. Where the logical qubit
    ///     is the index in the qubit index in the circuit.
    ///
    #[pyo3(text_signature = "(self, /)")]
    fn layout_mapping(&self) -> Vec<[usize; 2]> {
        (0..self.logic_to_phys.len())
            .map(|i| [i, self.logic_to_phys[i]])
            .collect()
    }

    /// Get physical bit from logical bit
    #[pyo3(text_signature = "(self, logical_bit, /)")]
    fn logical_to_physical(&self, logical_bit: usize) -> usize {
        self.logic_to_phys[logical_bit]
    }

    /// Get logical bit from physical bit
    #[pyo3(text_signature = "(self, physical_bit, /)")]
    pub fn physical_to_logical(&self, physical_bit: usize) -> usize {
        self.phys_to_logic[physical_bit]
    }

    /// Swap the specified virtual qubits
    #[pyo3(text_signature = "(self, bit_a, bit_b, /)")]
    pub fn swap_logical(&mut self, bit_a: usize, bit_b: usize) {
        self.logic_to_phys.swap(bit_a, bit_b);
        self.phys_to_logic[self.logic_to_phys[bit_a]] = bit_a;
        self.phys_to_logic[self.logic_to_phys[bit_b]] = bit_b;
    }

    /// Swap the specified physical qubits
    #[pyo3(text_signature = "(self, bit_a, bit_b, /)")]
    pub fn swap_physical(&mut self, bit_a: usize, bit_b: usize) {
        self.swap(bit_a, bit_b)
    }

    pub fn copy(&self) -> NLayout {
        self.clone()
    }
}

#[pymodule]
pub fn nlayout(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<NLayout>()?;
    Ok(())
}

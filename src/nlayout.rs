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

#[pyclass(module = "stoachstic_swap")]
#[pyo3(text_signature = "(/")]
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
            logic_to_phys: vec![0; logical_qubits],
            phys_to_logic: vec![0; physical_qubits],
        };
        for (key, value) in qubit_indices {
            res.logic_to_phys[key] = value;
            res.phys_to_logic[value] = key;
        }
        res
    }

    fn layout_mapping(&self) -> Vec<[usize; 2]> {
        (0..self.logic_to_phys.len())
            .map(|i| [self.logic_to_phys[i], self.phys_to_logic[i]])
            .collect()
    }
}

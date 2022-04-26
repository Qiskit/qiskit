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

use indexmap::IndexMap;

/// A mapping of swap candidates to their scores
///
/// This class tracks the mapping of swap candidates to their float score. The
/// scores can only be updated from rust but once initialized this can be
/// used to access the scores from a particular swap candidate.
#[pyclass(mapping, module = "qiskit._accelerate.sabre_swap")]
#[pyo3(text_signature = "(swap_candidates, /)")]
#[derive(Clone, Debug)]
pub struct SwapScores {
    pub scores: IndexMap<[usize; 2], f64>,
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

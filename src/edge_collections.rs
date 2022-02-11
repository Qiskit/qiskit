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

use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::Python;

/// A simple container that contains a vector representing edges in the
/// coupling map that are found to be optimal by the swap mapper.
#[pyclass(module = "stoachstic_swap")]
#[pyo3(text_signature = "(/")]
#[derive(Clone, Debug)]
pub struct EdgeCollection {
    pub edges: Vec<usize>,
}

impl Default for EdgeCollection {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl EdgeCollection {
    #[new]
    pub fn new() -> Self {
        EdgeCollection { edges: Vec::new() }
    }

    /// Add two edges, in order, to the collection.
    ///
    /// Args:
    ///     edge_start (int): The beginning edge.
    ///     edge_end (int): The end of the edge.
    pub fn add(&mut self, edge_start: usize, edge_end: usize) {
        self.edges.push(edge_start);
        self.edges.push(edge_end);
    }

    pub fn edges(&self, py: Python) -> PyObject {
        self.edges.clone().into_pyarray(py).into()
    }
}

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
#[pyclass(module = "qiskit._accelerate.stochastic_swap")]
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
    #[pyo3(text_signature = "(/)")]
    pub fn new() -> Self {
        EdgeCollection { edges: Vec::new() }
    }

    /// Add two edges, in order, to the collection.
    ///
    /// Args:
    ///     edge_start (int): The beginning edge.
    ///     edge_end (int): The end of the edge.
    #[pyo3(text_signature = "(self, edge_start, edge_end, /)")]
    pub fn add(&mut self, edge_start: usize, edge_end: usize) {
        self.edges.push(edge_start);
        self.edges.push(edge_end);
    }

    /// Return the numpy array of edges
    ///
    /// The out array is the flattened edge list from the coupling graph.
    /// For example, if the edge list were ``[(0, 1), (1, 2), (2, 3)]`` the
    /// output array here would be ``[0, 1, 1, 2, 2, 3]``.
    #[pyo3(text_signature = "(self, /)")]
    pub fn edges(&self, py: Python) -> PyObject {
        self.edges.clone().into_pyarray(py).into()
    }

    fn __getstate__(&self) -> Vec<usize> {
        self.edges.clone()
    }

    fn __setstate__(&mut self, state: Vec<usize>) {
        self.edges = state
    }
}

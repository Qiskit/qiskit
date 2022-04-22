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

use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;

/// A simple container that contains a vector representing edges in the
/// coupling map that are found to be optimal by the swap mapper.
#[pyclass(module = "qiskit._accelerate.stochastic_swap")]
#[pyo3(text_signature = "(/)")]
#[derive(Clone, Debug)]
pub struct EdgeList {
    pub edges: Vec<[usize; 2]>,
}

impl Default for EdgeList {
    fn default() -> Self {
        Self::new(None)
    }
}

#[pymethods]
impl EdgeList {
    #[new]
    pub fn new(capacity: Option<usize>) -> Self {
        match capacity {
            Some(size) => EdgeList {
                edges: Vec::with_capacity(size),
            },
            None => EdgeList { edges: Vec::new() },
        }
    }

    /// Append an edge to the list.
    ///
    /// Args:
    ///     edge_start (int): The start qubit of the edge.
    ///     edge_end (int): The end qubit of the edge.
    #[pyo3(text_signature = "(self, edge_start, edge_end, /)")]
    pub fn append(&mut self, edge_start: usize, edge_end: usize) {
        self.edges.push([edge_start, edge_end]);
    }

    pub fn __iter__(slf: PyRef<Self>) -> PyResult<Py<EdgeListIter>> {
        let iter = EdgeListIter {
            inner: slf.edges.clone().into_iter(),
        };
        Py::new(slf.py(), iter)
    }

    pub fn __len__(&self) -> usize {
        self.edges.len()
    }

    pub fn __contains__(&self, object: [usize; 2]) -> bool {
        self.edges.contains(&object)
    }

    pub fn __getitem__(&self, object: usize) -> PyResult<[usize; 2]> {
        if object > self.edges.len() {
            return Err(PyIndexError::new_err(format!(
                "Index {} out of range for this EdgeList",
                object
            )));
        }
        Ok(self.edges[object])
    }

    fn __getstate__(&self) -> Vec<[usize; 2]> {
        self.edges.clone()
    }

    fn __setstate__(&mut self, state: Vec<[usize; 2]>) {
        self.edges = state
    }
}

#[pyclass]
pub struct EdgeListIter {
    inner: std::vec::IntoIter<[usize; 2]>,
}

#[pymethods]
impl EdgeListIter {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<Self>) -> Option<[usize; 2]> {
        slf.inner.next()
    }
}

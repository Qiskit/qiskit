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

use crate::getenv_use_multiple_threads;
use ndarray::prelude::*;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use rayon::prelude::*;

/// A simple container that contains a vector of vectors representing
/// neighbors of each node in the coupling map
///
/// This object is typically created once from the adjacency matrix of
/// a coupling map, for example::
///
///     neigh_table = NeighborTable(rustworkx.adjacency_matrix(coupling_map.graph))
///
/// and used solely to represent neighbors of each node in qiskit-terra's rust
/// module.
#[pyclass(module = "qiskit._accelerate.sabre_swap")]
#[pyo3(text_signature = "(/)")]
#[derive(Clone, Debug)]
pub struct NeighborTable {
    pub neighbors: Vec<Vec<usize>>,
}

#[pymethods]
impl NeighborTable {
    #[new]
    pub fn new(adjacency_matrix: Option<PyReadonlyArray2<f64>>) -> Self {
        let run_in_parallel = getenv_use_multiple_threads();
        let neighbors = match adjacency_matrix {
            Some(adjacency_matrix) => {
                let adj_mat = adjacency_matrix.as_array();
                let build_neighbors = |row: ArrayView1<f64>| -> Vec<usize> {
                    row.iter()
                        .enumerate()
                        .filter_map(
                            |(row_index, value)| {
                                if *value == 0. {
                                    None
                                } else {
                                    Some(row_index)
                                }
                            },
                        )
                        .collect()
                };
                if run_in_parallel {
                    adj_mat
                        .axis_iter(Axis(0))
                        .into_par_iter()
                        .map(|row| build_neighbors(row))
                        .collect()
                } else {
                    adj_mat
                        .axis_iter(Axis(0))
                        .map(|row| build_neighbors(row))
                        .collect()
                }
            }
            None => Vec::new(),
        };
        NeighborTable { neighbors }
    }

    fn __getstate__(&self) -> Vec<Vec<usize>> {
        self.neighbors.clone()
    }

    fn __setstate__(&mut self, state: Vec<Vec<usize>>) {
        self.neighbors = state
    }
}

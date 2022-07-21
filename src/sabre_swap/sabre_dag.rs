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
use retworkx_core::petgraph::prelude::*;

/// A DAG object used to represent the 2q interactions from the circuit to
/// the sabre algorithm. The input is a list of the 2 qubits for each 2q gate
/// in the circuit in topological order.
#[pyclass(module = "qiskit._accelerate.sabre_swap")]
#[pyo3(text_signature = "(/)")]
#[derive(Clone, Debug)]
pub struct SabreDAG {
    // DAG as a directed graph with no node weight, qubit indices as edge weight and u32 indices
    pub dag: DiGraph<[usize; 3], usize>,
    pub first_layer: Vec<NodeIndex>,
}

#[pymethods]
impl SabreDAG {
    #[new]
    pub fn new(num_qubits: usize, nodes: Vec<[usize; 3]>) -> Self {
        let mut first_layer: Vec<NodeIndex> = Vec::new();
        let mut qubit_pos: Vec<usize> = vec![usize::MAX; num_qubits];
        let mut dag: DiGraph<[usize; 3], usize> =
            Graph::with_capacity(nodes.len(), 2 * nodes.len());
        for node in nodes {
            let u: usize = node[1];
            let v: usize = node[2];
            let gate_index = dag.add_node(node);
            if qubit_pos[u] == usize::MAX && qubit_pos[v] == usize::MAX {
                first_layer.push(gate_index);
            } else {
                if qubit_pos[u] != usize::MAX {
                    dag.add_edge(NodeIndex::new(qubit_pos[u]), gate_index, u);
                }
                if qubit_pos[v] != usize::MAX {
                    dag.add_edge(NodeIndex::new(qubit_pos[v]), gate_index, v);
                }
            }
            qubit_pos[u] = gate_index.index();
            qubit_pos[v] = gate_index.index();
        }
        SabreDAG { dag, first_layer }
    }
}

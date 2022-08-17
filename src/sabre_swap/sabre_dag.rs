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

use hashbrown::HashMap;
use numpy::PyReadonlyArray1;
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
    pub dag: DiGraph<(usize, Vec<usize>, Vec<usize>), usize>,
    pub first_layer: Vec<NodeIndex>,
}

#[pymethods]
impl SabreDAG {
    #[new]
    pub fn new(
        num_qubits: usize,
        num_clbits: usize,
        nodes: Vec<(usize, Vec<usize>, Vec<usize>)>,
        front_layer: PyReadonlyArray1<usize>,
    ) -> PyResult<Self> {
        let mut qubit_pos: Vec<usize> = vec![usize::MAX; num_qubits];
        let mut clbit_pos: Vec<usize> = vec![usize::MAX; num_clbits];
        let mut reverse_index_map: HashMap<usize, NodeIndex> = HashMap::with_capacity(nodes.len());
        let mut dag: DiGraph<(usize, Vec<usize>, Vec<usize>), usize> =
            Graph::with_capacity(nodes.len(), 2 * nodes.len());
        for node in &nodes {
            let qargs = &node.1;
            let cargs = &node.2;
            let gate_index = dag.add_node(node.clone());
            reverse_index_map.insert(node.0, gate_index);
            for x in qargs {
                if qubit_pos[*x] != usize::MAX {
                    dag.add_edge(NodeIndex::new(qubit_pos[*x]), gate_index, *x);
                }
                qubit_pos[*x] = gate_index.index();
            }
            for x in cargs {
                if clbit_pos[*x] != usize::MAX {
                    dag.add_edge(NodeIndex::new(qubit_pos[*x]), gate_index, *x);
                }
                clbit_pos[*x] = gate_index.index();
            }
        }
        let first_layer = front_layer
            .as_slice()?
            .iter()
            .map(|x| reverse_index_map[x])
            .collect();
        Ok(SabreDAG { dag, first_layer })
    }
}

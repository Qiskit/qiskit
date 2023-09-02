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
use hashbrown::HashSet;
use pyo3::prelude::*;
use rustworkx_core::petgraph::prelude::*;

/// Named access to the node elements in the [SabreDAG].
#[derive(Clone, Debug)]
pub struct DAGNode {
    pub py_node_id: usize,
    pub qubits: Vec<usize>,
}

/// A DAG representation of the logical circuit to be routed.  This represents the same dataflow
/// dependences as the Python-space [DAGCircuit], but without any information about _what_ the
/// operations being performed are. Note that all the qubit references here are to "virtual"
/// qubits, that is, the qubits are those specified by the user.  This DAG does not need to be
/// full-width on the hardware.
///
/// Control-flow operations are represented by the presence of the Python [DAGCircuit]'s node id
/// (the [DAGNode.py_node_id] field) as a key in [node_blocks], where the value is an array of the
/// inner dataflow graphs.
#[pyclass(module = "qiskit._accelerate.sabre_swap")]
#[derive(Clone, Debug)]
pub struct SabreDAG {
    pub num_qubits: usize,
    pub num_clbits: usize,
    pub dag: DiGraph<DAGNode, ()>,
    pub first_layer: Vec<NodeIndex>,
    pub nodes: Vec<(usize, Vec<usize>, HashSet<usize>)>,
    pub node_blocks: HashMap<usize, Vec<SabreDAG>>,
}

#[pymethods]
impl SabreDAG {
    #[new]
    #[pyo3(text_signature = "(num_qubits, num_clbits, nodes, node_blocks, /)")]
    pub fn new(
        num_qubits: usize,
        num_clbits: usize,
        nodes: Vec<(usize, Vec<usize>, HashSet<usize>)>,
        node_blocks: HashMap<usize, Vec<SabreDAG>>,
    ) -> Self {
        let mut qubit_pos: Vec<Option<NodeIndex>> = vec![None; num_qubits];
        let mut clbit_pos: Vec<Option<NodeIndex>> = vec![None; num_clbits];
        let mut dag = DiGraph::with_capacity(nodes.len(), 2 * nodes.len());
        let mut first_layer = Vec::<NodeIndex>::new();
        for node in &nodes {
            let qargs = &node.1;
            let cargs = &node.2;
            let gate_index = dag.add_node(DAGNode {
                py_node_id: node.0,
                qubits: qargs.clone(),
            });
            let mut is_front = true;
            for x in qargs {
                if let Some(predecessor) = qubit_pos[*x] {
                    is_front = false;
                    dag.add_edge(predecessor, gate_index, ());
                }
                qubit_pos[*x] = Some(gate_index);
            }
            for x in cargs {
                if let Some(predecessor) = clbit_pos[*x] {
                    is_front = false;
                    dag.add_edge(predecessor, gate_index, ());
                }
                clbit_pos[*x] = Some(gate_index);
            }
            if is_front {
                first_layer.push(gate_index);
            }
        }
        SabreDAG {
            num_qubits,
            num_clbits,
            dag,
            first_layer,
            nodes,
            node_blocks,
        }
    }
}

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
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use rustworkx_core::petgraph::prelude::*;

use crate::nlayout::VirtualQubit;

/// Named access to the node elements in the [SabreDAG].
#[derive(Clone, Debug)]
pub struct DAGNode {
    pub py_node_id: usize,
    pub qubits: Vec<VirtualQubit>,
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
    pub nodes: Vec<(usize, Vec<VirtualQubit>, HashSet<usize>)>,
    pub node_blocks: HashMap<usize, Vec<SabreDAG>>,
}

#[pymethods]
impl SabreDAG {
    #[new]
    #[pyo3(text_signature = "(num_qubits, num_clbits, nodes, node_blocks, /)")]
    pub fn new(
        num_qubits: usize,
        num_clbits: usize,
        nodes: Vec<(usize, Vec<VirtualQubit>, HashSet<usize>)>,
        node_blocks: HashMap<usize, Vec<SabreDAG>>,
    ) -> PyResult<Self> {
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
                let pos = qubit_pos.get_mut(x.index()).ok_or_else(|| {
                    PyIndexError::new_err(format!(
                        "qubit index {} is out of range for {} qubits",
                        x.index(),
                        num_qubits
                    ))
                })?;
                if let Some(predecessor) = *pos {
                    is_front = false;
                    dag.add_edge(predecessor, gate_index, ());
                }
                *pos = Some(gate_index);
            }
            for x in cargs {
                let pos = clbit_pos.get_mut(*x).ok_or_else(|| {
                    PyIndexError::new_err(format!(
                        "clbit index {} is out of range for {} clbits",
                        *x, num_qubits
                    ))
                })?;
                if let Some(predecessor) = *pos {
                    is_front = false;
                    dag.add_edge(predecessor, gate_index, ());
                }
                *pos = Some(gate_index);
            }
            if is_front {
                first_layer.push(gate_index);
            }
        }
        Ok(SabreDAG {
            num_qubits,
            num_clbits,
            dag,
            first_layer,
            nodes,
            node_blocks,
        })
    }
}

#[cfg(test)]
mod test {
    use super::SabreDAG;
    use crate::nlayout::VirtualQubit;
    use hashbrown::{HashMap, HashSet};

    #[test]
    fn no_panic_on_bad_qubits() {
        let bad_qubits = vec![VirtualQubit::new(0), VirtualQubit::new(2)];
        assert!(SabreDAG::new(2, 0, vec![(0, bad_qubits, HashSet::new())], HashMap::new()).is_err())
    }

    #[test]
    fn no_panic_on_bad_clbits() {
        let good_qubits = vec![VirtualQubit::new(0), VirtualQubit::new(1)];
        assert!(SabreDAG::new(2, 1, vec![(0, good_qubits, [0, 1].into())], HashMap::new()).is_err())
    }
}

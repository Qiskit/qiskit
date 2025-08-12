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
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use rustworkx_core::petgraph::prelude::*;
use thiserror::Error;

use qiskit_circuit::converters::circuit_to_dag;
use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType, Wire};
use qiskit_circuit::operations::{Operation, OperationRef, PyInstruction};
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::{Qubit, VirtualQubit};

// TODO: replace with simple map over `op.blocks()` once vars and stretches are in `CircuitData`
// (and then even simpler logic once control flow is in Rust).
fn control_flow_block_dags<'a>(
    py: Python<'a>,
    inst: &'a PyInstruction,
) -> PyResult<impl Iterator<Item = PyResult<DAGCircuit>> + 'a> {
    // Can't do `op.blocks()` because `CircuitData` doesn't track the vars.
    Ok(inst
        .instruction
        .bind(py)
        .getattr("blocks")?
        .downcast::<PyTuple>()?
        .iter()
        .map(move |block| circuit_to_dag(block.extract()?, false, None, None)))
}

/// The type of a node in the Sabre interactions graph.
#[derive(Clone, Debug)]
pub enum InteractionKind {
    /// The node is some sort of synchronisation point, but not something that imposes routing
    /// constraints itself.  It can be pushed into the output as soon as all its predecessors are
    /// satisfied.  This might be a directive, or a single-qubit gate conditioned on clbits that
    /// depend on more than one 2q gate, or even just a single-qubit gate. In the simple 1q case,
    /// the synchronisation is trivial, and the gate will be automatically folded into a parent.
    Synchronize,
    /// A two-qubit interaction.  This is the principal component that we're concerned with.
    TwoQ([VirtualQubit; 2]),
    /// A control-flow operation on more than two qubits.  If it's only on a single qubit, it'll be
    /// handled by `Synchronize`.  If it's on 2q, it'll be handled by `TwoQ`.  We need to store both
    /// the Sabre version and the DAG representation of the block, so we can reconstruct things
    /// later.  When control-flow ops are represented with native DAGs, we won't need to store the
    /// temporary.
    ControlFlow(Box<[(SabreDAG, DAGCircuit)]>),
}
impl InteractionKind {
    fn from_op(op: &PackedOperation, qargs: &[Qubit]) -> Result<Self, SabreDAGError> {
        if op.directive() {
            return Ok(Self::Synchronize);
        }
        if op.control_flow() {
            let OperationRef::Instruction(inst) = op.view() else {
                panic!("control-flow ops should always be PyInstruction");
            };
            let blocks = Python::with_gil(|py| {
                control_flow_block_dags(py, inst)?
                    .map(|dag| {
                        dag.and_then(|dag| Ok((SabreDAG::from_dag(&dag)?, dag)))
                            .map_err(SabreDAGError::from)
                    })
                    .collect::<Result<Box<[_]>, _>>()
            })?;
            return Ok(Self::ControlFlow(blocks));
        }
        match qargs {
            // We're assuming that if the instruction has classical wires (like a `PyInstruction` or
            // something), then it's still going to need routing, even though we can't see inside
            // the operation to actually _know_ what it is.
            &[left, right] => Ok(Self::TwoQ([
                VirtualQubit::new(left.0),
                VirtualQubit::new(right.0),
            ])),
            // TODO: multi-q gates _should_ be an error, but for the initial patch, we're
            // maintaining the historical behaviour of Sabre which is to ignore multi-q gates.
            _ => Ok(Self::Synchronize),
        }
    }
}

/// Named access to the node elements in the [SabreDAG].
#[derive(Clone, Debug)]
pub struct SabreNode {
    pub index: NodeIndex,
    pub kind: InteractionKind,
}
impl SabreNode {
    fn new(index: NodeIndex, kind: InteractionKind) -> Self {
        Self { index, kind }
    }
}

#[derive(Error, Debug)]
pub enum SabreDAGError {
    #[error("Python error: {0}")]
    Python(#[from] PyErr),
}
impl From<SabreDAGError> for PyErr {
    fn from(err: SabreDAGError) -> PyErr {
        match err {
            SabreDAGError::Python(err) => err,
        }
    }
}

/// A DAG representation of the logical circuit to be routed.
///
/// Note that all the qubit references here are to "virtual" qubits, that is, the qubits are those
/// specified by the user.  This DAG does not need to be full-width on the hardware.
#[derive(Clone, Debug)]
pub struct SabreDAG {
    /// An ordered sequence of node indices in the original [DAGCircuit] that are automatically
    /// routable immediately at the start of the circuit, regardless of layout (for example 1q
    /// gates and directives that are preceded only by 1q gates).
    pub initial: Vec<NodeIndex>,
    pub dag: DiGraph<SabreNode, ()>,
    /// The incoming externals of the `dag`.  These will form the first "front layer" of the Sabre
    /// search (unless they're already routable with the chosen layout or control-flow ops).  These
    /// are indices into the [dag] field.
    pub first_layer: Vec<NodeIndex>,
}

impl SabreDAG {
    pub fn from_dag(dag: &DAGCircuit) -> Result<Self, SabreDAGError> {
        // The `NodeIndex` here is into `dag`.
        let mut initial = Vec::<NodeIndex>::new();
        let mut sabre = DiGraph::new();
        // The `NodeIndex`es in here are into `sabre`.
        let mut wire_pos: HashMap<Wire, NodeIndex> = HashMap::with_capacity(dag.width());
        let mut first_layer = Vec::<NodeIndex>::new();

        for dag_node in dag
            .topological_op_nodes()
            .expect("infallible if DAG is in a valid state")
        {
            let NodeType::Operation(inst) = &dag[dag_node] else {
                panic!("op nodes should always be of type `Operation`");
            };
            let kind = InteractionKind::from_op(&inst.op, dag.get_qargs(inst.qubits))?;
            let sabre_node = sabre.add_node(SabreNode::new(dag_node, kind));
            for edge in dag.dag().edges_directed(dag_node, Direction::Incoming) {
                if let Some(parent) = wire_pos.insert(*edge.weight(), sabre_node) {
                    sabre.add_edge(parent, sabre_node, ());
                }
            }
            if sabre
                .edges_directed(sabre_node, Direction::Incoming)
                .next()
                .is_none()
            {
                match &sabre[sabre_node].kind {
                    InteractionKind::Synchronize => {
                        for edge in dag.dag().edges_directed(dag_node, Direction::Incoming) {
                            wire_pos.remove(edge.weight());
                        }
                        initial.push(sabre.remove_node(sabre_node).unwrap().index);
                    }
                    _ => {
                        first_layer.push(sabre_node);
                    }
                };
            }
        }
        Ok(SabreDAG {
            initial,
            dag: sabre,
            first_layer,
        })
    }

    /// Reduce the [SabreDAG] down to only consider the interaction structure.
    ///
    /// The resulting [SabreDAG] contains no actual node indices from the original [DAGCircuit], so
    /// cannot be used to reconstruct an output (you'd just get the swaps, no gates).
    ///
    /// This converts all [InteractionKind::ControlFlow] nodes in the graph into simple
    /// [InteractionKind::Synchronize] entries, because the current Sabre algorithm ensures
    /// control-flow blocks return the layout to their starting position, which makes them
    /// indistinguishable from simple synchronisation points from the top level of routing.
    pub fn only_interactions(&self) -> Self {
        let mut dag = DiGraph::with_capacity(self.dag.node_count(), self.dag.edge_count());
        for node in self.dag.node_weights() {
            let kind = match &node.kind {
                InteractionKind::Synchronize | InteractionKind::TwoQ(_) => node.kind.clone(),
                InteractionKind::ControlFlow(_) => InteractionKind::Synchronize,
            };
            dag.add_node(SabreNode {
                index: node.index,
                kind,
            });
        }
        for edge in self.dag.edge_references() {
            dag.add_edge(edge.source(), edge.target(), ());
        }
        Self {
            initial: vec![],
            dag,
            first_layer: self.first_layer.clone(),
        }
    }

    pub fn reverse_dag(&self) -> Self {
        let mut out_dag = self.clone();
        out_dag.dag.reverse();
        out_dag.first_layer = out_dag.dag.externals(Direction::Incoming).collect();
        out_dag
    }
}

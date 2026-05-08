// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

/// Remove diagonal gates (including diagonal 2Q gates) before a measurement.
use pyo3::prelude::*;
use rayon::prelude::*;
use rustworkx_core::petgraph::prelude::*;
use rustworkx_core::petgraph::visit::NodeIndexable;

use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType};
use qiskit_circuit::operations::OperationRef;
use qiskit_circuit::operations::StandardGate;
use qiskit_circuit::operations::StandardInstruction;
use qiskit_util::getenv_use_multiple_threads;

/// Run the RemoveDiagonalGatesBeforeMeasure pass on `dag`.
/// Args:
///     dag (DAGCircuit): the DAG to be optimized.
/// Returns:
///     DAGCircuit: the optimized DAG.
#[pyfunction]
#[pyo3(name = "remove_diagonal_gates_before_measure")]
pub fn run_remove_diagonal_before_measure(dag: &mut DAGCircuit) {
    if !dag.get_op_counts().contains_key("measure") {
        return;
    }
    let run_in_parallel = getenv_use_multiple_threads();
    let is_measure = |index: NodeIndex| -> bool {
        let Some(NodeType::Operation(inst)) = dag.dag().node_weight(index) else {
            return false;
        };
        matches!(
            inst.op.view(),
            OperationRef::StandardInstruction(StandardInstruction::Measure)
        )
    };
    let process_node = |index: NodeIndex| -> Option<NodeIndex> {
        if !is_measure(index) {
            return None;
        }
        let predecessor = dag
            .quantum_predecessors(index)
            .next()
            .expect("index is an operation node, so it must have a predecessor.");
        let NodeType::Operation(ref pred_inst) = dag[predecessor] else {
            return None;
        };
        match pred_inst.op.try_standard_gate()? {
            StandardGate::RZ
            | StandardGate::Z
            | StandardGate::T
            | StandardGate::S
            | StandardGate::Tdg
            | StandardGate::Sdg
            | StandardGate::U1
            | StandardGate::Phase => Some(predecessor),
            StandardGate::CZ
            | StandardGate::CRZ
            | StandardGate::CU1
            | StandardGate::RZZ
            | StandardGate::CPhase
            | StandardGate::CS
            | StandardGate::CSdg
            | StandardGate::CCZ => dag
                .quantum_successors(predecessor)
                .all(is_measure)
                .then_some(predecessor),
            _ => None,
        }
    };

    let nodes_to_remove: Vec<NodeIndex> = if run_in_parallel && dag.num_ops() >= 50_000 {
        (0..dag.dag().node_bound())
            .into_par_iter()
            .filter_map(|index| process_node(NodeIndex::new(index)))
            .collect()
    } else {
        dag.op_nodes(false)
            .filter_map(|(index, _)| process_node(index))
            .collect()
    };

    for node in nodes_to_remove {
        if dag.dag().node_weight(node).is_some() {
            dag.remove_op_node(node);
        }
    }
}

pub fn remove_diagonal_gates_before_measure_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(run_remove_diagonal_before_measure))?;
    Ok(())
}

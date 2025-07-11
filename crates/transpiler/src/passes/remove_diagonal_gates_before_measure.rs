// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

/// Remove diagonal gates (including diagonal 2Q gates) before a measurement.
use pyo3::prelude::*;
use rayon::prelude::*;
use rustworkx_core::petgraph::prelude::*;

use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType};
use qiskit_circuit::getenv_use_multiple_threads;
use qiskit_circuit::operations::OperationRef;
use qiskit_circuit::operations::StandardGate;
use qiskit_circuit::operations::StandardInstruction;
use qiskit_circuit::packed_instruction::PackedInstruction;

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

    let process_node = |index: NodeIndex, inst: &PackedInstruction| {
        if matches!(
            inst.op.view(),
            OperationRef::StandardInstruction(StandardInstruction::Measure)
        ) {
            let predecessor = (dag.quantum_predecessors(index))
                .next()
                .expect("index is an operation node, so it must have a predecessor.");

            let NodeType::Operation(ref pred_inst) = dag[predecessor] else {
                return None;
            };
            if let Some(gate) = pred_inst.standard_gate() {
                match gate {
                    StandardGate::RZ
                    | StandardGate::Z
                    | StandardGate::T
                    | StandardGate::S
                    | StandardGate::Tdg
                    | StandardGate::Sdg
                    | StandardGate::U1
                    | StandardGate::Phase => return Some(predecessor),
                    StandardGate::CZ
                    | StandardGate::CRZ
                    | StandardGate::CU1
                    | StandardGate::RZZ
                    | StandardGate::CPhase
                    | StandardGate::CS
                    | StandardGate::CSdg
                    | StandardGate::CCZ => {
                        let mut successors = dag.quantum_successors(predecessor);
                        if successors.all(|s| {
                            let node_s = &dag.dag()[s];
                            if let NodeType::Operation(inst_s) = node_s {
                                matches!(
                                    inst_s.op.view(),
                                    OperationRef::StandardInstruction(StandardInstruction::Measure)
                                )
                            } else {
                                false
                            }
                        }) {
                            return Some(predecessor);
                        }
                    }
                    _ => return None,
                }
            }
        }
        None
    };

    let nodes_to_remove: Vec<NodeIndex> = if run_in_parallel && dag.num_ops() >= 50_000 {
        let node_indices = dag.dag().node_indices().collect::<Vec<_>>();
        node_indices
            .into_par_iter()
            .filter_map(|index| {
                if let NodeType::Operation(ref inst) = dag.dag()[index] {
                    process_node(index, inst)
                } else {
                    None
                }
            })
            .collect()
    } else {
        dag.op_nodes(true)
            .filter_map(|x| process_node(x.0, x.1))
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

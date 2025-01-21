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

use indexmap::IndexSet;

use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::operations::Operation;
use qiskit_circuit::operations::StandardGate;

/// Run the RemoveDiagonalGatesBeforeMeasure pass on `dag`.
/// Args:
///     dag (DAGCircuit): the DAG to be optimized.
/// Returns:
///     DAGCircuit: the optimized DAG.
#[pyfunction]
#[pyo3(name = "remove_diagonal_gates_before_measure")]
fn run_remove_diagonal_before_measure(dag: &mut DAGCircuit) -> PyResult<()> {
    static DIAGONAL_1Q_GATES: [StandardGate; 8] = [
        StandardGate::RZGate,
        StandardGate::ZGate,
        StandardGate::TGate,
        StandardGate::SGate,
        StandardGate::TdgGate,
        StandardGate::SdgGate,
        StandardGate::U1Gate,
        StandardGate::PhaseGate,
    ];
    static DIAGONAL_MULTIQ_GATES: [StandardGate; 8] = [
        StandardGate::CZGate,
        StandardGate::CRZGate,
        StandardGate::CU1Gate,
        StandardGate::RZZGate,
        StandardGate::CPhaseGate,
        StandardGate::CSGate,
        StandardGate::CSdgGate,
        StandardGate::CCZGate,
    ];

    let nodes_to_remove = dag
        .op_nodes(true)
        .filter_map(|(index, inst)| {
            if inst.op.name() != "measure" {
                return None;
            }
            let predecessor = (dag.quantum_predecessors(index.node()))
                .next()
                .expect("index is an operation node, so it must have a predecessor.");
            let (pred_index, pred_inst) = dag.get_operation(predecessor)?;
            let gate = pred_inst.standard_gate()?;
            (DIAGONAL_1Q_GATES.contains(&gate)
                || (DIAGONAL_MULTIQ_GATES.contains(&gate)
                    && dag.quantum_successors(pred_index.node()).all(|s| {
                        dag.get_operation(s)
                            .is_some_and(|(_, inst)| inst.op.name() == "measure")
                    })))
            .then_some(pred_index)
        })
        .collect::<IndexSet<_>>();

    for node_to_remove in nodes_to_remove {
        dag.remove_op_node(node_to_remove);
    }

    Ok(())
}

pub fn remove_diagonal_gates_before_measure(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(run_remove_diagonal_before_measure))?;
    Ok(())
}

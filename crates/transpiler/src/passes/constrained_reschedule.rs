// This code is part of Qiskit.
//
// (C) Copyright IBM 2025.
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::target::Target;
use crate::TranspilerError;
use hashbrown::HashSet;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::{pyfunction, wrap_pyfunction, Bound, PyResult};
use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType};
use qiskit_circuit::operations::Param;
use qiskit_circuit::operations::{Operation, OperationRef, StandardInstruction};
use qiskit_circuit::PhysicalQubit;
use rustworkx_core::petgraph::stable_graph::NodeIndex;

/// Returns the immediate successor operation nodes of a given node in the DAG.
///
/// This function traverses the DAG to find all nodes that are direct successors
/// of the given node and filters them to return only operation nodes.
///
/// # Arguments
///
/// * `dag` - Reference to the DAGCircuit containing the quantum circuit
/// * `node_index` - Index of the node whose successors we want to find
///
/// # Returns
///
/// An iterator of `NodeIndex` values representing the immediate successor operation nodes.
fn get_next_gate(dag: &DAGCircuit, node_index: NodeIndex) -> impl Iterator<Item = NodeIndex> + '_ {
    dag.quantum_successors(node_index)
        .filter(|&idx| matches!(dag[idx], NodeType::Operation(_)))
}

/// Update the start time of the current node to satisfy alignment constraints.
/// Immediate successors are pushed back to avoid overlap and will be processed later.
///
/// Note:
/// This logic assumes that all bits in the qregs and cregs synchronously start and end,
/// i.e. occupy the same time slot, but qregs and cregs can take different time slots
/// due to classical I/O latencies.
///
/// # Args:
/// * `py` - Python interpreter reference for PyO3 operations
/// * `dag` - Reference to the DAGCircuit to be rescheduled with constraints
/// * `node_index` - Index of the current node to be processed
/// * `node_start_time` - Mutable Python dictionary mapping node indices to start times
/// * `clbit_write_latency` - Additional latency for classical bit write operations
/// * `pulse_align` - Alignment constraint for gate operations (in dt units)
/// * `acquire_align` - Alignment constraint for measurement/reset operations (in dt units)
/// * `target` - Optional target backend for duration information
fn push_node_back(
    dag: &DAGCircuit,
    node_index: NodeIndex,
    node_start_time: &Bound<PyDict>,
    clbit_write_latency: u32,
    pulse_align: u32,
    acquire_align: u32,
    target: Option<&Target>,
) -> PyResult<()> {
    let NodeType::Operation(op) = &dag[node_index] else {
        unreachable!("topological_op_nodes() should only return operations.")
    };

    let op_view = op.op.view();
    let alignment = match op_view {
        OperationRef::Gate(_) | OperationRef::StandardGate(_) => Some(pulse_align),
        OperationRef::StandardInstruction(StandardInstruction::Reset)
        | OperationRef::StandardInstruction(StandardInstruction::Measure) => Some(acquire_align),
        _ => None,
    };

    let obj = node_start_time.get_item(node_index.index())?;
    let mut this_t0: u32 = obj
        .as_ref()
        .ok_or_else(|| PyValueError::new_err("Missing value in node_start_time"))?
        .extract()?;

    if let Some(alignment) = alignment {
        let misalignment = this_t0 % alignment;
        let shift = if misalignment != 0 {
            (alignment - misalignment).max(0)
        } else {
            0
        };
        this_t0 += shift;
        node_start_time
            .set_item(node_index.index(), this_t0)
            .unwrap();
    }

    let new_t1q = if let Some(target) = target {
        let qargs: Vec<PhysicalQubit> = dag
            .qargs_interner()
            .get(op.qubits)
            .iter()
            .map(|q| PhysicalQubit(q.index() as u32))
            .collect();
        let duration = target.get_duration(op.op.name(), &qargs).unwrap_or(0.0);
        this_t0 + duration as u32
    } else if matches!(
        op_view,
        OperationRef::StandardInstruction(StandardInstruction::Delay(_))
    ) {
        let params = op.params_view();
        let param = params
            .first()
            .ok_or_else(|| PyValueError::new_err("Delay instruction missing duration parameter"))?;
        let duration = match param {
            Param::Obj(val) => {
                // Try to extract as different numeric types
                val.bind(node_start_time.py()).extract::<u32>()
            }
            Param::Float(f) => Ok(*f as u32),
            _ => Err(TranspilerError::new_err(
                "The provided Delay duration is not in terms of dt.",
            )),
        }?;

        this_t0 + duration
    } else {
        this_t0
    };

    let this_qubits: HashSet<_> = dag
        .qargs_interner()
        .get(op.qubits)
        .iter()
        .map(|q| q.index())
        .collect();

    // Handle classical bits based on operation type
    let (new_t1c, this_clbits) = if matches!(
        op_view,
        OperationRef::StandardInstruction(StandardInstruction::Measure)
            | OperationRef::StandardInstruction(StandardInstruction::Reset)
    ) {
        // creg access ends at the end of instruction
        let new_t1c = Some(new_t1q);
        let this_clbits: HashSet<_> = dag
            .cargs_interner()
            .get(op.clbits)
            .iter()
            .map(|c| c.index())
            .collect();
        (new_t1c, this_clbits)
    } else {
        (None, HashSet::new())
    };
    // Check immediate successors for overlap
    for next_node_index in get_next_gate(dag, node_index) {
        // Get the next node
        let NodeType::Operation(next_node) = &dag[next_node_index] else {
            unreachable!("topological_op_nodes() should only return operations.")
        };

        // Compute next node start time separately for qreg and creg
        let next_t0q_obj = node_start_time.get_item(next_node_index.index())?;
        let next_t0q: u32 = next_t0q_obj
            .as_ref()
            .expect("Expected value in node_start_time for next_node_index")
            .extract()?;

        let next_qubits: HashSet<_> = dag
            .qargs_interner()
            .get(next_node.qubits)
            .iter()
            .map(|q| q.index())
            .collect();

        let next_op_view = next_node.op.view();
        let (next_t0c, next_clbits) = if matches!(
            next_op_view,
            OperationRef::StandardInstruction(StandardInstruction::Measure)
                | OperationRef::StandardInstruction(StandardInstruction::Reset)
        ) {
            // creg access starts after write latency
            let next_t0c = Some(next_t0q + clbit_write_latency);
            let next_clbits: HashSet<_> = dag
                .cargs_interner()
                .get(next_node.clbits)
                .iter()
                .map(|c| c.index())
                .collect();
            (next_t0c, next_clbits)
        } else {
            (None, HashSet::new())
        };

        // Compute overlap if there is qubits overlap
        let qreg_overlap = if !this_qubits.is_disjoint(&next_qubits) {
            (new_t1q - next_t0q).max(0)
        } else {
            0
        };

        // Compute overlap if there is clbits overlap
        let creg_overlap = if !this_clbits.is_empty()
            && !next_clbits.is_empty()
            && !this_clbits.is_disjoint(&next_clbits)
        {
            if let (Some(t1c), Some(t0c)) = (new_t1c, next_t0c) {
                (t1c - t0c).max(0)
            } else {
                0
            }
        } else {
            0
        };

        // Shift next node if there is finite overlap in either qubits or clbits
        let overlap = qreg_overlap.max(creg_overlap);
        if overlap > 0 {
            let new_start_time = next_t0q + overlap;
            node_start_time.set_item(next_node_index.index(), new_start_time)?;
        }
    }
    Ok(())
}

#[pyfunction]
#[pyo3(name="constrained_reschedule", signature=(dag, node_start_time, clbit_write_latency, acquire_align, pulse_align, target))]
pub fn run_constrained_reschedule(
    dag: &DAGCircuit,
    node_start_time: &Bound<PyDict>,
    clbit_write_latency: u32,
    acquire_align: u32,
    pulse_align: u32,
    target: Option<&Target>,
) -> PyResult<Py<PyDict>> {
    for node_index in dag.topological_op_nodes()? {
        let start_time = node_start_time.get_item(node_index.index());
        let val = start_time
            .map_err(|e| {
                TranspilerError::new_err(format!(
                    "PyDict error for node {}: {}",
                    node_index.index(),
                    e
                ))
            })?
            .ok_or_else(|| {
                TranspilerError::new_err(format!(
                    "Missing start time for node {}. Run scheduler again.",
                    node_index.index()
                ))
            })?
            .extract::<u32>()
            .map_err(|e| {
                TranspilerError::new_err(format!(
                    "Extract error for node {}: {}",
                    node_index.index(),
                    e
                ))
            })?;

        if val == 0 {
            continue;
        }

        push_node_back(
            dag,
            node_index,
            node_start_time,
            clbit_write_latency,
            acquire_align,
            pulse_align,
            target,
        )?;
    }

    Ok(node_start_time.clone().into())
}

pub fn constrained_reschedule_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(run_constrained_reschedule))?;
    Ok(())
}

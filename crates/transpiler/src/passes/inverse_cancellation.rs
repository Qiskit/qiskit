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

use ahash::RandomState;
use hashbrown::HashSet;
use indexmap::IndexMap;
use pyo3::prelude::*;
use rustworkx_core::petgraph::stable_graph::NodeIndex;

use qiskit_circuit::circuit_instruction::OperationFromPython;
use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType};
use qiskit_circuit::operations::{Operation, OperationRef, StandardGate};
use qiskit_circuit::packed_instruction::PackedInstruction;

fn gate_eq(gate_a: &PackedInstruction, gate_b: &OperationFromPython) -> PyResult<bool> {
    if gate_a.op.name() != gate_b.operation.name() {
        return Ok(false);
    }
    let a_params = gate_a.params_view();
    if a_params.len() != gate_b.params.len() {
        return Ok(false);
    }
    let mut param_eq = true;
    for (a, b) in a_params.iter().zip(&gate_b.params) {
        if !a.is_close(b, 1e-10)? {
            param_eq = false;
            break;
        }
    }
    Ok(param_eq)
}

fn run_on_self_inverse(
    dag: &mut DAGCircuit,
    op_counts: &IndexMap<String, usize, RandomState>,
    self_inverse_gate_names: HashSet<String>,
    self_inverse_gates: Vec<OperationFromPython>,
) -> PyResult<()> {
    if !self_inverse_gate_names
        .iter()
        .any(|name| op_counts.contains_key(name))
    {
        return Ok(());
    }
    for gate in self_inverse_gates {
        let gate_count = op_counts.get(gate.operation.name()).unwrap_or(&0);
        if *gate_count <= 1 {
            continue;
        }
        let mut collect_set: HashSet<String> = HashSet::with_capacity(1);
        collect_set.insert(gate.operation.name().to_string());
        let gate_runs: Vec<Vec<NodeIndex>> = dag.collect_runs(collect_set).collect();
        for gate_cancel_run in gate_runs {
            let mut partitions: Vec<Vec<NodeIndex>> = Vec::new();
            let mut chunk: Vec<NodeIndex> = Vec::new();
            let max_index = gate_cancel_run.len() - 1;
            for (i, cancel_gate) in gate_cancel_run.iter().enumerate() {
                let node = &dag[*cancel_gate];
                if let NodeType::Operation(inst) = node {
                    if gate_eq(inst, &gate)? {
                        chunk.push(*cancel_gate);
                    } else {
                        if !chunk.is_empty() {
                            partitions.push(std::mem::take(&mut chunk));
                        }
                        continue;
                    }
                    if i == max_index {
                        partitions.push(std::mem::take(&mut chunk));
                    } else {
                        let next_qargs =
                            if let NodeType::Operation(next_inst) = &dag[gate_cancel_run[i + 1]] {
                                next_inst.qubits
                            } else {
                                panic!("Not an op node")
                            };
                        if inst.qubits != next_qargs {
                            partitions.push(std::mem::take(&mut chunk));
                        }
                    }
                } else {
                    panic!("Not an op node");
                }
            }
            for chunk in partitions {
                if chunk.len() % 2 == 0 {
                    dag.remove_op_node(chunk[0]);
                }
                for node in &chunk[1..] {
                    dag.remove_op_node(*node);
                }
            }
        }
    }
    Ok(())
}
fn run_on_inverse_pairs(
    dag: &mut DAGCircuit,
    op_counts: &IndexMap<String, usize, RandomState>,
    inverse_gate_names: HashSet<String>,
    inverse_gates: Vec<[OperationFromPython; 2]>,
) -> PyResult<()> {
    if !inverse_gate_names
        .iter()
        .any(|name| op_counts.contains_key(name))
    {
        return Ok(());
    }
    for [gate_0, gate_1] in inverse_gates {
        let gate_0_name = gate_0.operation.name();
        let gate_1_name = gate_1.operation.name();
        if !op_counts.contains_key(gate_0_name) || !op_counts.contains_key(gate_1_name) {
            continue;
        }
        let names: HashSet<String> = [&gate_0, &gate_1]
            .iter()
            .map(|x| x.operation.name().to_string())
            .collect();
        let runs: Vec<Vec<NodeIndex>> = dag.collect_runs(names).collect();
        for nodes in runs {
            let mut i = 0;
            while i < nodes.len() - 1 {
                if let NodeType::Operation(inst) = &dag[nodes[i]] {
                    if let NodeType::Operation(next_inst) = &dag[nodes[i + 1]] {
                        if inst.qubits == next_inst.qubits
                            && ((gate_eq(inst, &gate_0)? && gate_eq(next_inst, &gate_1)?)
                                || (gate_eq(inst, &gate_1)? && gate_eq(next_inst, &gate_0)?))
                        {
                            dag.remove_op_node(nodes[i]);
                            dag.remove_op_node(nodes[i + 1]);
                            i += 2;
                        } else {
                            i += 1;
                        }
                    } else {
                        panic!("Not an op node")
                    }
                } else {
                    panic!("Not an op node")
                }
            }
        }
    }
    Ok(())
}

static SELF_INVERSE_GATES_FOR_CANCELLATION: [StandardGate; 15] = [
    StandardGate::CX,
    StandardGate::ECR,
    StandardGate::CY,
    StandardGate::CZ,
    StandardGate::X,
    StandardGate::Y,
    StandardGate::Z,
    StandardGate::H,
    StandardGate::Swap,
    StandardGate::CH,
    StandardGate::CCX,
    StandardGate::CCZ,
    StandardGate::RCCX,
    StandardGate::CSwap,
    StandardGate::C3X,
];

static INVERSE_PAIRS_FOR_CANCELLATION: [[StandardGate; 2]; 4] = [
    [StandardGate::T, StandardGate::Tdg],
    [StandardGate::S, StandardGate::Sdg],
    [StandardGate::SX, StandardGate::SXdg],
    [StandardGate::CS, StandardGate::CSdg],
];

fn std_self_inverse(dag: &mut DAGCircuit) {
    if !SELF_INVERSE_GATES_FOR_CANCELLATION
        .iter()
        .any(|gate| dag.get_op_counts().contains_key(gate.name()))
    {
        return;
    }
    // Handle self inverse gates
    for self_inv_gate in SELF_INVERSE_GATES_FOR_CANCELLATION {
        if *dag.get_op_counts().get(self_inv_gate.name()).unwrap_or(&0) <= 1 {
            continue;
        }
        let filter = |inst: &PackedInstruction| -> bool {
            match inst.op.view() {
                OperationRef::StandardGate(gate) => gate == self_inv_gate,
                _ => false,
            }
        };
        let run_nodes: Vec<_> = dag.collect_runs_by(filter).collect();
        for gate_cancel_run in run_nodes {
            let mut partitions: Vec<Vec<NodeIndex>> = Vec::new();
            let mut chunk: Vec<NodeIndex> = Vec::new();
            let max_index = gate_cancel_run.len() - 1;
            for (i, cancel_gate) in gate_cancel_run.iter().enumerate() {
                let node = &dag[*cancel_gate];
                let NodeType::Operation(inst) = node else {
                    unreachable!("Not an op node");
                };
                chunk.push(*cancel_gate);
                if i == max_index {
                    partitions.push(std::mem::take(&mut chunk));
                } else {
                    let NodeType::Operation(next_inst) = &dag[gate_cancel_run[i + 1]] else {
                        unreachable!("Not an op node");
                    };
                    let next_qargs = next_inst.qubits;
                    if inst.qubits != next_qargs {
                        partitions.push(std::mem::take(&mut chunk));
                    }
                }
            }
            for chunk in partitions {
                if chunk.len() % 2 == 0 {
                    dag.remove_op_node(chunk[0]);
                }
                for node in &chunk[1..] {
                    dag.remove_op_node(*node);
                }
            }
        }
    }
}

fn std_inverse_pairs(dag: &mut DAGCircuit) {
    if !INVERSE_PAIRS_FOR_CANCELLATION.iter().any(|gate| {
        dag.get_op_counts().contains_key(gate[0].name())
            && dag.get_op_counts().contains_key(gate[1].name())
    }) {
        return;
    }
    // Handle inverse pairs
    for [gate_0, gate_1] in INVERSE_PAIRS_FOR_CANCELLATION {
        if !dag.get_op_counts().contains_key(gate_0.name())
            || !dag.get_op_counts().contains_key(gate_1.name())
        {
            continue;
        }
        let filter = |inst: &PackedInstruction| -> bool {
            match inst.op.view() {
                OperationRef::StandardGate(gate) => gate == gate_0 || gate == gate_1,
                _ => false,
            }
        };
        let run_nodes: Vec<_> = dag.collect_runs_by(filter).collect();
        for nodes in run_nodes {
            let mut i = 0;
            while i < nodes.len() - 1 {
                let NodeType::Operation(inst) = &dag[nodes[i]] else {
                    unreachable!("Not an op node");
                };
                let NodeType::Operation(next_inst) = &dag[nodes[i + 1]] else {
                    unreachable!("Not an op node");
                };
                if inst.qubits == next_inst.qubits
                    && (inst.op.try_standard_gate() == Some(gate_0)
                        && next_inst.op.try_standard_gate() == Some(gate_1))
                    || (inst.op.try_standard_gate() == Some(gate_1)
                        && next_inst.op.try_standard_gate() == Some(gate_0))
                {
                    dag.remove_op_node(nodes[i]);
                    dag.remove_op_node(nodes[i + 1]);
                    i += 2;
                } else {
                    i += 1;
                }
            }
        }
    }
}

#[pyfunction]
pub fn run_inverse_cancellation_standard_gates(dag: &mut DAGCircuit) {
    std_self_inverse(dag);
    std_inverse_pairs(dag);
}

#[pyfunction]
#[pyo3(name = "inverse_cancellation")]
pub fn py_run_inverse_cancellation(
    py: Python,
    dag: &mut DAGCircuit,
    inverse_gates: Vec<[OperationFromPython; 2]>,
    self_inverse_gates: Vec<OperationFromPython>,
    inverse_gate_names: HashSet<String>,
    self_inverse_gate_names: HashSet<String>,
    run_defaults: bool,
) -> PyResult<()> {
    if self_inverse_gate_names.is_empty() && inverse_gate_names.is_empty() {
        return Ok(());
    }
    let op_counts = dag.count_ops(py, true)?;
    if !self_inverse_gate_names.is_empty() {
        run_on_self_inverse(dag, &op_counts, self_inverse_gate_names, self_inverse_gates)?;
    }
    if !inverse_gate_names.is_empty() {
        run_on_inverse_pairs(dag, &op_counts, inverse_gate_names, inverse_gates)?;
    }
    if run_defaults {
        std_self_inverse(dag);
        std_inverse_pairs(dag);
    }
    Ok(())
}

pub fn inverse_cancellation_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_run_inverse_cancellation))?;
    m.add_wrapped(wrap_pyfunction!(run_inverse_cancellation_standard_gates))?;
    Ok(())
}

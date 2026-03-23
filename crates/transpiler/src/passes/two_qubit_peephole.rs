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

use std::cell::RefCell;
#[cfg(feature = "cache_pygates")]
use std::sync::OnceLock;

use dashmap::DashMap;
use hashbrown::HashSet;
use pyo3::Python;
use pyo3::intern;
use pyo3::prelude::*;
use rayon::prelude::*;
use rustworkx_core::petgraph::stable_graph::NodeIndex;

use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType};
use qiskit_circuit::instruction::Parameters;
use qiskit_circuit::operations::{
    Operation, OperationRef, Param, PyOperationTypes, PythonOperation,
};
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperation};
use qiskit_circuit::{BlocksMode, Qubit, VarsMode};

use crate::passes::unitary_synthesis::{
    Approximation, QpuConstraint, TwoQSynthesisResult, fidelity_2q_sequence, synthesize_2q_matrix,
};
use crate::passes::{UnitarySynthesisConfig, UnitarySynthesisState};
use crate::target::Target;
use qiskit_circuit::PhysicalQubit;
use qiskit_circuit::getenv_use_multiple_threads;
use qiskit_quantum_info::convert_2q_block_matrix::blocks_to_matrix;
use thread_local::ThreadLocal;

type MappingIterItem = Option<(TwoQSynthesisResult, [Qubit; 2])>;

// This is a separate function in case we need to handle any Python synchronization in the future
// (such as releasing the GIL). For right now this doesn't seem to be necessary, but keeping it
// separate enables any manipulation of the Py handle in the future.
#[pyfunction(name = "two_qubit_unitary_peephole_optimize")]
pub fn py_two_qubit_unitary_peephole_optimize(
    dag: &DAGCircuit,
    target: &Target,
    approximation_degree: Option<f64>,
) -> PyResult<Option<DAGCircuit>> {
    two_qubit_unitary_peephole_optimize(dag, target, approximation_degree)
}

/// This function runs the two qubit unitary peephole optimization pass
///
/// It returns None if there is no modifications/optimiations made to the input dag and the pass
/// function calling this should just return the input dag from the pass.
pub fn two_qubit_unitary_peephole_optimize(
    dag: &DAGCircuit,
    target: &Target,
    approximation_degree: Option<f64>,
) -> PyResult<Option<DAGCircuit>> {
    let runs: Vec<Vec<NodeIndex>> = dag.collect_2q_runs().unwrap();
    if runs.is_empty() {
        return Ok(None);
    }
    let node_mapping: DashMap<NodeIndex, usize, ahash::RandomState> =
        DashMap::with_capacity_and_hasher(
            runs.iter().map(|run| run.len()).sum(),
            ahash::RandomState::default(),
        );
    let physical_qubits = (0..dag.num_qubits() as u32)
        .map(PhysicalQubit::new)
        .collect::<Vec<_>>();
    let approximation = Approximation::from_py_approximation_degree(approximation_degree);
    let unitary_synthesis_config = UnitarySynthesisConfig {
        approximation,
        ..Default::default()
    };
    let thread_local_states = ThreadLocal::new();
    let find_best_sequence =
        |run_index: usize, node_indices: &[NodeIndex]| -> PyResult<MappingIterItem> {
            let q_virt = node_indices
                .iter()
                .find_map(|node_index| {
                    let inst = dag.dag()[*node_index].unwrap_operation();
                    let qubits = dag.get_qargs(inst.qubits);
                    if qubits.len() == 2 {
                        Some([qubits[0], qubits[1]])
                    } else {
                        None
                    }
                })
                .unwrap();
            let q_phys = q_virt.map(|q| physical_qubits[q.index()]);
            let matrix = blocks_to_matrix(dag, node_indices, q_virt)?;
            let synthesis_state: &RefCell<UnitarySynthesisState> = thread_local_states
                .get_or(|| RefCell::new(UnitarySynthesisState::new(unitary_synthesis_config)));

            let result = synthesize_2q_matrix(
                matrix.into(),
                q_phys,
                &mut synthesis_state.borrow_mut(),
                QpuConstraint::Target(target),
            )?;
            if result.is_none() {
                return Ok(None);
            }
            let result = result.unwrap();
            let mut original_fidelity: f64 = 1.;
            let mut original_2q_count: usize = 0;
            let original_total_count: usize = node_indices.len();
            let mut outside_target = false;
            for node_index in node_indices {
                let NodeType::Operation(ref inst) = dag.dag()[*node_index] else {
                    unreachable!("All run nodes will be ops")
                };
                let qubits: &[_] = match dag.get_qargs(inst.qubits) {
                    [q] => &[PhysicalQubit(q.0)],
                    [q0, q1] => &[PhysicalQubit(q0.0), PhysicalQubit(q1.0)],
                    _ => panic!("Runs should only contain 1q and 2q gates"),
                };
                if qubits.len() == 2 {
                    original_2q_count += 1;
                }
                let name = inst.op.name();
                let gate_fidelity = match target.get_error(name, qubits) {
                    Some(err) => 1. - err,
                    None => {
                        // If error rate is None this can mean either the gate is not supported
                        // in the target or the gate is ideal. We need to do a second lookup
                        // to determine if the gate is supported, and if it isn't we don't need
                        // to finish scoring because we know we'll use the synthesis output
                        if !target.instruction_supported(name, qubits, inst.params_view(), true) {
                            outside_target = true;
                            break;
                        }
                        1.
                    }
                };
                original_fidelity *= gate_fidelity;
            }
            let original_score = (
                original_2q_count,
                1. - original_fidelity,
                original_total_count,
            );
            let new_2q_count = result
                .sequence
                .gates
                .iter()
                .filter(|x| x.0.num_qubits() == 2)
                .count();
            let new_gate_count = result.sequence.gates.len();
            let new_score = (
                new_2q_count,
                1. - result.fidelity.unwrap_or_else(|| {
                    fidelity_2q_sequence(
                        &result.dir,
                        &result.sequence,
                        &QpuConstraint::Target(target),
                        q_phys,
                    )
                }),
                new_gate_count,
            );
            // If the we are not outside the target and the new score isn't any better just use the
            // original (this includes a tie).
            if !outside_target && new_score >= original_score {
                return Ok(None);
            }
            // This is done at the end of the map in some attempt to minimize
            // lock contention. If this were serial code it'd make more sense
            // to do this as part of the iteration building the
            for node in node_indices {
                node_mapping.insert(*node, run_index);
            }
            Ok(Some((result, q_virt)))
        };

    let run_mapping: PyResult<Vec<MappingIterItem>> = if getenv_use_multiple_threads() {
        // Build a vec of all the best synthesized two qubit gate sequences from the collected runs.
        // This is done in parallel
        runs.par_iter()
            .enumerate()
            .map(|(index, sequence)| find_best_sequence(index, sequence.as_slice()))
            .collect()
    } else {
        runs.iter()
            .enumerate()
            .map(|(index, sequence)| find_best_sequence(index, sequence.as_slice()))
            .collect()
    };
    let run_mapping = run_mapping?;
    // After we've computed all the sequences to execute now serially build up a new dag.
    let mut processed_runs: HashSet<usize> = HashSet::with_capacity(run_mapping.len());
    let out_dag = dag.copy_empty_like_with_same_capacity(VarsMode::Alike, BlocksMode::Keep)?;
    let mut out_dag_builder = out_dag.into_builder();
    if node_mapping.is_empty() {
        return Ok(None);
    }
    for node in dag.topological_op_nodes(false) {
        match node_mapping.get(&node) {
            Some(run_index) => {
                if processed_runs.contains(run_index.value()) {
                    continue;
                }
                // If this is not a two qubit gate then there is a chance this will cause the
                // insertion to happen too early. We skip the nodes in a run until we encounter
                // a 2q gate which ensure the block is inserted into the correct location in the
                // circuit.
                if dag.dag()[node].unwrap_operation().op.num_qubits() != 2 {
                    continue;
                }
                // A None is inserted into the run_mapping as the value for a run that we don't
                // substitute but was identified so we added an explicit None to preserve the
                // indexing with the vec. This shouldn't be possible to hit the else condition
                // since node mapping will never contain a value for a run_mapping index that
                // is set to None.
                let Some((result, qargs_virt)) = run_mapping[*run_index].as_ref() else {
                    unreachable!(
                        "node_mapping can't contain a value pointing to an unpoluated run in run_mapping"
                    );
                };
                let order = result.dir.as_indices();
                let out_qargs = [qargs_virt[order[0] as usize], qargs_virt[order[1] as usize]];
                let qubit_keys = [
                    out_dag_builder.insert_qargs(&[out_qargs[0]]),
                    out_dag_builder.insert_qargs(&[out_qargs[1]]),
                    out_dag_builder.insert_qargs(&[out_qargs[0], out_qargs[1]]),
                    out_dag_builder.insert_qargs(&[out_qargs[1], out_qargs[0]]),
                ];
                for (gate, params, local_qubits) in &result.sequence.gates {
                    let qubits = match local_qubits.as_slice() {
                        [0] => qubit_keys[0],
                        [1] => qubit_keys[1],
                        [0, 1] => qubit_keys[2],
                        [1, 0] => qubit_keys[3],
                        _ => panic!(
                            "internal logic error: decomposed sequence contained unexpected qargs"
                        ),
                    };
                    let op = match gate.view() {
                        OperationRef::StandardGate(gate) => PackedOperation::from(gate),
                        OperationRef::Gate(py_gate) => Python::attach(|py| -> PyResult<_> {
                            let gate = py_gate.py_copy(py)?;
                            gate.instruction
                                .setattr(py, intern!(py, "params"), params)?;
                            Ok(PackedOperation::from(Box::new(PyOperationTypes::Gate(
                                gate,
                            ))))
                        })?,
                        _ => {
                            panic!("internal logic error: decomposed sequence contains a non-gate")
                        }
                    };
                    let params = (!params.is_empty()).then(|| {
                        Box::new(Parameters::Params(
                            params.iter().copied().map(Param::Float).collect(),
                        ))
                    });
                    out_dag_builder.push_back(PackedInstruction {
                      op,
                      qubits,
                      clbits: Default::default(),
                      params,
                      label: None,
                      #[cfg(feature = "cache_pygates")] // W: code is inactive due to #[cfg] directives: feature …
                      py_op: OnceLock::new(),
                    })?;
                }
                out_dag_builder.add_global_phase(&Param::Float(result.sequence.global_phase()))?;
                processed_runs.insert(*run_index);
            }
            None => {
                let NodeType::Operation(ref instr) = dag.dag()[node] else {
                    unreachable!("Must be an op node")
                };
                out_dag_builder.push_back(instr.clone())?;
            }
        }
    }
    Ok(Some(out_dag_builder.build()))
}

pub fn two_qubit_peephole_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_two_qubit_unitary_peephole_optimize))?;
    Ok(())
}

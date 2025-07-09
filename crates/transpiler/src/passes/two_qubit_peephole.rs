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

use std::cmp::Ordering;
use std::sync::Mutex;

use hashbrown::{HashMap, HashSet};
use pyo3::prelude::*;
use rayon::prelude::*;
use rustworkx_core::petgraph::stable_graph::NodeIndex;
use smallvec::{smallvec, SmallVec};

use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType, VarsMode};
use qiskit_circuit::operations::{Operation, OperationRef, Param, StandardGate};
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::Qubit;

use super::two_qubit_unitary_synthesis_utils::{
    preferred_direction, synth_su4_sequence, DecomposerElement, DecomposerType,
    TwoQubitUnitarySequence,
};
use crate::target::{Qargs, Target, TargetOperation};
use crate::TranspilerError;
use qiskit_circuit::getenv_use_multiple_threads;
use qiskit_circuit::PhysicalQubit;
use qiskit_quantum_info::convert_2q_block_matrix::blocks_to_matrix;
use qiskit_synthesis::euler_one_qubit_decomposer::{
    EulerBasis, EulerBasisSet, EULER_BASES, EULER_BASIS_NAMES,
};
use qiskit_synthesis::two_qubit_decompose::{
    RXXEquivalent, TwoQubitBasisDecomposer, TwoQubitControlledUDecomposer,
    TwoQubitWeylDecomposition,
};

fn get_decomposers_from_target(
    target: &Target,
    qubits: &[Qubit],
    fidelity: f64,
) -> PyResult<Vec<DecomposerElement>> {
    let physical_qubits: SmallVec<[PhysicalQubit; 2]> =
        smallvec![PhysicalQubit(qubits[0].0), PhysicalQubit(qubits[1].0)];
    let reverse_qubits: SmallVec<[PhysicalQubit; 2]> =
        physical_qubits.iter().rev().copied().collect();
    let mut reverse_used = false;
    let mut gate_names: HashSet<&str> = match target.operation_names_for_qargs(&physical_qubits) {
        Ok(names) => names.into_iter().collect(),
        Err(err) => {
            reverse_used = true;
            target
                .operation_names_for_qargs(&reverse_qubits)
                .map_err(|_| TranspilerError::new_err(err.to_string()))?
                .into_iter()
                .collect()
        }
    };
    if !reverse_used {
        if let Ok(reverse_names) = target.operation_names_for_qargs(&reverse_qubits) {
            if !reverse_names.is_empty() {
                for name in reverse_names {
                    gate_names.insert(name);
                }
            }
        }
    }
    let available_kak_gate: Vec<(&str, &PackedOperation, &[Param])> = gate_names
        .iter()
        .filter_map(|name| match target.operation_from_name(name) {
            Some(raw_op) => {
                if let TargetOperation::Normal(op) = raw_op {
                    match op.operation.view() {
                        OperationRef::StandardGate(gate) => {
                            if matches!(
                                gate,
                                StandardGate::CX | StandardGate::CZ | StandardGate::ECR
                            ) {
                                Some((*name, &op.operation, op.params.as_slice()))
                            } else if let Some(matrix) = gate.matrix(&op.params) {
                                if let Ok(weyl) =
                                    TwoQubitWeylDecomposition::new_inner(matrix.view(), None, None)
                                {
                                    if weyl.is_supercontrolled() {
                                        Some((*name, &op.operation, op.params.as_slice()))
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        }
                        OperationRef::Gate(gate) => {
                            if let Some(matrix) = gate.matrix(&op.params) {
                                let weyl =
                                    TwoQubitWeylDecomposition::new_inner(matrix.view(), None, None)
                                        .unwrap();
                                if weyl.is_supercontrolled() {
                                    Some((*name, &op.operation, op.params.as_slice()))
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        }
                        _ => None,
                    }
                } else {
                    None
                }
            }
            None => None,
        })
        .collect();

    let single_qubit_basis_list = target.operation_names_for_qargs(&[physical_qubits[0]]);
    let mut target_basis_set = EulerBasisSet::new();
    match single_qubit_basis_list {
        Ok(basis_list) => {
            EULER_BASES
                .iter()
                .enumerate()
                .filter_map(|(idx, gates)| {
                    if !gates.iter().all(|gate| basis_list.contains(gate)) {
                        return None;
                    }
                    let basis = EULER_BASIS_NAMES[idx];
                    Some(basis)
                })
                .for_each(|basis| target_basis_set.add_basis(basis));
        }
        Err(_) => target_basis_set.support_all(),
    }
    if target_basis_set.basis_supported(EulerBasis::U3)
        && target_basis_set.basis_supported(EulerBasis::U321)
    {
        target_basis_set.remove(EulerBasis::U3);
    }
    if target_basis_set.basis_supported(EulerBasis::ZSX)
        && target_basis_set.basis_supported(EulerBasis::ZSXX)
    {
        target_basis_set.remove(EulerBasis::ZSX);
    }

    let decomposers: PyResult<Vec<DecomposerElement>> = available_kak_gate
        .iter()
        .filter_map(|(two_qubit_name, two_qubit_gate, params)| {
            let matrix = two_qubit_gate.matrix(params);
            matrix.map(|matrix| {
                target_basis_set.get_bases().filter_map(move |euler_basis| {
                    TwoQubitBasisDecomposer::new_inner(
                        two_qubit_name.to_string(),
                        matrix.view(),
                        fidelity,
                        euler_basis,
                        None,
                    )
                    .map(|decomp| {
                        if !decomp.super_controlled() {
                            None
                        } else {
                            Some(DecomposerElement {
                                decomposer: DecomposerType::TwoQubitBasis(Box::new(decomp)),
                                packed_op: (*two_qubit_gate).clone(),
                                params: params.iter().cloned().collect(),
                                target_name: two_qubit_name.to_string(),
                            })
                        }
                    })
                    .transpose()
                })
            })
        })
        .flatten()
        .collect();
    let mut decomposers = decomposers?;
    for gate in [
        StandardGate::RXX,
        StandardGate::RZZ,
        StandardGate::RYY,
        StandardGate::RZX,
    ] {
        if gate_names.contains(gate.name()) {
            let op = target.operation_from_name(gate.name()).unwrap();
            if op
                .params()
                .iter()
                .all(|x| matches!(x, Param::ParameterExpression(_)))
            {
                for euler_basis in target_basis_set.get_bases() {
                    decomposers.push(DecomposerElement {
                        decomposer: DecomposerType::TwoQubitControlledU(Box::new(
                            TwoQubitControlledUDecomposer::new(
                                RXXEquivalent::Standard(gate),
                                euler_basis.as_str(),
                            )?,
                        )),
                        packed_op: gate.into(),
                        // TODO: Add param when ParameterExpression doesn't
                        // need python. This is a corrupt param for the gates
                        // here, but it unused in the passes and needs to be
                        // an unbound  parameter. Do not use this value for
                        // constructing a circuit.
                        params: smallvec![],
                        target_name: gate.name().to_string(),
                    });
                }
            }
        }
    }
    Ok(decomposers)
}

/// Score a given sequence using the error rate reported in the target
///
/// Return a tuple of the predicted fidelity and the number of 2q gates in the sequence
#[inline]
fn score_sequence<'a>(
    target: &'a Target,
    kak_gate_name: &str,
    sequence: impl Iterator<Item = (Option<StandardGate>, SmallVec<[Qubit; 2]>)> + 'a,
) -> (usize, f64, usize) {
    let mut two_gate_count = 0;
    let mut total_gate_count = 0;
    let fidelity = sequence
        .filter_map(|(gate, local_qubits)| {
            total_gate_count += 1;
            let qubits = local_qubits
                .iter()
                .map(|qubit| PhysicalQubit(qubit.0))
                .collect::<Vec<_>>();
            if qubits.len() == 2 {
                two_gate_count += 1;
            }
            let name = match gate.as_ref() {
                Some(g) => g.name(),
                None => kak_gate_name,
            };
            let error = target.get_error(name, qubits.as_slice());
            error.map(|error| 1. - error)
        })
        .product::<f64>();
    (two_gate_count, 1. - fidelity, total_gate_count)
}

type MappingIterItem = Option<(TwoQubitUnitarySequence, [Qubit; 2])>;

/// This transpiler pass can only run in a context where we've translated the circuit gates (or
/// where we know all gates have a matrix). If any gate identified in the run fails to have a
/// matrix defined (either in rust or python) it will be skipped
#[pyfunction]
pub fn two_qubit_unitary_peephole_optimize(
    py: Python,
    dag: &DAGCircuit,
    target: &Target,
    fidelity: f64,
) -> PyResult<DAGCircuit> {
    let runs: Vec<Vec<NodeIndex>> = dag.collect_2q_runs().unwrap();
    let node_mapping: HashMap<NodeIndex, usize> =
        HashMap::with_capacity(runs.iter().map(|run| run.len()).sum());
    let locked_node_mapping = Mutex::new(node_mapping);
    let coupling_edges = match target.qargs() {
        Some(qargs) => qargs
            .filter_map(|qargs| match qargs {
                Qargs::Concrete(qargs) => {
                    if qargs.len() == 2 {
                        Some([qargs[0], qargs[1]])
                    } else {
                        None
                    }
                }
                Qargs::Global => None,
            })
            .collect(),
        None => HashSet::new(),
    };
    let find_best_sequence =
        |run_index: usize, node_indices: &[NodeIndex]| -> PyResult<MappingIterItem> {
            let block_qubit_map = node_indices
                .iter()
                .find_map(|node_index| {
                    let inst = dag.dag()[*node_index].unwrap_operation();
                    let qubits = dag.get_qargs(inst.qubits);
                    if qubits.len() == 2 {
                        if qubits[0] > qubits[1] {
                            Some([qubits[1], qubits[0]])
                        } else {
                            Some([qubits[0], qubits[1]])
                        }
                    } else {
                        None
                    }
                })
                .unwrap();
            let matrix = blocks_to_matrix(dag, node_indices, block_qubit_map)?;
            let decomposers = get_decomposers_from_target(target, &block_qubit_map, fidelity)?;
            let mut decomposer_scores: Vec<Option<(usize, f64, usize)>> =
                vec![None; decomposers.len()];

            let order_sequence =
                |(index_a, sequence_a): &(usize, TwoQubitUnitarySequence),
                 (index_b, sequence_b): &(usize, TwoQubitUnitarySequence)| {
                    let score_a = (
                        match decomposer_scores[*index_a] {
                            Some(score) => score,
                            None => {
                                let score: (usize, f64, usize) = score_sequence(
                                    target,
                                    sequence_a.target_name.as_str(),
                                    sequence_a.gate_sequence.gates.iter().map(
                                        |(gate, _params, local_qubits)| {
                                            let qubits = local_qubits
                                                .iter()
                                                .map(|qubit| block_qubit_map[*qubit as usize])
                                                .collect();
                                            (*gate, qubits)
                                        },
                                    ),
                                );
                                decomposer_scores[*index_a] = Some(score);
                                score
                            }
                        },
                        index_a,
                    );

                    let score_b = (
                        match decomposer_scores[*index_b] {
                            Some(score) => score,
                            None => {
                                let score: (usize, f64, usize) = score_sequence(
                                    target,
                                    sequence_b.target_name.as_str(),
                                    sequence_b.gate_sequence.gates.iter().map(
                                        |(gate, _params, local_qubits)| {
                                            let qubits = local_qubits
                                                .iter()
                                                .map(|qubit| block_qubit_map[*qubit as usize])
                                                .collect();
                                            (*gate, qubits)
                                        },
                                    ),
                                );
                                decomposer_scores[*index_b] = Some(score);
                                score
                            }
                        },
                        index_b,
                    );
                    score_a.partial_cmp(&score_b).unwrap_or(Ordering::Equal)
                };
            let sequence = decomposers
                .iter()
                .map(|decomposer| {
                    let physical_block_qubit_map: [PhysicalQubit; 2] = [
                        PhysicalQubit(block_qubit_map[0].0),
                        PhysicalQubit(block_qubit_map[1].0),
                    ];
                    let dir = preferred_direction(
                        &physical_block_qubit_map,
                        Some(true),
                        &coupling_edges,
                        Some(target),
                        decomposer,
                    )
                    .ok()
                    .flatten();
                    synth_su4_sequence(matrix.view(), decomposer, dir, Some(fidelity)).unwrap()
                })
                .enumerate()
                .min_by(order_sequence);
            if sequence.is_none() {
                return Ok(None);
            }
            let sequence = sequence.unwrap();
            let mut original_fidelity: f64 = 1.;
            let mut original_2q_count: usize = 0;
            let original_total_count: usize = node_indices.len();
            let mut outside_target = false;
            for node_index in node_indices {
                let NodeType::Operation(ref inst) = dag.dag()[*node_index] else {
                    unreachable!("All run nodes will be ops")
                };
                let qubits: SmallVec<[PhysicalQubit; 2]> = dag
                    .get_qargs(inst.qubits)
                    .iter()
                    .map(|qubit| PhysicalQubit(qubit.0))
                    .collect();
                if qubits.len() == 2 {
                    original_2q_count += 1;
                }
                let name = inst.op.name();
                let gate_fidelity = match target.get_error(name, qubits.as_slice()) {
                    Some(err) => 1. - err,
                    None => {
                        // If error rate is None this can mean either the gate is not supported
                        // in the target or the gate is ideal. We need to do a second lookup
                        // to determine if the gate is supported, and if it isn't we don't need
                        // to finish scoring because we know we'll use the synthesis output
                        if !target.instruction_supported(name, &qubits) {
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
            let new_score: (usize, f64, usize) =
                match decomposer_scores[sequence.0] {
                    Some(score) => score,
                    None => score_sequence(
                        target,
                        sequence.1.target_name.as_str(),
                        sequence.1.gate_sequence.gates.iter().map(
                            |(gate, _params, local_qubits)| {
                                let qubits = local_qubits
                                    .iter()
                                    .map(|qubit| block_qubit_map[*qubit as usize])
                                    .collect();
                                (*gate, qubits)
                            },
                        ),
                    ),
                };
            // If the we are not outside the target and the new score isn't any better just use the
            // original (this includes a tie).
            if !outside_target && new_score >= original_score {
                return Ok(None);
            }
            // This is done at the end of the map in some attempt to minimize
            // lock contention. If this were serial code it'd make more sense
            // to do this as part of the iteration building the
            let mut node_mapping = locked_node_mapping.lock().unwrap();
            for node in node_indices {
                node_mapping.insert(*node, run_index);
            }
            Ok(Some((sequence.1, block_qubit_map)))
        };

    let run_mapping: PyResult<Vec<MappingIterItem>> = if getenv_use_multiple_threads() {
        py.allow_threads(|| {
            // Build a vec of all the best synthesized two qubit gate sequences from the collected runs.
            // This is done in parallel
            runs.par_iter()
                .enumerate()
                .map(|(index, sequence)| find_best_sequence(index, sequence.as_slice()))
                .collect()
        })
    } else {
        runs.iter()
            .enumerate()
            .map(|(index, sequence)| find_best_sequence(index, sequence.as_slice()))
            .collect()
    };

    let run_mapping = run_mapping?;
    // After we've computed all the sequences to execute now serially build up a new dag.
    let mut processed_runs: HashSet<usize> = HashSet::with_capacity(run_mapping.len());
    let out_dag = dag.copy_empty_like(VarsMode::Alike)?;
    let mut out_dag_builder = out_dag.into_builder();
    let node_mapping = locked_node_mapping.into_inner().unwrap();
    for node in dag.topological_op_nodes()? {
        match node_mapping.get(&node) {
            Some(run_index) => {
                if processed_runs.contains(run_index) {
                    continue;
                }
                let run = run_mapping[*run_index].as_ref();
                if run.is_none() {
                    let NodeType::Operation(ref instr) = dag.dag()[node] else {
                        unreachable!("Must be an op node")
                    };
                    out_dag_builder.push_back(instr.clone())?;
                    continue;
                }
                let (sequence, qubit_map) = run.unwrap();
                for (gate, params, local_qubits) in &sequence.gate_sequence.gates {
                    let qubits: Vec<Qubit> = local_qubits
                        .iter()
                        .map(|index| qubit_map[*index as usize])
                        .collect();

                    let out_params = if params.is_empty() {
                        None
                    } else {
                        Some(params.into_iter().map(|val| Param::Float(*val)).collect())
                    };
                    match gate {
                        Some(gate) => out_dag_builder.apply_operation_back(
                            PackedOperation::from_standard_gate(*gate),
                            &qubits,
                            &[],
                            out_params,
                            None,
                            #[cfg(feature = "cache_pygates")]
                            None,
                        ),
                        None => out_dag_builder.apply_operation_back(
                            sequence.decomp_op.clone(),
                            &qubits,
                            &[],
                            Some(out_params.unwrap_or(sequence.decomp_params.clone())),
                            None,
                            #[cfg(feature = "cache_pygates")]
                            None,
                        ),
                    }?;
                }
                out_dag_builder
                    .add_global_phase(&Param::Float(sequence.gate_sequence.global_phase))?;
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
    Ok(out_dag_builder.build())
}

pub fn two_qubit_peephole_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(two_qubit_unitary_peephole_optimize))?;
    Ok(())
}

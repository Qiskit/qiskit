// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use std::f64::consts::PI;

use num_complex::Complex64;
use pyo3::prelude::*;
use pyo3::{Bound, PyResult, pyfunction, wrap_pyfunction};
use qiskit_circuit::instruction::Parameters;
use smallvec::smallvec;

use crate::commutation_checker::{CommutationChecker, try_matrix_with_definition};
use crate::passes::remove_identity_equiv::{average_gate_fidelity_below_tol, is_identity_equiv};
use qiskit_circuit::circuit_instruction::OperationFromPython;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::operations::{
    Operation, OperationRef, Param, StandardGate, multiply_param, radd_param,
};
use qiskit_circuit::{BlocksMode, Clbit, Qubit, imports};

use qiskit_circuit::VarsMode;
use qiskit_circuit::packed_instruction::PackedInstruction;

/// Holds the action for each node in the original DAGCircuit.
#[derive(Clone, Debug)]
enum NodeAction {
    /// The node's instruction is unchanged, and can be retrieved from the circuit.
    Keep,
    /// The node's instruction can be replaced by this representative
    /// (the second parameter specifies the global phase update).
    /// However, unless this representative gate is removed or merged,
    /// we will add the original instruction to the output circuit.
    Canonical(PackedInstruction, Param),
    /// The node's instruction has been removed.
    Drop,
    /// The node's instruction has been replaced by the current instruction.
    Replace(PackedInstruction),
}

/// Returns true if the two parameter lists are equal.
fn compare_params(params1: &[Param], params2: &[Param]) -> PyResult<bool> {
    if params1.len() != params2.len() {
        return Ok(false);
    }

    for (p1, p2) in params1.iter().zip(params2.iter()) {
        let eq = p1.eq(p2)?;
        if !eq {
            return Ok(false);
        }
    }

    Ok(true)
}

/// List of symmetric gates, that is the gate remains the same under all
/// permutations of its arguments.
static SYMMETRIC_GATES: [StandardGate; 13] = [
    StandardGate::CZ,
    StandardGate::Swap,
    StandardGate::ISwap,
    StandardGate::CPhase,
    StandardGate::CS,
    StandardGate::CSdg,
    StandardGate::CU1,
    StandardGate::RXX,
    StandardGate::RYY,
    StandardGate::RZZ,
    StandardGate::XXMinusYY,
    StandardGate::XXPlusYY,
    StandardGate::CCZ,
];

/// List of single-parameter rotation gates. This gates can be merged into a gate
/// of the same class with the summed parameter. The list should contain only the
/// "canonical" gates that remain after `canonicalize`.`
static MERGEABLE_ROTATION_GATES: [StandardGate; 12] = [
    StandardGate::RX,
    StandardGate::RY,
    StandardGate::RZ,
    StandardGate::RXX,
    StandardGate::RYY,
    StandardGate::RZX,
    StandardGate::RZZ,
    StandardGate::CRX,
    StandardGate::CRY,
    StandardGate::CRZ,
    StandardGate::CPhase,
    StandardGate::CU1,
];

/// Computes the canonical representative of a packed instruction, and in particular:
/// * replaces all types of Z-rotations by RZ-gates,
/// * replaces all types of X-rotations by RX-gates,
/// * sorts the qubits for symmetric gates.
///
/// # Arguments:
///
/// * `dag` - The output [DAGCircuit]. We use its `qargs_interner` to store sorted
///   qubits for symmetric gates.
/// * `inst` - The instruction to canonicalize.
///
/// # Returns:
///
/// The canonical instruction and the global phase update. `None` means that the
/// original instruction is already canonical.
fn canonicalize(
    dag: &mut DAGCircuit,
    inst: &PackedInstruction,
) -> Option<(PackedInstruction, Param)> {
    // ToDo: possibly consider other rotations as well (e.g. CS -> CRZ).
    let rotation = match inst.op.view() {
        OperationRef::StandardGate(StandardGate::Phase)
        | OperationRef::StandardGate(StandardGate::U1) => Some((
            StandardGate::RZ,
            inst.params_view()[0].clone(),
            multiply_param(&inst.params_view()[0], 0.5),
        )),
        OperationRef::StandardGate(StandardGate::Z) => {
            Some((StandardGate::RZ, Param::Float(PI), Param::Float(PI / 2.)))
        }
        OperationRef::StandardGate(StandardGate::S) => Some((
            StandardGate::RZ,
            Param::Float(PI / 2.),
            Param::Float(PI / 4.),
        )),
        OperationRef::StandardGate(StandardGate::Sdg) => Some((
            StandardGate::RZ,
            Param::Float(-PI / 2.),
            Param::Float(-PI / 4.),
        )),
        OperationRef::StandardGate(StandardGate::T) => Some((
            StandardGate::RZ,
            Param::Float(PI / 4.),
            Param::Float(PI / 8.),
        )),
        OperationRef::StandardGate(StandardGate::Tdg) => Some((
            StandardGate::RZ,
            Param::Float(-PI / 4.),
            Param::Float(-PI / 8.),
        )),
        OperationRef::StandardGate(StandardGate::X) => {
            Some((StandardGate::RX, Param::Float(PI), Param::Float(PI / 2.)))
        }
        OperationRef::StandardGate(StandardGate::SX) => Some((
            StandardGate::RX,
            Param::Float(PI / 2.),
            Param::Float(PI / 4.),
        )),
        OperationRef::StandardGate(StandardGate::SXdg) => Some((
            StandardGate::RX,
            Param::Float(-PI / 2.),
            Param::Float(-PI / 4.),
        )),
        _ => None,
    };

    if let Some((gate, param, phase_update)) = rotation {
        let params = Some(Box::new(smallvec![param]));
        return Some((
            PackedInstruction::from_standard_gate(gate, params, inst.qubits),
            phase_update,
        ));
    }

    if let OperationRef::StandardGate(standard_gate) = inst.op.view() {
        if SYMMETRIC_GATES.contains(&standard_gate) {
            let qargs = dag.get_qargs(inst.qubits);
            if !qargs.is_sorted() {
                let mut sorted_qargs = qargs.to_vec();
                sorted_qargs.sort();
                let sorted_qubits = dag.add_qargs(&sorted_qargs);
                let canonical_instruction = PackedInstruction {
                    op: standard_gate.into(),
                    qubits: sorted_qubits,
                    clbits: Default::default(),
                    params: inst.params.clone(),
                    label: None,
                    #[cfg(feature = "cache_pygates")]
                    py_op: std::sync::OnceLock::new(),
                };
                return Some((canonical_instruction, Param::Float(0.)));
            }
        }
    }
    None
}

/// Return `true` if two instructions commute (up to the specified tolerance).
///
/// # Arguments:
///
/// * `dag`: The output [DAGCircuit] that contains all interned qubits.
/// * `approximation_degree`: Specifies tolerance.
/// * `matrix_max_num_qubits`: The maximum number of qubits to use for more expensive
///   matrix-based checks.
fn commute(
    dag: &DAGCircuit,
    inst1: &PackedInstruction,
    inst2: &PackedInstruction,
    approximation_degree: f64,
    matrix_max_num_qubits: u32,
    commutation_checker: &mut CommutationChecker,
) -> PyResult<bool> {
    let qargs1 = dag.get_qargs(inst1.qubits);
    let qargs2 = dag.get_qargs(inst2.qubits);
    let cargs1 = dag.get_cargs(inst1.clbits);
    let cargs2 = dag.get_cargs(inst2.clbits);

    let op1 = inst1.op.view();
    let op2 = inst2.op.view();

    Ok(commutation_checker.commute(
        &op1,
        inst1.params.as_deref(),
        qargs1,
        cargs1,
        &op2,
        inst2.params.as_deref(),
        qargs2,
        cargs2,
        None,
        matrix_max_num_qubits,
        approximation_degree,
    )?)
}

/// Merge two instructions.
///
/// The two instructions have already been canonicalized.
///
/// Arguments:
///
/// * `dag`: The output [DAGCircuit] that contains all interned qubits.
/// * `tol`: Specifies tolerance to check whether the merged operation
///   is close to the identity.
/// * `matrix_max_num_qubits`: The maximum number of qubits to use for more expensive
///   matrix-based checks.
///
/// # Returns:
///
/// A triple, consisting of whether the two instructions can be merged, the
/// merged instruction (`None` if the two instructions cancel out, up to a
/// global phase), and the global phase update. In other words:
/// * (true, None, phase_update): the two instructions cancel out, producing
///   the given global phase.
/// * (true, Some(instruction), phase_update): the two instructions are merged.
/// * (false, None, 0.): the two instructions cannot be merged.
fn try_merge(
    dag: &DAGCircuit,
    inst1: &PackedInstruction,
    inst2: &PackedInstruction,
    tol: f64,
    matrix_max_num_qubits: u32,
) -> PyResult<(bool, Option<PackedInstruction>, f64)> {
    if inst1.op.num_qubits() != inst2.op.num_qubits() {
        return Ok((false, None, 0.));
    }

    let error_cutoff_fn = |_inst: &PackedInstruction| -> f64 { tol };

    let params1 = inst1.params_view();
    let params2 = inst2.params_view();
    let qargs1 = dag.get_qargs(inst1.qubits);
    let qargs2 = dag.get_qargs(inst2.qubits);
    let cargs1 = dag.get_cargs(inst1.clbits);
    let cargs2 = dag.get_cargs(inst2.clbits);

    if (qargs1 != qargs2) || (cargs1 != cargs2) {
        return Ok((false, None, 0.));
    }

    // Both instructions are standard gates.
    if let (OperationRef::StandardGate(gate1), OperationRef::StandardGate(gate2)) =
        (inst1.op.view(), inst2.op.view())
    {
        // Check wether the two gates are self-inverse.
        if let Some((gate1inv, params1inv)) = gate1.inverse(params1) {
            if (gate1inv == gate2) && compare_params(&params1inv, params2)? {
                return Ok((true, None, 0.));
            }
        }

        // Can merge two single-parameter standard rotation gates of the same type.
        if gate1 == gate2 && MERGEABLE_ROTATION_GATES.contains(&gate1) {
            let merged_param = radd_param(params1[0].clone(), params2[0].clone());
            let params = Some(Box::new(smallvec![merged_param]));
            let merged_instruction =
                PackedInstruction::from_standard_gate(gate1, params, inst1.qubits);
            if let Some(phase_update) = is_identity_equiv(
                &merged_instruction,
                true,
                Some(matrix_max_num_qubits),
                error_cutoff_fn,
            )? {
                return Ok((true, None, phase_update));
            } else {
                return Ok((true, Some(merged_instruction), 0.));
            }
        }
    }

    // Special handling for PauliEvolutionGates.
    if inst1.op.name() == "PauliEvolution" && inst2.op.name() == "PauliEvolution" {
        if let (OperationRef::Gate(py_gate1), OperationRef::Gate(py_gate2)) =
            (inst1.op.view(), inst2.op.view())
        {
            let merged_instruction = Python::attach(|py| -> PyResult<Option<PackedInstruction>> {
                let merge_result = imports::MERGE_TWO_PAULI_EVOLUTIONS
                    .get_bound(py)
                    .call1((py_gate1.gate.clone_ref(py), py_gate2.gate.clone_ref(py)))?;

                if merge_result.is_none() {
                    Ok(None)
                } else {
                    let instr: OperationFromPython = merge_result.extract()?;
                    let merged_param = instr
                        .params
                        .expect("PauliEvolution gate contains a parameter")
                        .unwrap_params()[0]
                        .clone();

                    let merged_params = Some(Box::new(Parameters::Params(smallvec![merged_param])));

                    Ok(Some(PackedInstruction {
                        op: instr.operation,
                        qubits: inst1.qubits,
                        clbits: inst1.clbits,
                        params: merged_params,
                        label: instr.label.clone(),
                        #[cfg(feature = "cache_pygates")]
                        py_op: std::sync::OnceLock::new(),
                    }))
                }
            })?;

            if let Some(merged_instruction) = merged_instruction {
                if let Some(phase_update) = is_identity_equiv(
                    &merged_instruction,
                    true,
                    Some(matrix_max_num_qubits),
                    error_cutoff_fn,
                )? {
                    return Ok((true, None, phase_update));
                } else {
                    return Ok((true, Some(merged_instruction), 0.));
                }
            } else {
                return Ok((false, None, 0.));
            }
        }
    }

    // Matrix-based check: the product matrix is equivalent to identity.
    if inst1.op.num_qubits() <= matrix_max_num_qubits {
        let view1 = inst1.op.view();
        let view2 = inst2.op.view();

        if let (Some(matrix1), Some(matrix2)) = (
            try_matrix_with_definition(&view1, inst1.params_view(), Some(matrix_max_num_qubits)),
            try_matrix_with_definition(&view2, inst2.params_view(), Some(matrix_max_num_qubits)),
        ) {
            let product_mat = matrix1.dot(&matrix2);
            let dim = product_mat.shape()[0] as f64;
            let tr_over_dim = product_mat.diag().iter().sum::<Complex64>() / dim;

            if let Some(phase_update) = average_gate_fidelity_below_tol(tr_over_dim, dim, tol) {
                return Ok((true, None, phase_update));
            }
        }
    }

    // Could not merge.
    Ok((false, None, 0.))
}

/// Returns whether qubits/clbits for one instruction are fully disjoint from qubit/clbits of
/// another instruction.
#[inline]
fn disjoint_instructions(
    qargs1: &[Qubit],
    cargs1: &[Clbit],
    qargs2: &[Qubit],
    cargs2: &[Clbit],
) -> bool {
    !qargs1.iter().any(|e| qargs2.contains(e)) && !cargs1.iter().any(|e| cargs2.contains(e))
}

#[pyfunction]
#[pyo3(name = "commutative_optimization")]
#[pyo3(signature = (dag, commutation_checker, approximation_degree=1., matrix_max_num_qubits=0))]
pub fn run_commutative_optimization(
    dag: &mut DAGCircuit,
    commutation_checker: &mut CommutationChecker,
    approximation_degree: f64,
    matrix_max_num_qubits: u32,
) -> PyResult<Option<DAGCircuit>> {
    let tol = 1e-12_f64.max(1. - approximation_degree);

    // Create output DAG.
    // We will use it to intern qubits of canonicalized instructions.
    // (In theory, we could also change qubits when merging instructions, however
    // this does not happen right now).
    let mut new_dag = dag.copy_empty_like_with_same_capacity(VarsMode::Alike, BlocksMode::Keep)?;

    let node_indices = dag.topological_op_nodes(false)?.collect::<Vec<_>>();
    let num_nodes = node_indices.len();

    let mut node_actions: Vec<NodeAction> = vec![NodeAction::Keep; num_nodes];
    let mut new_global_phase = dag.global_phase().clone();

    let mut modified: bool = false;

    for idx1 in 0..num_nodes {
        let node_index1 = node_indices[idx1];
        let instr1 = dag[node_index1].unwrap_operation();

        // For now, assume that control-flow operations do not commute with anything.
        if instr1.op.try_control_flow().is_some() {
            continue;
        }

        if let Some((new_instruction, phase_update)) = canonicalize(&mut new_dag, instr1) {
            node_actions[idx1] = NodeAction::Canonical(new_instruction, phase_update);
        }

        let (instr1, extraphase1) = match &node_actions[idx1] {
            NodeAction::Replace(instruction) => (instruction, Param::Float(0.)),
            NodeAction::Keep => (instr1, Param::Float(0.)),
            NodeAction::Canonical(instruction, phase) => (instruction, phase.clone()),
            NodeAction::Drop => {
                unreachable!("The current instruction should not be deleted.")
            }
        };

        let qargs1: &[Qubit] = new_dag.get_qargs(instr1.qubits);
        let cargs1: &[Clbit] = new_dag.get_cargs(instr1.clbits);

        for idx2 in (0..idx1).rev() {
            let node_index2 = node_indices[idx2];

            let (instr2, extraphase2) = match &node_actions[idx2] {
                NodeAction::Replace(instruction) => (instruction, Param::Float(0.)),
                NodeAction::Keep => (dag[node_index2].unwrap_operation(), Param::Float(0.)),
                NodeAction::Canonical(instruction, phase) => (instruction, phase.clone()),
                NodeAction::Drop => continue,
            };

            // For now, assume that control-flow operations do not commute with anything.
            if instr2.op.try_control_flow().is_some() {
                break;
            }

            let qargs2: &[Qubit] = new_dag.get_qargs(instr2.qubits);
            let cargs2: &[Clbit] = new_dag.get_cargs(instr2.clbits);

            // If the two sets of qubit/clbits instructions are fully disjoint, we assume
            // that the instructions cannot be merged and also that the instructions commute.
            if disjoint_instructions(qargs1, cargs1, qargs2, cargs2) {
                continue;
            }

            let (can_be_merged, merged_instruction, phase_update) =
                try_merge(&new_dag, instr1, instr2, tol, matrix_max_num_qubits)?;

            if can_be_merged {
                if let Some(merged_instruction) = merged_instruction {
                    node_actions[idx1] = NodeAction::Replace(merged_instruction);
                } else {
                    node_actions[idx1] = NodeAction::Drop;
                }

                node_actions[idx2] = NodeAction::Drop;
                new_global_phase = radd_param(new_global_phase, Param::Float(phase_update));
                new_global_phase = radd_param(new_global_phase, extraphase1.clone());
                new_global_phase = radd_param(new_global_phase, extraphase2.clone());

                modified = true;
                break;
            }

            if !commute(
                &new_dag,
                instr1,
                instr2,
                approximation_degree,
                matrix_max_num_qubits,
                commutation_checker,
            )? {
                break;
            }
        }
    }

    if !modified {
        // Nothing was changed
        return Ok(None);
    }

    new_dag.set_global_phase(new_global_phase)?;

    for idx in 0..num_nodes {
        match &node_actions[idx] {
            NodeAction::Drop => {}
            NodeAction::Keep | NodeAction::Canonical(_, _) => {
                new_dag.push_back(dag[node_indices[idx]].unwrap_operation().clone())?;
            }
            NodeAction::Replace(instruction) => {
                new_dag.push_back(instruction.clone())?;
            }
        }
    }

    Ok(Some(new_dag))
}

pub fn commutative_optimization_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(run_commutative_optimization))?;
    Ok(())
}

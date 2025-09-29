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

use ndarray::ArrayView2;
use num_complex::Complex64;
use num_complex::ComplexFloat;
use pyo3::prelude::*;
use pyo3::{pyfunction, wrap_pyfunction, Bound, PyResult};
use smallvec::smallvec;

use crate::commutation_checker::{get_matrix, CommutationChecker};
use crate::gate_metrics::rotation_trace_and_dim;
use qiskit_circuit::circuit_instruction::OperationFromPython;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::imports;
use qiskit_circuit::operations::{
    multiply_param, radd_param, Operation, OperationRef, Param, StandardGate,
};
use qiskit_circuit::packed_instruction::PackedInstruction;
use qiskit_circuit::VarsMode;

/// Check if the given matrix is equivalent to identity up to a global phase, up to
/// the specified tolerance `tol`. If this is the case, return the tuple
/// `(true, global_phase)`, and if not, return the tuple `(false, 0.)`.
fn is_mat_identity_equiv(mat: ArrayView2<Complex64>, tol: f64) -> (bool, f64) {
    let dim = mat.shape()[0] as f64;
    let tr_over_dim = mat.diag().iter().sum::<Complex64>() / dim;
    let f_pro = tr_over_dim.abs().powi(2);
    let gate_fidelity = (dim * f_pro + 1.) / (dim + 1.);

    if (1. - gate_fidelity).abs() < tol {
        (true, tr_over_dim.arg())
    } else {
        (false, 0.)
    }
}

/// Check if the given operation is equivalent to identity up to a global phase, up to
/// the specified tolerance `tol`. If this is the case, return the tuple
/// `(true, global_phase)`, and if not, return the tuple `(false, 0.)`.
fn is_identity_equiv(inst: &PackedInstruction, tol: f64, max_qubits: u32) -> PyResult<(bool, f64)> {
    if inst.is_parameterized() {
        // Skip parameterized gates
        return Ok((false, 0.));
    }

    let view = inst.op.view();

    // Handle standard gates
    if let OperationRef::StandardGate(gate) = view {
        let (tr_over_dim, dim) = match gate {
            StandardGate::RX
            | StandardGate::RY
            | StandardGate::RZ
            | StandardGate::Phase
            | StandardGate::RXX
            | StandardGate::RYY
            | StandardGate::RZX
            | StandardGate::RZZ
            | StandardGate::CRX
            | StandardGate::CRY
            | StandardGate::CRZ
            | StandardGate::CPhase => {
                if let Param::Float(angle) = inst.params_view()[0] {
                    let (tr_over_dim, dim) = rotation_trace_and_dim(gate, angle).expect(
                        "Since only supported rotation gates are given, the result is not None",
                    );
                    (tr_over_dim, dim)
                } else {
                    return Ok((false, 0.));
                }
            }
            _ => {
                if let Some(matrix) = get_matrix(&inst.op.view(), inst.params_view()) {
                    let dim = matrix.shape()[0] as f64;
                    let tr_over_dim = matrix.diag().iter().sum::<Complex64>() / dim;
                    (tr_over_dim, dim)
                } else {
                    return Ok((false, 0.));
                }
            }
        };

        let f_pro = tr_over_dim.abs().powi(2);
        let gate_fidelity = (dim * f_pro + 1.) / (dim + 1.);
        if (1. - gate_fidelity).abs() < tol {
            return Ok((true, tr_over_dim.arg()));
        } else {
            return Ok((false, 0.));
        }
    }

    // Perform matrix-based check.
    if inst.op.num_qubits() <= max_qubits {
        if let Some(matrix) = get_matrix(&view, inst.params_view()) {
            return Ok(is_mat_identity_equiv(matrix.view(), tol));
        }
    }

    // Special handling for large pauli rotation gates.
    if let OperationRef::Gate(py_gate) = view {
        let result = Python::attach(|py| -> PyResult<Option<(Complex64, usize)>> {
            let result = imports::PAULI_ROTATION_TRACE_AND_DIM
                .get_bound(py)
                .call1((py_gate.gate.clone_ref(py),))?
                .extract()?;
            Ok(result)
        })?;

        if let Some((tr_over_dim, dim)) = result {
            let f_pro = tr_over_dim.abs().powi(2);
            let gate_fidelity = (dim as f64 * f_pro + 1.) / (dim as f64 + 1.);
            if (1. - gate_fidelity).abs() < tol {
                return Ok((true, tr_over_dim.arg()));
            } else {
                return Ok((false, 0.));
            }
        }
    }

    Ok((false, 0.))
}

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
fn compare_params(params1: &[Param], params2: &[Param]) -> bool {
    if params1.len() != params2.len() {
        return false;
    }

    for (p1, p2) in params1.iter().zip(params2.iter()) {
        match p1.eq(p2) {
            Ok(true) => continue,
            Ok(false) => return false,
            Err(_) => return false,
        }
    }

    true
}

/// List of 2-qubit symmetric gates, that is `G(q1, q2) = G(q2, q1)`.
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
                let canonical_instruction = PackedInstruction::from_standard_gate(
                    standard_gate,
                    inst.params.clone(),
                    sorted_qubits,
                );
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
/// * `max_qubits`: The maximum number of qubits to use for more expensive
///   matrix-based checks.
fn commute(
    dag: &DAGCircuit,
    inst1: &PackedInstruction,
    inst2: &PackedInstruction,
    approximation_degree: f64,
    max_qubits: u32,
    commutation_checker: &mut CommutationChecker,
) -> bool {
    let op1 = inst1.op.view();
    let op2 = inst2.op.view();
    let params1 = inst1.params_view();
    let params2 = inst2.params_view();
    let qargs1 = dag.get_qargs(inst1.qubits);
    let qargs2 = dag.get_qargs(inst2.qubits);
    let cargs1 = dag.get_cargs(inst1.clbits);
    let cargs2 = dag.get_cargs(inst2.clbits);

    commutation_checker
        .commute(
            &op1,
            params1,
            qargs1,
            cargs1,
            &op2,
            params2,
            qargs2,
            cargs2,
            max_qubits,
            approximation_degree,
        )
        .expect("Commutation checker should work")
}

/// Merge two instructions.
///
/// The two instructions have already been canonicalized.
///
/// Arguments:
///
/// * `dag`: The output [DAGCircuit] that contains all interned qubits.
/// * `tol`: Specifies tolerance.
/// * `max_qubits`: The maximum number of qubits to use for more expensive
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
    max_qubits: u32,
) -> PyResult<(bool, Option<PackedInstruction>, f64)> {
    if inst1.op.num_qubits() != inst2.op.num_qubits() {
        return Ok((false, None, 0.));
    }

    let params1 = inst1.params_view();
    let params2 = inst2.params_view();
    let qargs1 = dag.get_qargs(inst1.qubits);
    let qargs2 = dag.get_qargs(inst2.qubits);
    let cargs1 = dag.get_cargs(inst1.clbits);
    let cargs2 = dag.get_cargs(inst2.clbits);

    if (qargs1 != qargs2) || (cargs1 != cargs2) {
        return Ok((false, None, 0.));
    }

    // Check: both instructions are standard gates which cancel out.
    if let (OperationRef::StandardGate(gate1), OperationRef::StandardGate(gate2)) =
        (inst1.op.view(), inst2.op.view())
    {
        // Handle the case when both are standard gates
        if let Some((gate1inv, params1inv)) = gate1.inverse(params1) {
            if (gate1inv == gate2) && compare_params(&params1inv, params2) {
                return Ok((true, None, 0.));
            }
        }
    }

    // Check: can merge special standard gates (currently RZ and RX rotations, but we
    // should include more gates).
    let merged_gate = match (inst1.op.view(), inst2.op.view()) {
        (
            OperationRef::StandardGate(StandardGate::RZ),
            OperationRef::StandardGate(StandardGate::RZ),
        ) => Some(StandardGate::RZ),
        (
            OperationRef::StandardGate(StandardGate::RX),
            OperationRef::StandardGate(StandardGate::RX),
        ) => Some(StandardGate::RX),
        _ => None,
    };
    if let Some(merged_gate) = merged_gate {
        let merged_param = radd_param(params1[0].clone(), params2[0].clone());
        let params = Some(Box::new(smallvec![merged_param]));
        let merged_instruction =
            PackedInstruction::from_standard_gate(merged_gate, params, inst1.qubits);
        let (can_be_removed, phase_update) =
            is_identity_equiv(&merged_instruction, tol, max_qubits)?;

        if can_be_removed {
            return Ok((true, None, phase_update));
        } else {
            return Ok((true, Some(merged_instruction), 0.));
        }
    }

    // Matrix-based check: the product matrix is equivalent to identity.
    if inst1.op.num_qubits() <= max_qubits {
        let view1 = inst1.op.view();
        let view2 = inst2.op.view();

        if let (Some(matrix1), Some(matrix2)) = (
            get_matrix(&view1, inst1.params_view()),
            get_matrix(&view2, inst2.params_view()),
        ) {
            let product_mat = matrix1.dot(&matrix2);
            let (can_be_removed, phase_update) = is_mat_identity_equiv(product_mat.view(), tol);
            if can_be_removed {
                return Ok((true, None, phase_update));
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

                    let merged_params = Some(Box::new(smallvec![instr.params[0].clone()]));
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
                let (can_be_removed, phase_update) =
                    is_identity_equiv(&merged_instruction, tol, max_qubits)?;
                if can_be_removed {
                    return Ok((true, None, phase_update));
                } else {
                    return Ok((true, Some(merged_instruction), 0.));
                }
            } else {
                return Ok((false, None, 0.));
            }
        }
    }

    // Could not merge.
    Ok((false, None, 0.))
}

#[pyfunction]
#[pyo3(name = "commutative_optimization")]
#[pyo3(signature = (dag, commutation_checker, approximation_degree=1., max_qubits=4))]
pub fn run_commutative_optimization(
    dag: &mut DAGCircuit,
    commutation_checker: &mut CommutationChecker,
    approximation_degree: f64,
    max_qubits: u32,
) -> PyResult<Option<DAGCircuit>> {
    let tol = 1e-12_f64.max(1. - approximation_degree);

    // Create output DAG.
    // We will use it to intern qubits of canonicalized instructions.
    // (In theory, we could also change qubits when merging instructions, however
    // this does not happen right now).
    let mut new_dag = dag.copy_empty_like(VarsMode::Alike)?;

    let node_indices = dag.topological_op_nodes()?.collect::<Vec<_>>();
    let num_nodes = node_indices.len();

    let mut node_actions: Vec<NodeAction> = vec![NodeAction::Keep; num_nodes];
    let mut new_global_phase = dag.global_phase().clone();

    let mut modified: bool = false;

    for idx1 in 0..num_nodes {
        let node_index1 = node_indices[idx1];
        let instr1 = dag[node_index1].unwrap_operation();

        let (can_be_removed, phase_update) = is_identity_equiv(instr1, tol, max_qubits)?;
        if can_be_removed {
            node_actions[idx1] = NodeAction::Drop;
            new_global_phase = radd_param(new_global_phase, Param::Float(phase_update));
            modified = true;
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

        for idx2 in (0..idx1).rev() {
            let node_index2 = node_indices[idx2];

            let (instr2, extraphase2) = match &node_actions[idx2] {
                NodeAction::Replace(instruction) => (instruction, Param::Float(0.)),
                NodeAction::Keep => (dag[node_index2].unwrap_operation(), Param::Float(0.)),
                NodeAction::Canonical(instruction, phase) => (instruction, phase.clone()),
                NodeAction::Drop => continue,
            };

            let (can_be_merged, merged_instruction, phase_update) =
                try_merge(&new_dag, instr1, instr2, tol, max_qubits)?;

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
                max_qubits,
                commutation_checker,
            ) {
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
            NodeAction::Keep => {
                new_dag.push_back(dag[node_indices[idx]].unwrap_operation().clone())?;
            }
            NodeAction::Canonical(_, _) => {
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

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

use pyo3::prelude::*;
use pyo3::{pyfunction, wrap_pyfunction, Bound, PyResult};
use smallvec::smallvec;

use crate::commutation_checker::CommutationChecker;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::operations::{radd_param, multiply_param, Operation, OperationRef, Param, StandardGate};
use qiskit_circuit::packed_instruction::PackedInstruction;
use qiskit_circuit::VarsMode;

// the maximum number of qubits we check commutativity for
const MAX_NUM_QUBITS: u32 = 3;

const APPROXIMATION_DEGREE: f64 = 1.;

/// RX and RZ rotation gates with smaller angles are considered as identity. 
// ToDo: should we update this precision?
const CUTOFF_PRECISION: f64 = 1e-5;




/// Holds the action for each node in the original DAGCircuit
// ToDo: see if we should store something other than PackedInstruction
#[derive(Clone, Debug)]
enum NodeAction {
    /// The node's instruction is unchanged, and can be retrieved from the circuit.
    Keep,
    /// The node's instruction has been removed.
    Drop,
    /// The node's instruction has been replaced by the current instruction.
    Replace(PackedInstruction),
}

/// Returns true if the two parameter lists are equal
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

/// Computes the canonical representative of a packed instruction, replacing all types of Z-rotations
/// by RZ-gates and all types of X-rotations by RX-gates (including the global phase update).
fn canonicalize(inst: &PackedInstruction) -> Option<(PackedInstruction, Param)> {
    let (gate, param, phase_update) = match inst.op.view() {
        OperationRef::StandardGate(StandardGate::Phase) | OperationRef::StandardGate(StandardGate::U1) => (
            StandardGate::RZ,
            inst.params_view()[0].clone(),
            multiply_param(&inst.params_view()[0], 0.5),
        ),
        OperationRef::StandardGate(StandardGate::Z) => (
            StandardGate::RZ,
            Param::Float(PI),
            Param::Float(PI / 2.),
        ),
        OperationRef::StandardGate(StandardGate::S) => (
            StandardGate::RZ,
            Param::Float(PI / 2.),
            Param::Float(PI / 4.),
        ),
        OperationRef::StandardGate(StandardGate::Sdg) => (
            StandardGate::RZ,
            Param::Float(-PI / 2.),
            Param::Float(-PI / 4.),
        ),
        OperationRef::StandardGate(StandardGate::T) => (
            StandardGate::RZ,
            Param::Float(PI / 4.),
            Param::Float(PI / 8.),
        ),
        OperationRef::StandardGate(StandardGate::Tdg) => (
            StandardGate::RZ,
            Param::Float(-PI / 4.),
            Param::Float(-PI / 8.),
        ),
        OperationRef::StandardGate(StandardGate::X) => (
            StandardGate::RX,
            Param::Float(PI),
            Param::Float(PI / 2.),
        ),
        OperationRef::StandardGate(StandardGate::SX) => (
            StandardGate::RX,
            Param::Float(PI / 2.),
            Param::Float(PI / 4.),
        ),
        OperationRef::StandardGate(StandardGate::SXdg) => (
            StandardGate::RX,
            Param::Float(-PI / 2.),
            Param::Float(-PI / 4.),
        ),
        _ => {
            return None;
        }
    };
    let params = Some(Box::new(smallvec![param]));
    Some((
        PackedInstruction::from_standard_gate(gate, params, inst.qubits),
        phase_update,
    ))
}

// Return true + global phase update if two instructions cancel out
// symmetric cases?
// handle more general gates, e.g. PauliEvo
// todo: handle up to phase, like CIC
// todo: handle symmetric gates, like rxx
fn can_cancel(dag: &DAGCircuit, inst1: &PackedInstruction, inst2: &PackedInstruction) -> bool {
    // if both standard gates, check inverse...
    // if not standard gates, also check inst1.inverse() == inst2
    // if matrix_based check (& have matrices), multiply matrices, see if up to phase
    // ALSO CHECK COMMUTATIVE_CANCELLATION

    let op1 = inst1.op.view();
    let op2 = inst2.op.view();
    let params1 = inst1.params_view();
    let params2 = inst2.params_view();
    let qargs1 = dag.get_qargs(inst1.qubits);
    let qargs2 = dag.get_qargs(inst2.qubits);
    let cargs1 = dag.get_cargs(inst1.clbits);
    let cargs2 = dag.get_cargs(inst2.clbits);

    if (qargs1 != qargs2) || (cargs1 != cargs2) {
        return false;
    }

    if let (OperationRef::StandardGate(gate1), OperationRef::StandardGate(gate2)) = (op1, op2) {
        // Handle the case when both are standard gates
        if let Some((gate1inv, params1inv)) = gate1.inverse(params1) {
            if (gate1inv == gate2) && compare_params(&*params1inv, params2) {
                println!("Can cancel {:?} and {:?}", inst1.op.name(), inst2.op.name());
                return true;
            }
        }
    }

    false
}

// Return true + new instruction if two instructions can be merged
// # standard Pauli rotations to merge
// symmetric_rotations = {"p", "rx", "ry", "rz", "rxx", "ryy", "rzz"}
// asymmetric_rotations = {"crx", "cry", "crz", "cp", "rzx"}
// todo: handle merging symmetric gates
// assumption: gates have already beeen canonicalized
// todo: check if merges are close to id
fn can_merge(
    dag: &DAGCircuit,
    inst1: &PackedInstruction,
    inst2: &PackedInstruction,
) -> Option<PackedInstruction> {
    // examine cases
    let op1 = inst1.op.view();
    let op2 = inst2.op.view();
    let params1 = inst1.params_view();
    let params2 = inst2.params_view();
    let qargs1 = dag.get_qargs(inst1.qubits);
    let qargs2 = dag.get_qargs(inst2.qubits);
    let cargs1 = dag.get_cargs(inst1.clbits);
    let cargs2 = dag.get_cargs(inst2.clbits);

    if (qargs1 != qargs2) || (cargs1 != cargs2) {
        return None;
    }

    // in every case (right now) merged params is the sum
    // more cases!
    let merged_gate = match (op1, op2) {
        (
            OperationRef::StandardGate(StandardGate::RZ),
            OperationRef::StandardGate(StandardGate::RZ),
        ) => StandardGate::RZ,
        (
            OperationRef::StandardGate(StandardGate::RX),
            OperationRef::StandardGate(StandardGate::RX),
        ) => StandardGate::RX,
        _ => {
            return None;
        }
    };

    // WHY DOES RADD_PARAM REQUIRE CLONING?
    let merged_param = radd_param(params1[0].clone(), params2[0].clone());

    let params = Some(Box::new(smallvec![merged_param]));
    Some(PackedInstruction::from_standard_gate(
        merged_gate,
        params,
        inst1.qubits,
    ))
}

/// Check if the instruction is close to the identity up to the global phase,
/// and thus can be removed.
// ToDo: handle all other standard rotation gates.
// ToDo; handle PauliEvoGate
fn can_remove(instr: &PackedInstruction) -> (bool, f64) {
    match instr.op.view() {
        OperationRef::StandardGate(StandardGate::RZ) | OperationRef::StandardGate(StandardGate::RX) => {
            match instr.params_view()[0] {
                Param::Float(angle) => {
                    let pi_multiple = angle / PI;
                    let mod4 = pi_multiple.rem_euclid(4.);
                    
                    if mod4 < CUTOFF_PRECISION || (4. - mod4) < CUTOFF_PRECISION {
                        // if the angle is close to a 4-pi multiple (from above or below), then the
                        // operator is equal to the identity
                        (true, 0.)
                    } 
                    else if (mod4 - 2.).abs() < CUTOFF_PRECISION {
                        // a 2-pi multiple has a phase of pi: RX(2pi) = RZ(2pi) = -I = I exp(i pi)
                        (true, PI)
                    }
                    else {
                        (false, 0.)
                    }
                },
                _ => { (false, 0.) }
            }
        },
        _ => { (false, 0.) }
    }
}

    
// Return true if two instructions commute
fn commute(
    dag: &DAGCircuit,
    commutation_checker: &mut CommutationChecker,
    inst1: &PackedInstruction,
    inst2: &PackedInstruction,
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
            MAX_NUM_QUBITS,
            APPROXIMATION_DEGREE,
        )
        .expect("Commutation checker should work")
}

// ToDo:
// uptophase inverse check -- as per CIC?
// ToDo: should we check if the gate in the circuit can be already removed?
// (ie apply can_remove on gates, not only on merged gates)
#[pyfunction]
#[pyo3(name = "commutation_optimization")]
pub fn run_commutation_optimization(
    dag: &mut DAGCircuit,
    commutation_checker: &mut CommutationChecker,
) -> PyResult<Option<DAGCircuit>> {
    println!("Running commutation optimization pass!");

    let node_indices = dag.topological_op_nodes()?.collect::<Vec<_>>();
    let num_nodes = node_indices.len();

    let mut node_actions: Vec<NodeAction> = vec![NodeAction::Keep; num_nodes];
    let mut new_global_phase = dag.global_phase().clone();

    let mut modified: bool = false;

    for idx1 in 0..num_nodes {
        let node_index1 = node_indices[idx1];
        let instr1 = dag[node_index1].unwrap_operation();

        if let Some((new_instruction, phase_update)) = canonicalize(&instr1) {
            node_actions[idx1] = NodeAction::Replace(new_instruction); // todo: remove this clone
                                                                       // see if can use add_param; i.e. if always have f64
            new_global_phase = radd_param(new_global_phase, phase_update);
        }

        let instr1 = match &node_actions[idx1] {
            NodeAction::Replace(instruction) => &instruction,
            NodeAction::Keep => instr1,
            NodeAction::Drop => {
                unreachable!("The current instruction could not possibly have been deleted yet")
            }
        };

        for idx2 in (0..idx1).rev() {
            let node_index2 = node_indices[idx2];

            let instr2 = match &node_actions[idx2] {
                NodeAction::Replace(instruction) => &instruction,
                NodeAction::Keep => dag[node_index2].unwrap_operation(),
                NodeAction::Drop => continue,
            };

            if can_cancel(dag, instr1, instr2) {
                //   mark both as canceled; exit from the inner loop
                node_actions[idx1] = NodeAction::Drop;
                node_actions[idx2] = NodeAction::Drop;
                modified = true;
                break;
            }

            if let Some(merged_instruction) = can_merge(dag, instr1, instr2) {
                
                let (can_be_removed, phase_update) = can_remove(&merged_instruction);
                if can_be_removed {
                    node_actions[idx1] = NodeAction::Drop;
                    new_global_phase = radd_param(new_global_phase, Param::Float(phase_update));
                } else {
                    node_actions[idx1] = NodeAction::Replace(merged_instruction);
                }
                node_actions[idx2] = NodeAction::Drop;
                modified = true;
                break;
            }

            if !commute(dag, commutation_checker, instr1, instr2) {
                break;
            }
        }
    }

    if !modified {
        // Nothing was changed
        return Ok(None);
    }

    // Create new DAG
    let mut new_dag = dag.copy_empty_like(VarsMode::Alike)?;
    new_dag.set_global_phase(new_global_phase)?;

    for idx in 0..num_nodes {
        match &node_actions[idx] {
            NodeAction::Drop => {}
            NodeAction::Keep => {
                new_dag.push_back(dag[node_indices[idx]].unwrap_operation().clone())?;
            }
            NodeAction::Replace(instruction) => {
                new_dag.push_back(instruction.clone())?;
            }
        }
    }

    Ok(Some(new_dag))
}

pub fn commutation_optimization_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(run_commutation_optimization))?;
    Ok(())
}

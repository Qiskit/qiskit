// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use smallvec::smallvec;
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, FRAC_PI_8, PI};

use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType};
use qiskit_circuit::imports::PAULI_EVOLUTION_GATE;
use qiskit_circuit::instruction::Parameters;
use qiskit_circuit::operations::{
    Operation, OperationRef, Param, PauliProductMeasurement, PyInstruction, PyOperationTypes,
    StandardGate, add_param, multiply_param, radd_param,
};
use qiskit_circuit::{BlocksMode, Qubit, VarsMode};
use qiskit_quantum_info::sparse_observable::PySparseObservable;

use crate::TranspilerError;

type GateToPBCType<'a> = (&'a [(&'a str, f64, &'a [u32])], f64);
type GateToPBCVec<'a> = (Vec<(&'static str, Param, &'static [u32])>, Param);

/// Map gates to a list of equivalent Pauli rotations and a global phase.
/// Each element of the list is of the form ((Pauli string, phase rescale factor, [qubit indices]), global_phase).
/// For gates that didn't have a phase (e.g. X)
/// the phase rescale factor is simply the phase of the rotation gate. The convention is
/// `original_gate = PauliEvolutionGate(pauli, phase) * e^{i global_phase * phase}`
static STANDARD_GATE_SUBSTITUTIONS: [Option<GateToPBCType>; 52] = [
    None,                                                                 // GlobalPhase
    Some((&[("Y", FRAC_PI_4, &[0]), ("X", FRAC_PI_2, &[0])], FRAC_PI_2)), // H
    Some((&[("I", 0.0, &[0])], 0.0)),                                     // I
    Some((&[("X", FRAC_PI_2, &[0])], FRAC_PI_2)),                         // X
    Some((&[("Y", FRAC_PI_2, &[0])], FRAC_PI_2)),                         // Y
    Some((&[("Z", FRAC_PI_2, &[0])], FRAC_PI_2)),                         // Z
    Some((&[("Z", 0.5, &[0])], 0.5)),                                     // Phase
    None,                                                                 // R
    Some((&[("X", 0.5, &[0])], 0.0)),                                     // RX
    Some((&[("Y", 0.5, &[0])], 0.0)),                                     // RY
    Some((&[("Z", 0.5, &[0])], 0.0)),                                     // RZ
    Some((&[("Z", FRAC_PI_4, &[0])], FRAC_PI_4)),                         // S
    Some((&[("Z", -FRAC_PI_4, &[0])], -FRAC_PI_4)),                       // Sdg
    Some((&[("X", FRAC_PI_4, &[0])], FRAC_PI_4)),                         // SX
    Some((&[("X", -FRAC_PI_4, &[0])], -FRAC_PI_4)),                       // SXdg
    Some((&[("Z", FRAC_PI_8, &[0])], FRAC_PI_8)),                         // T
    Some((&[("Z", -FRAC_PI_8, &[0])], -FRAC_PI_8)),                       // Tdg
    None,                                                                 // U
    Some((&[("Z", 0.5, &[0])], 0.5)),                                     // U1
    None,                                                                 // U2
    None,                                                                 // U3
    Some((
        &[
            ("X", -1.0 * FRAC_PI_4, &[1]),
            ("Z", -1.0 * FRAC_PI_8, &[1]),
            ("XZ", FRAC_PI_4, &[0, 1]),
            ("Z", -FRAC_PI_4, &[0]),
            ("Y", -FRAC_PI_8, &[1]),
        ],
        7.0 * FRAC_PI_4,
    )), // CH
    Some((
        &[
            ("XZ", FRAC_PI_4, &[0, 1]),
            ("Z", -FRAC_PI_4, &[0]),
            ("X", -FRAC_PI_4, &[1]),
        ],
        -FRAC_PI_4,
    )), // CX
    Some((
        &[
            ("YZ", FRAC_PI_4, &[0, 1]),
            ("Z", -FRAC_PI_4, &[0]),
            ("Y", -FRAC_PI_4, &[1]),
        ],
        -FRAC_PI_4,
    )), // CY
    Some((
        &[
            ("ZZ", FRAC_PI_4, &[0, 1]),
            ("Z", -FRAC_PI_4, &[0]),
            ("Z", -FRAC_PI_4, &[1]),
        ],
        -FRAC_PI_4,
    )), // CZ
    Some((
        &[
            ("XZ", FRAC_PI_4, &[0, 1]),
            ("Z", -FRAC_PI_4, &[0]),
            ("X", -FRAC_PI_4, &[1]),
            ("XZ", FRAC_PI_4, &[1, 0]),
            ("X", -FRAC_PI_4, &[0]),
            ("Z", -FRAC_PI_4, &[1]),
        ],
        -FRAC_PI_2,
    )), // DCX
    Some((
        &[
            ("XZ", -1.0 * FRAC_PI_4, &[0, 1]),
            ("Y", -FRAC_PI_2, &[0]),
            ("Z", FRAC_PI_2, &[1]),
            ("Y", FRAC_PI_2, &[1]),
        ],
        PI,
    )), // ECR
    Some((
        &[
            ("XX", FRAC_PI_4, &[0, 1]),
            ("YY", FRAC_PI_4, &[0, 1]),
            ("ZZ", FRAC_PI_4, &[0, 1]),
        ],
        FRAC_PI_4,
    )), // Swap
    Some((
        &[("XX", -FRAC_PI_4, &[0, 1]), ("YY", -FRAC_PI_4, &[0, 1])],
        0.0,
    )), // ISwap
    Some((
        &[("ZZ", -0.25, &[0, 1]), ("Z", 0.25, &[0]), ("Z", 0.25, &[1])],
        0.25,
    )), // CPhase
    Some((&[("XZ", -0.25, &[0, 1]), ("X", 0.25, &[1])], 0.0)),            // CRX
    Some((&[("YZ", -0.25, &[0, 1]), ("Y", 0.25, &[1])], 0.0)),            // CRY
    Some((&[("ZZ", -0.25, &[0, 1]), ("Z", 0.25, &[1])], 0.0)),            // CRZ
    Some((
        &[
            ("ZZ", -FRAC_PI_8, &[0, 1]),
            ("Z", FRAC_PI_8, &[0]),
            ("Z", FRAC_PI_8, &[1]),
        ],
        FRAC_PI_8,
    )), // CS
    Some((
        &[
            ("ZZ", FRAC_PI_8, &[0, 1]),
            ("Z", -FRAC_PI_8, &[0]),
            ("Z", -FRAC_PI_8, &[1]),
        ],
        -FRAC_PI_8,
    )), // CSdg
    Some((
        &[
            ("XZ", -FRAC_PI_8, &[0, 1]),
            ("Z", FRAC_PI_8, &[0]),
            ("X", FRAC_PI_8, &[1]),
        ],
        FRAC_PI_8,
    )), // CSX
    None,                                                                 // CU
    Some((
        &[("ZZ", -0.25, &[0, 1]), ("Z", 0.25, &[0]), ("Z", 0.25, &[1])],
        0.25,
    )), // CU1
    None,                                                                 // CU3
    Some((&[("XX", 0.5, &[0, 1])], 0.0)),                                 // RXX
    Some((&[("YY", 0.5, &[0, 1])], 0.0)),                                 // RYY
    Some((&[("ZZ", 0.5, &[0, 1])], 0.0)),                                 // RZZ
    Some((&[("XZ", 0.5, &[0, 1])], 0.0)),                                 // RZX
    None,                                                                 // XXMinusYY
    None,                                                                 // XXPlusYY
    None,                                                                 // CCX
    None,                                                                 // CCZ
    None,                                                                 // CSwap
    None,                                                                 // RCCX
    None,                                                                 // C3X
    None,                                                                 // C3SX
    None,                                                                 // RC3X
];

/// Map gates with more than one paramter to a list of equivalent Pauli rotations and a global phase.
/// Each element of the list is of the form ((Pauli string, [phases], [qubit indices]), global_phase).
/// The convention is
/// `original_gate = PauliEvolutionGate(pauli, phase) * e^{i global_phase}`
fn replace_gate_by_pauli_vec(gate: StandardGate, angles: &[Param]) -> GateToPBCVec<'static> {
    match gate {
        StandardGate::U | StandardGate::U3 => (
            vec![
                ("Z", multiply_param(&angles[2], 0.5), &[0]),
                ("Y", multiply_param(&angles[0], 0.5), &[0]),
                ("Z", multiply_param(&angles[1], 0.5), &[0]),
            ],
            multiply_param(&radd_param(angles[1].clone(), angles[2].clone()), 0.5),
        ),
        StandardGate::R => (
            vec![
                (
                    "Z",
                    multiply_param(&add_param(&angles[1], -FRAC_PI_2), -0.5),
                    &[0],
                ),
                ("Y", multiply_param(&angles[0], 0.5), &[0]),
                (
                    "Z",
                    multiply_param(&add_param(&angles[1], -FRAC_PI_2), 0.5),
                    &[0],
                ),
            ],
            Param::Float(0.0),
        ),
        StandardGate::U2 => (
            vec![
                ("Z", multiply_param(&angles[1], 0.5), &[0]),
                ("Y", Param::Float(FRAC_PI_4), &[0]),
                ("Z", multiply_param(&angles[0], 0.5), &[0]),
            ],
            multiply_param(&radd_param(angles[0].clone(), angles[1].clone()), 0.5),
        ),
        StandardGate::CU => (
            vec![
                (
                    "Z",
                    radd_param(
                        radd_param(
                            multiply_param(&angles[3], 0.5),
                            multiply_param(&angles[1], 0.25),
                        ),
                        multiply_param(&angles[2], 0.25),
                    ),
                    &[0],
                ),
                (
                    "Z",
                    radd_param(
                        multiply_param(&angles[2], 0.25),
                        multiply_param(&angles[1], -0.25),
                    ),
                    &[1],
                ),
                ("XZ", Param::Float(FRAC_PI_4), &[0, 1]),
                ("Z", Param::Float(-FRAC_PI_4), &[0]),
                ("X", Param::Float(-FRAC_PI_4), &[1]),
                (
                    "Z",
                    radd_param(
                        multiply_param(&angles[2], -0.25),
                        multiply_param(&angles[1], -0.25),
                    ),
                    &[1],
                ),
                ("Y", multiply_param(&angles[0], -0.25), &[1]),
                ("XZ", Param::Float(FRAC_PI_4), &[0, 1]),
                ("Z", Param::Float(-FRAC_PI_4), &[0]),
                ("X", Param::Float(-FRAC_PI_4), &[1]),
                ("Y", multiply_param(&angles[0], 0.25), &[1]),
                ("Z", multiply_param(&angles[1], 0.5), &[1]),
            ],
            radd_param(
                radd_param(
                    multiply_param(&angles[3], 0.5),
                    multiply_param(&angles[1], 0.25),
                ),
                add_param(&multiply_param(&angles[2], 0.25), -FRAC_PI_2),
            ),
        ),
        StandardGate::CU3 => (
            vec![
                (
                    "Z",
                    radd_param(
                        multiply_param(&angles[1], 0.25),
                        multiply_param(&angles[2], 0.25),
                    ),
                    &[0],
                ),
                (
                    "Z",
                    radd_param(
                        multiply_param(&angles[2], 0.25),
                        multiply_param(&angles[1], -0.25),
                    ),
                    &[1],
                ),
                ("XZ", Param::Float(FRAC_PI_4), &[0, 1]),
                ("Z", Param::Float(-FRAC_PI_4), &[0]),
                ("X", Param::Float(-FRAC_PI_4), &[1]),
                (
                    "Z",
                    radd_param(
                        multiply_param(&angles[2], -0.25),
                        multiply_param(&angles[1], -0.25),
                    ),
                    &[1],
                ),
                ("Y", multiply_param(&angles[0], -0.25), &[1]),
                ("XZ", Param::Float(FRAC_PI_4), &[0, 1]),
                ("Z", Param::Float(-FRAC_PI_4), &[0]),
                ("X", Param::Float(-FRAC_PI_4), &[1]),
                ("Y", multiply_param(&angles[0], 0.25), &[1]),
                ("Z", multiply_param(&angles[1], 0.5), &[1]),
            ],
            radd_param(
                multiply_param(&angles[1], 0.25),
                add_param(&multiply_param(&angles[2], 0.25), -FRAC_PI_2),
            ),
        ),
        StandardGate::XXPlusYY => (
            vec![
                ("Z", multiply_param(&angles[1], 0.5), &[0]),
                ("XX", multiply_param(&angles[0], 0.25), &[0, 1]),
                ("YY", multiply_param(&angles[0], 0.25), &[0, 1]),
                ("Z", multiply_param(&angles[1], -0.5), &[0]),
            ],
            Param::Float(0.0),
        ),
        StandardGate::XXMinusYY => (
            vec![
                ("Z", multiply_param(&angles[1], -0.5), &[0]),
                ("XX", multiply_param(&angles[0], 0.25), &[0, 1]),
                ("YY", multiply_param(&angles[0], -0.25), &[0, 1]),
                ("Z", multiply_param(&angles[1], 0.5), &[0]),
            ],
            Param::Float(0.0),
        ),
        _ => {
            panic!("This is only called for one and two qubit gates with more than one parameter.")
        }
    }
}

fn generate_pauli_evolution_gate(
    py_evo_cls: &Bound<PyAny>,
    paulis: &str,
    time: Param,
    qubits: &[u32],
) -> PyResult<PyOperationTypes> {
    let py_pauli = PySparseObservable::from_label(paulis.chars().collect::<String>().as_str())?;
    let py_evo = py_evo_cls.call1((py_pauli, time))?;
    let py_gate = PyOperationTypes::Gate(PyInstruction {
        qubits: qubits.len() as u32,
        clbits: 0,
        params: 1,
        op_name: "PauliEvolution".to_string(),
        instruction: py_evo.into(),
    });
    Ok(py_gate)
}

#[pyfunction]
#[pyo3(name = "pbc_transformation")]
pub fn py_pbc_transformation(py: Python, dag: &mut DAGCircuit) -> PyResult<DAGCircuit> {
    let mut new_dag = dag.copy_empty_like(VarsMode::Alike, BlocksMode::Drop)?;
    let py_evo_cls = PAULI_EVOLUTION_GATE.get_bound(py);

    // Iterate over nodes in the DAG and collect nodes
    let mut global_phase = Param::Float(0.0);

    for node_index in dag.topological_op_nodes(false) {
        let NodeType::Operation(inst) = &dag[node_index] else {
            unreachable!("dag.topological_op_nodes only returns Operations");
        };
        if (inst.op.name() == "barrier") | (inst.op.name() == "reset") | (inst.op.name() == "delay")
        {
            new_dag.push_back(inst.clone())?;
        } else if inst.op.name() == "measure" {
            let z = vec![true];
            let x = vec![false];
            let neg = false;
            let ppm = PauliProductMeasurement { z, x, neg };
            let ppm_qubits = dag.get_qargs(inst.qubits);
            let ppm_clbits = dag.get_cargs(inst.clbits);

            new_dag.apply_operation_back(
                ppm.into(),
                ppm_qubits,
                ppm_clbits,
                None,
                None,
                #[cfg(feature = "cache_pygates")]
                None,
            )?;
        } else if let OperationRef::StandardGate(gate) = inst.op.view() {
            if (gate.num_qubits() > 2) | (gate.num_qubits() < 1) {
                return Err(TranspilerError::new_err(format!(
                    "Unable to run PBC tranformation as the circuit contains instructions not supported by the pass: {:?}",
                    gate.name()
                )));
            }
            // handling only 1-qubit and 2-qubit gates with no parameter or with a single parameter
            if gate.num_params() <= 1 {
                if let Some((sequence, global_phase_update)) =
                    STANDARD_GATE_SUBSTITUTIONS[gate as usize]
                {
                    let angle: Param = if gate.num_params() == 1 {
                        inst.params_view()[0].clone()
                    } else {
                        Param::Float(1.0)
                    };
                    for (paulis, phase_rescale, qubits) in sequence {
                        let original_qubits = dag.get_qargs(inst.qubits);
                        let updated_qubits: Vec<Qubit> = qubits
                            .iter()
                            .map(|q| original_qubits[*q as usize])
                            .collect();
                        let time = multiply_param(&angle, *phase_rescale);
                        let py_gate = generate_pauli_evolution_gate(
                            py_evo_cls,
                            paulis,
                            time.clone(),
                            qubits,
                        )?;

                        new_dag.apply_operation_back(
                            py_gate.into(),
                            &updated_qubits,
                            &[],
                            Some(Parameters::Params(smallvec![time])),
                            None,
                            #[cfg(feature = "cache_pygates")]
                            None,
                        )?;
                    }
                    global_phase =
                        radd_param(multiply_param(&angle, global_phase_update), global_phase);
                } else {
                    return Err(TranspilerError::new_err(format!(
                        "Unable to run PBC tranformation as the circuit contains instructions not supported by the pass: {:?}",
                        inst.op.name()
                    )));
                }
            }
            // handling only 1-qubit and 2-qubit gates with more than one parameter
            else if matches!(
                gate,
                StandardGate::U
                    | StandardGate::U2
                    | StandardGate::U3
                    | StandardGate::R
                    | StandardGate::CU
                    | StandardGate::CU3
                    | StandardGate::XXPlusYY
                    | StandardGate::XXMinusYY
            ) {
                let angles = inst.params_view();
                let (sequence, global_phase_update) = replace_gate_by_pauli_vec(gate, angles);
                for (paulis, phase_rescale, qubits) in sequence {
                    let original_qubits = dag.get_qargs(inst.qubits);
                    let updated_qubits: Vec<Qubit> = qubits
                        .iter()
                        .map(|q| original_qubits[*q as usize])
                        .collect();
                    let time = phase_rescale;
                    let py_gate =
                        generate_pauli_evolution_gate(py_evo_cls, paulis, time.clone(), qubits)?;

                    new_dag.apply_operation_back(
                        py_gate.into(),
                        &updated_qubits,
                        &[],
                        Some(Parameters::Params(smallvec![time])),
                        None,
                        #[cfg(feature = "cache_pygates")]
                        None,
                    )?;
                }
                global_phase = radd_param(global_phase, global_phase_update);
            }
        } else {
            return Err(TranspilerError::new_err(format!(
                "Unable to run PBC tranformation as the circuit contains instructions not supported by the pass: {:?}",
                inst.op.name()
            )));
        }
    }

    new_dag.add_global_phase(&global_phase)?;

    Ok(new_dag)
}

pub fn pbc_transformation_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_pbc_transformation))?;
    Ok(())
}

// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
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
use qiskit_circuit::operations::{OperationRef, Param, PyGate, StandardGate};
use qiskit_circuit::{BlocksMode, Qubit, VarsMode};
use qiskit_quantum_info::sparse_observable::PySparseObservable;

type GateToPBCType<'a> = (&'static [(&'static str, f64, &'static [u32])], f64);

/// Map gates to a list of equivalent Pauli rotations and a global phase.
/// Each element of the list is of the form ((Pauli string, phase rescale factor, [qubit indices]), global phase).
/// For gates that didn't have a phase (e.g. X)
/// the phase rescale factor is simply the phase of the rotation gate. The convention is
/// `original_gate = PauliEvolutionGate(pauli, phase) * e^{i global_phase * phase}`
fn replace_gate_by_pauli_rotation(gate: StandardGate) -> GateToPBCType<'static> {
    match gate {
        StandardGate::I => (&[("I", 0.0, &[0])], 0.0),
        StandardGate::X => (&[("X", FRAC_PI_2, &[0])], FRAC_PI_2),
        StandardGate::Y => (&[("Y", FRAC_PI_2, &[0])], FRAC_PI_2),
        StandardGate::Z => (&[("Z", FRAC_PI_2, &[0])], FRAC_PI_2),
        StandardGate::S => (&[("Z", FRAC_PI_4, &[0])], FRAC_PI_4),
        StandardGate::Sdg => (&[("Z", -FRAC_PI_4, &[0])], -FRAC_PI_4),
        StandardGate::T => (&[("Z", FRAC_PI_8, &[0])], FRAC_PI_8),
        StandardGate::Tdg => (&[("Z", -FRAC_PI_8, &[0])], -FRAC_PI_8),
        StandardGate::SX => (&[("X", FRAC_PI_4, &[0])], FRAC_PI_4),
        StandardGate::SXdg => (&[("X", -FRAC_PI_4, &[0])], -FRAC_PI_4),
        StandardGate::H => (
            &[
                ("Z", FRAC_PI_4, &[0]),
                ("X", FRAC_PI_4, &[0]),
                ("Z", FRAC_PI_4, &[0]),
            ],
            FRAC_PI_2,
        ),
        StandardGate::RZ => (&[("Z", 0.5, &[0])], 0.0),
        StandardGate::RX => (&[("X", 0.5, &[0])], 0.0),
        StandardGate::RY => (&[("Y", 0.5, &[0])], 0.0),
        StandardGate::Phase => (&[("Z", 0.5, &[0])], 0.5),
        StandardGate::U1 => (&[("Z", 0.5, &[0])], 0.5),
        StandardGate::CX => (
            &[
                ("XZ", FRAC_PI_4, &[0, 1]),
                ("Z", -FRAC_PI_4, &[0]),
                ("X", -FRAC_PI_4, &[1]),
            ],
            -FRAC_PI_4,
        ),
        StandardGate::CZ => (
            &[
                ("ZZ", FRAC_PI_4, &[0, 1]),
                ("Z", -FRAC_PI_4, &[0]),
                ("Z", -FRAC_PI_4, &[1]),
            ],
            -FRAC_PI_4,
        ),
        StandardGate::CY => (
            &[
                ("YZ", FRAC_PI_4, &[0, 1]),
                ("Z", -FRAC_PI_4, &[0]),
                ("Y", -FRAC_PI_4, &[1]),
            ],
            -FRAC_PI_4,
        ),
        StandardGate::CH => (
            &[
                ("Z", FRAC_PI_2, &[1]),
                ("X", FRAC_PI_4, &[1]),
                ("Z", 3.0 * FRAC_PI_8, &[1]),
                ("XZ", FRAC_PI_4, &[0, 1]),
                ("Z", -FRAC_PI_4, &[0]),
                ("X", -FRAC_PI_4, &[1]),
                ("Z", FRAC_PI_8, &[1]),
                ("X", FRAC_PI_4, &[1]),
            ],
            3.0 * FRAC_PI_4,
        ),
        StandardGate::CS => (
            &[
                ("ZZ", -FRAC_PI_8, &[0, 1]),
                ("Z", FRAC_PI_8, &[0]),
                ("Z", FRAC_PI_8, &[1]),
            ],
            FRAC_PI_8,
        ),
        StandardGate::CSdg => (
            &[
                ("ZZ", FRAC_PI_8, &[0, 1]),
                ("Z", -FRAC_PI_8, &[0]),
                ("Z", -FRAC_PI_8, &[1]),
            ],
            -FRAC_PI_8,
        ),
        StandardGate::CSX => (
            &[
                ("XZ", -FRAC_PI_8, &[0, 1]),
                ("Z", FRAC_PI_8, &[0]),
                ("X", FRAC_PI_8, &[1]),
            ],
            FRAC_PI_8,
        ),
        StandardGate::Swap => (
            &[
                ("XX", FRAC_PI_4, &[0, 1]),
                ("YY", FRAC_PI_4, &[0, 1]),
                ("ZZ", FRAC_PI_4, &[0, 1]),
            ],
            FRAC_PI_4,
        ),
        StandardGate::ISwap => (
            &[("XX", -FRAC_PI_4, &[0, 1]), ("YY", -FRAC_PI_4, &[0, 1])],
            0.0,
        ),
        StandardGate::DCX => (
            &[
                ("XZ", FRAC_PI_4, &[0, 1]),
                ("Z", -FRAC_PI_4, &[0]),
                ("X", -FRAC_PI_4, &[1]),
                ("XZ", FRAC_PI_4, &[1, 0]),
                ("X", -FRAC_PI_4, &[0]),
                ("Z", -FRAC_PI_4, &[1]),
            ],
            -FRAC_PI_2,
        ),
        StandardGate::ECR => (
            &[
                ("Y", -FRAC_PI_2, &[0]),
                ("Z", FRAC_PI_4, &[0]),
                ("X", -FRAC_PI_4, &[1]),
                ("XZ", FRAC_PI_4, &[0, 1]),
                ("Z", -FRAC_PI_4, &[0]),
                ("X", -FRAC_PI_4, &[1]),
            ],
            -2.0 * PI,
        ),
        StandardGate::RZZ => (&[("ZZ", 0.5, &[0, 1])], 0.0),
        StandardGate::RXX => (&[("XX", 0.5, &[0, 1])], 0.0),
        StandardGate::RYY => (&[("YY", 0.5, &[0, 1])], 0.0),
        StandardGate::RZX => (&[("XZ", 0.5, &[0, 1])], 0.0),
        StandardGate::CPhase => (
            &[("ZZ", -0.25, &[0, 1]), ("Z", 0.25, &[0]), ("Z", 0.25, &[1])],
            0.25,
        ),
        StandardGate::CU1 => (
            &[("ZZ", -0.25, &[0, 1]), ("Z", 0.25, &[0]), ("Z", 0.25, &[1])],
            0.25,
        ),
        StandardGate::CRZ => (&[("ZZ", -0.25, &[0, 1]), ("Z", 0.25, &[1])], 0.0),
        StandardGate::CRX => (&[("XZ", -0.25, &[0, 1]), ("X", 0.25, &[1])], 0.0),
        StandardGate::CRY => (&[("YZ", -0.25, &[0, 1]), ("Y", 0.25, &[1])], 0.0),
        _ => unreachable!(
            "This is only called for one and two qubit gates with no paramers or with a single parameter."
        ),
    }
}

#[pyfunction]
#[pyo3(name = "pbc_transformation")]
pub fn py_pbc_transformation(py: Python, dag: &mut DAGCircuit) -> PyResult<DAGCircuit> {
    let mut new_dag = dag.copy_empty_like_with_capacity(0, 0, VarsMode::Alike, BlocksMode::Keep)?;

    // Iterate over nodes in the DAG and collect nodes
    let mut global_phase: f64 = 0.;

    for node_index in dag.topological_op_nodes(false)? {
        if let NodeType::Operation(inst) = &dag[node_index] {
            if let OperationRef::StandardGate(gate) = inst.op.view() {
                // handling only 1-qubit and 2-qubit gates with a single parameter
                if matches!(
                    gate,
                    StandardGate::RX
                        | StandardGate::RY
                        | StandardGate::RZ
                        | StandardGate::Phase
                        | StandardGate::U1
                        | StandardGate::RZZ
                        | StandardGate::RXX
                        | StandardGate::RZX
                        | StandardGate::RYY
                        | StandardGate::CPhase
                        | StandardGate::CU1
                        | StandardGate::CRZ
                        | StandardGate::CRX
                        | StandardGate::CRY
                ) {
                    if let Param::Float(angle) = inst.params_view()[0] {
                        let (sequence, global_phase_update) = replace_gate_by_pauli_rotation(gate);
                        for (paulis, phase_rescale, qubits) in sequence {
                            let original_qubits = dag.get_qargs(inst.qubits);
                            let updated_qubits: Vec<Qubit> = qubits
                                .iter()
                                .map(|q| original_qubits[*q as usize])
                                .collect();
                            let time = phase_rescale * angle;
                            let py_pauli = PySparseObservable::from_label(
                                paulis.chars().collect::<String>().as_str(),
                            )?;
                            let py_evo_cls = PAULI_EVOLUTION_GATE.get_bound(py);
                            let py_evo = py_evo_cls.call1((py_pauli, time))?;
                            let py_gate = PyGate {
                                qubits: qubits.len() as u32,
                                clbits: 0,
                                params: 1,
                                op_name: "PauliEvolution".to_string(),
                                gate: py_evo.into(),
                            };

                            new_dag.apply_operation_back(
                                py_gate.into(),
                                &updated_qubits,
                                &[],
                                Some(Parameters::Params(smallvec![Param::Float(time)])),
                                None,
                                #[cfg(feature = "cache_pygates")]
                                None,
                            )?;
                        }
                        global_phase += global_phase_update * angle;
                    } else {
                        panic!();
                    }
                }
                // handling only 1-qubit and 2-qubit gates with no parameters
                else if matches!(
                    gate,
                    StandardGate::I
                        | StandardGate::X
                        | StandardGate::Y
                        | StandardGate::Z
                        | StandardGate::S
                        | StandardGate::Sdg
                        | StandardGate::T
                        | StandardGate::Tdg
                        | StandardGate::SX
                        | StandardGate::SXdg
                        | StandardGate::H
                        | StandardGate::CX
                        | StandardGate::CZ
                        | StandardGate::CY
                        | StandardGate::CH
                        | StandardGate::CS
                        | StandardGate::CSdg
                        | StandardGate::CSX
                        | StandardGate::Swap
                        | StandardGate::ISwap
                        | StandardGate::DCX
                        | StandardGate::ECR
                ) {
                    let (sequence, global_phase_update) = replace_gate_by_pauli_rotation(gate);
                    for (paulis, phase_rescale, qubits) in sequence {
                        let original_qubits = dag.get_qargs(inst.qubits);
                        let updated_qubits: Vec<Qubit> = qubits
                            .iter()
                            .map(|q| original_qubits[*q as usize])
                            .collect();
                        let time = phase_rescale;
                        let py_pauli = PySparseObservable::from_label(
                            paulis.chars().collect::<String>().as_str(),
                        )?;
                        let py_evo_cls = PAULI_EVOLUTION_GATE.get_bound(py);
                        let py_evo = py_evo_cls.call1((py_pauli, time))?;
                        let py_gate = PyGate {
                            qubits: qubits.len() as u32,
                            clbits: 0,
                            params: 1,
                            op_name: "PauliEvolution".to_string(),
                            gate: py_evo.into(),
                        };

                        new_dag.apply_operation_back(
                            py_gate.into(),
                            &updated_qubits,
                            &[],
                            Some(Parameters::Params(smallvec![Param::Float(*time)])),
                            None,
                            #[cfg(feature = "cache_pygates")]
                            None,
                        )?;
                    }
                    global_phase += global_phase_update;
                }
            } else {
                panic!();
            }
        } else {
            unreachable!();
        }
    }

    new_dag.add_global_phase(&Param::Float(global_phase))?;

    Ok(new_dag)
}

pub fn pbc_transformation_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_pbc_transformation))?;
    Ok(())
}

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

use pyo3::prelude::*;

use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType};
use qiskit_circuit::imports::PAULI_EVOLUTION_GATE;
use qiskit_circuit::operations::{Operation, OperationRef, Param, PyGate, StandardGate};
use qiskit_circuit::packed_instruction::PackedInstruction;
use qiskit_circuit::{Clbit, Qubit, VarsMode};

use qiskit_quantum_info::clifford::Clifford;
use qiskit_quantum_info::sparse_observable::PySparseObservable;

use smallvec::smallvec;
use std::f64::consts::PI;

use crate::TranspilerError;

// List of gate names supported by the pass: the pass is skipped if the circuit
// contains gate names outside of this list.
const SUPPORTED_GATE_NAMES: &[&str; 19] = &[
    "id", "x", "y", "z", "h", "s", "sdg", "sx", "sxdg", "cx", "cz", "cy", "swap", "iswap", "ecr",
    "dcx", "t", "tdg", "rz",
];

// List of rotation gate names: the pass is skipped if the circuit contains
// no gate names in this list.
const ROTATION_GATE_NAMES: &[&str; 3] = &["t", "tdg", "rz"];

/// Expresses a given circuit as a sequence of Pauli rotations followed by a final Clifford operator.
/// Returns the list of rotations in the sparse format: (sign, paulis, indices).
pub fn extract_rotations(
    circuit: &[(&str, Vec<usize>)],
    nqubits: usize,
) -> Vec<(bool, String, Vec<u32>)> {
    let mut clifford = Clifford::identity(nqubits);
    let mut rotations: Vec<(bool, String, Vec<u32>)> = Vec::new();

    for (gate_name, qbits) in circuit.iter() {
        match *gate_name {
            "id" => {}
            "x" => clifford.append_x(qbits[0]),
            "y" => clifford.append_y(qbits[0]),
            "z" => clifford.append_z(qbits[0]),
            "h" => clifford.append_h(qbits[0]),
            "s" => clifford.append_s(qbits[0]),
            "sdg" => clifford.append_sdg(qbits[0]),
            "sx" => clifford.append_sx(qbits[0]),
            "sxdg" => clifford.append_sxdg(qbits[0]),
            "cx" => clifford.append_cx(qbits[0], qbits[1]),
            "cz" => clifford.append_cz(qbits[0], qbits[1]),
            "cy" => clifford.append_cy(qbits[0], qbits[1]),
            "swap" => clifford.append_swap(qbits[0], qbits[1]),
            "iswap" => clifford.append_iswap(qbits[0], qbits[1]),
            "ecr" => clifford.append_ecr(qbits[0], qbits[1]),
            "dcx" => clifford.append_dcx(qbits[0], qbits[1]),
            "rz" => {
                rotations.push(clifford.get_inverse_z(qbits[0]));
            }
            _ => panic!("Unsupported gate {}", gate_name),
        }
    }

    rotations
}

#[pyfunction]
#[pyo3(signature = (dag, fix_clifford=true))]
pub fn run_litinski_transformation(
    py: Python,
    dag: &DAGCircuit,
    fix_clifford: bool,
) -> PyResult<Option<DAGCircuit>> {
    let op_counts = dag.get_op_counts();

    // Skip the pass if there are no rotation gates.
    if op_counts
        .keys()
        .all(|k| !ROTATION_GATE_NAMES.contains(&k.as_str()))
    {
        return Ok(None);
    }

    // Skip the pass if there are unsupported gates.
    if !op_counts
        .keys()
        .all(|k| SUPPORTED_GATE_NAMES.contains(&k.as_str()))
    {
        let unsupported: Vec<_> = op_counts
            .keys()
            .filter(|k| !SUPPORTED_GATE_NAMES.contains(&k.as_str()))
            .collect();

        return Err(TranspilerError::new_err(format!(
            "Unable to run Litinski tranformation as the circuit contains gates not supported by the pass: {:?}",
            unsupported
        )));
    }

    let num_qubits = dag.num_qubits();

    // Turn the Qiskit circuit into a vector of (gate name, qubit indices).
    // Additionally, keep track of the rotation angles, an update to the global phase (produced when
    // converting T/Tdg gates to RZ-rotations), and Clifford gates in the circuit.
    let mut circuit: Vec<(&str, Vec<usize>)> = Vec::new();
    let mut angles: Vec<Param> = Vec::new();
    let mut global_phase_update = 0.;
    let mut clifford_ops: Vec<PackedInstruction> = Vec::new();
    for node_index in dag.topological_op_nodes()? {
        if let NodeType::Operation(inst) = &dag[node_index] {
            let (name, angle, phase_update) = match inst.op.view() {
                OperationRef::StandardGate(StandardGate::T) => {
                    ("rz", Some(Param::Float(PI / 8.)), PI / 8.)
                }
                OperationRef::StandardGate(StandardGate::Tdg) => {
                    ("rz", Some(Param::Float(-PI / 8.0)), -PI / 8.)
                }
                OperationRef::StandardGate(StandardGate::RZ) => {
                    let param = &inst.params_view()[0];
                    ("rz", Some(param.mul_f64(0.5)), 0.)
                }
                _ => (inst.op.name(), None, 0.),
            };

            global_phase_update += phase_update;

            let qubits: Vec<usize> = dag
                .get_qargs(inst.qubits)
                .iter()
                .map(|q| q.index())
                .collect();

            circuit.push((name, qubits));

            if let Some(angle) = angle {
                // This is a rotation, save the angle.
                angles.push(angle);
            } else {
                // This is a Clifford operation, save it.
                clifford_ops.push(inst.clone());
            }
        } else {
            unreachable!("Gate instructions should be either Clifford or T/Tdg/RZ at this point.");
        }
    }

    // Apply the Litinski transformation.
    // This returns a list of rotations with +1/-1 signs. Since we aim to preserve the
    // global phase of the circuit, we ignore the final Clifford operator, and instead
    // append the Clifford gates from the original circuit.
    let rotations = extract_rotations(&circuit, num_qubits);

    let py_evo_cls = PAULI_EVOLUTION_GATE.get_bound(py);
    let no_clbits: Vec<Clbit> = Vec::new();

    let mut new_dag = dag.copy_empty_like(VarsMode::Alike)?;
    new_dag.add_global_phase(&Param::Float(global_phase_update))?;

    // Add Pauli rotation gates to the Qiskit circuit.
    for ((sign, paulis, indices), angle) in rotations.iter().zip(angles) {
        let qubits: Vec<Qubit> = indices.iter().map(|index| Qubit(*index)).collect();

        let py_pauli =
            PySparseObservable::from_label(paulis.chars().rev().collect::<String>().as_str())?;

        let time = if *sign { angle.mul_f64(-1.) } else { angle };
        let py_evo = py_evo_cls.call1((py_pauli, time.clone()))?;
        let py_gate = PyGate {
            qubits: qubits.len() as u32,
            clbits: 0,
            params: 1,
            op_name: "PauliEvolution".to_string(),
            gate: py_evo.into(),
        };

        new_dag.apply_operation_back(
            py_gate.into(),
            &qubits,
            &no_clbits,
            Some(smallvec![time]),
            None,
            #[cfg(feature = "cache_pygates")]
            None,
        )?;
    }

    // Add Clifford gates to the Qiskit circuit (when required).
    if fix_clifford {
        for inst in clifford_ops {
            new_dag.apply_operation_back(
                inst.op,
                dag.get_qargs(inst.qubits),
                dag.get_cargs(inst.clbits),
                inst.params.as_deref().cloned(),
                inst.label.as_ref().map(|x| x.as_ref().clone()),
                #[cfg(feature = "cache_pygates")]
                inst.py_op.get().map(|x| x.clone_ref(py)),
            )?;
        }
    }

    Ok(Some(new_dag))
}

pub fn litinski_transformation_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(run_litinski_transformation))?;
    Ok(())
}

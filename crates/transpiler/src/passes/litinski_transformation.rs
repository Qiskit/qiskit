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
use qiskit_circuit::interner::{Interned, Interner};
use qiskit_circuit::operations::{
    Operation, OperationRef, Param, PyGate, StandardGate, multiply_param,
};
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
fn extract_rotations(
    gate: StandardGate,
    qbits: &Interned<[Qubit]>,
    interner: &Interner<[Qubit]>,
    clifford: &mut Clifford,
) -> Option<(bool, String, Vec<Qubit>)> {
    match gate {
        StandardGate::I => {}
        StandardGate::X => clifford.append_x(interner.get(*qbits)[0].index()),
        StandardGate::Y => clifford.append_y(interner.get(*qbits)[0].index()),
        StandardGate::Z => clifford.append_z(interner.get(*qbits)[0].index()),
        StandardGate::H => clifford.append_h(interner.get(*qbits)[0].index()),
        StandardGate::S => clifford.append_s(interner.get(*qbits)[0].index()),
        StandardGate::Sdg => clifford.append_sdg(interner.get(*qbits)[0].index()),
        StandardGate::SX => clifford.append_sx(interner.get(*qbits)[0].index()),
        StandardGate::SXdg => clifford.append_sxdg(interner.get(*qbits)[0].index()),
        StandardGate::CX => clifford.append_cx(
            interner.get(*qbits)[0].index(),
            interner.get(*qbits)[1].index(),
        ),
        StandardGate::CZ => clifford.append_cz(
            interner.get(*qbits)[0].index(),
            interner.get(*qbits)[1].index(),
        ),
        StandardGate::CY => clifford.append_cy(
            interner.get(*qbits)[0].index(),
            interner.get(*qbits)[1].index(),
        ),
        StandardGate::Swap => clifford.append_swap(
            interner.get(*qbits)[0].index(),
            interner.get(*qbits)[1].index(),
        ),
        StandardGate::ISwap => clifford.append_iswap(
            interner.get(*qbits)[0].index(),
            interner.get(*qbits)[1].index(),
        ),
        StandardGate::ECR => clifford.append_ecr(
            interner.get(*qbits)[0].index(),
            interner.get(*qbits)[1].index(),
        ),
        StandardGate::DCX => clifford.append_dcx(
            interner.get(*qbits)[0].index(),
            interner.get(*qbits)[1].index(),
        ),
        StandardGate::RZ => {
            return Some(clifford.get_inverse_z(interner.get(*qbits)[0].index()));
        }
        _ => panic!("Unsupported gate {}", gate.name()),
    };
    None
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
    let rotation_count = op_counts
        .iter()
        .filter_map(|(k, v)| {
            if ROTATION_GATE_NAMES.contains(&k.as_str()) {
                Some(v)
            } else {
                None
            }
        })
        .sum();
    let clifford_count = dag.size(false)? - rotation_count;

    let num_qubits = dag.num_qubits();

    // Turn the Qiskit circuit into a vector of (gate name, qubit indices).
    // Additionally, keep track of the rotation angles, an update to the global phase (produced when
    // converting T/Tdg gates to RZ-rotations), and Clifford gates in the circuit.
    let mut angles: Vec<Param> = Vec::with_capacity(rotation_count);
    let mut global_phase_update = 0.;
    let mut clifford_ops: Vec<&PackedInstruction> = Vec::with_capacity(clifford_count);
    let mut clifford = Clifford::identity(num_qubits);
    let rotations: Vec<_> = dag
        .topological_op_nodes()?
        .filter_map(|node_index| {
            let NodeType::Operation(inst) = &dag[node_index] else {
                unreachable!(
                    "Gate instructions should be either Clifford or T/Tdg/RZ at this point."
                );
            };
            let (gate, angle, phase_update) = match inst.op.view() {
                OperationRef::StandardGate(StandardGate::T) => {
                    (StandardGate::RZ, Some(Param::Float(PI / 8.)), PI / 8.)
                }
                OperationRef::StandardGate(StandardGate::Tdg) => {
                    (StandardGate::RZ, Some(Param::Float(-PI / 8.0)), -PI / 8.)
                }
                OperationRef::StandardGate(StandardGate::RZ) => {
                    let param = &inst.params_view()[0];
                    (StandardGate::RZ, Some(multiply_param(param, 0.5)), 0.)
                }
                _ => (inst.op.try_standard_gate().unwrap(), None, 0.),
            };

            global_phase_update += phase_update;

            if let Some(angle) = angle {
                // This is a rotation, save the angle.
                angles.push(angle);
            } else {
                // This is a Clifford operation, save it.
                clifford_ops.push(inst);
            }
            extract_rotations(gate, &inst.qubits, dag.qargs_interner(), &mut clifford)
        })
        .collect();

    // Apply the Litinski transformation.
    // This returns a list of rotations with +1/-1 signs. Since we aim to preserve the
    // global phase of the circuit, we ignore the final Clifford operator, and instead
    // append the Clifford gates from the original circuit.

    let py_evo_cls = PAULI_EVOLUTION_GATE.get_bound(py);
    let no_clbits: Vec<Clbit> = Vec::new();

    let new_dag = dag.copy_empty_like(VarsMode::Alike)?;
    let mut new_dag = new_dag.into_builder();
    new_dag.add_global_phase(&Param::Float(global_phase_update))?;

    // Add Pauli rotation gates to the Qiskit circuit.
    for ((sign, paulis, qubits), angle) in rotations.iter().zip(angles) {
        let py_pauli =
            PySparseObservable::from_label(paulis.chars().rev().collect::<String>().as_str())?;

        let time = if *sign {
            multiply_param(&angle, -1.)
        } else {
            angle
        };
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
            qubits,
            &no_clbits,
            Some(smallvec![time]),
            None,
            #[cfg(feature = "cache_pygates")]
            None,
        )?;
    }

    // Add Clifford gates to the Qiskit circuit (when required).
    if fix_clifford {
        for inst in clifford_ops.into_iter() {
            new_dag.push_back(inst.clone())?;
        }
    }

    Ok(Some(new_dag.build()))
}

pub fn litinski_transformation_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(run_litinski_transformation))?;
    Ok(())
}

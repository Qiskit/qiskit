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
use qiskit_circuit::operations::{
    Operation, OperationRef, Param, PauliProductMeasurement, PyGate, StandardGate, multiply_param,
};
use qiskit_circuit::packed_instruction::PackedInstruction;
use qiskit_circuit::{Clbit, Qubit, VarsMode};

use qiskit_quantum_info::clifford::Clifford;
use qiskit_quantum_info::sparse_observable::PySparseObservable;

use smallvec::smallvec;
use std::f64::consts::PI;

use crate::TranspilerError;

// List of gate/instruction names supported by the pass: the pass raises an error if the circuit
// contains instruction with names outside of this list.
static SUPPORTED_INSTRUCTION_NAMES: [&str; 20] = [
    "id", "x", "y", "z", "h", "s", "sdg", "sx", "sxdg", "cx", "cz", "cy", "swap", "iswap", "ecr",
    "dcx", "t", "tdg", "rz", "measure",
];

// List of supported Clifford gate names.
static SUPPORTED_CLIFFORD_NAMES: [&str; 16] = [
    "id", "x", "y", "z", "h", "s", "sdg", "sx", "sxdg", "cx", "cz", "cy", "swap", "iswap", "ecr",
    "dcx",
];

// List of instruction names which are modified by the pass: the pass is skipped if the circuit
// contains no instructions with names in this list.
static HANDLED_INSTRUCTION_NAMES: [&str; 4] = ["t", "tdg", "rz", "measure"];

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
        .all(|k| !HANDLED_INSTRUCTION_NAMES.contains(&k.as_str()))
    {
        return Ok(None);
    }

    // Skip the pass if there are unsupported gates.
    if !op_counts
        .keys()
        .all(|k| SUPPORTED_INSTRUCTION_NAMES.contains(&k.as_str()))
    {
        let unsupported: Vec<_> = op_counts
            .keys()
            .filter(|k| !SUPPORTED_INSTRUCTION_NAMES.contains(&k.as_str()))
            .collect();

        return Err(TranspilerError::new_err(format!(
            "Unable to run Litinski tranformation as the circuit contains instructions not supported by the pass: {:?}",
            unsupported
        )));
    }

    let mut new_dag = dag.copy_empty_like(VarsMode::Alike)?;

    let py_evo_cls = PAULI_EVOLUTION_GATE.get_bound(py);
    let no_clbits: Vec<Clbit> = Vec::new();

    let num_qubits = dag.num_qubits();
    let mut clifford = Clifford::identity(num_qubits);

    // Keep track of the update to the global phase (produced when converting T/Tdg gates
    // to RZ-rotations).
    let mut global_phase_update = 0.;

    // Keep track of the clifford operations in the circuit.
    let mut clifford_ops: Vec<PackedInstruction> = Vec::new();

    // Apply the Litinski transformation: that is, express a given circuit as a sequence of Pauli
    // product rotations and Pauli product measurements, followed by a final Clifford operator.
    for node_index in dag.topological_op_nodes()? {
        // Convert T and Tdg gates to RZ rotations.
        if let NodeType::Operation(inst) = &dag[node_index] {
            let name = inst.op.name();

            let qubits: Vec<usize> = dag
                .get_qargs(inst.qubits)
                .iter()
                .map(|q| q.index())
                .collect();

            match name {
                "id" => {}
                "x" => clifford.append_x(qubits[0]),
                "y" => clifford.append_y(qubits[0]),
                "z" => clifford.append_z(qubits[0]),
                "h" => clifford.append_h(qubits[0]),
                "s" => clifford.append_s(qubits[0]),
                "sdg" => clifford.append_sdg(qubits[0]),
                "sx" => clifford.append_sx(qubits[0]),
                "sxdg" => clifford.append_sxdg(qubits[0]),
                "cx" => clifford.append_cx(qubits[0], qubits[1]),
                "cz" => clifford.append_cz(qubits[0], qubits[1]),
                "cy" => clifford.append_cy(qubits[0], qubits[1]),
                "swap" => clifford.append_swap(qubits[0], qubits[1]),
                "iswap" => clifford.append_iswap(qubits[0], qubits[1]),
                "ecr" => clifford.append_ecr(qubits[0], qubits[1]),
                "dcx" => clifford.append_dcx(qubits[0], qubits[1]),
                "t" | "tdg" | "rz" => {
                    // Convert T and Tdg gates to RZ rotations
                    let (angle, phase_update) = match inst.op.view() {
                        OperationRef::StandardGate(StandardGate::T) => {
                            (Param::Float(PI / 4.), PI / 8.)
                        }
                        OperationRef::StandardGate(StandardGate::Tdg) => {
                            (Param::Float(-PI / 4.0), -PI / 8.)
                        }
                        OperationRef::StandardGate(StandardGate::RZ) => {
                            let param = &inst.params_view()[0];
                            (param.clone(), 0.)
                        }
                        _ => {
                            unreachable!("We cannot have gates other than T/Tdg/RZ at this point.");
                        }
                    };
                    global_phase_update += phase_update;

                    // Evolve the single-qubit Pauli-Z with Z on the given qubit.
                    // Returns the evolved Pauli in the sparse format: (sign, pauli z, pauli x, indices),
                    // where signs `true` and `false` correspond to coefficients `-1` and `+1` respectively.
                    let (sign, z, x, indices) = clifford.get_inverse_z(qubits[0]);

                    let evo_qubits: Vec<Qubit> =
                        indices.iter().map(|index| Qubit(*index)).collect();

                    let label = z
                        .iter()
                        .rev()
                        .zip(x.iter().rev())
                        .map(|(z, x)| match (z, x) {
                            (false, true) => 'X',
                            (true, false) => 'Z',
                            (true, true) => 'Y',
                            (false, false) => {
                                unreachable!("We do not produce 'I' paulis in the sparse format.");
                            }
                        })
                        .collect::<String>();

                    let py_pauli = PySparseObservable::from_label(label.as_str())?;

                    let time = if sign {
                        multiply_param(&angle, -0.5)
                    } else {
                        multiply_param(&angle, 0.5)
                    };
                    let py_evo = py_evo_cls.call1((py_pauli, time.clone()))?;
                    let py_gate = PyGate {
                        qubits: evo_qubits.len() as u32,
                        clbits: 0,
                        params: 1,
                        op_name: "PauliEvolution".to_string(),
                        gate: py_evo.into(),
                    };

                    new_dag.apply_operation_back(
                        py_gate.into(),
                        &evo_qubits,
                        &no_clbits,
                        Some(smallvec![time]),
                        None,
                        #[cfg(feature = "cache_pygates")]
                        None,
                    )?;
                }
                "measure" => {
                    // Returns the evolved Pauli in the sparse format: (sign, pauli z, pauli x, indices),
                    // where signs `true` and `false` correspond to coefficients `-1` and `+1` respectively.
                    let (sign, z, x, indices) = clifford.get_inverse_z(qubits[0]);

                    let ppm = PauliProductMeasurement { z, x, neg: sign };

                    let ppm_qubits: Vec<Qubit> =
                        indices.iter().map(|index| Qubit(*index)).collect();

                    let ppm_clbits = dag.get_cargs(inst.clbits);

                    new_dag.apply_operation_back(
                        ppm.into(),
                        &ppm_qubits,
                        ppm_clbits,
                        None,
                        None,
                        #[cfg(feature = "cache_pygates")]
                        None,
                    )?;
                }
                _ => unreachable!(
                    "We cannot have unsupported names at this step of Litinski Transformation: {}",
                    name
                ),
            }

            // Also save the Clifford operation
            if fix_clifford && SUPPORTED_CLIFFORD_NAMES.contains(&name) {
                clifford_ops.push(inst.clone());
            }
        }
    }

    new_dag.add_global_phase(&Param::Float(global_phase_update))?;

    // Add Clifford gates to the Qiskit circuit (when required).
    // Since we aim to preserve the global phase of the circuit, we add the Clifford operations from
    // the original circuit (and not the final Clifford operator).
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

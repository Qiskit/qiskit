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
use qiskit_circuit::instruction::Parameters;
use qiskit_circuit::operations::{
    Operation, OperationRef, Param, PauliProductMeasurement, PyGate, StandardGate,
    StandardInstruction, multiply_param,
};
use qiskit_circuit::packed_instruction::PackedInstruction;
use qiskit_circuit::{BlocksMode, VarsMode};

use qiskit_quantum_info::clifford::Clifford;
use qiskit_quantum_info::sparse_observable::SparseObservable;

use crate::TranspilerError;
use num_complex::Complex64;
use smallvec::smallvec;
use std::f64::consts::PI;

// List of gate/instruction names supported by the pass: the pass raises an error if the circuit
// contains instruction with names outside of this list.
static SUPPORTED_INSTRUCTION_NAMES: [&str; 20] = [
    "id", "x", "y", "z", "h", "s", "sdg", "sx", "sxdg", "cx", "cz", "cy", "swap", "iswap", "ecr",
    "dcx", "t", "tdg", "rz", "measure",
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
    let non_clifford_handled_count: usize = op_counts
        .iter()
        .filter_map(|(k, v)| {
            if HANDLED_INSTRUCTION_NAMES.contains(&k.as_str()) {
                Some(v)
            } else {
                None
            }
        })
        .sum();
    let clifford_count = dag.size(false)? - non_clifford_handled_count;

    let new_dag = dag.copy_empty_like_with_same_capacity(VarsMode::Alike, BlocksMode::Keep)?;
    let mut new_dag = new_dag.into_builder();

    let py_evo_cls = PAULI_EVOLUTION_GATE.get_bound(py);

    let num_qubits = dag.num_qubits();
    let mut clifford = Clifford::identity(num_qubits);

    // Keep track of the update to the global phase (produced when converting T/Tdg gates
    // to RZ-rotations).
    let mut global_phase_update = 0.;

    // Keep track of the clifford operations in the circuit.
    let mut clifford_ops: Vec<&PackedInstruction> = Vec::with_capacity(clifford_count);
    // Apply the Litinski transformation: that is, express a given circuit as a sequence of Pauli
    // product rotations and Pauli product measurements, followed by a final Clifford operator.
    for node_index in dag.topological_op_nodes(false)? {
        // Convert T and Tdg gates to RZ rotations.
        if let NodeType::Operation(inst) = &dag[node_index] {
            let name = inst.op.name();

            match inst.op.view() {
                OperationRef::StandardGate(StandardGate::I) => {
                    if fix_clifford {
                        clifford_ops.push(inst);
                    }
                }
                OperationRef::StandardGate(StandardGate::X) => {
                    if fix_clifford {
                        clifford_ops.push(inst);
                    }
                    clifford.append_x(dag.get_qargs(inst.qubits)[0].index())
                }
                OperationRef::StandardGate(StandardGate::Y) => {
                    if fix_clifford {
                        clifford_ops.push(inst);
                    }
                    clifford.append_y(dag.get_qargs(inst.qubits)[0].index())
                }
                OperationRef::StandardGate(StandardGate::Z) => {
                    if fix_clifford {
                        clifford_ops.push(inst);
                    }
                    clifford.append_z(dag.get_qargs(inst.qubits)[0].index())
                }
                OperationRef::StandardGate(StandardGate::H) => {
                    if fix_clifford {
                        clifford_ops.push(inst);
                    }
                    clifford.append_h(dag.get_qargs(inst.qubits)[0].index())
                }
                OperationRef::StandardGate(StandardGate::S) => {
                    if fix_clifford {
                        clifford_ops.push(inst);
                    }
                    clifford.append_s(dag.get_qargs(inst.qubits)[0].index())
                }
                OperationRef::StandardGate(StandardGate::Sdg) => {
                    if fix_clifford {
                        clifford_ops.push(inst);
                    }
                    clifford.append_sdg(dag.get_qargs(inst.qubits)[0].index())
                }
                OperationRef::StandardGate(StandardGate::SX) => {
                    if fix_clifford {
                        clifford_ops.push(inst);
                    }
                    clifford.append_sx(dag.get_qargs(inst.qubits)[0].index())
                }
                OperationRef::StandardGate(StandardGate::SXdg) => {
                    if fix_clifford {
                        clifford_ops.push(inst);
                    }
                    clifford.append_sxdg(dag.get_qargs(inst.qubits)[0].index())
                }
                OperationRef::StandardGate(StandardGate::CX) => {
                    if fix_clifford {
                        clifford_ops.push(inst);
                    }
                    clifford.append_cx(
                        dag.get_qargs(inst.qubits)[0].index(),
                        dag.get_qargs(inst.qubits)[1].index(),
                    )
                }
                OperationRef::StandardGate(StandardGate::CZ) => {
                    if fix_clifford {
                        clifford_ops.push(inst);
                    }
                    clifford.append_cz(
                        dag.get_qargs(inst.qubits)[0].index(),
                        dag.get_qargs(inst.qubits)[1].index(),
                    )
                }
                OperationRef::StandardGate(StandardGate::CY) => {
                    if fix_clifford {
                        clifford_ops.push(inst);
                    }
                    clifford.append_cy(
                        dag.get_qargs(inst.qubits)[0].index(),
                        dag.get_qargs(inst.qubits)[1].index(),
                    )
                }
                OperationRef::StandardGate(StandardGate::Swap) => {
                    if fix_clifford {
                        clifford_ops.push(inst);
                    }
                    clifford.append_swap(
                        dag.get_qargs(inst.qubits)[0].index(),
                        dag.get_qargs(inst.qubits)[1].index(),
                    )
                }
                OperationRef::StandardGate(StandardGate::ISwap) => {
                    if fix_clifford {
                        clifford_ops.push(inst);
                    }
                    clifford.append_iswap(
                        dag.get_qargs(inst.qubits)[0].index(),
                        dag.get_qargs(inst.qubits)[1].index(),
                    )
                }
                OperationRef::StandardGate(StandardGate::ECR) => {
                    if fix_clifford {
                        clifford_ops.push(inst);
                    }
                    clifford.append_ecr(
                        dag.get_qargs(inst.qubits)[0].index(),
                        dag.get_qargs(inst.qubits)[1].index(),
                    )
                }
                OperationRef::StandardGate(StandardGate::DCX) => {
                    if fix_clifford {
                        clifford_ops.push(inst);
                    }
                    clifford.append_dcx(
                        dag.get_qargs(inst.qubits)[0].index(),
                        dag.get_qargs(inst.qubits)[1].index(),
                    )
                }
                OperationRef::StandardGate(StandardGate::T)
                | OperationRef::StandardGate(StandardGate::Tdg)
                | OperationRef::StandardGate(StandardGate::RZ) => {
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
                    let (sign, terms, indices) =
                        clifford.get_inverse_z(dag.get_qargs(inst.qubits)[0].index());
                    let coeffs = vec![Complex64::new(1., 0.)];
                    let terms_len = terms.len() as u32;
                    let boundaries = vec![0, terms_len as usize];
                    // SAFETY: This is computed from the clifford and has a known size based on
                    // the returned terms that is always valid.
                    let obs = unsafe {
                        SparseObservable::new_unchecked(
                            terms_len,
                            coeffs,
                            terms,
                            (0..terms_len).collect(),
                            boundaries,
                        )
                    };

                    let time = if sign {
                        multiply_param(&angle, -0.5)
                    } else {
                        multiply_param(&angle, 0.5)
                    };
                    let py_evo = py_evo_cls.call1((obs, time.clone()))?;
                    let py_gate = PyGate {
                        qubits: indices.len() as u32,
                        clbits: 0,
                        params: 1,
                        op_name: "PauliEvolution".to_string(),
                        gate: py_evo.into(),
                    };

                    new_dag.apply_operation_back(
                        py_gate.into(),
                        &indices,
                        &[],
                        Some(Parameters::Params(smallvec![time])),
                        None,
                        #[cfg(feature = "cache_pygates")]
                        None,
                    )?;
                }
                OperationRef::StandardInstruction(StandardInstruction::Measure) => {
                    // Returns the evolved Pauli in the sparse format: (sign, pauli z, pauli x, indices),
                    // where signs `true` and `false` correspond to coefficients `-1` and `+1` respectively.
                    let (sign, z, x, indices) = clifford
                        .get_inverse_z_for_measurement(dag.get_qargs(inst.qubits)[0].index());
                    let ppm = PauliProductMeasurement { z, x, neg: sign };

                    let ppm_clbits = dag.get_cargs(inst.clbits);

                    new_dag.apply_operation_back(
                        ppm.into(),
                        &indices,
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
        }
    }

    new_dag.add_global_phase(&Param::Float(global_phase_update))?;

    // Add Clifford gates to the Qiskit circuit (when required).
    // Since we aim to preserve the global phase of the circuit, we add the Clifford operations from
    // the original circuit (and not the final Clifford operator).
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

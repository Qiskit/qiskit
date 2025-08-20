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

use std::sync::OnceLock;

use hashbrown::{HashMap, HashSet};
use indexmap::{IndexMap, IndexSet};
use nalgebra::{Matrix2, Matrix4};
use ndarray::Array;
use pyo3::prelude::*;
use qiskit_circuit::bit::QuantumRegister;
use qiskit_circuit::circuit_instruction::OperationFromPython;
use qiskit_circuit::imports::GATE;
use qiskit_circuit::operations::{
    get_standard_gate_names, ArrayType, StandardGate, StandardInstruction, UnitaryGate,
};
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::parameter::parameter_expression::ParameterExpression;
use qiskit_circuit::parameter::symbol_expr::Symbol;
use qiskit_circuit::parameter_table::ParameterUuid;
use qiskit_circuit::Qubit;
use qiskit_circuit::{
    circuit_data::CircuitData,
    dag_circuit::DAGCircuit,
    operations::{Operation, Param},
};
use smallvec::SmallVec;

use super::errors::BasisTranslatorError;

// Custom types
pub type GateIdentifier = (String, u32);
pub type BasisTransformIn = (SmallVec<[Param; 3]>, CircuitData);
pub type BasisTransformOut = (SmallVec<[Param; 3]>, DAGCircuit);

static STD_GATE_MAPPING: OnceLock<HashMap<&str, StandardGate>> = OnceLock::new();
static STD_INST_SET: OnceLock<HashSet<&str>> = OnceLock::new();

pub(super) fn compose_transforms<'a>(
    basis_transforms: &'a [(GateIdentifier, BasisTransformIn)],
    source_basis: &'a IndexSet<GateIdentifier, ahash::RandomState>,
    source_dag: &'a DAGCircuit,
) -> Result<IndexMap<GateIdentifier, BasisTransformOut, ahash::RandomState>, BasisTranslatorError> {
    let mut gate_param_counts: IndexMap<GateIdentifier, usize, ahash::RandomState> =
        IndexMap::default();
    get_gates_num_params(source_dag, &mut gate_param_counts);
    let mut mapped_instructions: IndexMap<GateIdentifier, BasisTransformOut, ahash::RandomState> =
        IndexMap::with_hasher(ahash::RandomState::default());

    for (gate_name, gate_num_qubits) in source_basis.iter().cloned() {
        let num_params = gate_param_counts[&(gate_name.clone(), gate_num_qubits)];
        let mut placeholder_params: SmallVec<[Param; 3]> = (0..num_params as u32)
            .map(|idx| {
                Param::ParameterExpression(
                    ParameterExpression::from_symbol(Symbol::new(&gate_name, None, Some(idx)))
                        .into(),
                )
            })
            .collect();

        let mut dag = DAGCircuit::new();
        // Create the mock gate and add to the circuit, use Python if necessary.
        let qubits = QuantumRegister::new_owning("q".to_string(), gate_num_qubits);
        dag.add_qreg(qubits)
            .map_err(|err| BasisTranslatorError::ComposeTransformsCircuitError(err.to_string()))?;
        let gate = if let Some(op) = name_to_packed_operation(&gate_name, gate_num_qubits) {
            op
        } else {
            let extract_py =
                Python::with_gil(|py| -> Result<OperationFromPython, BasisTranslatorError> {
                    let gate = || -> PyResult<OperationFromPython> {
                        GATE.get_bound(py)
                            .call1((&gate_name, gate_num_qubits, placeholder_params.as_ref()))?
                            .extract()
                    };
                    gate().map_err(|err| {
                        BasisTranslatorError::ComposeTransformsGateError(err.to_string())
                    })
                })?;
            placeholder_params = extract_py.params;
            extract_py.operation
        };
        let qubits: Vec<Qubit> = (0..dag.num_qubits() as u32).map(Qubit).collect();
        dag.apply_operation_back(
            gate,
            &qubits,
            &[],
            if placeholder_params.is_empty() {
                None
            } else {
                Some(placeholder_params.clone())
            },
            None,
            #[cfg(feature = "cache_pygates")]
            None,
        )
        .map_err(|err| BasisTranslatorError::ComposeTransformsCircuitError(err.to_string()))?;
        mapped_instructions.insert((gate_name, gate_num_qubits), (placeholder_params, dag));
    }

    for ((gate_name, gate_num_qubits), (equiv_params, equiv)) in basis_transforms {
        for (_, dag) in &mut mapped_instructions.values_mut() {
            let nodes_to_replace = dag
                .op_nodes(true)
                .filter(|(_, op)| {
                    (op.op.num_qubits() == *gate_num_qubits) && (op.op.name() == gate_name.as_str())
                })
                .map(|(node, op)| {
                    (
                        node,
                        op.params_view()
                            .iter()
                            .cloned()
                            .collect::<SmallVec<[Param; 3]>>(),
                    )
                })
                .collect::<Vec<_>>();
            for (node, params) in nodes_to_replace {
                let param_mapping: IndexMap<ParameterUuid, Param, ahash::RandomState> =
                    equiv_params
                        .iter()
                        .map(|x| match x {
                            Param::ParameterExpression(parameter_expression) => {
                                let symbol = parameter_expression.try_to_symbol().unwrap();
                                ParameterUuid::from_symbol(&symbol)
                            }
                            _ => unreachable!("A non parameter-expression has snuck in"),
                        })
                        .zip(params)
                        .collect();
                let mut replacement = equiv.clone();
                replacement
                    .assign_parameters_from_mapping(param_mapping)
                    .map_err(|err| {
                        BasisTranslatorError::ComposeTransformsCircuitError(err.to_string())
                    })?;
                let replace_dag: DAGCircuit =
                    DAGCircuit::from_circuit_data(&replacement, true, None, None, None, None)
                        .map_err(|err| {
                            BasisTranslatorError::ComposeTransformsCircuitError(err.to_string())
                        })?;
                dag.substitute_node_with_dag(node, &replace_dag, None, None, None)
                    .map_err(|err| {
                        BasisTranslatorError::ComposeTransformsCircuitError(err.to_string())
                    })?;
            }
        }
    }
    Ok(mapped_instructions)
}

/// Creates the placeholder gate as [PackedOperation].
fn name_to_packed_operation(name: &str, num_qubits: u32) -> Option<PackedOperation> {
    let std_gate_mapping = STD_GATE_MAPPING.get_or_init(|| {
        get_standard_gate_names()
            .iter()
            .enumerate()
            .map(|(k, v)| (*v, bytemuck::checked::cast(k as u8)))
            .collect()
    });
    let std_instruction_mapping =
        STD_INST_SET.get_or_init(|| HashSet::from_iter(["barrier", "delay", "measure", "reset"]));
    if let Some(operation) = std_gate_mapping.get(name) {
        Some((*operation).into())
    } else if std_instruction_mapping.contains(name) {
        let inst = match name {
            "barrier" => StandardInstruction::Barrier(num_qubits),
            "delay" => StandardInstruction::Delay(qiskit_circuit::operations::DelayUnit::DT),
            "measure" => StandardInstruction::Measure,
            "reset" => StandardInstruction::Reset,
            _ => unreachable!(),
        };
        Some(inst.into())
    } else if name == "unitary" {
        let matrix = match num_qubits {
            1 => ArrayType::OneQ(Matrix2::identity()),
            2 => ArrayType::TwoQ(Matrix4::identity()),
            _ => ArrayType::NDArray(Array::eye(2_usize.pow(num_qubits))),
        };
        Some(UnitaryGate { array: matrix }.into())
    } else {
        None
    }
}

/// `DAGCircuit` variant.
///
/// Gets the identifier of a gate instance (name, number of qubits) mapped to the
/// number of parameters it contains currently.
fn get_gates_num_params(
    dag: &DAGCircuit,
    example_gates: &mut IndexMap<GateIdentifier, usize, ahash::RandomState>,
) {
    for (_, inst) in dag.op_nodes(true) {
        example_gates.insert(
            (inst.op.name().to_string(), inst.op.num_qubits()),
            inst.params_view().len(),
        );
        if inst.op.control_flow() {
            let blocks = inst.op.blocks();
            for block in blocks {
                get_gates_num_params_circuit(&block, example_gates);
            }
        }
    }
}

/// `CircuitData` variant.
///
/// Gets the identifier of a gate instance (name, number of qubits) mapped to the
/// number of parameters it contains currently.
fn get_gates_num_params_circuit(
    circuit: &CircuitData,
    example_gates: &mut IndexMap<GateIdentifier, usize, ahash::RandomState>,
) {
    for inst in circuit.iter() {
        example_gates.insert(
            (inst.op.name().to_string(), inst.op.num_qubits()),
            inst.params_view().len(),
        );
        if inst.op.control_flow() {
            let blocks = inst.op.blocks();
            for block in blocks {
                get_gates_num_params_circuit(&block, example_gates);
            }
        }
    }
}

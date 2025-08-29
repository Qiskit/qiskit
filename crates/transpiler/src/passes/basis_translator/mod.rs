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

use compose_transforms::BasisTransformIn;
use compose_transforms::BasisTransformOut;
use compose_transforms::GateIdentifier;

use basis_search::basis_search;
use compose_transforms::compose_transforms;
use errors::BasisTranslatorError;
use hashbrown::{HashMap, HashSet};
use indexmap::{IndexMap, IndexSet};
use pyo3::intern;
use pyo3::prelude::*;

mod basis_search;
mod compose_transforms;
mod errors;

use qiskit_circuit::circuit_instruction::OperationFromPython;
use qiskit_circuit::converters::dag_to_circuit;
use qiskit_circuit::dag_circuit::DAGCircuitBuilder;
use qiskit_circuit::imports::QUANTUM_CIRCUIT;
use qiskit_circuit::operations::Param;
use qiskit_circuit::packed_instruction::PackedInstruction;
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::parameter::parameter_expression::ParameterError;
use qiskit_circuit::parameter::parameter_expression::ParameterExpression;
use qiskit_circuit::parameter::symbol_expr::Symbol;
use qiskit_circuit::parameter::symbol_expr::SymbolExpr;
use qiskit_circuit::parameter::symbol_expr::Value;
use qiskit_circuit::{
    circuit_data::CircuitData,
    dag_circuit::DAGCircuit,
    operations::{Operation, OperationRef, PythonOperation},
};
use qiskit_circuit::{Clbit, PhysicalQubit, Qubit, VarsMode};
use smallvec::SmallVec;

use crate::equivalence::EquivalenceLibrary;
use crate::target::Qargs;
use crate::target::QargsRef;
use crate::target::Target;

type AhashIndexMap<K, V> = IndexMap<K, V, ahash::RandomState>;
type AhashIndexSet<O> = IndexSet<O, ahash::RandomState>;
type InstMap = AhashIndexMap<GateIdentifier, BasisTransformOut>;
type ExtraInstructionMap<'a> = AhashIndexMap<&'a PhysicalQargs, InstMap>;
type PhysicalQargs = SmallVec<[PhysicalQubit; 2]>;

#[pyfunction(name = "base_run", signature = (dag, equiv_lib, min_qubits, target=None, target_basis=None))]
fn py_run_basis_translator(
    dag: &DAGCircuit,
    equiv_lib: &mut EquivalenceLibrary,
    min_qubits: usize,
    target: Option<&Target>,
    target_basis: Option<HashSet<String>>,
) -> PyResult<Option<DAGCircuit>> {
    let target_basis_ref: Option<HashSet<&str>> = target_basis
        .as_ref()
        .map(|set| set.iter().map(|obj| obj.as_str()).collect());
    run_basis_translator(dag, equiv_lib, min_qubits, target, target_basis_ref).map_err(|e| e.into())
}

pub fn run_basis_translator(
    dag: &DAGCircuit,
    equiv_lib: &mut EquivalenceLibrary,
    min_qubits: usize,
    target: Option<&Target>,
    target_basis: Option<HashSet<&str>>,
) -> Result<Option<DAGCircuit>, BasisTranslatorError> {
    if target_basis.is_none() && target.is_none() {
        return Ok(None);
    }

    let (non_global_operations, qargs_with_non_global_operation): (
        Option<AhashIndexSet<&str>>,
        AhashIndexMap<Qargs, AhashIndexSet<&str>>,
    ) = if let Some(target) = target {
        let mut qargs_mapping: AhashIndexMap<Qargs, AhashIndexSet<&str>> = AhashIndexMap::default();
        let global_set: AhashIndexSet<&str> =
            AhashIndexSet::from_iter(target.get_non_global_operation_names(false));
        for name in global_set.iter() {
            for qarg in target[name].keys().cloned() {
                qargs_mapping
                    .entry(qarg)
                    .and_modify(|val| {
                        val.insert(name);
                    })
                    .or_insert(AhashIndexSet::from_iter([*name]));
            }
        }
        (Some(global_set), qargs_mapping)
    } else {
        (None, AhashIndexMap::default())
    };

    let basic_instrs: AhashIndexSet<&str>;
    let mut source_basis: AhashIndexSet<GateIdentifier> = AhashIndexSet::default();
    let mut new_target_basis: AhashIndexSet<&str>;
    let mut qargs_local_source_basis: AhashIndexMap<PhysicalQargs, AhashIndexSet<GateIdentifier>> =
        AhashIndexMap::default();
    if let Some(target) = target.as_ref() {
        basic_instrs = ["barrier", "snapshot", "store"].into_iter().collect();
        let non_global_str: AhashIndexSet<&str> =
            if let Some(operations) = non_global_operations.as_ref() {
                operations.clone()
            } else {
                AhashIndexSet::default()
            };
        let target_keys = target.keys().collect::<AhashIndexSet<_>>();
        new_target_basis = target_keys.difference(&non_global_str).copied().collect();
        extract_basis_target(
            dag,
            &mut source_basis,
            &mut qargs_local_source_basis,
            min_qubits,
            &qargs_with_non_global_operation,
        );
    } else {
        basic_instrs = ["measure", "reset", "barrier", "snapshot", "delay", "store"]
            .into_iter()
            .collect();
        source_basis = extract_basis(dag, min_qubits);
        new_target_basis = target_basis
            .as_ref()
            .unwrap()
            .into_iter()
            .copied()
            .collect();
    }
    new_target_basis = new_target_basis.union(&basic_instrs).copied().collect();
    // If the source basis is a subset of the target basis and we have no circuit
    // instructions on qargs that have non-global operations there is nothing to
    // translate and we can exit early.
    let source_basis_names: AhashIndexSet<&str> =
        source_basis.iter().map(|x| x.0.as_str()).collect();
    if source_basis_names.is_subset(&new_target_basis) && qargs_local_source_basis.is_empty() {
        return Ok(None);
    }
    let basis_transforms = basis_search(equiv_lib, &source_basis, &new_target_basis);
    let mut qarg_local_basis_transforms: AhashIndexMap<
        &PhysicalQargs,
        Vec<(GateIdentifier, BasisTransformIn)>,
    > = AhashIndexMap::default();
    for (qargs, local_source_basis) in qargs_local_source_basis.iter() {
        // For any multiqubit operation that contains a subset of qubits that
        // has a non-local operation, include that non-local operation in the
        // search. This matches with the check we did above to include those
        // subset non-local operations in the check here.
        let mut expanded_target = new_target_basis.clone();
        // Qargs are always guaranteed to be concrete based on `extract_basis_target`.
        if qargs.len() > 1 {
            let qarg_as_set: AhashIndexSet<PhysicalQubit> =
                AhashIndexSet::from_iter(qargs.iter().copied());
            for (non_local_qarg, local_basis) in qargs_with_non_global_operation.iter() {
                if let Qargs::Concrete(non_local_qarg) = non_local_qarg {
                    let non_local_qarg_as_set: AhashIndexSet<PhysicalQubit> =
                        AhashIndexSet::from_iter(non_local_qarg.iter().copied());
                    if qarg_as_set.is_superset(&non_local_qarg_as_set) {
                        expanded_target = expanded_target.union(local_basis).cloned().collect();
                    }
                }
            }
        } else {
            expanded_target = expanded_target
                .union(&qargs_with_non_global_operation[&Qargs::from_iter(qargs.iter().copied())])
                .cloned()
                .collect();
        }
        let local_basis_transforms = basis_search(equiv_lib, local_source_basis, &expanded_target);
        if let Some(local_basis_transforms) = local_basis_transforms {
            qarg_local_basis_transforms.insert(qargs, local_basis_transforms);
        } else {
            return Err(BasisTranslatorError::TargetMissingEquivalence {
                basis: format!("{:?}", local_source_basis),
                expanded: format!("{:?}", expanded_target),
            });
        }
    }

    let Some(basis_transforms) = basis_transforms else {
        return Err(BasisTranslatorError::TargetMissingEquivalence {
            basis: format!("{:?}", source_basis),
            expanded: format!("{:?}", new_target_basis),
        });
    };

    let instr_map: InstMap = compose_transforms(&basis_transforms, &source_basis, dag)?;
    let extra_inst_map: ExtraInstructionMap = qarg_local_basis_transforms
        .iter()
        .map(|(qarg, transform)| -> Result<_, BasisTranslatorError> {
            Ok((
                *qarg,
                compose_transforms(transform, &qargs_local_source_basis[*qarg], dag)?,
            ))
        })
        .collect::<Result<_, BasisTranslatorError>>()?;

    let (out_dag, _) = apply_translation(
        dag,
        &new_target_basis,
        &instr_map,
        &extra_inst_map,
        min_qubits,
        &qargs_with_non_global_operation,
    )?;
    Ok(Some(out_dag))
}

/// Method that extracts all gate instances identifiers from a DAGCircuit.
fn extract_basis(circuit: &DAGCircuit, min_qubits: usize) -> AhashIndexSet<GateIdentifier> {
    let mut basis = AhashIndexSet::default();
    // Recurse for DAGCircuit
    fn recurse_dag(
        circuit: &DAGCircuit,
        basis: &mut AhashIndexSet<GateIdentifier>,
        min_qubits: usize,
    ) {
        for (_node, operation) in circuit.op_nodes(true) {
            if circuit.get_qargs(operation.qubits).len() >= min_qubits {
                basis.insert((operation.op.name().to_string(), operation.op.num_qubits()));
            }
            if operation.op.control_flow() {
                for block in operation.op.blocks() {
                    recurse_circuit(&block, basis, min_qubits);
                }
            }
        }
    }

    // Recurse for CircuitData
    fn recurse_circuit(
        circuit: &CircuitData,
        basis: &mut AhashIndexSet<GateIdentifier>,
        min_qubits: usize,
    ) {
        for inst in circuit.iter() {
            if circuit.get_qargs(inst.qubits).len() >= min_qubits {
                basis.insert((inst.op.name().to_string(), inst.op.num_qubits()));
            }
            if inst.op.control_flow() {
                for block in inst.op.blocks() {
                    recurse_circuit(&block, basis, min_qubits);
                }
            }
        }
    }

    recurse_dag(circuit, &mut basis, min_qubits);
    basis
}

/// Method that extracts a mapping of all the qargs in the local_source basis
/// obtained from the [Target], to all gate instances identifiers from a DAGCircuit.
/// When dealing with `ControlFlowOp` instances the function will perform a recursion call
/// to a variant design to handle instances of `QuantumCircuit`.
fn extract_basis_target(
    dag: &DAGCircuit,
    source_basis: &mut AhashIndexSet<GateIdentifier>,
    qargs_local_source_basis: &mut AhashIndexMap<PhysicalQargs, AhashIndexSet<GateIdentifier>>,
    min_qubits: usize,
    qargs_with_non_global_operation: &AhashIndexMap<Qargs, AhashIndexSet<&str>>,
) {
    for (_, node_obj) in dag.op_nodes(true) {
        let qargs: &[Qubit] = dag.get_qargs(node_obj.qubits);
        if qargs.len() < min_qubits {
            continue;
        }
        // Treat the instruction as on an incomplete basis if the qargs are in the
        // qargs_with_non_global_operation dictionary or if any of the qubits in qargs
        // are a superset for a non-local operation. For example, if the qargs
        // are (0, 1) and that's a global (ie no non-local operations on (0, 1)
        // operation but there is a non-local operation on (1,) we need to
        // do an extra non-local search for this op to ensure we include any
        // single qubit operation for (1,) as valid. This pattern also holds
        // true for > 2q ops too (so for 4q operations we need to check for 3q, 2q,
        // and 1q operations in the same manner)
        let physical_qargs: PhysicalQargs = qargs.iter().map(|x| PhysicalQubit(x.0)).collect();
        let physical_qargs_as_set: AhashIndexSet<PhysicalQubit> =
            AhashIndexSet::from_iter(physical_qargs.iter().copied());
        let physical_qargs: Qargs = physical_qargs.into();
        if qargs_with_non_global_operation.contains_key(&physical_qargs)
            || qargs_with_non_global_operation
                .keys()
                .filter_map(|qargs| {
                    if let QargsRef::Concrete(qargs) = qargs.as_ref() {
                        Some(qargs)
                    } else {
                        None
                    }
                })
                .any(|incomplete_qargs| {
                    let incomplete_qargs: AhashIndexSet<PhysicalQubit> =
                        AhashIndexSet::from_iter(incomplete_qargs.iter().copied());
                    physical_qargs_as_set.is_superset(&incomplete_qargs)
                })
        {
            let qargs_from_set: PhysicalQargs = physical_qargs_as_set.into_iter().collect();
            qargs_local_source_basis
                .entry(qargs_from_set)
                .and_modify(|set| {
                    set.insert((node_obj.op.name().to_string(), node_obj.op.num_qubits()));
                })
                .or_insert(AhashIndexSet::from_iter([(
                    node_obj.op.name().to_string(),
                    node_obj.op.num_qubits(),
                )]));
        } else {
            source_basis.insert((node_obj.op.name().to_string(), node_obj.op.num_qubits()));
        }
        if node_obj.op.control_flow() {
            for block in node_obj.op.blocks() {
                extract_basis_target_circ(
                    &block,
                    source_basis,
                    qargs_local_source_basis,
                    min_qubits,
                    qargs_with_non_global_operation,
                );
            }
        }
    }
}

/// Variant of extract_basis_target that takes an instance of [CircuitData].
fn extract_basis_target_circ(
    circuit: &CircuitData,
    source_basis: &mut AhashIndexSet<GateIdentifier>,
    qargs_local_source_basis: &mut AhashIndexMap<PhysicalQargs, AhashIndexSet<GateIdentifier>>,
    min_qubits: usize,
    qargs_with_non_global_operation: &AhashIndexMap<Qargs, AhashIndexSet<&str>>,
) {
    for node_obj in circuit.iter() {
        let qargs = circuit.get_qargs(node_obj.qubits);
        if qargs.len() < min_qubits {
            continue;
        }
        // Treat the instruction as on an incomplete basis if the qargs are in the
        // qargs_with_non_global_operation dictionary or if any of the qubits in qargs
        // are a superset for a non-local operation. For example, if the qargs
        // are (0, 1) and that's a global (ie no non-local operations on (0, 1)
        // operation but there is a non-local operation on (1,) we need to
        // do an extra non-local search for this op to ensure we include any
        // single qubit operation for (1,) as valid. This pattern also holds
        // true for > 2q ops too (so for 4q operations we need to check for 3q, 2q,
        // and 1q operations in the same manner)
        let physical_qargs: PhysicalQargs = qargs.iter().map(|x| PhysicalQubit(x.0)).collect();
        let physical_qargs_as_set: AhashIndexSet<PhysicalQubit> =
            AhashIndexSet::from_iter(physical_qargs.iter().copied());
        let physical_qargs: Qargs = physical_qargs.into();
        if qargs_with_non_global_operation.contains_key(&physical_qargs)
            || qargs_with_non_global_operation
                .keys()
                .filter_map(|qargs| {
                    if let QargsRef::Concrete(qargs) = qargs.as_ref() {
                        Some(qargs)
                    } else {
                        None
                    }
                })
                .any(|incomplete_qargs| {
                    let incomplete_qargs: AhashIndexSet<PhysicalQubit> =
                        AhashIndexSet::from_iter(incomplete_qargs.iter().copied());
                    physical_qargs_as_set.is_superset(&incomplete_qargs)
                })
        {
            let qargs_from_set: PhysicalQargs = physical_qargs_as_set.into_iter().collect();
            qargs_local_source_basis
                .entry(qargs_from_set)
                .and_modify(|set| {
                    set.insert((node_obj.op.name().to_string(), node_obj.op.num_qubits()));
                })
                .or_insert(AhashIndexSet::from_iter([(
                    node_obj.op.name().to_string(),
                    node_obj.op.num_qubits(),
                )]));
        } else {
            source_basis.insert((node_obj.op.name().to_string(), node_obj.op.num_qubits()));
        }
        if node_obj.op.control_flow() {
            for block in node_obj.op.blocks() {
                extract_basis_target_circ(
                    &block,
                    source_basis,
                    qargs_local_source_basis,
                    min_qubits,
                    qargs_with_non_global_operation,
                );
            }
        }
    }
}

fn apply_translation(
    dag: &DAGCircuit,
    target_basis: &AhashIndexSet<&str>,
    instr_map: &InstMap,
    extra_inst_map: &ExtraInstructionMap,
    min_qubits: usize,
    qargs_with_non_global_operation: &AhashIndexMap<Qargs, AhashIndexSet<&str>>,
) -> Result<(DAGCircuit, bool), BasisTranslatorError> {
    let mut is_updated = false;
    let out_dag = dag.copy_empty_like(VarsMode::Alike).map_err(|_| {
        BasisTranslatorError::BasisDAGCircuitError("Error copying DAGCircuit instance".to_string())
    })?;
    let mut out_dag_builder = out_dag.into_builder();
    for node in dag.topological_op_nodes().map_err(|_| {
        BasisTranslatorError::BasisDAGCircuitError("Error retrieving Op nodes from DAG".to_string())
    })? {
        let node_obj = dag[node].unwrap_operation();
        let node_qarg = dag.get_qargs(node_obj.qubits);
        let node_carg = dag.get_cargs(node_obj.clbits);
        let qubit_set: AhashIndexSet<Qubit> = AhashIndexSet::from_iter(node_qarg.iter().copied());
        let mut new_op: Option<OperationFromPython> = None;
        if target_basis.contains(node_obj.op.name()) || node_qarg.len() < min_qubits {
            if node_obj.op.control_flow() {
                Python::with_gil(|py| -> PyResult<()> {
                    // This part is only executed through python because `ControlFlowOp`
                    // does not exist in Rust space yet, and we need the method `replace_blocks`.
                    // TODO: Refactor this condition block once https://github.com/Qiskit/qiskit/pull/14568 merges.
                    let OperationRef::Instruction(control_op) = node_obj.op.view() else {
                        unreachable!("This instruction {} says it is of control flow type, but is not an Instruction instance", node_obj.op.name())
                    };
                    let mut flow_blocks = vec![];
                    let bound_obj = control_op.instruction.bind(py);
                    for block in node_obj.op.blocks() {
                        let dag_block: DAGCircuit = DAGCircuit::from_circuit_data(&block, true, None, None, None, None)?;
                        let updated_dag: DAGCircuit;
                        (updated_dag, is_updated) = apply_translation(
                            &dag_block,
                            target_basis,
                            instr_map,
                            extra_inst_map,
                            min_qubits,
                            qargs_with_non_global_operation,
                        ).map_err(PyErr::from)?;
                        let flow_circ_block = if is_updated {
                            QUANTUM_CIRCUIT.get_bound(py).call_method1(intern!(py, "_from_circuit_data"), (dag_to_circuit(&updated_dag, true)?,))?
                        } else {
                            QUANTUM_CIRCUIT.get_bound(py).call_method1(intern!(py, "_from_circuit_data"), (block,))?
                        };
                        flow_blocks.push(flow_circ_block);
                    }
                    let replaced_blocks =
                        bound_obj.call_method1("replace_blocks", (flow_blocks,))?;
                    new_op = Some(replaced_blocks.extract()?);
                    Ok(())
                }).map_err(|_| BasisTranslatorError::BasisCircuitError("Error replacing control flow operation blocks".to_string()))?;
            }
            if let Some(new_op) = new_op {
                out_dag_builder
                    .apply_operation_back(
                        new_op.operation,
                        node_qarg,
                        node_carg,
                        if new_op.params.is_empty() {
                            None
                        } else {
                            Some(new_op.params)
                        },
                        new_op.label.as_deref().cloned(),
                        #[cfg(feature = "cache_pygates")]
                        None,
                    )
                    .map_err(|_| {
                        BasisTranslatorError::BasisDAGCircuitError(
                            "Error applying operation to DAGCircuit".to_string(),
                        )
                    })?;
            } else {
                out_dag_builder
                    .apply_operation_back(
                        node_obj.op.clone(),
                        node_qarg,
                        node_carg,
                        node_obj.params.as_ref().map(|x| *x.clone()),
                        node_obj.label.as_deref().cloned(),
                        #[cfg(feature = "cache_pygates")]
                        None,
                    )
                    .map_err(|_| {
                        BasisTranslatorError::BasisDAGCircuitError(
                            "Error applying operation to DAGCircuit".to_string(),
                        )
                    })?;
            }
            continue;
        }
        let node_qarg_as_physical: Qargs = node_qarg.iter().map(|x| PhysicalQubit(x.0)).collect();
        if qargs_with_non_global_operation.contains_key(&node_qarg_as_physical)
            && qargs_with_non_global_operation[&node_qarg_as_physical].contains(node_obj.op.name())
        {
            out_dag_builder
                .apply_operation_back(
                    node_obj.op.clone(),
                    node_qarg,
                    node_carg,
                    node_obj.params.as_ref().map(|x| *x.clone()),
                    node_obj.label.as_deref().cloned(),
                    #[cfg(feature = "cache_pygates")]
                    None,
                )
                .map_err(|_| {
                    BasisTranslatorError::BasisDAGCircuitError(
                        "Error applying operation to DAGCircuit".to_string(),
                    )
                })?;
            continue;
        }

        let unique_qargs: PhysicalQargs = qubit_set.iter().map(|x| PhysicalQubit(x.0)).collect();
        if extra_inst_map.contains_key(&unique_qargs) {
            replace_node(
                &mut out_dag_builder,
                node_obj.clone(),
                &extra_inst_map[&unique_qargs],
            )?;
        } else if instr_map
            .contains_key(&(node_obj.op.name().to_string(), node_obj.op.num_qubits()))
        {
            replace_node(&mut out_dag_builder, node_obj.clone(), instr_map)?;
        } else {
            return Err(BasisTranslatorError::ApplyTranslationMappingError(
                node_obj.op.name().to_string(),
            ));
        }
        is_updated = true;
    }
    Ok((out_dag_builder.build(), is_updated))
}

fn replace_node(
    dag: &mut DAGCircuitBuilder,
    node: PackedInstruction,
    instr_map: &AhashIndexMap<GateIdentifier, (SmallVec<[Param; 3]>, DAGCircuit)>,
) -> Result<(), BasisTranslatorError> {
    // Method to check if the operation is Rust native.
    // Should be removed in the future.
    let is_native = |op: &PackedOperation| -> bool {
        op.try_standard_gate().is_some()
            || op.try_standard_instruction().is_some()
            || matches!(op.view(), OperationRef::Unitary(_))
    };
    let (target_params, target_dag) =
        &instr_map[&(node.op.name().to_string(), node.op.num_qubits())];
    if node.params_view().len() != target_params.len() {
        return Err(BasisTranslatorError::ReplaceNodeParamMismatch {
            node_params: format!("{:?}", node.params_view()),
            node_name: node.op.name().to_string(),
            target_params: format!("{:?}", target_params),
            target_dag: format!("{:?}", target_dag),
        });
    }
    if node.params_view().is_empty() {
        for inner_index in target_dag.topological_op_nodes().map_err(|_| {
            BasisTranslatorError::BasisDAGCircuitError(
                "Error retrieving Op nodes from DAG".to_string(),
            )
        })? {
            let inner_node = &target_dag[inner_index].unwrap_operation();
            let old_qargs = dag.qargs_interner().get(node.qubits);
            let old_cargs = dag.cargs_interner().get(node.clbits);
            let new_qubits: Vec<Qubit> = target_dag
                .get_qargs(inner_node.qubits)
                .iter()
                .map(|qubit| old_qargs[qubit.0 as usize])
                .collect();
            let new_clbits: Vec<Clbit> = target_dag
                .get_cargs(inner_node.clbits)
                .iter()
                .map(|clbit| old_cargs[clbit.0 as usize])
                .collect();
            let new_op: PackedOperation = match inner_node.op.view() {
                OperationRef::Gate(gate) => {
                    Python::with_gil(|py| gate.py_copy(py).map(|op| op.into()))
                        .expect("Error while copying gate instance.")
                }
                OperationRef::Instruction(instruction) => {
                    Python::with_gil(|py| instruction.py_copy(py).map(|op| op.into()))
                        .expect("Error while copying instruction instance.")
                }
                OperationRef::Operation(operation) => {
                    Python::with_gil(|py| operation.py_copy(py).map(|op| op.into()))
                        .expect("Error while copying operation instance.")
                }
                OperationRef::StandardGate(gate) => gate.into(),
                OperationRef::StandardInstruction(instruction) => instruction.into(),
                OperationRef::Unitary(unitary) => unitary.clone().into(),
            };
            let new_params: SmallVec<[Param; 3]> =
                inner_node.params_view().iter().cloned().collect();
            dag.apply_operation_back(
                new_op,
                &new_qubits,
                &new_clbits,
                if new_params.is_empty() {
                    None
                } else {
                    Some(new_params)
                },
                node.label.as_deref().cloned(),
                #[cfg(feature = "cache_pygates")]
                None,
            )
            .map_err(|_| {
                BasisTranslatorError::BasisDAGCircuitError(
                    "Error applying operation to DAGCircuit".to_string(),
                )
            })?;
        }
        dag.add_global_phase(target_dag.global_phase())
            .map_err(|_| {
                BasisTranslatorError::BasisDAGCircuitError(
                    "Error while adding a new global phase".to_string(),
                )
            })?;
    } else {
        let parameter_map: HashMap<Symbol, Param> = HashMap::from_iter(
            target_params
                .iter()
                .zip(node.params_view())
                .filter_map(|(key, val)| match key {
                    Param::ParameterExpression(param) => param
                        .try_to_symbol()
                        .ok()
                        .map(|param| (param.clone(), val.clone())),
                    _ => None,
                }),
        );
        for inner_index in target_dag.topological_op_nodes().map_err(|_| {
            BasisTranslatorError::BasisDAGCircuitError(
                "Error retrieving Op nodes from DAG".to_string(),
            )
        })? {
            let inner_node = &target_dag[inner_index].unwrap_operation();
            let old_qargs = dag.qargs_interner().get(node.qubits);
            let old_cargs = dag.cargs_interner().get(node.clbits);
            let new_qubits: Vec<Qubit> = target_dag
                .get_qargs(inner_node.qubits)
                .iter()
                .map(|qubit| old_qargs[qubit.0 as usize])
                .collect();
            let new_clbits: Vec<Clbit> = target_dag
                .get_cargs(inner_node.clbits)
                .iter()
                .map(|clbit| old_cargs[clbit.0 as usize])
                .collect();
            let new_op: PackedOperation = match inner_node.op.view() {
                OperationRef::Gate(gate) => Python::with_gil(|py| {
                    gate.py_copy(py).map(|op| op.into())
                })
                .map_err(|err| BasisTranslatorError::BasisDAGCircuitError(err.to_string()))?,
                OperationRef::Instruction(instruction) => {
                    Python::with_gil(|py| instruction.py_copy(py).map(|op| op.into())).map_err(
                        |err| BasisTranslatorError::BasisDAGCircuitError(err.to_string()),
                    )?
                }
                OperationRef::Operation(operation) => {
                    Python::with_gil(|py| operation.py_copy(py).map(|op| op.into())).map_err(
                        |err| BasisTranslatorError::BasisDAGCircuitError(err.to_string()),
                    )?
                }
                OperationRef::StandardGate(gate) => gate.into(),
                OperationRef::StandardInstruction(instruction) => instruction.into(),
                OperationRef::Unitary(unitary) => unitary.clone().into(),
            };

            let mut new_params: SmallVec<[Param; 3]> =
                inner_node.params_view().iter().cloned().collect();
            if inner_node
                .params_view()
                .iter()
                .any(|param| matches!(param, Param::ParameterExpression(_)))
            {
                new_params = inner_node
                    .params_view()
                    .iter()
                    .map(|param| match param {
                        Param::ParameterExpression(parameter_expression) => {
                            param_assignment_expr(parameter_expression, &parameter_map, true)
                        }
                        _ => Ok(param.clone()),
                    })
                    .collect::<Result<_, BasisTranslatorError>>()?;
                if !is_native(&new_op) {
                    // TODO: Remove this.
                    // Acquire the gil if the operation is not native to set the operation parameters in
                    // Python.
                    Python::with_gil(|py| -> Result<(), BasisTranslatorError> {
                        match new_op.view() {
                            OperationRef::Instruction(inst) => inst
                                .instruction
                                .bind(py)
                                .setattr("params", new_params.clone())
                                .map_err(|err| {
                                    BasisTranslatorError::BasisDAGCircuitError(err.to_string())
                                }),
                            OperationRef::Gate(gate) => gate
                                .gate
                                .bind(py)
                                .setattr("params", new_params.clone())
                                .map_err(|err| {
                                    BasisTranslatorError::BasisDAGCircuitError(err.to_string())
                                }),
                            OperationRef::Operation(oper) => oper
                                .operation
                                .bind(py)
                                .setattr("params", new_params.clone())
                                .map_err(|err| {
                                    BasisTranslatorError::BasisDAGCircuitError(err.to_string())
                                }),
                            _ => Ok(()),
                        }
                    })?;
                }
            }
            dag.apply_operation_back(
                new_op,
                &new_qubits,
                &new_clbits,
                if new_params.is_empty() {
                    None
                } else {
                    Some(new_params)
                },
                inner_node.label.as_deref().cloned(),
                #[cfg(feature = "cache_pygates")]
                None,
            )
            .map_err(|err| BasisTranslatorError::BasisDAGCircuitError(err.to_string()))?;
        }

        match target_dag.global_phase() {
            Param::ParameterExpression(expr) => {
                let param = param_assignment_expr(expr, &parameter_map, false)?;
                dag.add_global_phase(&param)
                    .map_err(|e| BasisTranslatorError::BasisDAGCircuitError(e.to_string()))
            }
            Param::Float(_) => dag
                .add_global_phase(target_dag.global_phase())
                .map_err(|e| BasisTranslatorError::BasisDAGCircuitError(e.to_string())),
            Param::Obj(_) => Ok(()),
        }?
    }

    Ok(())
}

fn param_expr_assignment(
    param_obj: &ParameterExpression,
    parameter_map: &HashMap<Symbol, Param>,
) -> Result<ParameterExpression, ParameterError> {
    let mut subs_map: HashMap<Symbol, ParameterExpression> = HashMap::new();
    let mut bind_map: HashMap<&Symbol, Value> = HashMap::new();
    for key in param_obj.iter_symbols() {
        match &parameter_map[key].clone() {
            Param::ParameterExpression(val) => {
                subs_map.insert(key.clone(), val.as_ref().clone());
            }
            Param::Float(val) => {
                bind_map.insert(key, Value::Real(*val));
            }
            Param::Obj(val) => {
                let val = Python::with_gil(|py| val.extract::<Value>(py))
                    .map_err(|_| ParameterError::InvalidValue)?;
                bind_map.insert(key, val);
            }
        }
    }
    // Apply substitution and binding in that order.
    let mut new_value: ParameterExpression = param_obj.subs(&subs_map, true)?;
    new_value = new_value.bind(&bind_map, true)?;
    Ok(new_value)
}

fn param_assignment_expr(
    param: &ParameterExpression,
    parameter_map: &HashMap<Symbol, Param>,
    allow_complex: bool,
) -> Result<Param, BasisTranslatorError> {
    let new_value = param_expr_assignment(param, parameter_map)
        .map_err(BasisTranslatorError::BasisParameterError)?;
    match (new_value.try_to_value(true), allow_complex) {
        (Ok(Value::Complex(parsed)), false) => Err(
            BasisTranslatorError::ReplaceNodeGlobalPhaseComplex(parsed.to_string()),
        ),
        (Ok(Value::Real(num)), _) => Ok(Param::Float(num)),
        (Ok(parsed), _) => Ok(Param::ParameterExpression(
            ParameterExpression::from_symbol_expr(SymbolExpr::Value(parsed)).into(),
        )),
        (Err(_), _) => Ok(Param::ParameterExpression(new_value.into())),
    }
}

pub fn basis_translator_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_run_basis_translator))?;
    Ok(())
}

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
use hashbrown::{HashMap, HashSet};
use indexmap::{IndexMap, IndexSet};
use itertools::Itertools;
use pyo3::intern;
use pyo3::prelude::*;

mod basis_search;
mod compose_transforms;

use pyo3::types::{IntoPyDict, PyComplex, PyDict, PyTuple};
use pyo3::PyTypeInfo;
use qiskit_circuit::circuit_instruction::OperationFromPython;
use qiskit_circuit::converters::circuit_to_dag;
use qiskit_circuit::dag_circuit::DAGCircuitBuilder;
use qiskit_circuit::imports::DAG_TO_CIRCUIT;
use qiskit_circuit::imports::PARAMETER_EXPRESSION;
use qiskit_circuit::operations::Param;
use qiskit_circuit::packed_instruction::PackedInstruction;
use qiskit_circuit::PhysicalQubit;
use qiskit_circuit::{
    circuit_data::CircuitData,
    dag_circuit::DAGCircuit,
    operations::{Operation, OperationRef},
};
use qiskit_circuit::{Clbit, Qubit};
use smallvec::SmallVec;

use crate::equivalence::EquivalenceLibrary;
use crate::target::Qargs;
use crate::target::QargsRef;
use crate::target::Target;
use crate::TranspilerError;

type InstMap = IndexMap<GateIdentifier, BasisTransformOut, ahash::RandomState>;
type ExtraInstructionMap<'a> = IndexMap<&'a PhysicalQargs, InstMap, ahash::RandomState>;
type PhysicalQargs = SmallVec<[PhysicalQubit; 2]>;

#[allow(clippy::too_many_arguments)]
#[pyfunction(name = "base_run", signature = (dag, equiv_lib, qargs_with_non_global_operation, min_qubits, target_basis=None, target=None, non_global_operations=None))]
pub fn run_basis_translator(
    py: Python<'_>,
    dag: DAGCircuit,
    equiv_lib: &mut EquivalenceLibrary,
    qargs_with_non_global_operation: HashMap<Qargs, HashSet<String>>,
    min_qubits: usize,
    target_basis: Option<HashSet<String>>,
    target: Option<&Target>,
    non_global_operations: Option<HashSet<String>>,
) -> PyResult<DAGCircuit> {
    if target_basis.is_none() && target.is_none() {
        return Ok(dag);
    }

    let qargs_with_non_global_operation: IndexMap<
        Qargs,
        IndexSet<String, ahash::RandomState>,
        ahash::RandomState,
    > = qargs_with_non_global_operation
        .into_iter()
        .map(|(k, v)| {
            (
                k,
                v.into_iter().collect::<IndexSet<_, ahash::RandomState>>(),
            )
        })
        .collect();

    let basic_instrs: IndexSet<String, ahash::RandomState>;
    let mut source_basis: IndexSet<GateIdentifier, ahash::RandomState> = IndexSet::default();
    let mut new_target_basis: IndexSet<String, ahash::RandomState>;
    let mut qargs_local_source_basis: IndexMap<
        PhysicalQargs,
        IndexSet<GateIdentifier, ahash::RandomState>,
        ahash::RandomState,
    > = IndexMap::default();
    if let Some(target) = target.as_ref() {
        basic_instrs = ["barrier", "snapshot", "store"]
            .into_iter()
            .map(|x| x.to_string())
            .collect();
        let non_global_str: IndexSet<&str, ahash::RandomState> =
            if let Some(operations) = non_global_operations.as_ref() {
                operations.iter().map(|x| x.as_str()).collect()
            } else {
                IndexSet::default()
            };
        let target_keys = target.keys().collect::<IndexSet<_, ahash::RandomState>>();
        new_target_basis = target_keys
            .difference(&non_global_str)
            .map(|x| x.to_string())
            .collect();
        extract_basis_target(
            py,
            &dag,
            &mut source_basis,
            &mut qargs_local_source_basis,
            min_qubits,
            &qargs_with_non_global_operation,
        )?;
    } else {
        basic_instrs = ["measure", "reset", "barrier", "snapshot", "delay", "store"]
            .into_iter()
            .map(|x| x.to_string())
            .collect();
        source_basis = extract_basis(py, &dag, min_qubits)?;
        new_target_basis = target_basis.unwrap().into_iter().collect();
    }
    new_target_basis = new_target_basis
        .union(&basic_instrs)
        .map(|x| x.to_string())
        .collect();
    // If the source basis is a subset of the target basis and we have no circuit
    // instructions on qargs that have non-global operations there is nothing to
    // translate and we can exit early.
    let source_basis_names: IndexSet<String> = source_basis.iter().map(|x| x.0.clone()).collect();
    if source_basis_names.is_subset(&new_target_basis) && qargs_local_source_basis.is_empty() {
        return Ok(dag);
    }
    let basis_transforms = basis_search(equiv_lib, &source_basis, &new_target_basis);
    let mut qarg_local_basis_transforms: IndexMap<
        &PhysicalQargs,
        Vec<(GateIdentifier, BasisTransformIn)>,
        ahash::RandomState,
    > = IndexMap::default();
    for (qargs, local_source_basis) in qargs_local_source_basis.iter() {
        // For any multiqubit operation that contains a subset of qubits that
        // has a non-local operation, include that non-local operation in the
        // search. This matches with the check we did above to include those
        // subset non-local operations in the check here.
        let mut expanded_target = new_target_basis.clone();
        // Qargs are always guaranteed to be concrete based on `extract_basis_target`.
        if qargs.len() > 1 {
            let qarg_as_set: IndexSet<PhysicalQubit> = IndexSet::from_iter(qargs.iter().copied());
            for (non_local_qarg, local_basis) in qargs_with_non_global_operation.iter() {
                if let Qargs::Concrete(non_local_qarg) = non_local_qarg {
                    let non_local_qarg_as_set: IndexSet<PhysicalQubit, ahash::RandomState> =
                        IndexSet::from_iter(non_local_qarg.iter().copied());
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
            return Err(TranspilerError::new_err(format!(
                "Unable to translate the operations in the circuit: \
            {:?} to the backend's (or manually specified) target \
            basis: {:?}. This likely means the target basis is not universal \
            or there are additional equivalence rules needed in the EquivalenceLibrary being \
            used. For more details on this error see: \
            https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.passes.\
            BasisTranslator#translation-errors",
                local_source_basis
                    .iter()
                    .map(|x| x.0.as_str())
                    .collect_vec(),
                &expanded_target
            )));
        }
    }

    let Some(basis_transforms) = basis_transforms else {
        return Err(TranspilerError::new_err(format!(
            "Unable to translate the operations in the circuit: \
        {:?} to the backend's (or manually specified) target \
        basis: {:?}. This likely means the target basis is not universal \
        or there are additional equivalence rules needed in the EquivalenceLibrary being \
        used. For more details on this error see: \
        https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.passes. \
        BasisTranslator#translation-errors",
            source_basis.iter().map(|x| x.0.as_str()).collect_vec(),
            &new_target_basis
        )));
    };

    let instr_map: InstMap = compose_transforms(py, &basis_transforms, &source_basis, &dag)?;
    let extra_inst_map: ExtraInstructionMap = qarg_local_basis_transforms
        .iter()
        .map(|(qarg, transform)| -> PyResult<_> {
            Ok((
                *qarg,
                compose_transforms(py, transform, &qargs_local_source_basis[*qarg], &dag)?,
            ))
        })
        .collect::<PyResult<_>>()?;

    let (out_dag, _) = apply_translation(
        py,
        &dag,
        &new_target_basis,
        &instr_map,
        &extra_inst_map,
        min_qubits,
        &qargs_with_non_global_operation,
    )?;
    Ok(out_dag)
}

/// Method that extracts all gate instances identifiers from a DAGCircuit.
fn extract_basis(
    py: Python,
    circuit: &DAGCircuit,
    min_qubits: usize,
) -> PyResult<IndexSet<GateIdentifier, ahash::RandomState>> {
    let mut basis = IndexSet::default();
    // Recurse for DAGCircuit
    fn recurse_dag(
        py: Python,
        circuit: &DAGCircuit,
        basis: &mut IndexSet<GateIdentifier, ahash::RandomState>,
        min_qubits: usize,
    ) -> PyResult<()> {
        for (_node, operation) in circuit.op_nodes(true) {
            if circuit.get_qargs(operation.qubits).len() >= min_qubits {
                basis.insert((operation.op.name().to_string(), operation.op.num_qubits()));
            }
            if operation.op.control_flow() {
                let OperationRef::Instruction(inst) = operation.op.view() else {
                    unreachable!("Control flow operation is not an instance of PyInstruction.")
                };
                let inst_bound = inst.instruction.bind(py);
                for block in inst_bound.getattr("blocks")?.try_iter()? {
                    recurse_circuit(py, block?, basis, min_qubits)?;
                }
            }
        }
        Ok(())
    }

    // Recurse for QuantumCircuit
    fn recurse_circuit(
        py: Python,
        circuit: Bound<PyAny>,
        basis: &mut IndexSet<GateIdentifier, ahash::RandomState>,
        min_qubits: usize,
    ) -> PyResult<()> {
        let circuit_data: PyRef<CircuitData> = circuit
            .getattr(intern!(py, "_data"))?
            .downcast_into()?
            .borrow();
        for (index, inst) in circuit_data.iter().enumerate() {
            let instruction_object = circuit.get_item(index)?;
            if circuit_data.get_qargs(inst.qubits).len() >= min_qubits {
                basis.insert((inst.op.name().to_string(), inst.op.num_qubits()));
            }
            if inst.op.control_flow() {
                let operation_ob = instruction_object.getattr(intern!(py, "operation"))?;
                let blocks = operation_ob.getattr("blocks")?;
                for block in blocks.try_iter()? {
                    recurse_circuit(py, block?, basis, min_qubits)?;
                }
            }
        }
        Ok(())
    }

    recurse_dag(py, circuit, &mut basis, min_qubits)?;
    Ok(basis)
}

/// Method that extracts a mapping of all the qargs in the local_source basis
/// obtained from the [Target], to all gate instances identifiers from a DAGCircuit.
/// When dealing with `ControlFlowOp` instances the function will perform a recursion call
/// to a variant design to handle instances of `QuantumCircuit`.
fn extract_basis_target(
    py: Python,
    dag: &DAGCircuit,
    source_basis: &mut IndexSet<GateIdentifier, ahash::RandomState>,
    qargs_local_source_basis: &mut IndexMap<
        PhysicalQargs,
        IndexSet<GateIdentifier, ahash::RandomState>,
        ahash::RandomState,
    >,
    min_qubits: usize,
    qargs_with_non_global_operation: &IndexMap<
        Qargs,
        IndexSet<String, ahash::RandomState>,
        ahash::RandomState,
    >,
) -> PyResult<()> {
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
        let physical_qargs_as_set: IndexSet<PhysicalQubit, ahash::RandomState> =
            IndexSet::from_iter(physical_qargs.iter().copied());
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
                    let incomplete_qargs: IndexSet<PhysicalQubit, ahash::RandomState> =
                        IndexSet::from_iter(incomplete_qargs.iter().copied());
                    physical_qargs_as_set.is_superset(&incomplete_qargs)
                })
        {
            let qargs_from_set: PhysicalQargs = physical_qargs_as_set.into_iter().collect();
            qargs_local_source_basis
                .entry(qargs_from_set)
                .and_modify(|set| {
                    set.insert((node_obj.op.name().to_string(), node_obj.op.num_qubits()));
                })
                .or_insert(IndexSet::from_iter([(
                    node_obj.op.name().to_string(),
                    node_obj.op.num_qubits(),
                )]));
        } else {
            source_basis.insert((node_obj.op.name().to_string(), node_obj.op.num_qubits()));
        }
        if node_obj.op.control_flow() {
            let OperationRef::Instruction(op) = node_obj.op.view() else {
                unreachable!("Control flow op is not a control flow op. But control_flow is `true`")
            };
            let bound_inst = op.instruction.bind(py);
            // TODO: Use Rust method `op.blocks` instead of Python side extraction now that
            // the python-space method `QuantumCircuit.has_calibration_for`
            // has been removed and we don't need to account for it.
            let blocks = bound_inst.getattr("blocks")?.try_iter()?;
            for block in blocks {
                extract_basis_target_circ(
                    &block?,
                    source_basis,
                    qargs_local_source_basis,
                    min_qubits,
                    qargs_with_non_global_operation,
                )?;
            }
        }
    }
    Ok(())
}

/// Variant of extract_basis_target that takes an instance of QuantumCircuit.
/// This needs to use a Python instance of `QuantumCircuit` due to it needing
/// to access `has_calibration_for()` which is unavailable through rust. However,
/// this API will be removed with the deprecation of `Pulse`.
/// TODO: pulse is removed, we can use op.blocks
fn extract_basis_target_circ(
    circuit: &Bound<PyAny>,
    source_basis: &mut IndexSet<GateIdentifier, ahash::RandomState>,
    qargs_local_source_basis: &mut IndexMap<
        PhysicalQargs,
        IndexSet<GateIdentifier, ahash::RandomState>,
        ahash::RandomState,
    >,
    min_qubits: usize,
    qargs_with_non_global_operation: &IndexMap<
        Qargs,
        IndexSet<String, ahash::RandomState>,
        ahash::RandomState,
    >,
) -> PyResult<()> {
    let py = circuit.py();
    let circ_data_bound = circuit.getattr("_data")?.downcast_into::<CircuitData>()?;
    let circ_data = circ_data_bound.borrow();
    for node_obj in circ_data.iter() {
        let qargs = circ_data.get_qargs(node_obj.qubits);
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
        let physical_qargs_as_set: IndexSet<PhysicalQubit, ahash::RandomState> =
            IndexSet::from_iter(physical_qargs.iter().copied());
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
                    let incomplete_qargs: IndexSet<PhysicalQubit, ahash::RandomState> =
                        IndexSet::from_iter(incomplete_qargs.iter().copied());
                    physical_qargs_as_set.is_superset(&incomplete_qargs)
                })
        {
            let qargs_from_set: PhysicalQargs = physical_qargs_as_set.into_iter().collect();
            qargs_local_source_basis
                .entry(qargs_from_set)
                .and_modify(|set| {
                    set.insert((node_obj.op.name().to_string(), node_obj.op.num_qubits()));
                })
                .or_insert(IndexSet::from_iter([(
                    node_obj.op.name().to_string(),
                    node_obj.op.num_qubits(),
                )]));
        } else {
            source_basis.insert((node_obj.op.name().to_string(), node_obj.op.num_qubits()));
        }
        if node_obj.op.control_flow() {
            let OperationRef::Instruction(op) = node_obj.op.view() else {
                unreachable!("Control flow op is not a control flow op. But control_flow is `true`")
            };
            let bound_inst = op.instruction.bind(py);
            let blocks = bound_inst.getattr("blocks")?.try_iter()?;
            for block in blocks {
                extract_basis_target_circ(
                    &block?,
                    source_basis,
                    qargs_local_source_basis,
                    min_qubits,
                    qargs_with_non_global_operation,
                )?;
            }
        }
    }
    Ok(())
}

fn apply_translation(
    py: Python,
    dag: &DAGCircuit,
    target_basis: &IndexSet<String, ahash::RandomState>,
    instr_map: &InstMap,
    extra_inst_map: &ExtraInstructionMap,
    min_qubits: usize,
    qargs_with_non_global_operation: &IndexMap<
        Qargs,
        IndexSet<String, ahash::RandomState>,
        ahash::RandomState,
    >,
) -> PyResult<(DAGCircuit, bool)> {
    let mut is_updated = false;
    let out_dag = dag.copy_empty_like("alike")?;
    let mut out_dag_builder = out_dag.into_builder();
    for node in dag.topological_op_nodes()? {
        let node_obj = dag[node].unwrap_operation();
        let node_qarg = dag.get_qargs(node_obj.qubits);
        let node_carg = dag.get_cargs(node_obj.clbits);
        let qubit_set: IndexSet<Qubit, ahash::RandomState> =
            IndexSet::from_iter(node_qarg.iter().copied());
        let mut new_op: Option<OperationFromPython> = None;
        if target_basis.contains(node_obj.op.name()) || node_qarg.len() < min_qubits {
            if node_obj.op.control_flow() {
                let OperationRef::Instruction(control_op) = node_obj.op.view() else {
                    unreachable!("This instruction {} says it is of control flow type, but is not an Instruction instance", node_obj.op.name())
                };
                let mut flow_blocks = vec![];
                let bound_obj = control_op.instruction.bind(py);
                let blocks = bound_obj.getattr("blocks")?;
                for block in blocks.try_iter()? {
                    let block = block?;
                    let dag_block: DAGCircuit =
                        circuit_to_dag(py, block.extract()?, true, None, None)?;
                    let updated_dag: DAGCircuit;
                    (updated_dag, is_updated) = apply_translation(
                        py,
                        &dag_block,
                        target_basis,
                        instr_map,
                        extra_inst_map,
                        min_qubits,
                        qargs_with_non_global_operation,
                    )?;
                    let flow_circ_block = if is_updated {
                        DAG_TO_CIRCUIT
                            .get_bound(py)
                            .call1((updated_dag,))?
                            .extract()?
                    } else {
                        block
                    };
                    flow_blocks.push(flow_circ_block);
                }
                let replaced_blocks = bound_obj.call_method1("replace_blocks", (flow_blocks,))?;
                new_op = Some(replaced_blocks.extract()?);
            }
            if let Some(new_op) = new_op {
                out_dag_builder.apply_operation_back(
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
                )?;
            } else {
                out_dag_builder.apply_operation_back(
                    node_obj.op.clone(),
                    node_qarg,
                    node_carg,
                    node_obj.params.as_ref().map(|x| *x.clone()),
                    node_obj.label.as_deref().cloned(),
                    #[cfg(feature = "cache_pygates")]
                    None,
                )?;
            }
            continue;
        }
        let node_qarg_as_physical: Qargs = node_qarg.iter().map(|x| PhysicalQubit(x.0)).collect();
        if qargs_with_non_global_operation.contains_key(&node_qarg_as_physical)
            && qargs_with_non_global_operation[&node_qarg_as_physical].contains(node_obj.op.name())
        {
            out_dag_builder.apply_operation_back(
                node_obj.op.clone(),
                node_qarg,
                node_carg,
                node_obj.params.as_ref().map(|x| *x.clone()),
                node_obj.label.as_deref().cloned(),
                #[cfg(feature = "cache_pygates")]
                None,
            )?;
            continue;
        }

        let unique_qargs: PhysicalQargs = qubit_set.iter().map(|x| PhysicalQubit(x.0)).collect();
        if extra_inst_map.contains_key(&unique_qargs) {
            replace_node(
                py,
                &mut out_dag_builder,
                node_obj.clone(),
                &extra_inst_map[&unique_qargs],
            )?;
        } else if instr_map
            .contains_key(&(node_obj.op.name().to_string(), node_obj.op.num_qubits()))
        {
            replace_node(py, &mut out_dag_builder, node_obj.clone(), instr_map)?;
        } else {
            return Err(TranspilerError::new_err(format!(
                "BasisTranslator did not map {}",
                node_obj.op.name()
            )));
        }
        is_updated = true;
    }
    Ok((out_dag_builder.build(), is_updated))
}

fn replace_node(
    py: Python,
    dag: &mut DAGCircuitBuilder,
    node: PackedInstruction,
    instr_map: &IndexMap<GateIdentifier, (SmallVec<[Param; 3]>, DAGCircuit), ahash::RandomState>,
) -> PyResult<()> {
    let (target_params, target_dag) =
        &instr_map[&(node.op.name().to_string(), node.op.num_qubits())];
    if node.params_view().len() != target_params.len() {
        return Err(TranspilerError::new_err(format!(
            "Translation num_params not equal to op num_params. \
            Op: {:?} {} Translation: {:?}\n{:?}",
            node.params_view(),
            node.op.name(),
            &target_params,
            &target_dag
        )));
    }
    if node.params_view().is_empty() {
        for inner_index in target_dag.topological_op_nodes()? {
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
            let new_op = if inner_node.op.try_standard_gate().is_none() {
                inner_node.op.py_copy(py)?
            } else {
                inner_node.op.clone()
            };
            let new_params: SmallVec<[Param; 3]> = inner_node
                .params_view()
                .iter()
                .map(|param| param.clone_ref(py))
                .collect();
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
            )?;
        }
        dag.add_global_phase(target_dag.global_phase())?;
    } else {
        let parameter_map = target_params
            .iter()
            .zip(node.params_view())
            .into_py_dict(py)?;
        for inner_index in target_dag.topological_op_nodes()? {
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
            let new_op = if inner_node.op.try_standard_gate().is_none() {
                inner_node.op.py_copy(py)?
            } else {
                inner_node.op.clone()
            };
            let mut new_params: SmallVec<[Param; 3]> = inner_node
                .params_view()
                .iter()
                .map(|param| param.clone_ref(py))
                .collect();
            if inner_node
                .params_view()
                .iter()
                .any(|param| matches!(param, Param::ParameterExpression(_)))
            {
                new_params = SmallVec::new();
                for param in inner_node.params_view() {
                    if let Param::ParameterExpression(param_obj) = param {
                        let bound_param = param_obj.bind(py);
                        let exp_params = param.iter_parameters(py)?;
                        let bind_dict = PyDict::new(py);
                        for key in exp_params {
                            let key = key?;
                            bind_dict.set_item(&key, parameter_map.get_item(&key)?)?;
                        }
                        let mut new_value: Bound<PyAny>;
                        let comparison = bind_dict.values().iter().any(|param| {
                            param
                                .is_instance(PARAMETER_EXPRESSION.get_bound(py))
                                .is_ok_and(|x| x)
                        });
                        if comparison {
                            new_value = bound_param.clone();
                            for items in bind_dict.items() {
                                new_value = new_value.call_method1(
                                    intern!(py, "assign"),
                                    items.downcast::<PyTuple>()?,
                                )?;
                            }
                        } else {
                            new_value =
                                bound_param.call_method1(intern!(py, "bind"), (&bind_dict,))?;
                        }
                        let eval = new_value.getattr(intern!(py, "parameters"))?;
                        if eval.is_empty()? {
                            new_value = new_value.call_method0(intern!(py, "numeric"))?;
                        }
                        new_params.push(new_value.extract()?);
                    } else {
                        new_params.push(param.clone_ref(py));
                    }
                }
                if new_op.try_standard_gate().is_none() {
                    match new_op.view() {
                        OperationRef::Instruction(inst) => inst
                            .instruction
                            .bind(py)
                            .setattr("params", new_params.clone())?,
                        OperationRef::Gate(gate) => {
                            gate.gate.bind(py).setattr("params", new_params.clone())?
                        }
                        OperationRef::Operation(oper) => oper
                            .operation
                            .bind(py)
                            .setattr("params", new_params.clone())?,
                        _ => (),
                    }
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
            )?;
        }

        match target_dag.global_phase() {
            Param::ParameterExpression(old_phase) => {
                let bound_old_phase = old_phase.bind(py);
                let bind_dict = PyDict::new(py);
                for key in target_dag.global_phase().iter_parameters(py)? {
                    let key = key?;
                    bind_dict.set_item(&key, parameter_map.get_item(&key)?)?;
                }
                let mut new_phase: Bound<PyAny>;
                if bind_dict.values().iter().any(|param| {
                    param
                        .is_instance(PARAMETER_EXPRESSION.get_bound(py))
                        .is_ok_and(|x| x)
                }) {
                    new_phase = bound_old_phase.clone();
                    for key_val in bind_dict.items() {
                        new_phase =
                            new_phase.call_method1(intern!(py, "assign"), key_val.downcast()?)?;
                    }
                } else {
                    new_phase = bound_old_phase.call_method1(intern!(py, "bind"), (bind_dict,))?;
                }
                if !new_phase.getattr(intern!(py, "parameters"))?.is_truthy()? {
                    new_phase = new_phase.call_method0(intern!(py, "numeric"))?;
                    if new_phase.is_instance(&PyComplex::type_object(py))? {
                        return Err(TranspilerError::new_err(format!(
                            "Global phase must be real, but got {}",
                            new_phase.repr()?
                        )));
                    }
                }
                let new_phase: Param = new_phase.extract()?;
                dag.add_global_phase(&new_phase)?;
            }

            Param::Float(_) => {
                dag.add_global_phase(target_dag.global_phase())?;
            }

            _ => {}
        }
    }

    Ok(())
}

pub fn basis_translator_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(run_basis_translator))?;
    Ok(())
}

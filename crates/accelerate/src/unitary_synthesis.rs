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
#![allow(clippy::too_many_arguments)]

#[cfg(feature = "cache_pygates")]
use std::cell::OnceCell;

use std::f64::consts::PI;

use approx::relative_eq;
use core::panic;
use hashbrown::{HashMap, HashSet};
use indexmap::IndexMap;
use ndarray::prelude::*;
use num_complex::{Complex, Complex64};
use numpy::IntoPyArray;
use pyo3::conversion::FromPyObjectBound;
use qiskit_circuit::circuit_instruction::ExtraInstructionAttributes;
use smallvec::{smallvec, SmallVec};

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict, PyList, PyString};
use pyo3::wrap_pyfunction;
use pyo3::Python;

use rustworkx_core::petgraph::stable_graph::NodeIndex;

use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType};
use qiskit_circuit::imports;
use qiskit_circuit::operations::{Operation, OperationRef, Param, StandardGate};
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperation, PackedOperationType};
use qiskit_circuit::Qubit;

use crate::euler_one_qubit_decomposer::{
    unitary_to_gate_sequence_inner, EulerBasis, EulerBasisSet, EULER_BASES, EULER_BASIS_NAMES,
};
use crate::nlayout::PhysicalQubit;
use crate::target_transpiler::{NormalOperation, Target};
use crate::two_qubit_decompose::{
    TwoQubitBasisDecomposer, TwoQubitGateSequence, TwoQubitWeylDecomposition,
};
use crate::QiskitError;

const PI2: f64 = PI / 2.;
const PI4: f64 = PI / 4.;

#[derive(Clone, Debug)]
enum DecomposerType {
    TwoQubitBasisDecomposer(Box<TwoQubitBasisDecomposer>),
    XXDecomposer(PyObject),
}

struct DecomposerElement {
    decomposer: DecomposerType,
    gate: NormalOperation,
}

#[derive(Clone, Debug)]
enum UnitarySynthesisReturnType {
    DAGType(Box<DAGCircuit>),
    TwoQSequenceType(TwoQubitUnitarySequence),
}
#[derive(Clone, Debug)]
struct TwoQubitUnitarySequence {
    gate_sequence: TwoQubitGateSequence,
    decomp_gate: NormalOperation,
}

static GOODBYE_SET: [&str; 3] = ["cx", "cz", "ecr"];

fn dag_from_2q_gate_sequence(
    py: Python<'_>,
    sequence: TwoQubitUnitarySequence,
    out_dag_info: Option<(DAGCircuit, &[Qubit])>,
) -> PyResult<DAGCircuit> {
    let qubit: &Bound<PyAny> = imports::QUBIT.get_bound(py);

    let (mut target_dag, mut out_qargs) = match out_dag_info {
        Some((dag, qargs)) => (dag, qargs.to_vec()),
        None => {
            let mut out_qargs: Vec<Qubit> = vec![];
            let mut target_dag = DAGCircuit::new(py)?;
            let qubit_obj = qubit.call0()?;
            out_qargs.push(target_dag.add_qubit_unchecked(py, &qubit_obj)?);
            (target_dag, out_qargs)
        }
    };

    let _ = target_dag.add_global_phase(py, &Param::Float(sequence.gate_sequence.global_phase));

    let mut instructions = Vec::new();
    for (gate, params, qubit_ids) in &sequence.gate_sequence.gates {
        let gate_node = match gate {
            None => sequence.decomp_gate.operation.standard_gate(),
            Some(gate) => *gate,
        };

        let mut mapped_qargs = Vec::new();
        for id in qubit_ids {
            while *id as usize >= out_qargs.len() {
                let qubit_obj = qubit.call0()?;
                out_qargs.push(target_dag.add_qubit_unchecked(py, &qubit_obj)?);
            }
            mapped_qargs.push(out_qargs[*id as usize]);
        }

        let new_params: Option<Box<SmallVec<[Param; 3]>>> = match gate {
            Some(_) => Some(Box::new(params.iter().map(|p| Param::Float(*p)).collect())),
            None => Some(Box::new(sequence.decomp_gate.params.clone())),
        };

        let pi = PackedInstruction {
            op: PackedOperation::from_standard(gate_node),
            qubits: target_dag.qargs_interner.insert(&mapped_qargs),
            clbits: target_dag.cargs_interner.get_default(),
            params: new_params,
            extra_attrs: ExtraInstructionAttributes::new(None, None, None, None),
            #[cfg(feature = "cache_pygates")]
            py_op: OnceCell::new(),
        };
        instructions.push(pi);
    }

    let _ = target_dag.extend(py, instructions.into_iter());

    Ok(target_dag.clone())
}

fn get_target_basis_set(target: &Target, qubit: PhysicalQubit) -> EulerBasisSet {
    let mut target_basis_set: EulerBasisSet = EulerBasisSet::new();
    let target_basis_list = target.operation_names_for_qargs(Some(&smallvec![qubit]));
    match target_basis_list {
        Ok(basis_list) => {
            EULER_BASES
                .iter()
                .enumerate()
                .filter_map(|(idx, gates)| {
                    if !gates.iter().all(|gate| basis_list.contains(gate)) {
                        return None;
                    }
                    let basis = EULER_BASIS_NAMES[idx];
                    Some(basis)
                })
                .for_each(|basis| target_basis_set.add_basis(basis));
        }
        Err(_) => target_basis_set.support_all(),
    }

    if target_basis_set.basis_supported(EulerBasis::U3)
        && target_basis_set.basis_supported(EulerBasis::U321)
    {
        target_basis_set.remove(EulerBasis::U3);
    }
    if target_basis_set.basis_supported(EulerBasis::ZSX)
        && target_basis_set.basis_supported(EulerBasis::ZSXX)
    {
        target_basis_set.remove(EulerBasis::ZSX);
    }
    target_basis_set
}
// This is the outer-most run function. It is meant to be called from Python inside `UnitarySynthesis.run()`
// This loop iterates over the dag and calls `run_2q_unitary_synthesis`
#[pyfunction]
#[pyo3(name = "run_default_main_loop")]
fn py_run_default_main_loop(
    py: Python,
    dag: &mut DAGCircuit,
    qubit_indices: &Bound<'_, PyList>,
    min_qubits: usize,
    target: &Target,
    approximation_degree: Option<f64>,
    coupling_edges: Option<&Bound<'_, PyAny>>,
    natural_direction: Option<bool>,
) -> PyResult<DAGCircuit> {
    let circuit_to_dag = imports::CIRCUIT_TO_DAG.get_bound(py);
    let dag_to_circuit = imports::DAG_TO_CIRCUIT.get_bound(py);

    let node_ids: Vec<NodeIndex> = dag.op_nodes(false).collect();
    for node in node_ids {
        if let NodeType::Operation(inst) = &dag.dag()[node] {
            if inst.op.control_flow() {
                if let OperationRef::Instruction(py_inst) = inst.op.view() {
                    let raw_blocks: Vec<PyResult<Bound<PyAny>>> = py_inst
                        .instruction
                        .getattr(py, "blocks")?
                        .bind(py)
                        .iter()?
                        .collect();
                    let mut new_blocks = Vec::with_capacity(raw_blocks.len());
                    for raw_block in raw_blocks {
                        let new_ids = dag.get_qargs(inst.qubits).iter().map(|qarg| {
                            qubit_indices
                                .get_item(qarg.0 as usize)
                                .expect("Unexpected index error in DAG")
                        });
                        let res = py_run_default_main_loop(
                            py,
                            &mut circuit_to_dag.call1((raw_block?,))?.extract()?,
                            &PyList::new_bound(py, new_ids),
                            min_qubits,
                            target,
                            approximation_degree,
                            coupling_edges,
                            natural_direction,
                        )?;
                        new_blocks.push(dag_to_circuit.call1((res,))?);
                    }
                    let old_node = dag.get_node(py, node)?.clone();
                    let new_node = py_inst
                        .instruction
                        .bind(py)
                        .call_method1("replace_blocks", (new_blocks,))?;
                    let _ = dag.substitute_node(old_node.bind(py), &new_node, true, false);
                }
            }
        }
    }

    let mut out_dag = dag.copy_empty_like(py, "alike")?;

    // Iterate over nodes, find decomposers and run synthesis
    for node in dag.topological_op_nodes()? {
        if let NodeType::Operation(packed_instr) = &dag.dag()[node] {
            if packed_instr.op.name() == "unitary"
                && packed_instr.op.num_qubits() >= min_qubits as u32
            {
                let unitary: Array<Complex<f64>, Dim<[usize; 2]>> =
                    match packed_instr.op.matrix(&[]) {
                        Some(unitary) => unitary,
                        None => return Err(QiskitError::new_err("Unitary not found")),
                    };
                match unitary.shape() {
                    // Run 1Q synthesis
                    [2, 2] => {
                        let qubit = dag.get_qargs(packed_instr.qubits)[0];
                        let target_basis_set =
                            get_target_basis_set(target, PhysicalQubit::new(qubit.0));
                        let sequence = unitary_to_gate_sequence_inner(
                            unitary.view(),
                            &target_basis_set,
                            qubit.0 as usize,
                            None,
                            true,
                            None,
                        )
                        .unwrap();

                        for gate in sequence.gates {
                            out_dag.insert_1q_on_incoming_qubit(
                                (gate.0, &gate.1),
                                NodeIndex::new(qubit.0 as usize),
                            );
                        }
                        out_dag.add_global_phase(py, &Param::Float(sequence.global_phase))?;
                    }
                    // Run 2Q synthesis
                    [4, 4] => {
                        let ref_qargs = dag.get_qargs(packed_instr.qubits);
                        // How to use ref_qubits:
                        // * index = output qubit from synthesis algorithm
                        // * value = correspoding physical qubit in dag/out_dag
                        let ref_qubits: [PhysicalQubit; 2] = [
                            PhysicalQubit::new(
                                qubit_indices.get_item(ref_qargs[0].0 as usize)?.extract()?,
                            ),
                            PhysicalQubit::new(
                                qubit_indices.get_item(ref_qargs[1].0 as usize)?.extract()?,
                            ),
                        ];
                        // The 2Q synth. output can be None, a DAGCircuit or a TwoQubitGateSequence
                        let raw_synth_output: Option<UnitarySynthesisReturnType> =
                            run_2q_unitary_synthesis(
                                py,
                                unitary,
                                &ref_qubits,
                                approximation_degree,
                                &coupling_edges,
                                natural_direction,
                                target,
                            )?;

                        let out_qargs = dag.get_qargs(packed_instr.qubits);
                        match raw_synth_output {
                            None => {
                                let _ = out_dag.push_back(py, packed_instr.clone());
                            }

                            Some(synth_output) => match synth_output {
                                UnitarySynthesisReturnType::DAGType(synth_dag) => {
                                    out_dag.add_global_phase(py, &synth_dag.get_global_phase())?;

                                    for out_node in synth_dag.topological_op_nodes()? {
                                        if let NodeType::Operation(mut out_packed_instr) =
                                            synth_dag.dag()[out_node].clone()
                                        {
                                            let synth_qargs =
                                                synth_dag.get_qargs(out_packed_instr.qubits);
                                            let mapped_qargs: Vec<Qubit> = synth_qargs
                                                .iter()
                                                .map(|qarg| out_qargs[qarg.0 as usize])
                                                .collect();

                                            out_packed_instr.qubits =
                                                out_dag.qargs_interner.insert(&mapped_qargs);

                                            let _ = out_dag.push_back(py, out_packed_instr.clone());
                                        }
                                    }
                                }
                                UnitarySynthesisReturnType::TwoQSequenceType(sequence) => {
                                    out_dag = dag_from_2q_gate_sequence(
                                        py,
                                        sequence,
                                        Some((out_dag, out_qargs)),
                                    )?;
                                }
                            },
                        }
                    }
                    // Run 3Q+ synthesis
                    _ => {
                        let qs_decomposition: &Bound<'_, PyAny> =
                            imports::QS_DECOMPOSITION.get_bound(py);
                        let synth_circ =
                            qs_decomposition.call1((unitary.clone().into_pyarray_bound(py),))?;
                        let synth_dag = circuit_to_dag.call1((synth_circ,))?.extract()?;
                        out_dag = synth_dag;
                    }
                }
            } else {
                let _ = out_dag.push_back(py, packed_instr.clone());
            }
        }
    }
    Ok(out_dag)
}

fn run_2q_unitary_synthesis(
    py: Python,
    unitary: Array2<Complex64>,
    ref_qubits: &[PhysicalQubit; 2],
    approximation_degree: Option<f64>,
    coupling_edges: &Option<&Bound<'_, PyAny>>,
    natural_direction: Option<bool>,
    target: &Target,
) -> PyResult<Option<UnitarySynthesisReturnType>> {
    // run 2q decomposition (in Rust except for XXDecomposer) -> Return types will vary.
    // step1: select decomposers
    let decomposers = {
        // let ref_qubits = ref_qubits;
        let decomposers_2q =
            get_2q_decomposers_from_target(py, target, ref_qubits, approximation_degree)?;
        match decomposers_2q {
            Some(decomp) => decomp,
            None => Vec::new(),
        }
    };

    // If we have a single TwoQubitBasisDecomposer, skip dag creation as we don't need to
    // store and can instead manually create the synthesized gates directly in the output dag
    if decomposers.len() == 1 {
        let decomposer_item = decomposers.first().unwrap();
        if let DecomposerType::TwoQubitBasisDecomposer(_) = decomposer_item.decomposer {
            let preferred_dir = preferred_direction(
                decomposer_item,
                ref_qubits,
                natural_direction,
                coupling_edges,
                target,
            )?;
            let synth = synth_su4_no_dag(
                py,
                &unitary,
                decomposer_item,
                preferred_dir,
                approximation_degree,
            )?;
            return Ok(Some(synth));
        }
    }

    let synth_circuits: Vec<PyResult<UnitarySynthesisReturnType>> = decomposers
        .iter()
        .map(|decomposer| {
            let preferred_dir = preferred_direction(
                decomposer,
                ref_qubits,
                natural_direction,
                coupling_edges,
                target,
            )?;
            synth_su4(
                py,
                &unitary,
                decomposer,
                preferred_dir,
                approximation_degree,
            )
        })
        .collect();

    fn compute_2q_error(
        py: Python<'_>,
        synth_circuit: &UnitarySynthesisReturnType,
        target: &Target,
        ref_qubits: &[PhysicalQubit; 2],
    ) -> f64 {
        let mut gate_fidelities = Vec::new();
        let mut score_instruction = |instruction: &PackedInstruction,
                                     inst_qubits: &SmallVec<[PhysicalQubit; 2]>|
         -> PyResult<()> {
            match target.operation_names_for_qargs(Some(inst_qubits)) {
                    Ok(names) => {
                        for name in names {
                            let target_op = target.operation_from_name(name).unwrap();
                            let are_params_close = if let Some(params) = &instruction.params {
                                    params.iter().zip(target_op.params.iter()).all(|(p1, p2)| {
                                        p1.is_close(py, p2, 1e-10)
                                            .expect("Unexpected parameter expression error.")
                                    })
                            } else {
                                false
                            };
                            let is_parametrized = target_op
                                .params
                                .iter()
                                .any(|param| matches!(param, Param::ParameterExpression(_)));
                            if target_op.operation.name() == instruction.op.name()
                                && (is_parametrized || are_params_close)
                            {
                                match target[name].get(Some(inst_qubits)) {
                                    Some(Some(props)) => gate_fidelities.push(
                                        1.0 - props.error.unwrap_or(0.0)
                                    ),
                                    _ => gate_fidelities.push(1.0),
                                }
                                break;
                            }
                        }
                        Ok(())
                    }
                    Err(_) => {
                        Err(QiskitError::new_err(
                            format!("Encountered a bad synthesis. Target has no {instruction:?} on qubits {inst_qubits:?}.")
                        ))
                    }
                }
        };

        if let UnitarySynthesisReturnType::DAGType(synth_dag) = synth_circuit {
            for node in synth_dag
                .topological_op_nodes()
                .expect("Unexpected error in dag.topological_op_nodes()")
            {
                if let NodeType::Operation(inst) = &synth_dag.dag()[node] {
                    let inst_qubits = synth_dag
                        .get_qargs(inst.qubits)
                        .iter()
                        .map(|q| ref_qubits[q.0 as usize])
                        .collect();
                    let _ = score_instruction(inst, &inst_qubits);
                }
            }
        } else {
            panic!("Synth output is not a DAG");
        }
        1.0 - gate_fidelities.into_iter().product::<f64>()
    }

    let synth_circuit: Option<UnitarySynthesisReturnType> = if !synth_circuits.is_empty() {
        let mut synth_errors = Vec::new();
        let mut synth_circuits_filt = Vec::new();

        for circuit in synth_circuits.iter().flatten() {
            let error = compute_2q_error(py, circuit, target, ref_qubits);
            synth_errors.push(error);
            synth_circuits_filt.push(circuit);
        }

        synth_errors
            .iter()
            .enumerate()
            .min_by(|error1, error2| error1.1.partial_cmp(error2.1).unwrap())
            .map(|(index, _)| synth_circuits_filt[index].clone())
    } else {
        None
    };
    Ok(synth_circuit)
    // The output at this point will be a DAG, the sequence may be returned in the special case for TwoQubitBasisDecomposer
}

// This function collects a bunch of decomposer instances that will be used in `run_2q_unitary_synthesis`
fn get_2q_decomposers_from_target(
    py: Python,
    target: &Target,
    qubits: &[PhysicalQubit; 2],
    approximation_degree: Option<f64>,
) -> PyResult<Option<Vec<DecomposerElement>>> {
    let qubits: SmallVec<[PhysicalQubit; 2]> = SmallVec::from_buf(*qubits);
    let reverse_qubits: SmallVec<[PhysicalQubit; 2]> = qubits.iter().rev().copied().collect();
    // HERE: caching
    // TODO: here return cache --> implementation?
    let mut available_2q_basis: IndexMap<&str, NormalOperation> = IndexMap::new();
    let mut available_2q_props: IndexMap<&str, (Option<f64>, Option<f64>)> = IndexMap::new();

    // try both directions for the qubits tuple
    let mut qubit_gate_map = IndexMap::new();
    match target.operation_names_for_qargs(Some(&qubits)) {
        Ok(direct_keys) => {
            qubit_gate_map.insert(qubits.clone(), direct_keys);
            if let Ok(reverse_keys) = target.operation_names_for_qargs(Some(&reverse_qubits)) {
                qubit_gate_map.insert(reverse_qubits.clone(), reverse_keys);
            }
        }
        Err(_) => {
            if let Ok(reverse_keys) = target.operation_names_for_qargs(Some(&reverse_qubits)) {
                qubit_gate_map.insert(reverse_qubits.clone(), reverse_keys);
            } else {
                return Err(QiskitError::new_err(
                    "Target has no gates available on qubits to synthesize over.",
                ));
            }
        }
    }

    fn replace_parametrized_gate(mut op: NormalOperation) -> NormalOperation {
        if let Some(std_gate) = op.operation.try_standard_gate() {
            match std_gate.name() {
                "rxx" => {
                    if let Param::ParameterExpression(_) = op.params[0] {
                        op.params[0] = Param::Float(PI2)
                    }
                }
                "rzx" => {
                    if let Param::ParameterExpression(_) = op.params[0] {
                        op.params[0] = Param::Float(PI4)
                    }
                }
                "rzz" => {
                    if let Param::ParameterExpression(_) = op.params[0] {
                        op.params[0] = Param::Float(PI2)
                    }
                }
                _ => (),
            }
        }
        op
    }

    for (q_pair, gates) in &qubit_gate_map {
        for key in gates {
            match target.operation_from_name(key) {
                Ok(op) => {
                    // if it's not a gate, move on to next iteration
                    match op.operation.discriminant() {
                        PackedOperationType::Gate => (),
                        PackedOperationType::StandardGate => (),
                        _ => continue,
                    }

                    available_2q_basis.insert(key, replace_parametrized_gate(op.clone()));

                    match target.qargs_for_operation_name(key) {
                        Ok(_) => {
                            available_2q_props.insert(
                                key,
                                match &target[key].get(Some(q_pair)) {
                                    Some(Some(props)) => (props.duration, props.error),
                                    _ => (None, None),
                                },
                            );
                        }
                        _ => continue,
                    };
                }
                _ => continue,
            }
        }
    }

    if available_2q_basis.is_empty() {
        return Err(QiskitError::new_err(
            "Target has no gates available on qubits to synthesize over.",
        ));
    }

    let target_basis_set = get_target_basis_set(target, qubits[0]);

    let available_1q_basis: HashSet<&str> =
        HashSet::from_iter(target_basis_set.get_bases().map(|basis| basis.as_str()));

    // find all decomposers
    let mut decomposers: Vec<DecomposerElement> = Vec::new();

    #[inline]
    fn is_supercontrolled(op: &NormalOperation) -> bool {
        match op.operation.matrix(&op.params) {
            None => false,
            Some(unitary_matrix) => {
                let kak = TwoQubitWeylDecomposition::new_inner(unitary_matrix.view(), None, None)
                    .unwrap();
                relative_eq!(*kak.a(), PI4) && relative_eq!(*kak.c(), 0.0)
            }
        }
    }

    #[inline]
    fn is_controlled(op: &NormalOperation) -> bool {
        match op.operation.matrix(&op.params) {
            None => false,
            Some(unitary_matrix) => {
                let kak = TwoQubitWeylDecomposition::new_inner(unitary_matrix.view(), None, None)
                    .unwrap();
                relative_eq!(*kak.b(), 0.0) && relative_eq!(*kak.c(), 0.0)
            }
        }
    }

    // Iterate over 1q and 2q supercontrolled basis, append TwoQubitBasisDecomposers
    let supercontrolled_basis: IndexMap<&str, NormalOperation> = available_2q_basis
        .clone()
        .into_iter()
        .filter(|(_, v)| is_supercontrolled(v))
        .collect();

    for basis_1q in &available_1q_basis {
        for basis_2q in supercontrolled_basis.keys() {
            let mut basis_2q_fidelity: f64 = match available_2q_props.get(basis_2q) {
                Some(&(_, Some(e))) => 1.0 - e,
                _ => 1.0,
            };
            if let Some(approx_degree) = approximation_degree {
                basis_2q_fidelity *= approx_degree;
            }
            let gate = &supercontrolled_basis[basis_2q];
            let decomposer = TwoQubitBasisDecomposer::new_inner(
                gate.operation.name().to_owned(),
                gate.operation.matrix(&gate.params).unwrap().view(),
                basis_2q_fidelity,
                basis_1q,
                None,
            )?;

            decomposers.push(DecomposerElement {
                decomposer: DecomposerType::TwoQubitBasisDecomposer(Box::new(decomposer)),
                gate: gate.clone(),
            });
        }
    }

    // If our 2q basis gates are a subset of cx, ecr, or cz then we know TwoQubitBasisDecomposer
    // is an ideal decomposition and there is no need to bother calculating the XX embodiments
    // or try the XX decomposer
    let available_basis_set: HashSet<&str> = available_2q_basis.keys().copied().collect();

    #[inline]
    fn check_goodbye(basis_set: &HashSet<&str>) -> bool {
        !basis_set.iter().any(|gate| !GOODBYE_SET.contains(gate))
    }

    if check_goodbye(&available_basis_set) {
        // TODO: decomposer cache thingy
        return Ok(Some(decomposers));
    }

    // Let's now look for possible controlled decomposers (i.e. XXDecomposer)
    let controlled_basis: IndexMap<&str, NormalOperation> = available_2q_basis
        .into_iter()
        .filter(|(_, v)| is_controlled(v))
        .collect();
    let mut pi2_basis: Option<&str> = None;
    let xx_embodiments: &Bound<'_, PyAny> = imports::XX_EMBODIMENTS.get_bound(py);

    // The xx decomposer args are the interaction strength (f64), basis_2q_fidelity (f64),
    // and embodiments (Bound<'_, PyAny>).
    let xx_decomposer_args = controlled_basis.iter().map(
        |(name, op)| -> PyResult<(f64, f64, pyo3::Bound<'_, pyo3::PyAny>)> {
            let strength = 2.0
                * TwoQubitWeylDecomposition::new_inner(
                    op.operation.matrix(&op.params).unwrap().view(),
                    None,
                    None,
                )
                .unwrap()
                .a();
            let mut fidelity_value = match available_2q_props.get(name) {
                Some(&(_, error)) => 1.0 - error.unwrap_or(0.0),
                None => 1.0,
            };
            if let Some(approx_degree) = approximation_degree {
                fidelity_value *= approx_degree;
            }
            let mut embodiment =
                xx_embodiments.get_item(op.clone().into_py(py).getattr(py, "base_class")?)?; //XXEmbodiments[v.base_class];

            if embodiment.getattr("parameters")?.len()? == 1 {
                embodiment = embodiment.call_method1("assign_parameters", (vec![strength],))?;
            }
            // basis equivalent to CX are well optimized so use for the pi/2 angle if available
            if relative_eq!(strength, PI2) && supercontrolled_basis.contains_key(name) {
                pi2_basis = Some(op.operation.name());
            }
            Ok((strength, fidelity_value, embodiment))
        },
    );

    let basis_2q_fidelity_dict = PyDict::new_bound(py);
    let embodiments_dict = PyDict::new_bound(py);
    let mut not_empty = false;
    for (strength, fidelity, embodiment) in xx_decomposer_args.flatten() {
        not_empty = true;
        basis_2q_fidelity_dict.set_item(strength, fidelity)?;
        embodiments_dict.set_item(strength, embodiment.into_py(py))?;
    }

    // Iterate over 2q fidelities ans select decomposers
    if not_empty {
        let xx_decomposer: &Bound<'_, PyAny> = imports::XX_DECOMPOSER.get_bound(py);
        for basis_1q in &available_1q_basis {
            let pi2_decomposer = if let Some(pi_2_basis) = pi2_basis {
                if pi_2_basis == "cx" && *basis_1q == "ZSX" {
                    let fidelity = match approximation_degree {
                        Some(approx_degree) => approx_degree,
                        None => match &target["cx"][Some(&qubits)] {
                            Some(props) => 1.0 - props.error.unwrap_or(0.0),
                            None => 1.0,
                        },
                    };
                    Some(TwoQubitBasisDecomposer::new_inner(
                        pi_2_basis.to_string(),
                        StandardGate::CXGate.matrix(&[]).unwrap().view(),
                        fidelity,
                        basis_1q,
                        Some(true),
                    )?)
                } else {
                    None
                }
            } else {
                None
            };

            let decomposer = xx_decomposer.call1((
                &basis_2q_fidelity_dict,
                PyString::new_bound(py, basis_1q),
                &embodiments_dict,
                pi2_decomposer,
            ))?;
            let decomposer_gate = decomposer
                .getattr(intern!(py, "gate"))?
                .extract::<NormalOperation>()?;

            decomposers.push(DecomposerElement {
                decomposer: DecomposerType::XXDecomposer(decomposer.into()),
                gate: decomposer_gate,
            });
        }
    }
    Ok(Some(decomposers))
}

fn preferred_direction(
    decomposer: &DecomposerElement,
    ref_qubits: &[PhysicalQubit; 2],
    natural_direction: Option<bool>,
    coupling_edges: &Option<&Bound<'_, PyAny>>,
    target: &Target,
) -> PyResult<Option<bool>> {
    // Returns:
    // * true if gate qubits are in the hardware-native direction
    // * false if gate qubits must be flipped to match hardware-native direction

    let qubits: [PhysicalQubit; 2] = *ref_qubits;
    let mut reverse_qubits: [PhysicalQubit; 2] = qubits;
    reverse_qubits.reverse();

    let compute_cost =
        |lengths: bool, q_tuple: [PhysicalQubit; 2], in_cost: f64| -> PyResult<f64> {
            let cost = match target.qargs_for_operation_name(decomposer.gate.operation.name()) {
                Ok(_) => match target[decomposer.gate.operation.name()].get(Some(
                    &q_tuple
                        .into_iter()
                        .collect::<SmallVec<[PhysicalQubit; 2]>>(),
                )) {
                    Some(Some(_props)) => {
                        if lengths {
                            _props.duration.unwrap_or(in_cost)
                        } else {
                            _props.error.unwrap_or(in_cost)
                        }
                    }
                    _ => in_cost,
                },
                Err(_) => in_cost,
            };
            Ok(cost)
        };

    let preferred_direction = match natural_direction {
        Some(false) => None,
        _ => {
            // None or Some(true)
            if let Some(edges) = coupling_edges {
                let edge_set: HashSet<(u32, u32)> =
                    HashSet::from_py_object_bound(edges.as_borrowed())?;
                let zero_one = edge_set.contains(&(qubits[0].0, qubits[1].0));
                let one_zero = edge_set.contains(&(qubits[1].0, qubits[0].0));

                match (zero_one, one_zero) {
                    (true, false) => Some(true),
                    (false, true) => Some(false),
                    _ => {
                        let mut cost_0_1: f64 = f64::INFINITY;
                        let mut cost_1_0: f64 = f64::INFINITY;

                        // Try to find the cost in gate_lengths
                        cost_0_1 = compute_cost(true, qubits, cost_0_1)?;
                        cost_1_0 = compute_cost(true, reverse_qubits, cost_1_0)?;

                        // If no valid cost was found in gate_lengths, check gate_errors
                        if !(cost_0_1 < f64::INFINITY || cost_1_0 < f64::INFINITY) {
                            cost_0_1 = compute_cost(false, qubits, cost_0_1)?;
                            cost_1_0 = compute_cost(false, reverse_qubits, cost_1_0)?;
                        }

                        if cost_0_1 < cost_1_0 {
                            Some(true)
                        } else if cost_1_0 < cost_0_1 {
                            Some(false)
                        } else {
                            None
                        }
                    }
                }
            } else {
                None
            }
        }
    };

    if natural_direction == Some(true) && preferred_direction.is_none() {
        return Err(QiskitError::new_err(format!(
            concat!(
                "No preferred direction of gate on qubits {:?} ",
                "could be determined from coupling map or gate lengths / gate errors."
            ),
            qubits
        )));
    }

    Ok(preferred_direction)
}

// generic synth function for 2q gates (4x4)
// used in `run_2q_unitary_synthesis`
fn synth_su4(
    py: Python,
    su4_mat: &Array2<Complex64>,
    decomposer_2q: &DecomposerElement,
    preferred_direction: Option<bool>,
    approximation_degree: Option<f64>,
) -> PyResult<UnitarySynthesisReturnType> {
    // double check approximation_degree None
    let is_approximate = approximation_degree.is_none() || approximation_degree.unwrap() != 1.0;
    let synth_dag = match &decomposer_2q.decomposer {
        // the output will be a dag  in the relative basis
        DecomposerType::XXDecomposer(decomposer) => {
            let kwargs: HashMap<&str, bool> = [("approximate", is_approximate), ("use_dag", true)]
                .into_iter()
                .collect();
            // can we avoid cloning the matrix to pass it to python?
            decomposer
                .call_method_bound(
                    py,
                    intern!(py, "__call__"),
                    (su4_mat.clone().into_pyarray_bound(py),),
                    Some(&kwargs.into_py_dict_bound(py)),
                )?
                .extract::<DAGCircuit>(py)?
        }
        // the output will be a sequence in the relative basis
        DecomposerType::TwoQubitBasisDecomposer(decomposer) => {
            // we don't have access to basis_fidelity, right???
            let synth = decomposer.call_inner(su4_mat.view(), None, is_approximate, None)?;
            let sequence = TwoQubitUnitarySequence {
                gate_sequence: synth,
                decomp_gate: decomposer_2q.gate.clone(),
            };
            dag_from_2q_gate_sequence(py, sequence, None)?
        }
    };

    match preferred_direction {
        None => Ok(UnitarySynthesisReturnType::DAGType(Box::new(synth_dag))),
        Some(preferred_dir) => {
            let mut synth_direction: Option<Vec<u32>> = None;
            for node in synth_dag.topological_op_nodes()? {
                if let NodeType::Operation(inst) = &synth_dag.dag()[node] {
                    if inst.op.num_qubits() == 2 {
                        // not sure if these are the right qargs
                        let qargs = synth_dag.get_qargs(inst.qubits);
                        synth_direction = Some(vec![qargs[0].0, qargs[1].0]);
                    }
                }
            }
            // synth direction is in the relative basis
            match synth_direction {
                None => Ok(UnitarySynthesisReturnType::DAGType(Box::new(synth_dag))),
                Some(synth_direction) => {
                    let synth_dir = match synth_direction.as_slice() {
                        [0, 1] => true,
                        [1, 0] => false,
                        _ => panic!("Only 2 possible synth directions."),
                    };
                    if synth_dir != preferred_dir {
                        reversed_synth_su4(py, su4_mat, decomposer_2q, approximation_degree)
                    } else {
                        Ok(UnitarySynthesisReturnType::DAGType(Box::new(synth_dag)))
                    }
                }
            }
        }
    }
}

// special-case synth function for the TwoQubitBasisDecomposer
// used in `run_2q_unitary_synthesis`
fn synth_su4_no_dag(
    py: Python<'_>,
    su4_mat: &Array2<Complex64>,
    decomposer_2q: &DecomposerElement,
    preferred_direction: Option<bool>,
    approximation_degree: Option<f64>,
) -> PyResult<UnitarySynthesisReturnType> {
    let is_approximate = approximation_degree.is_none() || approximation_degree.unwrap() != 1.0;
    let synth = if let DecomposerType::TwoQubitBasisDecomposer(decomp) = &decomposer_2q.decomposer {
        decomp.call_inner(su4_mat.view(), None, is_approximate, None)?
    } else {
        panic!("synth su4 no dag should only be called for TwoQubitBasisDecomposer")
    };

    let sequence = TwoQubitUnitarySequence {
        gate_sequence: synth.clone(),
        decomp_gate: decomposer_2q.gate.clone(),
    };

    //synth_direction is calculated in terms of logical qubits
    match preferred_direction {
        None => Ok(UnitarySynthesisReturnType::TwoQSequenceType(sequence)),
        Some(preferred_dir) => {
            let mut synth_direction: Option<SmallVec<[u8; 2]>> = None;
            for (gate, _, qubits) in synth.gates {
                if gate.is_none() || gate.unwrap().name() == "cx" {
                    synth_direction = Some(qubits);
                }
            }

            match synth_direction {
                None => Ok(UnitarySynthesisReturnType::TwoQSequenceType(sequence)),
                Some(synth_direction) => {
                    let synth_dir = match synth_direction.as_slice() {
                        [0, 1] => true,
                        [1, 0] => false,
                        _ => panic!(),
                    };
                    if synth_dir != preferred_dir {
                        reversed_synth_su4(py, su4_mat, decomposer_2q, approximation_degree)
                    } else {
                        Ok(UnitarySynthesisReturnType::TwoQSequenceType(sequence))
                    }
                }
            }
        }
    }
}

// generic synth function for 2q gates (4x4) called from synth_su4
fn reversed_synth_su4(
    py: Python<'_>,
    su4_mat: &Array2<Complex64>,
    decomposer_2q: &DecomposerElement,
    approximation_degree: Option<f64>,
) -> PyResult<UnitarySynthesisReturnType> {
    let is_approximate = approximation_degree.is_none() || approximation_degree.unwrap() != 1.0;
    let mut su4_mat_mm = su4_mat.clone();

    // Swap rows 1 and 2
    let (mut row_1, mut row_2) = su4_mat_mm.multi_slice_mut((s![1, ..], s![2, ..]));
    azip!((x in &mut row_1, y in &mut row_2) (*x, *y) = (*y, *x));

    // Swap columns 1 and 2
    let (mut col_1, mut col_2) = su4_mat_mm.multi_slice_mut((s![.., 1], s![.., 2]));
    azip!((x in &mut col_1, y in &mut col_2) (*x, *y) = (*y, *x));

    let synth_dag = match &decomposer_2q.decomposer {
        DecomposerType::XXDecomposer(decomposer) => {
            // the output will be a dag in the relative basis
            let mut kwargs = HashMap::<&str, bool>::new();
            kwargs.insert("approximate", is_approximate);
            kwargs.insert("use_dag", true);
            decomposer
                .call_method_bound(
                    py,
                    "__call__",
                    (su4_mat_mm.clone().into_pyarray_bound(py),),
                    Some(&kwargs.into_py_dict_bound(py)),
                )?
                .extract::<DAGCircuit>(py)?
        }
        DecomposerType::TwoQubitBasisDecomposer(decomposer) => {
            // we don't have access to basis_fidelity, right???
            let synth = decomposer.call_inner(su4_mat_mm.view(), None, is_approximate, None)?;
            let sequence = TwoQubitUnitarySequence {
                gate_sequence: synth,
                decomp_gate: decomposer_2q.gate.clone(),
            };
            dag_from_2q_gate_sequence(py, sequence, None)?
        }
    };

    let mut target_dag = synth_dag.copy_empty_like(py, "alike")?;
    let flip_bits: [Qubit; 2] = [Qubit(1), Qubit(0)];

    for node in synth_dag.topological_op_nodes()? {
        if let NodeType::Operation(mut inst) = synth_dag.dag()[node].clone() {
            let qubits = synth_dag
                .qargs_interner
                .get(inst.qubits)
                .iter()
                // .rev()
                .map(|x| flip_bits[x.0 as usize])
                .collect();
            inst.qubits = target_dag.qargs_interner.insert_owned(qubits);
            let _ = target_dag.push_back(py, inst.clone());
        }
    }
    Ok(UnitarySynthesisReturnType::DAGType(Box::new(target_dag)))
}

#[pymodule]
pub fn unitary_synthesis(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_run_default_main_loop))?;
    Ok(())
}

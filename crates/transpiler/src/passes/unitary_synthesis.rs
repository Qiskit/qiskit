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

use std::f64::consts::PI;

use approx::relative_eq;
use hashbrown::{HashMap, HashSet};
use indexmap::{IndexMap, IndexSet};
use itertools::Itertools;
use nalgebra::{DMatrix, Matrix2};
use ndarray::prelude::*;
use num_complex::Complex64;
use numpy::PyReadonlyArray2;
use numpy::{IntoPyArray, ToPyArray};
use smallvec::SmallVec;

use pyo3::Python;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict, PyString, PyType};
use pyo3::wrap_pyfunction;

use qiskit_circuit::bit::QuantumRegister;
use qiskit_circuit::dag_circuit::{DAGCircuit, DAGCircuitBuilder, NodeType};
use qiskit_circuit::instruction::{Instruction, Parameters};
use qiskit_circuit::operations::{Operation, OperationRef, Param, PythonOperation, StandardGate};
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperation};
use qiskit_circuit::{BlocksMode, Qubit, VarsMode, imports};

use crate::QiskitError;
use crate::passes::optimize_clifford_t::CLIFFORD_T_GATE_NAMES;
use crate::target::{NormalOperation, Target, TargetOperation};
use crate::target::{Qargs, QargsRef};
use qiskit_circuit::PhysicalQubit;
use qiskit_synthesis::discrete_basis::solovay_kitaev::SolovayKitaevSynthesis;
use qiskit_synthesis::euler_one_qubit_decomposer::{
    EULER_BASES, EULER_BASIS_NAMES, EulerBasis, EulerBasisSet, unitary_to_gate_sequence_inner,
};
use qiskit_synthesis::qsd::quantum_shannon_decomposition;
use qiskit_synthesis::two_qubit_decompose::{
    RXXEquivalent, TwoQubitBasisDecomposer, TwoQubitControlledUDecomposer, TwoQubitGateSequence,
    TwoQubitWeylDecomposition,
};

const PI2: f64 = PI / 2.;
const PI4: f64 = PI / 4.;

/// The matcher for the set of standard gates that the TwoQubitControlledUDecomposer
/// supports
macro_rules! PARAM_SET {
    // Make sure that this is kept in sync with the static array PARAM_GATES below
    () => {
        StandardGate::RXX
            | StandardGate::RYY
            | StandardGate::RZZ
            | StandardGate::RZX
            | StandardGate::CRX
            | StandardGate::CRY
            | StandardGate::CRZ
            | StandardGate::CPhase
    };
}

// Make sure that this is kept in sync with the macro PARAM_SET above
static PARAM_SET_BASIS_GATES: [StandardGate; 8] = [
    StandardGate::RXX,
    StandardGate::RYY,
    StandardGate::RZZ,
    StandardGate::RZX,
    StandardGate::CRX,
    StandardGate::CRY,
    StandardGate::CRZ,
    StandardGate::CPhase,
];

/// The matcher for the set of standard gates that the TwoQubitBasisDecomposer
/// supports
macro_rules! TWO_QUBIT_BASIS_SET {
    // Make sure that this is kept in sync with the static array TWO_QUBIT_BASIS_SET_GATES below
    () => {
        StandardGate::CX
            | StandardGate::CY
            | StandardGate::CZ
            | StandardGate::CH
            | StandardGate::DCX
            | StandardGate::ISwap
            | StandardGate::ECR
    };
}

// Make sure this is kept in sync with the macro TWO_QUBIT_BASIS_SET above
static TWO_QUBIT_BASIS_SET_GATES: [StandardGate; 7] = [
    StandardGate::CX,
    StandardGate::CY,
    StandardGate::CZ,
    StandardGate::CH,
    StandardGate::DCX,
    StandardGate::ISwap,
    StandardGate::ECR,
];

pub(crate) use {PARAM_SET, TWO_QUBIT_BASIS_SET};

#[derive(Clone, Debug)]
enum DecomposerType {
    TwoQubitBasis(Box<TwoQubitBasisDecomposer>),
    TwoQubitControlledU(Box<TwoQubitControlledUDecomposer>),
    XX(Py<PyAny>),
}

#[derive(Clone, Debug)]
struct DecomposerElement {
    decomposer: DecomposerType,
    packed_op: PackedOperation,
}

#[derive(Clone, Debug)]
struct TwoQubitUnitarySequence {
    gate_sequence: TwoQubitGateSequence,
}

/// Stores unitary synthesis cache, to avoid recomputing the same information for
/// every synthesized gate. (In particular, we want to compute the basic approximations
/// used in the Solovay-Kitaev algorithm at most once).
struct UnitarySynthesisCache {
    solovay_kitaev: Option<SolovayKitaevSynthesis>,
}

impl UnitarySynthesisCache {
    fn new() -> Self {
        UnitarySynthesisCache {
            solovay_kitaev: None,
        }
    }

    fn get_solovay_kitaev(&mut self) -> &SolovayKitaevSynthesis {
        if self.solovay_kitaev.is_none() {
            self.solovay_kitaev = Some(
                SolovayKitaevSynthesis::new(
                    &[StandardGate::T, StandardGate::Tdg, StandardGate::H],
                    12,
                    None,
                    false,
                )
                .unwrap(),
            )
        }
        self.solovay_kitaev.as_ref().unwrap()
    }
}

/// Given a list of basis gates, find a corresponding euler basis to use.
/// This will determine the available 1q synthesis basis for different decomposers.
fn get_euler_basis_set(basis_list: &IndexSet<&str, ::ahash::RandomState>) -> EulerBasisSet {
    let mut euler_basis_set: EulerBasisSet = EulerBasisSet::new();
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
        .for_each(|basis| euler_basis_set.add_basis(basis));

    if euler_basis_set.basis_supported(EulerBasis::U3)
        && euler_basis_set.basis_supported(EulerBasis::U321)
    {
        euler_basis_set.remove(EulerBasis::U3);
    }
    if euler_basis_set.basis_supported(EulerBasis::ZSX)
        && euler_basis_set.basis_supported(EulerBasis::ZSXX)
    {
        euler_basis_set.remove(EulerBasis::ZSX);
    }
    euler_basis_set
}

/// Given a `Target`, find an euler basis that is supported for a specific `PhysicalQubit`.
/// This will determine the available 1q synthesis basis for different decomposers.
fn get_target_basis_set(target: &Target, qubit: PhysicalQubit) -> EulerBasisSet {
    let mut target_basis_set: EulerBasisSet = EulerBasisSet::new();
    let target_basis_list = target.operation_names_for_qargs(&[qubit]);
    match target_basis_list {
        Ok(basis_list) => {
            target_basis_set = get_euler_basis_set(&basis_list.into_iter().collect());
        }
        Err(_) => {
            target_basis_set.support_all();
            target_basis_set.remove(EulerBasis::U3);
            target_basis_set.remove(EulerBasis::ZSX);
        }
    }
    target_basis_set
}

static SKIP_NAMES_FOR_CLIFFORD_T: [&str; 9] = [
    "for_loop",
    "while_loop",
    "if_else",
    "switch_case",
    "box",
    "delay",
    "barrier",
    "reset",
    "measure",
];

/// Find whether the basis supported for a specific `PhysicalQubit` is is of the form Clifford+T.
fn is_clifford_t_basis_set(
    target: Option<&Target>,
    basis_gates: &HashSet<String>,
    qubit: PhysicalQubit,
) -> bool {
    match target {
        // If the target is specified, it is used.
        Some(target) => {
            let target_basis_list = target.operation_names_for_qargs(&[qubit]);
            match target_basis_list {
                Ok(basis_list) => basis_list.into_iter().all(|k| {
                    CLIFFORD_T_GATE_NAMES.contains(&k) || SKIP_NAMES_FOR_CLIFFORD_T.contains(&k)
                }),
                Err(_) => false,
            }
        }
        // Otherwise, basis_gates is used.
        None => basis_gates.into_iter().all(|k| {
            CLIFFORD_T_GATE_NAMES.contains(&k.as_str())
                || SKIP_NAMES_FOR_CLIFFORD_T.contains(&k.as_str())
        }),
    }
}

/// Apply synthesis output (`synth_dag`) to final `DAGCircuit` (`out_dag`).
/// `synth_dag` is a subgraph, and the `qubit_ids` are relative to the subgraph
///  size/orientation, so `out_qargs` is used to track the final qubit ids where
/// it should be applied.
fn apply_synth_dag(
    out_dag: &mut DAGCircuitBuilder,
    out_qargs: &[Qubit],
    synth_dag: &DAGCircuit,
) -> PyResult<()> {
    for out_node in synth_dag.topological_op_nodes(false)? {
        let mut out_packed_instr = synth_dag[out_node].unwrap_operation().clone();
        let synth_qargs = synth_dag.get_qargs(out_packed_instr.qubits);
        let mapped_qargs: Vec<Qubit> = synth_qargs
            .iter()
            .map(|qarg| out_qargs[qarg.0 as usize])
            .collect();
        out_packed_instr.qubits = out_dag.insert_qargs(&mapped_qargs);
        out_dag.push_back(out_packed_instr)?;
    }
    out_dag.add_global_phase(&synth_dag.get_global_phase())?;
    Ok(())
}

/// Apply synthesis output (`sequence`) to final `DAGCircuit` (`out_dag`).
/// `sequence` contains a representation of gates to be applied to a subgraph,
/// and the `qubit_ids` are relative to the subgraph size/orientation,
/// so `out_qargs` is used to track the final qubit ids where they should be applied.
fn apply_synth_sequence(
    out_dag: &mut DAGCircuitBuilder,
    out_qargs: &[Qubit],
    sequence: &TwoQubitUnitarySequence,
) -> PyResult<()> {
    for (gate, params, qubit_ids) in sequence.gate_sequence.gates() {
        let packed_op = gate;
        let mapped_qargs: Vec<Qubit> = qubit_ids.iter().map(|id| out_qargs[*id as usize]).collect();

        let new_op: PackedOperation = match packed_op.view() {
            OperationRef::Gate(gate) => Python::attach(|py| -> PyResult<PackedOperation> {
                let new_gate = gate.py_copy(py)?;
                new_gate.gate.setattr(py, "params", params)?;
                Ok(Box::new(new_gate).into())
            })?,
            OperationRef::StandardGate(_) => packed_op.clone(),
            _ => {
                return Err(QiskitError::new_err(
                    "Decomposed gate sequence contains unexpected operations.",
                ));
            }
        };

        out_dag.apply_operation_back(
            new_op,
            &mapped_qargs,
            &[],
            Some(Parameters::Params(
                params.iter().map(|x| Param::Float(*x)).collect(),
            )),
            None,
            #[cfg(feature = "cache_pygates")]
            None,
        )?;
    }
    out_dag.add_global_phase(&Param::Float(sequence.gate_sequence.global_phase()))?;
    Ok(())
}

/// Iterate over `DAGCircuit` to perform unitary synthesis.
/// For each eligible gate: find decomposers, select the synthesis
/// method with the highest fidelity score and apply decompositions. The available methods are:
///     * 1q synthesis: OneQubitEulerDecomposer, SolovayKitaevSynthesis
///     * 2q synthesis: TwoQubitBasisDecomposer, TwoQubitControlledUDecomposer, XXDecomposer (Python, only if target is provided)
///     * 3q+ synthesis: QuantumShannonDecomposer
/// This function is currently used in the Python `UnitarySynthesis`` transpiler pass as a replacement for the `_run_main_loop` method.
/// It returns a new `DAGCircuit` with the different synthesized gates.
#[pyfunction]
#[pyo3(name = "run_main_loop", signature=(dag, qubit_indices, min_qubits, target, basis_gates, synth_gates, coupling_edges, approximation_degree=None, natural_direction=None, pulse_optimize=None))]
pub fn py_unitary_synthesis(
    dag: &mut DAGCircuit,
    qubit_indices: Vec<usize>,
    min_qubits: usize,
    target: Option<&Target>,
    basis_gates: HashSet<String>,
    synth_gates: HashSet<String>,
    coupling_edges: HashSet<[PhysicalQubit; 2]>,
    approximation_degree: Option<f64>,
    natural_direction: Option<bool>,
    pulse_optimize: Option<bool>,
) -> PyResult<DAGCircuit> {
    run_unitary_synthesis(
        dag,
        qubit_indices,
        min_qubits,
        target,
        basis_gates,
        synth_gates,
        coupling_edges,
        approximation_degree,
        natural_direction,
        pulse_optimize,
        true,
    )
}

/// Synthesize a unitary matrix
fn synthesize_unitary_matrix(
    matrix: CowArray<Complex64, Ix2>,
    num_qubits: u32,
    qubit_indices: &[usize],
    coupling_edges: &HashSet<[PhysicalQubit; 2]>,
    target: Option<&Target>,
    basis_gates: &HashSet<String>,
    approximation_degree: Option<f64>,
    natural_direction: Option<bool>,
    pulse_optimize: Option<bool>,
    out_dag: &mut DAGCircuitBuilder,
    out_qargs: &[Qubit],
    run_python_decomposers: bool,
    mut apply_original_op: impl FnMut(&mut DAGCircuitBuilder) -> PyResult<()>,
    unitary_synthesis_cache: &mut UnitarySynthesisCache,
) -> PyResult<()> {
    match num_qubits {
        // Run 1q synthesis
        1 => {
            let qubit = out_qargs[0];

            // Special case when the basis set is of the form Clifford+T.
            if is_clifford_t_basis_set(target, basis_gates, PhysicalQubit::new(qubit.0)) {
                let solovay_kitaev = unitary_synthesis_cache.get_solovay_kitaev();
                let matrix_nalgebra = Matrix2::from_fn(|i, j| matrix[[i, j]]);
                let circuit = solovay_kitaev.synthesize_matrix(&matrix_nalgebra, 5);

                match circuit {
                    Ok(circuit) => {
                        for inst in circuit.data() {
                            let new_params: SmallVec<[Param; 3]> =
                                inst.params_view().iter().cloned().collect();

                            out_dag.apply_operation_back(
                                inst.op.clone(),
                                &[qubit],
                                &[],
                                Some(Parameters::Params(new_params)),
                                None,
                                #[cfg(feature = "cache_pygates")]
                                None,
                            )?;
                        }
                        out_dag.add_global_phase(circuit.global_phase())?;
                    }
                    Err(_) => {
                        apply_original_op(out_dag)?;
                    }
                }
            } else {
                let target_basis_set = match target {
                    Some(target) => get_target_basis_set(target, PhysicalQubit::new(qubit.0)),
                    None => {
                        let basis_gates: IndexSet<&str, ::ahash::RandomState> =
                            basis_gates.iter().map(String::as_str).collect();
                        get_euler_basis_set(&basis_gates)
                    }
                };

                let sequence = unitary_to_gate_sequence_inner(
                    matrix.view(),
                    &target_basis_set,
                    qubit.0 as usize,
                    None,
                    true,
                    None,
                );

                match sequence {
                    Some(sequence) => {
                        for (gate, params) in sequence.gates {
                            let new_params: SmallVec<[Param; 3]> =
                                params.iter().map(|p| Param::Float(*p)).collect();
                            out_dag.apply_operation_back(
                                gate.into(),
                                &[qubit],
                                &[],
                                Some(Parameters::Params(new_params)),
                                None,
                                #[cfg(feature = "cache_pygates")]
                                None,
                            )?;
                        }
                        out_dag.add_global_phase(&Param::Float(sequence.global_phase))?;
                    }
                    None => {
                        apply_original_op(out_dag)?;
                    }
                }
            }
        }
        // Run 2q synthesis
        2 => {
            // "ref_qubits" is used to access properties in the target. It accounts for control flow mapping.
            let ref_qubits: &[PhysicalQubit; 2] = &[
                PhysicalQubit::new(qubit_indices[out_qargs[0].0 as usize] as u32),
                PhysicalQubit::new(qubit_indices[out_qargs[1].0 as usize] as u32),
            ];

            run_2q_unitary_synthesis(
                matrix.view(),
                ref_qubits,
                coupling_edges,
                target,
                basis_gates,
                approximation_degree,
                natural_direction,
                pulse_optimize,
                out_dag,
                out_qargs,
                apply_original_op,
                run_python_decomposers,
            )?;
        }
        // Run 3q+ synthesis
        _ => {
            if basis_gates.is_empty() && target.is_none() {
                apply_original_op(out_dag)?;
            } else {
                let array = matrix.view();
                let shape = array.shape();
                let matrix_nalgebra = DMatrix::from_fn(shape[0], shape[1], |i, j| array[[i, j]]);
                let synth_circ =
                    quantum_shannon_decomposition(&matrix_nalgebra, None, None, None, None)?;
                let synth_dag =
                    DAGCircuit::from_circuit_data(&synth_circ, false, None, None, None, None)?;
                apply_synth_dag(out_dag, out_qargs, &synth_dag)?;
            }
        }
    }
    Ok(())
}

pub fn run_unitary_synthesis(
    dag: &mut DAGCircuit,
    qubit_indices: Vec<usize>,
    min_qubits: usize,
    target: Option<&Target>,
    basis_gates: HashSet<String>,
    synth_gates: HashSet<String>,
    coupling_edges: HashSet<[PhysicalQubit; 2]>,
    approximation_degree: Option<f64>,
    natural_direction: Option<bool>,
    pulse_optimize: Option<bool>,
    run_python_decomposers: bool,
) -> PyResult<DAGCircuit> {
    let out_dag = dag.copy_empty_like(VarsMode::Alike, BlocksMode::Keep)?;
    let mut out_dag = out_dag.into_builder();
    let mut unitary_synthesis_cache = UnitarySynthesisCache::new();

    // Iterate over dag nodes and determine unitary synthesis approach
    for node in dag.topological_op_nodes(false)? {
        let packed_instr = dag[node].unwrap_operation();
        let packed_instr = if let Some(control_flow) = dag.try_view_control_flow(packed_instr) {
            let blocks = control_flow.blocks();
            let mut new_blocks = Vec::with_capacity(blocks.len());
            for block in blocks {
                let new_ids = dag
                    .get_qargs(packed_instr.qubits)
                    .iter()
                    .map(|qarg| qubit_indices[qarg.0 as usize])
                    .collect_vec();
                let res = run_unitary_synthesis(
                    &mut block.clone(),
                    new_ids,
                    min_qubits,
                    target,
                    basis_gates.clone(),
                    synth_gates.clone(),
                    coupling_edges.clone(),
                    approximation_degree,
                    natural_direction,
                    pulse_optimize,
                    run_python_decomposers,
                )?;
                new_blocks.push(out_dag.add_block(res));
            }
            PackedInstruction::from_control_flow(
                packed_instr.op.control_flow().clone(),
                new_blocks,
                packed_instr.qubits,
                packed_instr.clbits,
                packed_instr.label.as_deref().cloned(),
            )
        } else {
            packed_instr.clone()
        };
        if !(synth_gates.contains(packed_instr.op.name())
            && packed_instr.op.num_qubits() >= min_qubits as u32)
        {
            out_dag.push_back(packed_instr)?;
            continue;
        }

        let matrix = match packed_instr.op.view() {
            OperationRef::Unitary(gate) => CowArray::from(gate.matrix_view()),
            _ => packed_instr
                .try_matrix()
                .map(CowArray::from)
                .ok_or_else(|| QiskitError::new_err("Unitary not found"))?,
        };

        // "out_qargs" is used to append the synthesized instructions to the output dag
        let out_qargs = dag.get_qargs(packed_instr.qubits);

        let apply_original_op = |out_dag: &mut DAGCircuitBuilder| -> PyResult<()> {
            out_dag.push_back(packed_instr.clone())?;
            Ok(())
        };

        synthesize_unitary_matrix(
            matrix,
            packed_instr.op.num_qubits(),
            &qubit_indices,
            &coupling_edges,
            target,
            &basis_gates,
            approximation_degree,
            natural_direction,
            pulse_optimize,
            &mut out_dag,
            out_qargs,
            run_python_decomposers,
            apply_original_op,
            &mut unitary_synthesis_cache,
        )?;
    }
    Ok(out_dag.build())
}

/// Return a single decomposer for the given `basis_gates`. If no decomposer is found,
/// return `None``. If a decomposer is found, the return type will be either
/// `DecomposerElement::TwoQubitBasis` or `DecomposerElement::TwoQubitControlledU`.
fn get_2q_decomposer_from_basis(
    basis_gates: IndexSet<&str, ::ahash::RandomState>,
    approximation_degree: Option<f64>,
    pulse_optimize: Option<bool>,
) -> PyResult<Option<DecomposerElement>> {
    // Non-parametrized 2q basis candidates (TwoQubitBasisDecomposer)
    let basis_names: IndexMap<&str, StandardGate, ::ahash::RandomState> = TWO_QUBIT_BASIS_SET_GATES
        .iter()
        .map(|x| (x.name(), *x))
        .collect();
    // Parametrized 2q basis candidates (TwoQubitControlledUDecomposer)
    let param_basis_names: IndexMap<&str, StandardGate, ::ahash::RandomState> =
        PARAM_SET_BASIS_GATES
            .iter()
            .map(|x| (x.name(), *x))
            .collect();
    // 1q basis (both decomposers)
    let euler_basis = match get_euler_basis_set(&basis_gates)
        .get_bases()
        .map(|basis| basis.as_str())
        .next()
    {
        Some(basis) => basis,
        None => return Ok(None),
    };

    // Try TwoQubitControlledUDecomposer first.
    let kak_gates: Vec<&str> = param_basis_names
        .keys()
        .copied()
        .collect::<IndexSet<&str, ::ahash::RandomState>>()
        .intersection(&basis_gates)
        .copied()
        .collect();
    if !kak_gates.is_empty() {
        let std_gate = *param_basis_names.get(kak_gates[0]).unwrap();
        let rxx_equivalent_gate = RXXEquivalent::Standard(std_gate);
        if let Ok(decomposer) =
            TwoQubitControlledUDecomposer::new_inner(rxx_equivalent_gate, euler_basis)
        {
            return Ok(Some(DecomposerElement {
                decomposer: DecomposerType::TwoQubitControlledU(Box::new(decomposer)),
                packed_op: PackedOperation::from_standard_gate(std_gate),
            }));
        };
    };

    // If there is no suitable TwoQubitControlledUDecomposer, try TwoQubitBasisDecomposer.
    let kak_gates: Vec<&str> = basis_names
        .keys()
        .copied()
        .collect::<IndexSet<&str, ::ahash::RandomState>>()
        .intersection(&basis_gates)
        .copied()
        .collect();
    if !kak_gates.is_empty() {
        let std_gate = *basis_names.get(kak_gates[0]).unwrap();
        let decomposer = TwoQubitBasisDecomposer::new_inner(
            std_gate.into(),
            SmallVec::new(),
            std_gate.matrix(&[]).unwrap().view(),
            approximation_degree.unwrap_or(1.0),
            euler_basis,
            pulse_optimize,
        )?;
        return Ok(Some(DecomposerElement {
            decomposer: DecomposerType::TwoQubitBasis(Box::new(decomposer)),
            packed_op: PackedOperation::from_standard_gate(std_gate),
        }));
    }
    Ok(None)
}

/// Return a list of decomposers for the given `target`. If no decomposer is found,
/// return `None``. The list can contain any `DecomposerElement`. This function
/// will exit early if an ideal decomposition is found.
fn get_2q_decomposers_from_target(
    target: &Target,
    qubits: &[PhysicalQubit; 2],
    approximation_degree: Option<f64>,
    pulse_optimize: Option<bool>,
    run_python_decomposers: bool,
) -> PyResult<Option<Vec<DecomposerElement>>> {
    // Store elegible basis gates (1q and 2q) with corresponding qargs (PhysicalQubit)
    let qargs: Qargs = Qargs::from_iter(*qubits);
    let reverse_qargs: Qargs = qubits.iter().rev().copied().collect();
    let mut qubit_gate_map: IndexMap<&Qargs, HashSet<&str>, ::ahash::RandomState> =
        IndexMap::default();
    match target.operation_names_for_qargs(&qargs) {
        Ok(direct_keys) => {
            qubit_gate_map.insert(&qargs, direct_keys);
            if let Ok(reverse_keys) = target.operation_names_for_qargs(&reverse_qargs) {
                qubit_gate_map.insert(&reverse_qargs, reverse_keys);
            }
        }
        Err(_) => {
            if let Ok(reverse_keys) = target.operation_names_for_qargs(&reverse_qargs) {
                qubit_gate_map.insert(&reverse_qargs, reverse_keys);
            } else {
                return Err(QiskitError::new_err(
                    "Target has no gates available on qubits to synthesize over.",
                ));
            }
        }
    }

    // Define available 1q basis
    let available_1q_basis: IndexSet<&str, ::ahash::RandomState> = IndexSet::from_iter(
        get_target_basis_set(target, qubits[0])
            .get_bases()
            .map(|basis| basis.as_str()),
    );

    // Define available 2q basis (setting apart parametrized 2q gates)
    let mut available_2q_basis: IndexMap<
        &str,
        (NormalOperation, Option<f64>),
        ::ahash::RandomState,
    > = IndexMap::default();
    let mut available_2q_param_basis: IndexMap<
        &str,
        (NormalOperation, Option<f64>),
        ::ahash::RandomState,
    > = IndexMap::default();
    for (q_pair, gates) in qubit_gate_map {
        for key in gates {
            let Some(TargetOperation::Normal(op)) = target.operation_from_name(key) else {
                continue;
            };
            match op.operation.view() {
                OperationRef::Gate(_) => (),
                OperationRef::StandardGate(_) => (),
                _ => continue,
            }
            // Filter out non-2q-gate candidates
            if op.operation.num_qubits() != 2 {
                continue;
            }
            // Add to param_basis if the gate parameters aren't bound (not Float)
            if !op
                .params_view()
                .iter()
                .all(|p| matches!(p, Param::Float(_)))
            {
                available_2q_param_basis.insert(
                    key,
                    (
                        op.clone(),
                        match &target[key].get(q_pair) {
                            Some(Some(props)) => props.error,
                            _ => None,
                        },
                    ),
                );
            }
            available_2q_basis.insert(
                key,
                (
                    op.clone(),
                    match &target[key].get(q_pair) {
                        Some(Some(props)) => props.error,
                        _ => None,
                    },
                ),
            );
        }
    }
    if available_2q_basis.is_empty() && available_2q_param_basis.is_empty() {
        return Err(QiskitError::new_err(
            "Target has no gates available on qubits to synthesize over.",
        ));
    }

    // If there are available 2q gates, start search for decomposers:
    let mut decomposers: Vec<DecomposerElement> = Vec::new();

    // Step 1: Try TwoQubitControlledUDecomposers
    for basis_1q in &available_1q_basis {
        for (_, (gate, _)) in available_2q_param_basis.iter() {
            let rxx_equivalent_gate = if let Some(std_gate) = gate.operation.try_standard_gate() {
                RXXEquivalent::Standard(std_gate)
            } else {
                let gate_type: Py<PyType> = Python::attach(|py| -> PyResult<Py<PyType>> {
                    let module = PyModule::import(py, "builtins")?;
                    let py_type = module.getattr("type")?;
                    Ok(py_type
                        .call1((gate.clone().into_pyobject(py)?,))?
                        .cast_into::<PyType>()?
                        .unbind())
                })?;
                RXXEquivalent::CustomPython(gate_type)
            };

            match TwoQubitControlledUDecomposer::new_inner(rxx_equivalent_gate, basis_1q) {
                Ok(decomposer) => {
                    decomposers.push(DecomposerElement {
                        decomposer: DecomposerType::TwoQubitControlledU(Box::new(decomposer)),
                        packed_op: gate.operation.clone(),
                    });
                }
                Err(_) => continue,
            };
        }
    }
    if available_2q_param_basis
        .values()
        .all(|(gate, _props)| matches!(gate.operation.try_standard_gate(), Some(PARAM_SET!())))
        && !available_2q_param_basis.is_empty()
    {
        return Ok(Some(decomposers));
    }

    // Step 2: Try TwoQubitBasisDecomposers
    #[inline]
    fn is_supercontrolled(op: &NormalOperation) -> bool {
        match op.try_matrix() {
            None => false,
            Some(unitary_matrix) => {
                let kak = TwoQubitWeylDecomposition::new_inner(unitary_matrix.view(), None, None)
                    .unwrap();
                relative_eq!(kak.a(), PI4) && relative_eq!(kak.c(), 0.0)
            }
        }
    }
    let supercontrolled_basis: IndexMap<
        &str,
        (NormalOperation, Option<f64>),
        ::ahash::RandomState,
    > = available_2q_basis
        .iter()
        .filter_map(|(k, (gate, props))| {
            if matches!(
                gate.operation.try_standard_gate(),
                Some(TWO_QUBIT_BASIS_SET!())
            ) || is_supercontrolled(gate)
            {
                Some((*k, (gate.clone(), *props)))
            } else {
                None
            }
        })
        .collect();
    for basis_1q in &available_1q_basis {
        for (_, (gate, props)) in supercontrolled_basis.iter() {
            let mut basis_2q_fidelity: f64 = match props {
                Some(error) => 1.0 - error,
                _ => 1.0,
            };
            if let Some(approx_degree) = approximation_degree {
                basis_2q_fidelity *= approx_degree;
            }
            let decomposer = TwoQubitBasisDecomposer::new_inner(
                gate.operation.clone(),
                gate.params_view()
                    .iter()
                    .map(|x| match x {
                        Param::Float(angle) => *angle,
                        _ => unreachable!(),
                    })
                    .collect(),
                gate.try_matrix().unwrap().view(),
                basis_2q_fidelity,
                basis_1q,
                pulse_optimize,
            )?;

            decomposers.push(DecomposerElement {
                decomposer: DecomposerType::TwoQubitBasis(Box::new(decomposer)),
                packed_op: gate.operation.clone(),
            });
        }
    }
    // If the 2q basis gates are a subset of KAK_STANDARD_GATE_SET, exit here.
    if available_2q_basis.values().all(|(gate, _props)| {
        matches!(
            gate.operation.try_standard_gate(),
            Some(TWO_QUBIT_BASIS_SET!())
        )
    }) && !available_2q_basis.is_empty()
    {
        return Ok(Some(decomposers));
    }

    // Step 3: Try XXDecomposers (Python)
    if !run_python_decomposers {
        if !decomposers.is_empty() {
            return Ok(Some(decomposers));
        } else {
            return Err(QiskitError::new_err(
                "Target has no gates available on qubits to synthesize over without Python",
            ));
        }
    }
    #[inline]
    fn is_controlled(op: &NormalOperation) -> bool {
        match op.try_matrix() {
            None => false,
            Some(unitary_matrix) => {
                let kak = TwoQubitWeylDecomposition::new_inner(unitary_matrix.view(), None, None)
                    .unwrap();
                relative_eq!(kak.b(), 0.0) && relative_eq!(kak.c(), 0.0)
            }
        }
    }
    let controlled_basis: IndexMap<&str, (NormalOperation, Option<f64>), ::ahash::RandomState> =
        available_2q_basis
            .iter()
            .filter(|(_, (gate, _))| is_controlled(gate))
            .map(|(k, (gate, props))| (*k, (gate.clone(), *props)))
            .collect();

    let mut pi2_basis: Option<&str> = None;
    Python::attach(|py| -> PyResult<()> {
        let xx_embodiments: &Bound<'_, PyAny> = imports::XX_EMBODIMENTS.get_bound(py);
        // The Python XXDecomposer args are the interaction strength (f64), basis_2q_fidelity (f64),
        // and embodiments (Bound<'_, PyAny>).
        let xx_decomposer_args = controlled_basis.iter().map(
            |(name, (op, props))| -> PyResult<(f64, f64, pyo3::Bound<'_, pyo3::PyAny>)> {
                let strength = 2.0
                    * TwoQubitWeylDecomposition::new_inner(
                        op.try_matrix().unwrap().view(),
                        None,
                        None,
                    )
                    .unwrap()
                    .a();
                let mut fidelity_value = match props {
                    Some(error) => 1.0 - error,
                    None => 1.0,
                };
                if let Some(approx_degree) = approximation_degree {
                    fidelity_value *= approx_degree;
                }
                let mut embodiment =
                    xx_embodiments.get_item(op.into_pyobject(py)?.getattr("base_class")?)?;

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
        let basis_2q_fidelity_dict = PyDict::new(py);
        let embodiments_dict = PyDict::new(py);
        for (strength, fidelity, embodiment) in xx_decomposer_args.flatten() {
            basis_2q_fidelity_dict.set_item(strength, fidelity)?;
            embodiments_dict.set_item(strength, embodiment)?;
        }
        if basis_2q_fidelity_dict.len() > 0 {
            let xx_decomposer: &Bound<'_, PyAny> = imports::XX_DECOMPOSER.get_bound(py);
            for basis_1q in available_1q_basis {
                let pi2_decomposer = if let Some(pi_2_basis) = pi2_basis {
                    if pi_2_basis == "cx" && basis_1q == "ZSX" {
                        let fidelity = match approximation_degree {
                            Some(approx_degree) => approx_degree,
                            None => match &target["cx"][&qargs] {
                                Some(props) => 1.0 - props.error.unwrap_or_default(),
                                None => 1.0,
                            },
                        };
                        Some(TwoQubitBasisDecomposer::new_inner(
                            StandardGate::CX.into(),
                            SmallVec::new(),
                            StandardGate::CX.matrix(&[]).unwrap().view(),
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
                    PyString::new(py, basis_1q),
                    &embodiments_dict,
                    pi2_decomposer,
                ))?;
                let decomposer_gate = decomposer
                    .getattr(intern!(py, "gate"))?
                    .extract::<NormalOperation>()?;

                decomposers.push(DecomposerElement {
                    decomposer: DecomposerType::XX(decomposer.into()),
                    packed_op: decomposer_gate.operation,
                });
            }
        }
        Ok(())
    })?;
    Ok(Some(decomposers))
}

/// Function to evaluate hardware-native direction, this allows to correct
/// the synthesis output to match the target constraints.
/// Returns:
///     * `true` if gate qubits are in the hardware-native direction
///     * `false` if gate qubits must be flipped to match hardware-native direction
fn preferred_direction(
    ref_qubits: &[PhysicalQubit; 2],
    natural_direction: Option<bool>,
    coupling_edges: &HashSet<[PhysicalQubit; 2]>,
    target: Option<&Target>,
    decomposer: &DecomposerElement,
) -> PyResult<Option<bool>> {
    let qubits: [PhysicalQubit; 2] = *ref_qubits;
    let mut reverse_qubits: [PhysicalQubit; 2] = qubits;
    reverse_qubits.reverse();

    let preferred_direction = match natural_direction {
        Some(false) => None,
        _ => {
            // None or Some(true)
            let zero_one = coupling_edges.contains(&qubits);
            let one_zero = coupling_edges.contains(&[qubits[1], qubits[0]]);

            match (zero_one, one_zero) {
                (true, false) => Some(true),
                (false, true) => Some(false),
                _ => {
                    match target {
                        Some(target) => {
                            let mut cost_0_1: f64 = f64::INFINITY;
                            let mut cost_1_0: f64 = f64::INFINITY;

                            let compute_cost = |lengths: bool,
                                                q_tuple: [PhysicalQubit; 2],
                                                in_cost: f64|
                             -> PyResult<f64> {
                                let cost = match target
                                    .qargs_for_operation_name(decomposer.packed_op.name())
                                {
                                    Ok(_) => match target[decomposer.packed_op.name()]
                                        .get(&Qargs::from(q_tuple))
                                    {
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
                        None => None,
                    }
                }
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

/// Apply synthesis for decomposers that return a SEQUENCE (TwoQubitBasis and TwoQubitControlledU).
fn synth_su4_sequence(
    su4_mat: ArrayView2<Complex64>,
    decomposer_2q: &DecomposerElement,
    preferred_direction: Option<bool>,
    approximation_degree: Option<f64>,
) -> PyResult<TwoQubitUnitarySequence> {
    let is_approximate = approximation_degree.is_none() || approximation_degree.unwrap() != 1.0;
    let synth = if let DecomposerType::TwoQubitBasis(decomp) = &decomposer_2q.decomposer {
        decomp.call_inner(su4_mat.view(), None, is_approximate, None)?
    } else if let DecomposerType::TwoQubitControlledU(decomp) = &decomposer_2q.decomposer {
        decomp.call_inner(su4_mat.view(), None)?
    } else {
        unreachable!("synth_su4_sequence should only be called for TwoQubitBasisDecomposer.")
    };
    let sequence = TwoQubitUnitarySequence {
        gate_sequence: synth,
    };
    match preferred_direction {
        None => Ok(sequence),
        Some(preferred_dir) => {
            let mut synth_direction: Option<SmallVec<[u8; 2]>> = None;
            // if the gates in synthesis are in the opposite direction of the preferred direction
            // resynthesize a new operator which is the original conjugated by swaps.
            // this new operator is doubly mirrored from the original and is locally equivalent.
            for (gate, _, qubits) in sequence.gate_sequence.gates() {
                if gate.num_qubits() == 2 {
                    synth_direction = Some(qubits.clone());
                }
            }
            match synth_direction {
                None => Ok(sequence),
                Some(synth_direction) => {
                    let synth_dir = match synth_direction.as_slice() {
                        [0, 1] => true,
                        [1, 0] => false,
                        _ => unreachable!(),
                    };
                    if synth_dir != preferred_dir {
                        reversed_synth_su4_sequence(
                            su4_mat.to_owned(),
                            decomposer_2q,
                            approximation_degree,
                        )
                    } else {
                        Ok(sequence)
                    }
                }
            }
        }
    }
}

/// Apply reverse synthesis for decomposers that return a SEQUENCE (TwoQubitBasis and TwoQubitControlledU).
/// This function is called by `synth_su4_sequence`` if the "direct" synthesis
/// doesn't match the hardware restrictions.
fn reversed_synth_su4_sequence(
    mut su4_mat: Array2<Complex64>,
    decomposer_2q: &DecomposerElement,
    approximation_degree: Option<f64>,
) -> PyResult<TwoQubitUnitarySequence> {
    let is_approximate = approximation_degree.is_none() || approximation_degree.unwrap() != 1.0;
    // Swap rows 1 and 2
    let (mut row_1, mut row_2) = su4_mat.multi_slice_mut((s![1, ..], s![2, ..]));
    azip!((x in &mut row_1, y in &mut row_2) (*x, *y) = (*y, *x));

    // Swap columns 1 and 2
    let (mut col_1, mut col_2) = su4_mat.multi_slice_mut((s![.., 1], s![.., 2]));
    azip!((x in &mut col_1, y in &mut col_2) (*x, *y) = (*y, *x));

    let synth = if let DecomposerType::TwoQubitBasis(decomp) = &decomposer_2q.decomposer {
        decomp.call_inner(su4_mat.view(), None, is_approximate, None)?
    } else if let DecomposerType::TwoQubitControlledU(decomp) = &decomposer_2q.decomposer {
        decomp.call_inner(su4_mat.view(), None)?
    } else {
        unreachable!(
            "reversed_synth_su4_sequence should only be called for TwoQubitBasisDecomposer."
        )
    };
    let flip_bits: [u8; 2] = [1, 0];
    let mut reversed_gates = Vec::with_capacity(synth.gates().len());
    for (gate, params, qubit_ids) in synth.gates() {
        let new_qubit_ids = qubit_ids
            .into_iter()
            .map(|x| flip_bits[*x as usize])
            .collect::<SmallVec<[u8; 2]>>();
        reversed_gates.push((gate.clone(), params.clone(), new_qubit_ids.clone()));
    }
    let mut reversed_synth: TwoQubitGateSequence = TwoQubitGateSequence::new();
    reversed_synth.set_state((reversed_gates, synth.global_phase()));
    let sequence = TwoQubitUnitarySequence {
        gate_sequence: reversed_synth,
    };
    Ok(sequence)
}

/// Apply synthesis for decomposers that return a DAG (XX).
fn synth_su4_xx_decomposer(
    py: Python,
    su4_mat: ArrayView2<Complex64>,
    decomposer_2q: &Py<PyAny>,
    preferred_direction: Option<bool>,
    approximation_degree: Option<f64>,
    packed_op: PackedOperation,
) -> PyResult<DAGCircuit> {
    let is_approximate = approximation_degree.is_none() || approximation_degree.unwrap() != 1.0;
    let kwargs: HashMap<&str, bool> = [("approximate", is_approximate), ("use_dag", true)]
        .into_iter()
        .collect();
    let synth_dag = decomposer_2q
        .call(
            py,
            (su4_mat.to_pyarray(py),),
            Some(&kwargs.into_py_dict(py)?),
        )?
        .extract::<DAGCircuit>(py)?;
    match preferred_direction {
        None => Ok(synth_dag),
        Some(preferred_dir) => {
            let mut synth_direction: Option<Vec<u32>> = None;
            for node in synth_dag.topological_op_nodes(false)? {
                let inst = &synth_dag[node].unwrap_operation();
                if inst.op.num_qubits() == 2 {
                    let qargs = synth_dag.get_qargs(inst.qubits);
                    synth_direction = Some(vec![qargs[0].0, qargs[1].0]);
                }
            }
            match synth_direction {
                None => Ok(synth_dag),
                Some(synth_direction) => {
                    let synth_dir = match synth_direction.as_slice() {
                        [0, 1] => true,
                        [1, 0] => false,
                        _ => unreachable!("There are no more than 2 possible synth directions."),
                    };
                    let new_elem = DecomposerElement {
                        decomposer: DecomposerType::XX(decomposer_2q.clone_ref(py)),
                        packed_op,
                    };
                    if synth_dir != preferred_dir {
                        reversed_synth_su4_dag(
                            py,
                            su4_mat.to_owned(),
                            &new_elem,
                            approximation_degree,
                        )
                    } else {
                        Ok(synth_dag)
                    }
                }
            }
        }
    }
}

/// Apply reverse synthesis for decomposers that return a DAG (XX).
/// This function is called by `synth_su4_xx_decomposer` if the "direct" synthesis
/// doesn't match the hardware restrictions.
fn reversed_synth_su4_dag(
    py: Python<'_>,
    mut su4_mat: Array2<Complex64>,
    decomposer_2q: &DecomposerElement,
    approximation_degree: Option<f64>,
) -> PyResult<DAGCircuit> {
    let is_approximate = approximation_degree.is_none() || approximation_degree.unwrap() != 1.0;

    // Swap rows 1 and 2
    let (mut row_1, mut row_2) = su4_mat.multi_slice_mut((s![1, ..], s![2, ..]));
    azip!((x in &mut row_1, y in &mut row_2) (*x, *y) = (*y, *x));

    // Swap columns 1 and 2
    let (mut col_1, mut col_2) = su4_mat.multi_slice_mut((s![.., 1], s![.., 2]));
    azip!((x in &mut col_1, y in &mut col_2) (*x, *y) = (*y, *x));

    let synth_dag = if let DecomposerType::XX(decomposer) = &decomposer_2q.decomposer {
        let kwargs: HashMap<&str, bool> = [("approximate", is_approximate), ("use_dag", true)]
            .into_iter()
            .collect();
        decomposer
            .call(
                py,
                (su4_mat.clone().into_pyarray(py),),
                Some(&kwargs.into_py_dict(py)?),
            )?
            .extract::<DAGCircuit>(py)?
    } else {
        unreachable!("reversed_synth_su4_dag should only be called for XXDecomposer")
    };

    let target_dag = synth_dag.copy_empty_like(VarsMode::Alike, BlocksMode::Keep)?;
    let flip_bits: [Qubit; 2] = [Qubit(1), Qubit(0)];
    let mut target_dag_builder = target_dag.into_builder();
    for node in synth_dag.topological_op_nodes(false)? {
        let mut inst = synth_dag[node].unwrap_operation().clone();
        let qubits: Vec<Qubit> = synth_dag
            .qargs_interner()
            .get(inst.qubits)
            .iter()
            .map(|x| flip_bits[x.0 as usize])
            .collect();
        inst.qubits = target_dag_builder.insert_qargs(&qubits);
        target_dag_builder.push_back(inst)?;
    }
    Ok(target_dag_builder.build())
}

/// Score the synthesis output (DAG or sequence) based on the expected gate fidelity/error score.
fn synth_error(
    synth_circuit: impl Iterator<
        Item = (
            String,
            Option<SmallVec<[Param; 3]>>,
            SmallVec<[PhysicalQubit; 2]>,
        ),
    >,
    target: &Target,
) -> f64 {
    let (lower_bound, upper_bound) = synth_circuit.size_hint();
    let mut gate_fidelities = match upper_bound {
        Some(bound) => Vec::with_capacity(bound),
        None => Vec::with_capacity(lower_bound),
    };
    let mut score_instruction =
        |inst_name: &str,
         inst_params: &Option<SmallVec<[Param; 3]>>,
         inst_qubits: &SmallVec<[PhysicalQubit; 2]>| {
            if let Ok(names) = target.operation_names_for_qargs(inst_qubits) {
                for name in names {
                    let Some(TargetOperation::Normal(target_op)) = target.operation_from_name(name)
                    else {
                        continue;
                    };
                    let are_params_close = if let Some(params) = inst_params {
                        params
                            .iter()
                            .zip(target_op.params_view().iter())
                            .all(|(p1, p2)| {
                                p1.is_close(p2, 1e-10)
                                    .expect("Unexpected parameter expression error.")
                            })
                    } else {
                        false
                    };
                    let is_parametrized = target_op
                        .params_view()
                        .iter()
                        .any(|param| matches!(param, Param::ParameterExpression(_)));
                    if target_op.operation.name() == inst_name
                        && (is_parametrized || are_params_close)
                    {
                        match target[name].get(&QargsRef::from(inst_qubits)) {
                            Some(Some(props)) => {
                                gate_fidelities.push(1.0 - props.error.unwrap_or(0.0))
                            }
                            _ => gate_fidelities.push(1.0),
                        }
                        break;
                    }
                }
            }
        };

    for (inst_name, inst_params, inst_qubits) in synth_circuit {
        score_instruction(&inst_name, &inst_params, &inst_qubits);
    }
    1.0 - gate_fidelities.into_iter().product::<f64>()
}

/// Perform 2q unitary synthesis for a given `unitary`. If some `target` is provided,
/// the decomposition will be hardware-aware and take into the account the reported
/// gate errors to select the best method among the options. If `target` is `None``,
/// the decomposition will use the given `basis_gates` and the first valid decomposition
/// will be returned (no selection).
fn run_2q_unitary_synthesis(
    unitary: ArrayView2<Complex64>,
    ref_qubits: &[PhysicalQubit; 2],
    coupling_edges: &HashSet<[PhysicalQubit; 2]>,
    target: Option<&Target>,
    basis_gates: &HashSet<String>,
    approximation_degree: Option<f64>,
    natural_direction: Option<bool>,
    pulse_optimize: Option<bool>,
    out_dag: &mut DAGCircuitBuilder,
    out_qargs: &[Qubit],
    mut apply_original_op: impl FnMut(&mut DAGCircuitBuilder) -> PyResult<()>,
    run_python_decomposers: bool,
) -> PyResult<()> {
    // Find decomposer candidates
    let decomposers = match target {
        Some(target) => {
            let decomposers_2q = get_2q_decomposers_from_target(
                target,
                ref_qubits,
                approximation_degree,
                pulse_optimize,
                run_python_decomposers,
            )?;
            decomposers_2q.unwrap_or_default()
        }
        None => {
            let basis_gates: IndexSet<&str, ::ahash::RandomState> =
                basis_gates.iter().map(String::as_str).collect();
            let decomposer_item: Option<DecomposerElement> =
                get_2q_decomposer_from_basis(basis_gates, approximation_degree, pulse_optimize)?;
            if decomposer_item.is_none() {
                apply_original_op(out_dag)?;
                return Ok(());
            };
            vec![decomposer_item.unwrap()]
        }
    };

    // If there's a single decomposer candidate, avoid computing synthesis score.
    // This will ALWAYS be the path if the `target` is `None` (`basis_gates` used).
    if decomposers.len() == 1 {
        let decomposer_item = decomposers.first().unwrap();
        let preferred_dir = preferred_direction(
            ref_qubits,
            natural_direction,
            coupling_edges,
            target,
            decomposer_item,
        )?;

        match &decomposer_item.decomposer {
            DecomposerType::TwoQubitBasis(_) => {
                let synth = synth_su4_sequence(
                    unitary,
                    decomposer_item,
                    preferred_dir,
                    approximation_degree,
                )?;
                apply_synth_sequence(out_dag, out_qargs, &synth)?;
            }
            DecomposerType::TwoQubitControlledU(_) => {
                let synth = synth_su4_sequence(
                    unitary,
                    decomposer_item,
                    preferred_dir,
                    approximation_degree,
                )?;
                apply_synth_sequence(out_dag, out_qargs, &synth)?;
            }
            DecomposerType::XX(xx_decomposer) => {
                if !run_python_decomposers {
                    return Err(QiskitError::new_err(
                        "No valid decomposer is present without Python",
                    ));
                }
                Python::attach(|py| -> PyResult<()> {
                    let synth = synth_su4_xx_decomposer(
                        py,
                        unitary,
                        xx_decomposer,
                        preferred_dir,
                        approximation_degree,
                        decomposer_item.packed_op.clone(),
                    )?;
                    apply_synth_dag(out_dag, out_qargs, &synth)?;
                    Ok(())
                })?;
            }
        }
        return Ok(());
    }

    // If there is more than one available decomposer, select the one with the best synthesis score.
    // This will only happen if `target` is not `None`, so we can assume that there is some target from
    // this point onwards. The scored SEQUENCEs and DAGs are stored in independent vectors to avoid defining
    // yet another custom type.
    let mut synth_errors_sequence = Vec::new();
    let mut synth_errors_dag = Vec::new();

    // The sequence synthesis logic can be shared between TwoQubitBasis and TwoQubitControlledU,
    // but the DAG logic needs to stay independent.
    let synth_sequence = |decomposer, preferred_dir| -> PyResult<(TwoQubitUnitarySequence, f64)> {
        let sequence =
            synth_su4_sequence(unitary, decomposer, preferred_dir, approximation_degree)?;
        let scoring_info =
            sequence
                .gate_sequence
                .gates()
                .iter()
                .map(|(gate, params, qubit_ids)| {
                    let inst_qubits = qubit_ids.iter().map(|q| ref_qubits[*q as usize]).collect();
                    (
                        gate.name().to_string(),
                        Some(params.iter().map(|p| Param::Float(*p)).collect()),
                        inst_qubits,
                    )
                });
        let score = synth_error(scoring_info, target.unwrap());
        Ok((sequence, score))
    };

    for decomposer in &decomposers {
        let preferred_dir = preferred_direction(
            ref_qubits,
            natural_direction,
            coupling_edges,
            target,
            decomposer,
        )?;
        match &decomposer.decomposer {
            DecomposerType::TwoQubitBasis(_) => {
                synth_errors_sequence.push(synth_sequence(decomposer, preferred_dir)?);
            }
            DecomposerType::TwoQubitControlledU(_) => {
                synth_errors_sequence.push(synth_sequence(decomposer, preferred_dir)?);
            }
            DecomposerType::XX(xx_decomposer) => {
                Python::attach(|py| -> PyResult<()> {
                    let synth_dag = synth_su4_xx_decomposer(
                        py,
                        unitary,
                        xx_decomposer,
                        preferred_dir,
                        approximation_degree,
                        decomposer.packed_op.clone(),
                    )?;
                    let scoring_info = synth_dag
                        .topological_op_nodes(false)
                        .expect("Unexpected error in dag.topological_op_nodes()")
                        .map(|node| {
                            let NodeType::Operation(inst) = &synth_dag[node] else {
                                unreachable!("DAG node must be an instruction")
                            };
                            let params = inst.params_view();
                            let inst_qubits = synth_dag
                                .get_qargs(inst.qubits)
                                .iter()
                                .map(|q| ref_qubits[q.0 as usize])
                                .collect();
                            (
                                inst.op.name().to_string(),
                                (!params.is_empty()).then(|| {
                                    params.iter().cloned().collect::<SmallVec<[Param; 3]>>()
                                }),
                                inst_qubits,
                            )
                        });
                    let score = synth_error(scoring_info, target.unwrap());
                    synth_errors_dag.push((synth_dag, score));
                    Ok(())
                })?;
            }
        }
    }

    // Resolve synthesis scores between sequence and DAG.
    let synth_sequence = synth_errors_sequence
        .iter()
        .enumerate()
        .min_by(|error1, error2| error1.1.1.partial_cmp(&error2.1.1).unwrap())
        .map(|(index, _)| &synth_errors_sequence[index]);

    let synth_dag = synth_errors_dag
        .iter()
        .enumerate()
        .min_by(|error1, error2| error1.1.1.partial_cmp(&error2.1.1).unwrap())
        .map(|(index, _)| &synth_errors_dag[index]);

    match (synth_sequence, synth_dag) {
        (None, None) => apply_original_op(out_dag)?,
        (Some((sequence, _)), None) => apply_synth_sequence(out_dag, out_qargs, sequence)?,
        (None, Some((dag, _))) => apply_synth_dag(out_dag, out_qargs, dag)?,
        (Some((sequence, sequence_error)), Some((dag, dag_error))) => {
            if sequence_error > dag_error {
                apply_synth_dag(out_dag, out_qargs, dag)?
            } else {
                apply_synth_sequence(out_dag, out_qargs, sequence)?
            }
        }
    };
    Ok(())
}

#[pyfunction]
#[pyo3(name = "synthesize_unitary_matrix", signature=(unitary, qubit_indices, target, basis_gates, coupling_edges, approximation_degree=None, natural_direction=None, pulse_optimize=None))]
pub fn py_synthesize_unitary_matrix(
    unitary: PyReadonlyArray2<Complex64>,
    qubit_indices: Vec<usize>,
    target: Option<&Target>,
    basis_gates: HashSet<String>,
    coupling_edges: HashSet<[PhysicalQubit; 2]>,
    approximation_degree: Option<f64>,
    natural_direction: Option<bool>,
    pulse_optimize: Option<bool>,
) -> PyResult<DAGCircuit> {
    let mat = unitary.as_array();

    let shape = mat.shape();
    let num_qubits = shape[0].trailing_zeros();

    let mut out_dag = DAGCircuit::new();
    let qubits = QuantumRegister::new_owning("q", num_qubits);
    out_dag.add_qreg(qubits)?;
    let mut out_dag = out_dag.into_builder();

    let out_qargs: Vec<Qubit> = (0..num_qubits).map(Qubit).collect();

    let apply_original_op = |_: &mut DAGCircuitBuilder| -> PyResult<()> {
        Err(QiskitError::new_err(
            "Did not succeed to decompose unitary.",
        ))
    };

    let mut unitary_synthesis_cache = UnitarySynthesisCache::new();

    synthesize_unitary_matrix(
        CowArray::from(mat.view()),
        num_qubits,
        &qubit_indices,
        &coupling_edges,
        target,
        &basis_gates,
        approximation_degree,
        natural_direction,
        pulse_optimize,
        &mut out_dag,
        &out_qargs,
        true,
        apply_original_op,
        &mut unitary_synthesis_cache,
    )?;

    Ok(out_dag.build())
}

pub fn unitary_synthesis_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_unitary_synthesis))?;
    m.add_wrapped(wrap_pyfunction!(py_synthesize_unitary_matrix))?;

    Ok(())
}

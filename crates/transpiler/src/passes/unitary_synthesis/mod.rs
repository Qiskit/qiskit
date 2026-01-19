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

mod decomposers;

use hashbrown::HashSet;
use indexmap::IndexSet;
use nalgebra::{DMatrix, Matrix2};
use ndarray::prelude::*;
use num_complex::Complex64;

use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use pyo3::{intern, wrap_pyfunction};

use self::decomposers::{Decomposer2q, DecomposerCache, Direction2q, FlipDirection};
use crate::QiskitError;
use crate::target::Target;
use qiskit_circuit::bit::QuantumRegister;
use qiskit_circuit::dag_circuit::{DAGCircuit, DAGCircuitBuilder};
use qiskit_circuit::instruction::Parameters;
use qiskit_circuit::operations::{Operation, OperationRef, Param, PythonOperation, StandardGate};
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperation};
use qiskit_circuit::{BlocksMode, PhysicalQubit, Qubit, VarsMode};
use qiskit_synthesis::euler_one_qubit_decomposer::unitary_to_gate_sequence_inner;
use qiskit_synthesis::qsd::quantum_shannon_decomposition;
use qiskit_synthesis::two_qubit_decompose::TwoQubitGateSequence;

#[cfg(feature = "cache_pygates")]
use std::sync::OnceLock;

/// A borrowed view onto the hardawre constraint.
#[derive(Clone, Copy, Debug)]
pub enum QpuConstraint<'a> {
    Target(&'a Target),
    Loose {
        basis_gates: &'a IndexSet<&'a str, ::ahash::RandomState>,
        coupling: &'a HashSet<[PhysicalQubit; 2]>,
    },
}
impl QpuConstraint<'_> {
    /// Which kind of coupling constraint is this?
    fn kind(&self) -> QpuConstraintKind {
        match self {
            Self::Target(_) => QpuConstraintKind::Target,
            Self::Loose { .. } => QpuConstraintKind::Loose,
        }
    }
}
impl<'a> From<&'a Target> for QpuConstraint<'a> {
    fn from(value: &'a Target) -> Self {
        Self::Target(value)
    }
}
/// Which kind of hardware constraint is in use?
///
/// This doesn't actually store the constraint or a reference to it, it's just a type homomoprhic to
/// the discriminant of [QpuConstraint] while discarding its lifetime, so it can be a hashed key.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum QpuConstraintKind {
    Target,
    Loose,
}

/// The settings to apply to 2q pulse-aware decomposers.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum UsePulseOptimizer {
    /// Require the use of a pulse-optimal decomposer.  This may produce compilation errors if the
    /// hardware target uses basis gates that the pulse-optimal decomposer cannot handle.
    Require,
    /// Never use a pulse-optimizing decomposer.
    Forbid,
    /// Use the pulse-optimising decomposer if it's available for a given hardware link, and
    /// produces better results.
    IfBetter,
}
impl UsePulseOptimizer {
    /// Convert the legacy Python-space `pulse_optimize` argument into the Rust-native explicit form.
    pub fn from_py_pulse_optimize(val: Option<bool>) -> Self {
        match val {
            None => Self::IfBetter,
            Some(false) => Self::Forbid,
            Some(true) => Self::Require,
        }
    }
    /// Convert back to the legacy Python-space `pulse_optimize` argument.
    pub fn to_py_pulse_optimize(self) -> Option<bool> {
        match self {
            Self::IfBetter => None,
            Self::Forbid => Some(false),
            Self::Require => Some(true),
        }
    }
}
/// Which directions to allow for a 2q gate in a decomposition.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum DecompositionDirection2q {
    /// The 2q gate can be used in either direction, regardless of whether the hardware is
    /// bidirectional or not.
    Any,
    /// If only one direction is hardware-native, then use that direction.  If both are valid,
    /// choose the one that minimizes some error heuristic (allowing either direction if they are
    /// both the same).
    BestValid,
    /// The same as [BestValid], except that an error is raised if both directions are valid and
    /// have the same (or no) error heuristics.
    UniquelyBestValid,
}
impl DecompositionDirection2q {
    pub fn from_py_natural_direction(val: Option<bool>) -> Self {
        match val {
            None => Self::BestValid,
            Some(false) => Self::Any,
            Some(true) => Self::UniquelyBestValid,
        }
    }
}

/// Configuration options for the default unitary synthesis plugin.
///
/// This implements `Default`, which is a convenient constructor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnitarySynthesisConfig {
    /// Whether to allow approximations (`Some`) or not (`None`).
    ///
    /// If `Some`, the weight is a multiplicative multiplier on fidelity, such that `1.0` means "use
    /// the gate fidelity exactly" and `0.5` would mean "treat the gate as having half its natural
    /// fidelity", etc.
    pub approximation_degree: Option<f64>,
    pub use_pulse_optimizer: UsePulseOptimizer,
    pub decomposition_direction_2q: DecompositionDirection2q,
    /// Whether to allow use of Python-space decomposers.
    pub run_python_decomposers: bool,
}
impl Default for UnitarySynthesisConfig {
    fn default() -> Self {
        Self {
            approximation_degree: None,
            use_pulse_optimizer: UsePulseOptimizer::IfBetter,
            decomposition_direction_2q: DecompositionDirection2q::BestValid,
            run_python_decomposers: false,
        }
    }
}

/// State of a unitary synthesis run.
#[derive(Clone, Debug, Default)]
pub struct UnitarySynthesisState {
    config: UnitarySynthesisConfig,
    cache: DecomposerCache,
}
impl UnitarySynthesisState {
    pub fn new(config: UnitarySynthesisConfig) -> Self {
        Self {
            config,
            cache: Default::default(),
        }
    }
}

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

/// Iterate over `DAGCircuit` to perform unitary synthesis.  For each eligible gate: find
/// decomposers, select the synthesis method with the highest fidelity score and apply
/// decompositions. The available methods are:
///
/// * 1q synthesis:
///     * OneQubitEulerDecomposer
///     * SolovayKitaevSynthesis
/// * 2q synthesis:
///     * TwoQubitBasisDecomposer
///     * TwoQubitControlledUDecomposer
///     * XXDecomposer (Python, only if target is provided)
/// * 3q+ synthesis: QuantumShannonDecomposer
pub fn run_unitary_synthesis(
    dag: &DAGCircuit,
    synth_gates: &HashSet<String>,
    min_qubits: usize,
    qubit_indices: &[PhysicalQubit],
    state: &mut UnitarySynthesisState,
    constraint: QpuConstraint,
) -> PyResult<DAGCircuit> {
    // This method is the actual distribution logic of unitary synthesis, but there are several
    // paths through it that return `Ok(false)`, meaning "no error and no synthesis needed", so the
    // caller is responsible for propagating the old instruction through to wherever is necessary.
    let synthesize_onto = |out: &mut DAGCircuitBuilder,
                           state: &mut UnitarySynthesisState,
                           inst: &PackedInstruction|
     -> PyResult<bool> {
        if !(synth_gates.contains(inst.op.name()) && inst.op.num_qubits() >= min_qubits as u32) {
            return Ok(false);
        }
        let Some(unitary) = inst.try_cow_array() else {
            return Ok(false);
        };
        synthesize_matrix_onto(
            out,
            unitary,
            qubit_indices,
            dag.get_qargs(inst.qubits),
            state,
            constraint,
        )
    };

    let mut out = dag
        .copy_empty_like(VarsMode::Alike, BlocksMode::Drop)?
        .into_builder();
    for node in dag.topological_op_nodes(false) {
        let inst = dag[node].unwrap_operation();
        let Some(cf) = dag.try_view_control_flow(inst) else {
            // Handle regular instructions - this path is where we end up most of the time.
            if !synthesize_onto(&mut out, state, inst)? {
                // No synthesis was necessary, so reinstate the operation.
                out.push_back(inst.clone())?;
            }
            continue;
        };
        // If we make it here, we've got control flow and have to set ourselves up to recurse.
        let blocks = cf
            .blocks()
            .into_iter()
            .map(|block| {
                let qubit_indices = dag
                    .get_qargs(inst.qubits)
                    .iter()
                    .map(|q| qubit_indices[q.index()])
                    .collect::<Vec<_>>();
                run_unitary_synthesis(
                    block,
                    synth_gates,
                    min_qubits,
                    &qubit_indices,
                    state,
                    constraint,
                )
                .map(|block| out.add_block(block))
            })
            .collect::<PyResult<_>>()?;
        out.push_back(PackedInstruction::from_control_flow(
            inst.op.control_flow().clone(),
            blocks,
            inst.qubits,
            inst.clbits,
            inst.label.as_deref().cloned(),
        ))?;
    }
    Ok(out.build())
}

/// Synthesise a matrix onto the DAG.
fn synthesize_matrix_onto(
    out: &mut DAGCircuitBuilder,
    unitary: CowArray<Complex64, Ix2>,
    qubits_phys: &[PhysicalQubit],
    qubits_local: &[Qubit],
    state: &mut UnitarySynthesisState,
    constraint: QpuConstraint,
) -> PyResult<bool> {
    let num_qubits = qubits_local.len();
    debug_assert_eq!(unitary.shape(), &[1 << num_qubits, 1 << num_qubits]);
    match *qubits_local {
        [] => {
            out.add_global_phase(&Param::Float(unitary[[0, 0]].arg()))?;
            Ok(true)
        }
        [q_virt] => {
            let q_phys = qubits_phys[q_virt.index()];
            synthesize_1q_matrix_onto(out, unitary.view(), q_phys, q_virt, state, constraint)
        }
        [q1_virt, q2_virt] => {
            let q_virt = [q1_virt, q2_virt];
            let q_phys = q_virt.map(|q| qubits_phys[q.index()]);
            synthesize_2q_matrix_onto(out, unitary, q_phys, q_virt, state, constraint)
        }
        _ => {
            if let QpuConstraint::Loose { basis_gates, .. } = constraint {
                if basis_gates.is_empty() {
                    return Ok(false);
                }
            }
            let unitary =
                DMatrix::from_fn(1 << num_qubits, 1 << num_qubits, |i, j| unitary[[i, j]]);
            let circuit = quantum_shannon_decomposition(&unitary, None, None, None, None)?;
            let map = out.merge_qargs(circuit.qargs_interner(), |q| Some(qubits_local[q.index()]));
            out.add_global_phase(circuit.global_phase())?;
            for inst in circuit.data() {
                out.push_back(PackedInstruction {
                    qubits: map[inst.qubits],
                    ..inst.clone()
                })?;
            }
            Ok(true)
        }
    }
}

fn synthesize_1q_matrix_onto(
    out: &mut DAGCircuitBuilder,
    unitary: ArrayView2<Complex64>,
    qubit_phys: PhysicalQubit,
    qubit_virt: Qubit,
    state: &mut UnitarySynthesisState,
    constraint: QpuConstraint,
) -> PyResult<bool> {
    // TODO: we possibly want to invert this logic and do Euler synthesis if possible, and SK only
    // if we have to.  If nothing else, it simplifies the logic of `try_solovay_kitaev` - instead of
    // "is this _only_ Clifford+T?" it can become "does this permit a Clifford+T decomposition",
    // which would allow us to remove all the "ignored" non-Clifford+T instructions.
    // If the qubit permits only known discrete-basis instructions, we'll do SK synthesis...
    if let Some(sk) = state.cache.try_solovay_kitaev(qubit_phys, constraint) {
        let circuit = sk
            .synthesize_matrix(&Matrix2::from_fn(|i, j| unitary[[i, j]]), 5)
            .expect("hardcoded standard gates should not include parametric gates");
        let qubits = out.insert_qargs(&[qubit_virt]);
        let clbits = out.insert_cargs(&[]);
        for inst in circuit.data() {
            out.push_back(PackedInstruction {
                qubits,
                clbits,
                ..inst.clone()
            })?;
        }
        out.add_global_phase(circuit.global_phase())?;
        return Ok(true);
    };
    // ... otherwise, we do continuous Euler-angle synthesis.
    let sequence = unitary_to_gate_sequence_inner(
        unitary,
        &state.cache.get_euler_1q(qubit_phys, constraint),
        qubit_phys.index(),
        None,
        true,
        None,
    );
    let Some(sequence) = sequence else {
        return Ok(false);
    };
    let qubits = out.insert_qargs(&[qubit_virt]);
    let clbits = out.insert_cargs(&[]);
    out.add_global_phase(&Param::Float(sequence.global_phase))?;
    for (gate, params) in sequence.gates {
        let params = (!params.is_empty()).then(|| {
            Box::new(Parameters::Params(
                params.iter().map(|p| Param::Float(*p)).collect(),
            ))
        });
        out.push_back(PackedInstruction {
            op: gate.into(),
            qubits,
            clbits,
            params,
            label: None,
            #[cfg(feature = "cache_pygates")]
            py_op: OnceLock::new(),
        })?;
    }
    Ok(true)
}

fn synthesize_2q_matrix_onto(
    out: &mut DAGCircuitBuilder,
    mut unitary: CowArray<Complex64, Ix2>,
    qargs_phys: [PhysicalQubit; 2],
    qargs_virt: [Qubit; 2],
    state: &mut UnitarySynthesisState,
    constraint: QpuConstraint,
) -> PyResult<bool> {
    let decomposer_cache = &mut state.cache;
    let config = &state.config;

    let mut cur_direction = Direction2q::Forwards;
    // This closure takes ownership of its two stateful values, `cur_direction` and `unitary`, via
    // the `move`, so that the rest of the function can't accidentally view them in unknown states.
    let mut single_decomposition = move |decomposer: &Decomposer2q, direction: Direction2q| {
        if cur_direction != direction {
            conjugate_with_swaps(unitary.view_mut());
        }
        cur_direction = direction;
        decomposer.decompose(unitary.view())
    };
    let mut sequences = decomposer_cache
        .get_2q(qargs_phys, config, constraint)?
        .map(|(decomposer, flip)| -> PyResult<_> {
            match flip {
                FlipDirection::No => single_decomposition(decomposer, Direction2q::Forwards)
                    .map(|seq| (Direction2q::Forwards, seq)),
                FlipDirection::Yes => single_decomposition(decomposer, Direction2q::Backwards)
                    .map(|seq| (Direction2q::Backwards, seq)),
                FlipDirection::Ensure(dir) => {
                    let normal = single_decomposition(decomposer, Direction2q::Forwards)?;
                    if normal
                        .gates()
                        .iter()
                        .filter_map(|(_, _, qubits)| (qubits.len() == 2).then_some(qubits))
                        .all(|qubits| qubits.as_slice() == dir.as_indices())
                    {
                        Ok((Direction2q::Forwards, normal))
                    } else {
                        // TODO: we assume that if the decomposer flipped when requesting one
                        // direction, then it'll flip when requesting the other too.  This is a
                        // historical assumption, and we'd be better just fixing the decomposers so
                        // they're not flaky with respect to direction.
                        single_decomposition(decomposer, Direction2q::Backwards)
                            .map(|seq| (Direction2q::Backwards, seq))
                    }
                }
            }
        });

    let Some(first) = sequences.next().transpose()? else {
        // TODO: The historical behaviour of this pass is to treat "failed to synthesise" the same
        // way as "synthesis was not necessary" (i.e. propagate the base unitary/gate to the output
        // without any synthesis).  This is probably not an ideal choice.
        //
        // Actually, there were also some convoluted circumstances specifically in the case of only
        // a `Target` being specified where it could instead raise an error, but these were
        // inconsistent; either it should be an error in _all_ circumstances if synthesis fails or
        // in _none_.  It's tricky to recreate the pre-Qiskit-2.4 behaviour bug-for-bug in the new
        // refactor because of how the split between decomposer construction and use works now.
        return Ok(false);
    };

    let fidelity = |pair: &(Direction2q, TwoQubitGateSequence)| -> f64 {
        let QpuConstraint::Target(target) = &constraint else {
            return 1.;
        };
        let (dir, sequence) = pair;
        let order = dir.as_indices();
        let phys = [qargs_phys[order[0] as usize], qargs_phys[order[1] as usize]];
        sequence
            .gates()
            .iter()
            .map(|(op, _, qubits)| {
                let qargs: &[_] = match *qubits.as_slice() {
                    [q] => &[phys[q as usize]],
                    [q1, q2] => &[phys[q1 as usize], phys[q2 as usize]],
                    _ => panic!("sequences should only contain 1q and 2q gates"),
                };
                // TODO: this does not handle the possibility of a 2q decomposer (like the
                // XXDecomposer) using specialised instructions whose operation names do not match
                // their target key.
                1. - target.get_error(op.name(), qargs).unwrap_or(0.)
            })
            .product()
    };

    // We only need to calculate the best score if there's more than one sequence.
    let mut best_fidelity = None;
    let mut best_pair = first;
    for sequence in sequences {
        let sequence = sequence?;
        let prev_fidelity = best_fidelity.unwrap_or_else(|| fidelity(&best_pair));
        let this_fidelity = fidelity(&sequence);
        if this_fidelity > prev_fidelity {
            best_fidelity = Some(this_fidelity);
            best_pair = sequence;
        } else {
            best_fidelity = Some(prev_fidelity);
        }
    }

    // ... now apply the best sequence.
    let (dir, sequence) = best_pair;
    let order = dir.as_indices();
    let out_qargs = [qargs_virt[order[0] as usize], qargs_virt[order[1] as usize]];
    let qubit_keys = [
        out.insert_qargs(&[out_qargs[0]]),
        out.insert_qargs(&[out_qargs[1]]),
        out.insert_qargs(&[out_qargs[0], out_qargs[1]]),
        out.insert_qargs(&[out_qargs[1], out_qargs[0]]),
    ];
    out.add_global_phase(&Param::Float(sequence.global_phase()))?;
    for (gate, params, qubits) in sequence.gates() {
        let qubits = match qubits.as_slice() {
            [0] => qubit_keys[0],
            [1] => qubit_keys[1],
            [0, 1] => qubit_keys[2],
            [1, 0] => qubit_keys[3],
            _ => panic!("internal logic error: decomposed sequence contained unexpected qargs"),
        };
        let op = match gate.view() {
            OperationRef::StandardGate(gate) => PackedOperation::from(gate),
            OperationRef::Gate(py_gate) => Python::attach(|py| -> PyResult<_> {
                let py_gate = Box::new(py_gate.py_copy(py)?);
                py_gate.gate.setattr(py, intern!(py, "params"), params)?;
                Ok(PackedOperation::from(py_gate))
            })?,
            _ => panic!("internal logic error: decomposed sequence contains a non-gate"),
        };
        let params = (!params.is_empty()).then(|| {
            Box::new(Parameters::Params(
                params.iter().copied().map(Param::Float).collect(),
            ))
        });
        out.push_back(PackedInstruction {
            op,
            qubits,
            clbits: Default::default(),
            params,
            label: None,
            #[cfg(feature = "cache_pygates")]
            py_op: OnceLock::new(),
        })?;
    }
    Ok(true)
}

/// Mutate the given 2q matrix in place to be `swap @ m @ swap`, i.e. "`m`, but applied to the qubit
/// arguments in reverse".
fn conjugate_with_swaps(mut m: ArrayViewMut2<Complex64>) {
    // Swap rows 1 and 2
    let (mut row_1, mut row_2) = m.multi_slice_mut((s![1, ..], s![2, ..]));
    azip!((x in &mut row_1, y in &mut row_2) (*x, *y) = (*y, *x));

    // Swap columns 1 and 2
    let (mut col_1, mut col_2) = m.multi_slice_mut((s![.., 1], s![.., 2]));
    azip!((x in &mut col_1, y in &mut col_2) (*x, *y) = (*y, *x));
}

/// Python entry point to [run_unitary_synthesis].
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(name = "run_main_loop", signature=(dag, qubit_indices, min_qubits, target, basis_gates, synth_gates, coupling_edges, approximation_degree=None, natural_direction=None, pulse_optimize=None))]
pub fn py_unitary_synthesis(
    dag: &DAGCircuit,
    qubit_indices: Vec<PhysicalQubit>,
    min_qubits: usize,
    target: Option<&Target>,
    basis_gates: HashSet<String>,
    synth_gates: HashSet<String>,
    coupling_edges: HashSet<[PhysicalQubit; 2]>,
    approximation_degree: Option<f64>,
    natural_direction: Option<bool>,
    pulse_optimize: Option<bool>,
) -> PyResult<DAGCircuit> {
    let config = UnitarySynthesisConfig {
        approximation_degree,
        use_pulse_optimizer: UsePulseOptimizer::from_py_pulse_optimize(pulse_optimize),
        decomposition_direction_2q: DecompositionDirection2q::from_py_natural_direction(
            natural_direction,
        ),
        run_python_decomposers: true,
    };
    let mut state = UnitarySynthesisState::new(config);
    let mut basis_gates_set: IndexSet<&str, ::ahash::RandomState>;
    let constraint = match target {
        Some(target) => QpuConstraint::Target(target),
        None => {
            basis_gates_set = basis_gates.iter().map(String::as_str).collect();
            basis_gates_set.sort();
            QpuConstraint::Loose {
                basis_gates: &basis_gates_set,
                coupling: &coupling_edges,
            }
        }
    };
    run_unitary_synthesis(
        dag,
        &synth_gates,
        min_qubits,
        &qubit_indices,
        &mut state,
        constraint,
    )
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(name = "synthesize_unitary_matrix", signature=(unitary, qubit_indices, target, basis_gates, coupling_edges, approximation_degree=None, natural_direction=None, pulse_optimize=None))]
pub fn py_synthesize_unitary_matrix(
    unitary: PyReadonlyArray2<Complex64>,
    qubit_indices: Vec<PhysicalQubit>,
    target: Option<&Target>,
    basis_gates: HashSet<String>,
    coupling_edges: HashSet<[PhysicalQubit; 2]>,
    approximation_degree: Option<f64>,
    natural_direction: Option<bool>,
    pulse_optimize: Option<bool>,
) -> PyResult<DAGCircuit> {
    let config = UnitarySynthesisConfig {
        approximation_degree,
        use_pulse_optimizer: UsePulseOptimizer::from_py_pulse_optimize(pulse_optimize),
        decomposition_direction_2q: DecompositionDirection2q::from_py_natural_direction(
            natural_direction,
        ),
        run_python_decomposers: true,
    };
    let mut state = UnitarySynthesisState::new(config);
    let mut basis_gates_set: IndexSet<&str, ::ahash::RandomState>;
    let constraint = match target {
        Some(target) => QpuConstraint::Target(target),
        None => {
            basis_gates_set = basis_gates.iter().map(String::as_str).collect();
            basis_gates_set.sort();
            QpuConstraint::Loose {
                basis_gates: &basis_gates_set,
                coupling: &coupling_edges,
            }
        }
    };

    let mat = unitary.as_array();

    let num_qubits = mat.shape()[0].trailing_zeros();

    let mut out_dag = DAGCircuit::new();
    let qubits = QuantumRegister::new_owning("q", num_qubits);
    out_dag.add_qreg(qubits)?;
    let mut out_dag = out_dag.into_builder();
    let out_qargs: Vec<Qubit> = (0..num_qubits).map(Qubit).collect();
    if !synthesize_matrix_onto(
        &mut out_dag,
        CowArray::from(mat.view()),
        &qubit_indices,
        &out_qargs,
        &mut state,
        constraint,
    )? {
        return Err(QiskitError::new_err("Failed to decompose unitary"));
    }
    Ok(out_dag.build())
}

pub fn unitary_synthesis_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_unitary_synthesis))?;
    m.add_wrapped(wrap_pyfunction!(py_synthesize_unitary_matrix))?;
    Ok(())
}

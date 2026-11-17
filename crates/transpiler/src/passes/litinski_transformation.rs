// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use pyo3::prelude::*;

use hashbrown::{DefaultHashBuilder, HashMap};
use std::hash::BuildHasher;

use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType, PyDAGCircuit};
use qiskit_circuit::imports::PAULI_EVOLUTION_GATE;
use qiskit_circuit::instruction::Parameters;
use qiskit_circuit::interner::Interned;
use qiskit_circuit::operations::{
    Operation, OperationRef, Param, PauliBased, PauliProductMeasurement, PauliProductRotation,
    PyInstruction, PyOpKind, StandardGate, StandardInstruction, multiply_param, radd_param,
};
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperation};
use qiskit_circuit::{BlocksMode, Clbit, Qubit, VarsMode};

use super::common::{
    MINIMUM_TOL, average_gate_fidelity_below_tol, is_angle_close_to_multiple_of_pi_k,
    sparse_paulis_commute,
};
use crate::TranspilerError;
use num_complex::Complex64;
use qiskit_quantum_info::clifford::{Clifford, Pauli1q};
use qiskit_quantum_info::sparse_observable::{BitTerm, SparseObservable};

use smallvec::smallvec;
use std::f64::consts::{FRAC_PI_4, FRAC_PI_8, PI};

// List of gate/instruction names supported by the pass: the pass raises an error if the circuit
// contains instruction with names outside of this list.
static SUPPORTED_INSTRUCTION_NAMES: [&str; 26] = [
    "id",
    "x",
    "y",
    "z",
    "h",
    "s",
    "sdg",
    "sx",
    "sxdg",
    "cx",
    "cz",
    "cy",
    "swap",
    "iswap",
    "ecr",
    "dcx",
    "t",
    "tdg",
    "rz",
    "rx",
    "ry",
    "p",
    "u1",
    "measure",
    "pauli_product_rotation",
    "pauli_product_measurement",
];

// List of instruction names which are modified by the pass: the pass is skipped if the circuit
// contains no instructions with names in this list.
static HANDLED_INSTRUCTION_NAMES: [&str; 10] = [
    "t",
    "tdg",
    "rz",
    "rx",
    "ry",
    "p",
    "u1",
    "measure",
    "pauli_product_rotation",
    "pauli_product_measurement",
];

// A Pauli product rotation or a Pauli product measurement produced by the pass,
// together with its qubit indices and its (interned) clbits. Qubit indices are
// assumed to be sorted.
struct PauliBasedEntry {
    op: PackedOperation,
    qubits: Vec<u32>,
    clbits: Interned<[Clbit]>,
}

#[pyfunction(name = "run_litinski_transformation")]
#[pyo3(signature = (dag, fix_clifford=true, insert_barrier=false, use_ppr=false, combine_paulis=false, approximation_degree=1.0))]
pub fn py_run_litinski_transformation(
    dag: &PyDAGCircuit,
    fix_clifford: bool,
    insert_barrier: bool,
    use_ppr: bool,
    combine_paulis: bool,
    approximation_degree: f64,
) -> PyResult<Option<PyDAGCircuit>> {
    let opts = LitinskiTransformationOptions {
        fix_clifford,
        insert_barrier,
        use_ppr,
        combine_paulis,
        tol: MINIMUM_TOL.max(1.0 - approximation_degree),
    };

    Ok(
        run_litinski_transformation(dag.try_read()?, &opts)?.map(|out_dag| {
            // Preserve metadata
            PyDAGCircuit::from_dagcircuit_with_cloned_metadata(out_dag, dag)
        }),
    )
}

/// Represents the current state of the transformation.
struct LitinskiTransformationState {
    // Keep track of the update to the global phase (produced when converting T/Tdg gates
    // to RZ-rotations).
    global_phase_update: Param,

    // Keep track of the Pauli product rotations and measurements produced by the pass.
    // `None` when the operation is removed (e.g. when it cancels with another one).
    pauli_based_ops: Vec<Option<PauliBasedEntry>>,

    // Keep track of the clifford operations in the circuit.
    clifford_ops: Vec<PackedInstruction>,

    // Clifford operator corresponding to `clifford_ops`.
    clifford: Clifford,

    // Used when combining rotations: for each (hashed) Pauli and qubits, we store the number of
    // non-removed Pauli product rotations in `pauli_based_ops` with this Pauli and qubits.
    ppr_counts: HashMap<u64, usize>,
    ppr_hasher: DefaultHashBuilder,
}

impl LitinskiTransformationState {
    pub fn new(
        num_qubits: usize,
        non_clifford_handled_count: usize,
        clifford_count: usize,
    ) -> Self {
        Self {
            global_phase_update: Param::Float(0.),
            pauli_based_ops: Vec::with_capacity(non_clifford_handled_count),
            clifford_ops: Vec::with_capacity(clifford_count),
            clifford: Clifford::identity(num_qubits),
            ppr_counts: HashMap::new(),
            ppr_hasher: DefaultHashBuilder::default(),
        }
    }

    /// The key of a Pauli product rotation in `ppr_counts`.
    fn ppr_key(&self, ppr: &PauliProductRotation, qubits: &[u32]) -> u64 {
        self.ppr_hasher
            .hash_one((ppr.z.as_slice(), ppr.x.as_slice(), qubits))
    }

    /// Add a Pauli product rotation (with the given key) to `pauli_based_ops`.
    fn push_ppr(&mut self, ppr: PauliProductRotation, qubits: Vec<u32>, key: u64) {
        *self.ppr_counts.entry(key).or_insert(0) += 1;
        self.pauli_based_ops.push(Some(PauliBasedEntry {
            op: PauliBased::PauliProductRotation(ppr).into(),
            qubits,
            clbits: Interned::default(),
        }));
    }

    /// Record that a rotation with the given key was removed from `pauli_based_ops`.
    fn remove_ppr_key(&mut self, key: u64) {
        if let hashbrown::hash_map::Entry::Occupied(mut count) = self.ppr_counts.entry(key) {
            *count.get_mut() -= 1;
            if *count.get() == 0 {
                count.remove();
            }
        }
    }
}

/// Represents the options to the pass.
pub struct LitinskiTransformationOptions {
    /// If `true`, the Clifford operations of the original circuit are appended to the output
    /// circuit (after the Pauli product rotations and measurements).
    fix_clifford: bool,
    /// If `true` (and `fix_clifford` is `true`), insert a barrier before the Clifford operations.
    insert_barrier: bool,
    /// Controls the representation of the non-Clifford rotations in the output circuit. These
    /// include the rotations obtained from single-qubit rotation gates (such as T, RZ, RX and
    /// RY), the non-Clifford Pauli product rotations of the input circuit, and (when
    /// `combine_paulis` is `true`) the merged rotations. If `true`, all of them are represented
    /// as Pauli product rotations. If `false`, all of them are represented as (Python-space)
    /// `PauliEvolutionGate`s. This option does not affect the Clifford operations (which are
    /// kept as in the original circuit) or the Pauli product measurements.
    use_ppr: bool,
    /// If `true`, merge (or cancel) rotations with the same Pauli, exploiting commutativity
    /// between Pauli product rotations and measurements.
    ///
    /// Note that this may increase the running time of the pass: when an earlier rotation with
    /// the same Pauli exists, the new rotation is compared against the previously collected
    /// operations, until reaching either that rotation or an operation it does not commute
    /// with (see `add_ppr`). In the worst case, where rotations with the same Pauli repeatedly
    /// appear far apart, separated by many commuting rotations, the running time is quadratic
    /// in the number of non-Clifford rotations.
    combine_paulis: bool,
    /// The tolerance used to check whether a rotation is a Clifford or the identity.
    tol: f64,
}

impl LitinskiTransformationOptions {
    /// Default options for the given value of `fix_clifford`: no barrier, all non-Clifford
    /// rotations represented as Pauli product rotations, no combining of rotations, and
    /// the minimum tolerance.
    pub fn default(fix_clifford: bool) -> Self {
        Self {
            fix_clifford,
            insert_barrier: false,
            use_ppr: true,
            combine_paulis: false,
            tol: MINIMUM_TOL,
        }
    }
}

pub fn run_litinski_transformation(
    dag: &DAGCircuit,
    opts: &LitinskiTransformationOptions,
) -> PyResult<Option<DAGCircuit>> {
    let op_counts = dag.get_op_counts();

    // Skip the pass if there are no rotation or measurement gates, including PPRs and PPMs.
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
            "Unable to run Litinski transformation as the circuit contains instructions not supported by the pass: {:?}",
            unsupported
        )));
    }
    // note that this count may not be accurate since non-clifford gates with pi/2 angles
    // are treated as cliffords by this pass
    let non_clifford_handled_count: usize = HANDLED_INSTRUCTION_NAMES
        .iter()
        .filter_map(|name| op_counts.get(*name))
        .sum();
    let clifford_count = dag.size(false)? - non_clifford_handled_count;

    // TODO: move to the end -- construction part
    let new_dag = dag.copy_empty_like_with_same_capacity(VarsMode::Alike, BlocksMode::Keep);
    let mut new_dag = new_dag.into_builder();

    let num_qubits = dag.num_qubits();
    let mut state =
        LitinskiTransformationState::new(num_qubits, non_clifford_handled_count, clifford_count);

    // Apply the Litinski transformation: that is, express a given circuit as a sequence of Pauli
    // product rotations and Pauli product measurements, followed by a final Clifford operator.
    for node_index in dag.topological_op_nodes(false) {
        // Convert T and Tdg gates to RZ rotations.
        if let NodeType::Operation(inst) = &dag[node_index] {
            let name = inst.op.name();
            let mut is_clifford = false; // indicates if it is a pi/2 rotation gate which is a clifford

            match inst.op.view() {
                OperationRef::StandardGate(StandardGate::I) => is_clifford = true,
                OperationRef::StandardGate(StandardGate::X) => {
                    state
                        .clifford
                        .append_x(dag.get_qargs(inst.qubits)[0].index());
                    is_clifford = true
                }
                OperationRef::StandardGate(StandardGate::Y) => {
                    state
                        .clifford
                        .append_y(dag.get_qargs(inst.qubits)[0].index());
                    is_clifford = true
                }
                OperationRef::StandardGate(StandardGate::Z) => {
                    state
                        .clifford
                        .append_z(dag.get_qargs(inst.qubits)[0].index());
                    is_clifford = true
                }
                OperationRef::StandardGate(StandardGate::H) => {
                    state
                        .clifford
                        .append_h(dag.get_qargs(inst.qubits)[0].index());
                    is_clifford = true
                }
                OperationRef::StandardGate(StandardGate::S) => {
                    state
                        .clifford
                        .append_s(dag.get_qargs(inst.qubits)[0].index());
                    is_clifford = true
                }
                OperationRef::StandardGate(StandardGate::Sdg) => {
                    state
                        .clifford
                        .append_sdg(dag.get_qargs(inst.qubits)[0].index());
                    is_clifford = true
                }
                OperationRef::StandardGate(StandardGate::SX) => {
                    state
                        .clifford
                        .append_sx(dag.get_qargs(inst.qubits)[0].index());
                    is_clifford = true
                }
                OperationRef::StandardGate(StandardGate::SXdg) => {
                    state
                        .clifford
                        .append_sxdg(dag.get_qargs(inst.qubits)[0].index());
                    is_clifford = true
                }
                OperationRef::StandardGate(StandardGate::CX) => {
                    state.clifford.append_cx(
                        dag.get_qargs(inst.qubits)[0].index(),
                        dag.get_qargs(inst.qubits)[1].index(),
                    );
                    is_clifford = true
                }
                OperationRef::StandardGate(StandardGate::CZ) => {
                    state.clifford.append_cz(
                        dag.get_qargs(inst.qubits)[0].index(),
                        dag.get_qargs(inst.qubits)[1].index(),
                    );
                    is_clifford = true
                }
                OperationRef::StandardGate(StandardGate::CY) => {
                    state.clifford.append_cy(
                        dag.get_qargs(inst.qubits)[0].index(),
                        dag.get_qargs(inst.qubits)[1].index(),
                    );
                    is_clifford = true
                }
                OperationRef::StandardGate(StandardGate::Swap) => {
                    state.clifford.append_swap(
                        dag.get_qargs(inst.qubits)[0].index(),
                        dag.get_qargs(inst.qubits)[1].index(),
                    );
                    is_clifford = true
                }
                OperationRef::StandardGate(StandardGate::ISwap) => {
                    state.clifford.append_iswap(
                        dag.get_qargs(inst.qubits)[0].index(),
                        dag.get_qargs(inst.qubits)[1].index(),
                    );
                    is_clifford = true
                }
                OperationRef::StandardGate(StandardGate::ECR) => {
                    state.clifford.append_ecr(
                        dag.get_qargs(inst.qubits)[0].index(),
                        dag.get_qargs(inst.qubits)[1].index(),
                    );
                    is_clifford = true
                }
                OperationRef::StandardGate(StandardGate::DCX) => {
                    state.clifford.append_dcx(
                        dag.get_qargs(inst.qubits)[0].index(),
                        dag.get_qargs(inst.qubits)[1].index(),
                    );
                    is_clifford = true
                }
                OperationRef::StandardGate(StandardGate::T)
                | OperationRef::StandardGate(StandardGate::Tdg)
                | OperationRef::StandardGate(StandardGate::RZ)
                | OperationRef::StandardGate(StandardGate::RX)
                | OperationRef::StandardGate(StandardGate::RY)
                | OperationRef::StandardGate(StandardGate::Phase)
                | OperationRef::StandardGate(StandardGate::U1) => {
                    let qubit = dag.get_qargs(inst.qubits)[0].index();
                    let param = inst.params_view();

                    // closure to process the rotation gates
                    let mut process_rot_gate = |gate: StandardGate,
                                                pauli: Pauli1q|
                     -> Option<(Pauli1q, Param)> {
                        if let Param::Float(angle) = param[0]
                            && let Some(multiple) =
                                is_angle_close_to_multiple_of_pi_k(gate, 2, angle, opts.tol)
                        {
                            is_clifford = true;
                            match gate {
                                StandardGate::RZ | StandardGate::Phase | StandardGate::U1 => {
                                    state.clifford.append_rz(qubit, multiple)
                                }
                                StandardGate::RX => state.clifford.append_rx(qubit, multiple),
                                StandardGate::RY => state.clifford.append_ry(qubit, multiple),
                                _ => unreachable!(
                                    "We cannot have gates other than RZ/RX/RY/P/U1 at this point."
                                ),
                            }
                            return None;
                        }
                        Some((pauli, param[0].clone()))
                    };

                    // Convert T and Tdg gates to RZ rotations
                    let (result, phase_update) = match inst.op.view() {
                        OperationRef::StandardGate(StandardGate::T) => (
                            Some((Pauli1q::Z, Param::Float(FRAC_PI_4))),
                            Param::Float(FRAC_PI_8),
                        ),
                        OperationRef::StandardGate(StandardGate::Tdg) => (
                            Some((Pauli1q::Z, Param::Float(-FRAC_PI_4))),
                            Param::Float(-FRAC_PI_8),
                        ),
                        OperationRef::StandardGate(StandardGate::RZ) => (
                            process_rot_gate(StandardGate::RZ, Pauli1q::Z),
                            Param::Float(0.),
                        ),
                        OperationRef::StandardGate(StandardGate::Phase)
                        | OperationRef::StandardGate(StandardGate::U1) => (
                            process_rot_gate(StandardGate::RZ, Pauli1q::Z),
                            multiply_param(&param[0], 0.5),
                        ),
                        OperationRef::StandardGate(StandardGate::RX) => (
                            process_rot_gate(StandardGate::RX, Pauli1q::X),
                            Param::Float(0.),
                        ),
                        OperationRef::StandardGate(StandardGate::RY) => (
                            process_rot_gate(StandardGate::RY, Pauli1q::Y),
                            Param::Float(0.),
                        ),
                        _ => {
                            unreachable!(
                                "We cannot have gates other than T/Tdg/RZ/RX/RY/P/U1 at this point."
                            );
                        }
                    };

                    // rotation gate is non-clifford
                    if let Some((pauli, angle)) = result {
                        state.global_phase_update =
                            radd_param(state.global_phase_update, phase_update);

                        // Evolving the single qubit pauli (X, Y or Z) by the Clifford.
                        // Returns the evolved Pauli in the sparse format: (sign, pauli z, pauli x, indices),
                        // where signs `true` and `false` correspond to coefficients `-1` and `+1` respectively.
                        let (sign, z, x, indices) = state.clifford.evolve_single_qubit_pauli(
                            pauli,
                            dag.get_qargs(inst.qubits)[0].index(),
                        );
                        let angle = if sign {
                            multiply_param(&angle, -1.0)
                        } else {
                            angle
                        };
                        let ppr = PauliProductRotation { z, x, angle };

                        let (orig_z, orig_x) = match pauli {
                            Pauli1q::Z => (true, false),
                            Pauli1q::X => (false, true),
                            Pauli1q::Y => (true, true),
                        };

                        add_ppr(
                            &mut state,
                            opts,
                            ppr,
                            indices,
                            sign,
                            (&[orig_z], &[orig_x], &[qubit as u32]),
                            inst.qubits,
                        );
                    }
                }
                OperationRef::PauliProductRotation(rotation) => {
                    // Synthesize PPR
                    let in_z = &rotation.z;
                    let in_x = &rotation.x;
                    let angle = &rotation.angle;
                    let qargs_in = dag.get_qargs(inst.qubits);
                    let indices_in: Vec<u32> = (0..qargs_in.len())
                        .map(|i| qargs_in[i].index() as u32)
                        .collect();

                    if let Param::Float(angle) = angle {
                        // PPR has pi/2 angle, and so is a clifford
                        if let Some(multiple) =
                            is_ppr_angle_close_to_multiple_of_pi2(in_z, in_x, *angle, opts.tol)
                        {
                            is_clifford = true;
                            state.clifford.append_ppr(in_z, in_x, &indices_in, multiple)
                        }
                    }
                    if !is_clifford {
                        // PPR is not clifford
                        // Evolve PPR by the clifford
                        let (sign, z, x, indices) =
                            state.clifford.evolve_pauli(in_z, in_x, &indices_in);

                        if indices.is_empty() {
                            // An all-identity PPR (with a parameterized angle, as otherwise it is
                            // treated as a Clifford) only contributes to the global phase, since
                            // exp(-i angle/2 I) = exp(-i angle/2) I.
                            state.global_phase_update =
                                radd_param(state.global_phase_update, multiply_param(angle, -0.5));
                        } else {
                            let out_sign = if sign { -1.0 } else { 1.0 };
                            let angle = multiply_param(angle, out_sign);
                            let ppr = PauliProductRotation { z, x, angle };

                            add_ppr(
                                &mut state,
                                opts,
                                ppr,
                                indices,
                                sign,
                                (in_z, in_x, &indices_in),
                                inst.qubits,
                            );
                        }
                    }
                }
                OperationRef::StandardInstruction(StandardInstruction::Measure) => {
                    // Evolve a measurement in the Z-basis by a Clifford.
                    // Returns the evolved Pauli in the sparse format: (sign, pauli z, pauli x, indices),
                    // where signs `true` and `false` correspond to coefficients `-1` and `+1` respectively.
                    let (sign, z, x, indices) = state.clifford.evolve_single_qubit_pauli(
                        Pauli1q::Z,
                        dag.get_qargs(inst.qubits)[0].index(),
                    );
                    let ppm = PauliProductMeasurement { z, x, neg: sign };
                    state.pauli_based_ops.push(Some(PauliBasedEntry {
                        op: PauliBased::PauliProductMeasurement(ppm).into(),
                        qubits: indices,
                        clbits: inst.clbits,
                    }));
                }
                OperationRef::PauliProductMeasurement(pp_meas) => {
                    // Evolve PPM by the clifford
                    let in_z = &pp_meas.z;
                    let in_x = &pp_meas.x;

                    let qargs_in = dag.get_qargs(inst.qubits);
                    let indices_in: Vec<u32> = (0..qargs_in.len())
                        .map(|i| qargs_in[i].index() as u32)
                        .collect();

                    let (sign, z, x, indices) =
                        state.clifford.evolve_pauli(in_z, in_x, &indices_in);
                    let ppm = PauliProductMeasurement {
                        z,
                        x,
                        neg: pp_meas.neg ^ sign,
                    };

                    state.pauli_based_ops.push(Some(PauliBasedEntry {
                        op: PauliBased::PauliProductMeasurement(ppm).into(),
                        qubits: indices,
                        clbits: inst.clbits,
                    }));
                }
                _ => unreachable!(
                    "We cannot have unsupported names at this step of Litinski Transformation: {}",
                    name
                ),
            }
            if is_clifford && opts.fix_clifford {
                state.clifford_ops.push(inst.clone());
            }
        }
    }

    new_dag.add_global_phase(&state.global_phase_update)?;

    // Add the collected Pauli product rotations and measurements to the output circuit.
    // Internally, all rotations are tracked as Pauli product rotations, regardless of whether
    // they originate from single-qubit rotation gates or from Pauli product rotations in the
    // input circuit. When `use_ppr` is `false`, each of them is converted to a
    // PauliEvolutionGate here.
    for PauliBasedEntry {
        op: packed_op,
        qubits,
        clbits,
    } in state.pauli_based_ops.into_iter().flatten()
    {
        let (new_op, params) = match packed_op.view() {
            OperationRef::PauliProductRotation(ppr) => {
                if !opts.use_ppr {
                    // Legacy path: implement the rotation as a PauliEvolutionGate,
                    // using exp(-i angle/2 P) = PauliEvolutionGate(P, time=angle/2).
                    let time = multiply_param(&ppr.angle, 0.5);
                    let py_gate = pauli_evolution_gate(&ppr.z, &ppr.x, &time)?;
                    (Some(py_gate.into()), Some(time))
                } else {
                    (None, Some(ppr.angle.clone()))
                }
            }
            OperationRef::PauliProductMeasurement(_) => (None, None),
            _ => unreachable!(
                "Only PPRs and PPMs are collected by the Litinski transformation pass."
            ),
        };

        new_dag.apply_operation_back(
            new_op.unwrap_or(packed_op),
            bytemuck::cast_slice(&qubits),
            dag.get_cargs(clbits),
            params.map(|param| Parameters::Params(smallvec![param])),
            None,
            #[cfg(feature = "cache_pygates")]
            None,
        )?;
    }

    // Add Clifford gates to the Qiskit circuit (when required).
    // Since we aim to preserve the global phase of the circuit, we add the Clifford operations from
    // the original circuit (and not the final Clifford operator).
    if opts.fix_clifford {
        // If specified, insert barriers between the Clifford and the rest of the circuit.
        if opts.insert_barrier {
            let barrier = StandardInstruction::Barrier(dag.num_qubits() as u32).into();
            let qubits = (0..dag.num_qubits() as u32).map(Qubit).collect::<Vec<_>>();
            new_dag.apply_operation_back(
                barrier,
                &qubits,
                &[],
                None,
                None,
                #[cfg(feature = "cache_pygates")]
                None,
            )?;
        }
        for inst in state.clifford_ops.into_iter() {
            new_dag.push_back(inst)?;
        }
    }

    Ok(Some(new_dag.build()))
}

/// Add a Pauli product rotation to the collected Pauli-based operations, merging it with an
/// earlier Pauli product rotation when possible.
///
/// The collected operations are scanned backwards. If an earlier rotation acts on the same qubits
/// with the same Pauli, the two rotations are merged into the earlier entry. Otherwise, the scan
/// stops at the first operation that does not commute with the new rotation, and the new rotation
/// is appended to the collected operations.
///
/// The scan is skipped altogether when no collected rotation has the same Pauli and qubits as the
/// new rotation (as tracked by `ppr_counts`), which is the common case. Otherwise, the scan is
/// linear in the number of collected operations that commute with the new rotation, so adding
/// all of the rotations of a circuit takes quadratic time in the worst case (when rotations with
/// the same Pauli repeatedly appear far apart, separated by many commuting rotations).
///
/// If the merged rotation is equivalent to the identity (up to a global phase), the entry is
/// removed altogether.
///
/// If the merged rotation `M` is a Clifford, the entry is removed and `M` is absorbed into the
/// Clifford. The rotation `M` has the same Pauli as the new rotation, which commutes with all
/// the later collected operations, so `M` can be moved to the end of the collected operations.
/// If `C` is the current Clifford, the circuit is now of the form `(rotations) -- M -- C`.
/// However, instead of prepending `M` to the list of Clifford gates and the current Clifford C,
/// we can append the original Pauli rotation (with the summed angle).
///
/// # Arguments
///
/// * `pauli_based_ops`: The collected Pauli-based operations.
/// * `clifford_ops`: The Clifford operations, or `None` if these are not added to the output
///   circuit.
/// * `clifford`: The current Clifford.
/// * `global_phase_update`: The update to the global phase, which is modified in-place when
///   the two rotations cancel out.
/// * `ppr`: The new Pauli product rotation, without I-terms.
/// * `qubits`: The sorted qubit indices of the new rotation.
/// * `sign`: The sign obtained when evolving the original Pauli of the new rotation.
/// * `orig_pauli`: The original Pauli `(z, x, indices)` of the new rotation, before evolving it
///   by the Clifford.
/// * `orig_qubits`: The interned qubits of the original instruction.
/// * `use_pauli_evo`: Whether the new rotation should be implemented as a PauliEvolutionGate.
///   A merged rotation is always implemented as a PauliProductRotation.
/// * `tol`: The tolerance to check whether the merged rotation is equivalent to the identity
///   or is a Clifford.
#[allow(clippy::too_many_arguments)]
fn add_ppr(
    state: &mut LitinskiTransformationState,
    opts: &LitinskiTransformationOptions,
    ppr: PauliProductRotation,
    qubits: Vec<u32>,
    sign: bool,
    orig_pauli: (&[bool], &[bool], &[u32]),
    orig_qubits: Interned<[Qubit]>,
) {
    if !opts.combine_paulis {
        state.pauli_based_ops.push(Some(PauliBasedEntry {
            op: PauliBased::PauliProductRotation(ppr).into(),
            qubits,
            clbits: Interned::default(),
        }));
        return;
    }

    // Only scan the collected operations if there is a rotation that may be merged with the
    // new one (that is, a rotation with the same Pauli and qubits).
    let key = state.ppr_key(&ppr, &qubits);
    if !state.ppr_counts.contains_key(&key) {
        state.push_ppr(ppr, qubits, key);
        return;
    }

    for entry in state.pauli_based_ops.iter_mut().rev() {
        let Some(PauliBasedEntry {
            op: packed_op,
            qubits: entry_qubits,
            ..
        }) = entry.as_ref()
        else {
            // The entry has already been removed.
            continue;
        };

        let (entry_z, entry_x, merged) = match packed_op.view() {
            OperationRef::PauliProductRotation(entry_ppr) => {
                let merged = if *entry_qubits == qubits {
                    entry_ppr.merge_with(&ppr)
                } else {
                    None
                };
                (&entry_ppr.z, &entry_ppr.x, merged)
            }
            OperationRef::PauliProductMeasurement(entry_ppm) => (&entry_ppm.z, &entry_ppm.x, None),
            _ => unreachable!(
                "Only PPRs and PPMs are collected by the Litinski transformation pass."
            ),
        };

        if let Some(merged) = merged {
            // Check whether the merged rotation is equivalent to the identity.
            if let Some(phase_update) =
                merged
                    .rotation_trace_and_dim()
                    .and_then(|(tr_over_dim, dim)| {
                        average_gate_fidelity_below_tol(tr_over_dim, dim, opts.tol)
                    })
            {
                *entry = None;
                state.remove_ppr_key(key);
                let old_phase = std::mem::replace(&mut state.global_phase_update, Param::Float(0.));
                state.global_phase_update = radd_param(old_phase, Param::Float(phase_update));
                return;
            }
            // Check whether the merged rotation is a Clifford, and if so absorb it into
            // the Clifford.
            if let Param::Float(angle) = merged.angle
                && let Some(multiple) =
                    is_ppr_angle_close_to_multiple_of_pi2(&merged.z, &merged.x, angle, opts.tol)
            {
                *entry = None;
                state.remove_ppr_key(key);
                let (orig_z, orig_x, orig_indices) = orig_pauli;
                let orig_multiple = if sign { (4 - multiple) % 4 } else { multiple };
                state
                    .clifford
                    .append_ppr(orig_z, orig_x, orig_indices, orig_multiple);
                if opts.fix_clifford {
                    let orig_angle = Param::Float(if sign { -angle } else { angle });
                    let orig_ppr = PauliProductRotation {
                        z: orig_z.to_vec(),
                        x: orig_x.to_vec(),
                        angle: orig_angle.clone(),
                    };
                    state.clifford_ops.push(PackedInstruction {
                        op: PauliBased::PauliProductRotation(orig_ppr).into(),
                        qubits: orig_qubits,
                        clbits: Interned::default(),
                        params: Some(Box::new(Parameters::Params(smallvec![orig_angle]))),
                        label: None,
                        #[cfg(feature = "cache_pygates")]
                        py_op: std::sync::OnceLock::new(),
                    });
                }
                return;
            }
            *entry = Some(PauliBasedEntry {
                op: PauliBased::PauliProductRotation(merged).into(),
                qubits,
                clbits: Interned::default(),
            });
            return;
        }

        if !sparse_paulis_commute(&ppr.z, &ppr.x, &qubits, entry_z, entry_x, entry_qubits) {
            break;
        }
    }

    state.push_ppr(ppr, qubits, key);
}

/// Create a PauliEvolutionGate for the Pauli given in the sparse ZX format (without I-terms),
/// and the evolution time.
fn pauli_evolution_gate(z: &[bool], x: &[bool], time: &Param) -> PyResult<PyInstruction> {
    let obs = sparse_obs_from_zx(z, x);
    Python::attach(|py| -> PyResult<_> {
        let py_evo = PAULI_EVOLUTION_GATE
            .get_bound(py)
            .call1((obs, time.clone()))?;
        Ok(PyInstruction {
            qubits: z.len() as u32,
            clbits: 0,
            params: 1,
            op_name: "PauliEvolution".to_string(),
            ob: py_evo.into(),
            kind: PyOpKind::Gate,
        })
    })
}

fn sparse_obs_from_zx(z: &[bool], x: &[bool]) -> SparseObservable {
    let bit_terms: Vec<BitTerm> = z
        .iter()
        .zip(x)
        .filter_map(|(&zi, &xi)| {
            if zi || xi {
                Some(non_identity_zx_to_bitterm(zi, xi))
            } else {
                None
            }
        })
        .collect();

    let boundaries = vec![0, bit_terms.len()];
    let coeffs = vec![Complex64::new(1.0, 0.0)];
    let num_qubits = bit_terms.len() as u32;
    let indices = (0..num_qubits).collect();

    // SAFETY: We manually built the internal data checking it is consistent.
    unsafe { SparseObservable::new_unchecked(num_qubits, coeffs, bit_terms, indices, boundaries) }
}

fn non_identity_zx_to_bitterm(z: bool, x: bool) -> BitTerm {
    match (z, x) {
        (false, false) => panic!("Identity terms not allowed."),
        (false, true) => BitTerm::X,
        (true, true) => BitTerm::Y,
        (true, false) => BitTerm::Z,
    }
}

/// For a given angle, if it is a multiple of PI/2, calculate the multiple mod (4),
/// Otherwise, return `None`.
/// I.e, if the angle is a multiple m of PI/2 then it returns m, where 0 <= m < 4.
fn is_ppr_angle_close_to_multiple_of_pi2(
    z: &[bool],
    x: &[bool],
    angle: f64,
    tol: f64,
) -> Option<usize> {
    let closest_ratio = 2.0 * angle / PI;
    let closest_integer = closest_ratio.round();
    let closest_angle = closest_integer * PI / 2.0;
    let theta = angle - closest_angle;

    // direct calculation of dim and tr_over_dim
    let num_qubits = z.iter().zip(x.iter()).filter(|(z, x)| **z || **x).count();
    let dim = (num_qubits as f64).exp2();
    let tr_over_dim = if num_qubits == 0 {
        // This is an identity Pauli rotation.
        (Complex64::new(0.0, -theta / 2.)).exp()
    } else {
        Complex64::new((theta / 2.).cos(), 0.)
    };

    if average_gate_fidelity_below_tol(tr_over_dim, dim, tol).is_some() {
        Some((closest_integer as i64).rem_euclid(4) as usize)
    } else {
        None
    }
}

pub fn litinski_transformation_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_run_litinski_transformation))?;
    Ok(())
}

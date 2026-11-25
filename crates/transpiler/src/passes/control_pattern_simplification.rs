// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use hashbrown::{HashMap, HashSet};
use itertools::Itertools;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use qiskit_util::IndexMap;
use rustworkx_core::petgraph::stable_graph::NodeIndex;
use smallvec::{SmallVec, smallvec};

use crate::gate_metrics::rotation_trace_and_dim;
use crate::passes::common::{MINIMUM_TOL, average_gate_fidelity_below_tol};
use qiskit_circuit::circuit_instruction::OperationFromPython;
use qiskit_circuit::dag_circuit::{DAGCircuit, PyDAGCircuit};
use qiskit_circuit::imports;
use qiskit_circuit::instruction::Parameters;
use qiskit_circuit::operations::{
    Operation, OperationRef, Param, PyInstruction, StandardGate, radd_param,
};
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::{BlocksMode, NoBlocks, Qubit, VarsMode};

type Params = SmallVec<[Param; 3]>;
type BaseGate = (StandardGate, Params);
/// The control states whose bits in `mask` (second item) equal those of `value` (first item).
type Cube = (u64, u64);
/// A base gate on target qubits, controlled on `(qubit, value)` pairs.
type ReplacementGate = (BaseGate, Vec<(Qubit, bool)>, Vec<Qubit>);

/// The largest number of control qubits of a group, so that its 2^n control states fit in a u64.
const MAX_CTRL_QUBITS: usize = 63;

/// An instruction of the DAG, split into its base gate, control qubits, control state and
/// target qubits.
struct ControlledInstruction {
    base_gate: Option<BaseGate>,
    ctrl_qubits: Vec<Qubit>,
    target_qubits: Vec<Qubit>,
    ctrl_state: u64,
}

/// The gate that a standard controlled gate applies when all its control qubits are 1, if it
/// is fully described by it (not `CU` or the relative-phase gates).
fn standard_base_gate(gate: StandardGate) -> Option<StandardGate> {
    use StandardGate::*;
    Some(match gate {
        CX | CCX | C3X => X,
        CY => Y,
        CZ | CCZ => Z,
        CH => H,
        CS => S,
        CSdg => Sdg,
        CSX | C3SX => SX,
        CRX => RX,
        CRY => RY,
        CRZ => RZ,
        CPhase => Phase,
        CU1 => U1,
        CU3 => U3,
        CSwap => Swap,
        _ => return None,
    })
}

/// Whether two base gates are the same gate, with parameters equal up to `tol`.
fn same_base_gate(a: &BaseGate, b: &BaseGate, tol: f64) -> bool {
    a.0 == b.0
        && a.1
            .iter()
            .zip(&b.1)
            .all(|(x, y)| x.is_close(y, tol).unwrap_or(false))
}

/// Whether applying the gate twice is the identity.
fn is_self_inverse(gate: &BaseGate, tol: f64) -> bool {
    gate.0
        .inverse(&gate.1)
        .is_some_and(|inverse| same_base_gate(&inverse, gate, tol))
}

/// Whether the gate is a rotation, whose angles add up when it is repeated.
fn is_rotation(gate: StandardGate) -> bool {
    rotation_trace_and_dim(gate, 0.).is_some()
}

/// Whether the rotation is the identity, including the global phase, which matters once
/// the gate is controlled.
fn is_identity_rotation(gate: StandardGate, angle: f64, tol: f64) -> bool {
    rotation_trace_and_dim(gate, angle)
        .and_then(|(trace, dim)| average_gate_fidelity_below_tol(trace, dim, tol))
        .is_some_and(|phase| phase.abs() < tol)
}

/// Whether two angles of the rotation `gate` give the same gate, up to `tol`.
fn same_angle(gate: StandardGate, a: &Param, b: &Param, tol: f64) -> bool {
    match (a, b) {
        (Param::Float(a), Param::Float(b)) => is_identity_rotation(gate, a - b, tol),
        _ => a.eq(b).unwrap_or(false),
    }
}

/// The base gate, number of control qubits and control state of a Python controlled gate
/// or annotated operation, if its base gate is standard.
///
/// A `ControlledGate` is read through `PyInstruction::ctrl_state`, which returns a `u32`, so
/// one with more than 32 control qubits is not recognized and is left unchanged. Only
/// annotated operations can have up to `MAX_CTRL_QUBITS` control qubits. Native standard
/// gates have at most 3.
fn python_controlled_gate(
    py: Python,
    op: &PyInstruction,
) -> PyResult<Option<(BaseGate, usize, u64)>> {
    let ob = op.ob.bind(py);
    let (base, num_ctrl_qubits, ctrl_state) = if let (Some(num_ctrl_qubits), Some(ctrl_state)) =
        (op.num_ctrl_qubits(), op.ctrl_state())
    {
        (
            ob.getattr("base_gate")?,
            num_ctrl_qubits as usize,
            ctrl_state as u64,
        )
    } else if ob.is_instance(imports::ANNOTATED_OPERATION.get_bound(py))? {
        // The canonical form combines all control modifiers into one.
        let modifiers = py
            .import("qiskit.circuit.annotated_operation")?
            .getattr("_canonicalize_modifiers")?
            .call1((ob.getattr("modifiers")?,))?;
        if modifiers.len()? != 1 {
            return Ok(None);
        }
        let modifier = modifiers.get_item(0)?;
        if !modifier.is_instance(imports::CONTROL_MODIFIER.get_bound(py))? {
            return Ok(None);
        }
        let num_ctrl_qubits: usize = modifier.getattr("num_ctrl_qubits")?.extract()?;
        if num_ctrl_qubits > MAX_CTRL_QUBITS {
            return Ok(None);
        }
        let ctrl_state = modifier.getattr("ctrl_state")?.extract()?;
        (ob.getattr("base_op")?, num_ctrl_qubits, ctrl_state)
    } else {
        return Ok(None);
    };
    let Ok(base) = base.extract::<OperationFromPython<NoBlocks>>() else {
        return Ok(None);
    };
    let Some(gate) = base.operation.try_standard_gate() else {
        return Ok(None);
    };
    // Skip gates with extra parameters (e.g. CUGate) or qubits (e.g. MCXVChain).
    let params: Params = base.params.map(|p| p.unwrap_params()).unwrap_or_default();
    if gate.num_qubits() == 0
        || op.params as usize != params.len()
        || op.qubits as usize != num_ctrl_qubits + gate.num_qubits() as usize
    {
        return Ok(None);
    }
    Ok(Some(((gate, params), num_ctrl_qubits, ctrl_state)))
}

/// The instruction at `node` as a controlled gate, with its control qubits sorted and its
/// control state remapped to that order. An instruction that is not a supported controlled
/// gate has no base gate, and all its qubits are target qubits, so that it blocks every
/// group it acts on.
fn controlled_instruction(
    py: Python,
    dag: &DAGCircuit,
    node: NodeIndex,
) -> PyResult<ControlledInstruction> {
    let inst = dag[node].unwrap_operation();
    let qargs = dag.get_qargs(inst.qubits);
    let controlled = match inst.op.view() {
        OperationRef::StandardGate(gate) => standard_base_gate(gate).map(|base_gate| {
            let num_ctrl_qubits = gate.num_ctrl_qubits() as usize;
            let params = inst.params_view().into();
            (
                (base_gate, params),
                num_ctrl_qubits,
                (1 << num_ctrl_qubits) - 1,
            )
        }),
        OperationRef::PyCustom(op) => python_controlled_gate(py, op)?,
        _ => None,
    };
    let Some((base_gate, num_ctrl_qubits, ctrl_state)) = controlled else {
        return Ok(ControlledInstruction {
            base_gate: None,
            ctrl_qubits: vec![],
            target_qubits: qargs.to_vec(),
            ctrl_state: 0,
        });
    };
    let order = (0..num_ctrl_qubits)
        .sorted_by_key(|k| qargs[*k])
        .collect_vec();
    Ok(ControlledInstruction {
        base_gate: Some(base_gate),
        ctrl_qubits: order.iter().map(|k| qargs[*k]).collect(),
        target_qubits: qargs[num_ctrl_qubits..].to_vec(),
        ctrl_state: order
            .iter()
            .enumerate()
            .fold(0, |acc, (new, old)| acc | (ctrl_state >> old & 1) << new),
    })
}

/// Collect groups of controlled gates with the same base gate and target qubits, in one pass.
/// A gate joins a group if nothing between them acts on its target qubits or targets its
/// control qubits. With `nested`, control qubits may be a subset or superset of those of the
/// first gate, and only groups with different numbers of control qubits are returned.
fn collect_groups(
    instructions: &[ControlledInstruction],
    excluded: &HashSet<usize>,
    nested: bool,
    tol: f64,
) -> Vec<Vec<usize>> {
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut open_group: HashMap<Qubit, usize> = HashMap::new();
    let mut last_targeted: HashMap<Qubit, usize> = HashMap::new();
    for (pos, inst) in instructions.iter().enumerate() {
        let base_gate = inst.base_gate.as_ref().filter(|_| !excluded.contains(&pos));
        let joined = base_gate.and_then(|base_gate| {
            let index = *open_group.get(&inst.target_qubits[0])?;
            let first_pos = groups[index][0];
            let first = &instructions[first_pos];
            let first_base_gate = first.base_gate.as_ref().unwrap();
            let ctrl_qubits_match = if nested {
                first
                    .ctrl_qubits
                    .iter()
                    .all(|q| inst.ctrl_qubits.contains(q))
                    || inst
                        .ctrl_qubits
                        .iter()
                        .all(|q| first.ctrl_qubits.contains(q))
            } else {
                inst.ctrl_qubits == first.ctrl_qubits
            };
            let matches = ctrl_qubits_match
                && inst.target_qubits == first.target_qubits
                && base_gate.0 == first_base_gate.0
                && (is_rotation(base_gate.0) || same_base_gate(base_gate, first_base_gate, tol))
                && inst
                    .ctrl_qubits
                    .iter()
                    .all(|q| last_targeted.get(q).is_none_or(|p| *p < first_pos));
            matches.then_some(index)
        });
        for qubit in inst.target_qubits.iter().chain(&inst.ctrl_qubits) {
            let index = open_group.get(qubit).copied();
            if let Some(index) = index.filter(|index| Some(*index) != joined) {
                for target in &instructions[groups[index][0]].target_qubits {
                    open_group.remove(target);
                }
            }
        }
        for qubit in &inst.target_qubits {
            last_targeted.insert(*qubit, pos);
        }
        match (joined, base_gate) {
            (Some(index), _) => groups[index].push(pos),
            (None, Some(_)) => {
                open_group.extend(inst.target_qubits.iter().map(|q| (*q, groups.len())));
                groups.push(vec![pos]);
            }
            (None, None) => (),
        }
    }
    let mixed = |group: &Vec<usize>| {
        !group
            .iter()
            .map(|k| instructions[*k].ctrl_qubits.len())
            .all_equal()
    };
    groups.retain(|group| group.len() > 1 && (!nested || mixed(group)));
    groups
}

/// The control states over `all_ctrl_qubits` that `ctrl_state` over `ctrl_qubits` covers.
fn expand_ctrl_state(
    ctrl_state: u64,
    ctrl_qubits: &[Qubit],
    all_ctrl_qubits: &[Qubit],
) -> Vec<u64> {
    (0..1u64 << (all_ctrl_qubits.len() - ctrl_qubits.len()))
        .map(|free_bits| {
            let (mut state, mut fixed, mut free) = (0, 0, 0);
            for (position, qubit) in all_ctrl_qubits.iter().enumerate() {
                if ctrl_qubits.contains(qubit) {
                    state |= (ctrl_state >> fixed & 1) << position;
                    fixed += 1;
                } else {
                    state |= (free_bits >> free & 1) << position;
                    free += 1;
                }
            }
            state
        })
        .collect()
}

/// The base gates the group applies, each with its control states, or `None` if gates on the
/// same control state do not combine.
fn split_by_base_gate(
    group: &[&ControlledInstruction],
    all_ctrl_qubits: &[Qubit],
    tol: f64,
) -> Option<Vec<(BaseGate, Vec<u64>)>> {
    let base_gate = group[0].base_gate.as_ref().unwrap();
    let states_of = |inst: &ControlledInstruction| {
        expand_ctrl_state(inst.ctrl_state, &inst.ctrl_qubits, all_ctrl_qubits)
    };
    if !is_rotation(base_gate.0) {
        let counts = group.iter().flat_map(|inst| states_of(inst)).counts();
        if !is_self_inverse(base_gate, tol) && counts.values().any(|count| *count > 1) {
            return None;
        }
        let states = group
            .iter()
            .flat_map(|inst| states_of(inst))
            .unique()
            .filter(|state| counts[state] % 2 == 1)
            .collect_vec();
        return Some(if states.is_empty() {
            vec![]
        } else {
            vec![(base_gate.clone(), states)]
        });
    }
    // Add up the angles on each control state, then group the states by angle.
    let mut angle_of_state: IndexMap<u64, Param> = IndexMap::default();
    for inst in group {
        let angle = &inst.base_gate.as_ref().unwrap().1[0];
        for state in states_of(inst) {
            match angle_of_state.get_mut(&state) {
                Some(total) => *total = radd_param(total.clone(), angle.clone()),
                None => drop(angle_of_state.insert(state, angle.clone())),
            }
        }
    }
    let mut split: Vec<(BaseGate, Vec<u64>)> = Vec::new();
    for (state, angle) in angle_of_state {
        if matches!(angle, Param::Float(value) if is_identity_rotation(base_gate.0, value, tol)) {
            continue;
        }
        match split
            .iter_mut()
            .find(|(other, _)| same_angle(base_gate.0, &angle, &other.1[0], tol))
        {
            Some((_, states)) => states.push(state),
            None => split.push(((base_gate.0, smallvec![angle]), vec![state])),
        }
    }
    Some(split)
}

/// A disjoint cover of `ctrl_states` from merging cubes that differ in one qubit, as in
/// Quine-McCluskey, trying each rotation of the qubit order and keeping the cover with the
/// fewest control qubits.
fn disjoint_cover(ctrl_states: &[u64], num_bits: usize) -> Vec<Cube> {
    let full_mask = (1u64 << num_bits) - 1;
    let cover_from = |start: usize| {
        let mut cubes: HashSet<Cube> = ctrl_states.iter().map(|s| (*s, full_mask)).collect();
        let mut changed = true;
        while changed {
            changed = false;
            for k in 0..num_bits {
                let bit = 1u64 << ((start + k) % num_bits);
                let mergeable = cubes
                    .iter()
                    .filter(|(value, mask)| mask & bit != 0 && value & bit == 0)
                    .filter(|(value, mask)| cubes.contains(&(value | bit, *mask)))
                    .copied()
                    .collect_vec();
                for (value, mask) in &mergeable {
                    cubes.remove(&(*value, *mask));
                    cubes.remove(&(value | bit, *mask));
                    cubes.insert((*value, mask & !bit));
                }
                changed |= !mergeable.is_empty();
            }
        }
        cubes.into_iter().sorted().collect_vec()
    };
    let num_ctrl_qubits = |cubes: &Vec<Cube>| cubes.iter().map(|c| c.1.count_ones()).sum::<u32>();
    (0..num_bits.max(1))
        .map(cover_from)
        .min_by_key(num_ctrl_qubits)
        .unwrap()
}

/// The gates for `cubes`, with their estimated CX count, two per control qubit. Two cubes that
/// fix the same qubits and differ in several of them agree on a parity, which a CX ladder
/// computes onto one qubit so that they merge, as for the Walsh operators of Welch et al.,
/// `<https://arxiv.org/abs/1306.3991>`.
fn replacement_gates(
    base_gate: &BaseGate,
    cubes: Vec<Cube>,
    ctrl_qubits: &[Qubit],
    target_qubits: &[Qubit],
) -> (u32, Vec<ReplacementGate>) {
    let gate = |cube: Cube| -> ReplacementGate {
        let fixed = ctrl_qubits
            .iter()
            .enumerate()
            .filter(|(k, _)| cube.1 >> k & 1 == 1);
        let controls = fixed
            .map(|(k, qubit)| (*qubit, cube.0 >> k & 1 == 1))
            .collect();
        (base_gate.clone(), controls, target_qubits.to_vec())
    };
    // Pair cubes that fix the same qubits, closest first.
    let mut paired: HashSet<Cube> = HashSet::new();
    let mut pairs = Vec::new();
    let candidates = cubes
        .iter()
        .array_combinations()
        .filter(|[a, b]| a.1 == b.1)
        .map(|[a, b]| ((a.0 ^ b.0).count_ones(), *a, *b))
        .sorted();
    for (_, a, b) in candidates {
        if !paired.contains(&a) && !paired.contains(&b) {
            paired.extend([a, b]);
            pairs.push((a, b));
        }
    }
    let unpaired = cubes
        .into_iter()
        .filter(|cube| !paired.contains(cube))
        .collect_vec();
    let mut cost = unpaired.iter().map(|cube| 2 * cube.1.count_ones()).sum();
    let mut gates = unpaired.into_iter().map(gate).collect_vec();
    for (a, b) in pairs {
        let diff = a.0 ^ b.0;
        let anchor = diff.trailing_zeros();
        let cx_gates = (anchor + 1..64)
            .filter(|k| diff >> k & 1 == 1)
            .map(|k| {
                let control = vec![(ctrl_qubits[anchor as usize], true)];
                (
                    (StandardGate::X, smallvec![]),
                    control,
                    vec![ctrl_qubits[k as usize]],
                )
            })
            .collect_vec();
        let value = if a.0 >> anchor & 1 == 1 {
            a.0 ^ diff
        } else {
            a.0
        };
        let merged = (value & !(1 << anchor), a.1 & !(1 << anchor));
        cost += 2 * (cx_gates.len() as u32 + merged.1.count_ones());
        gates.extend(cx_gates.iter().cloned());
        gates.push(gate(merged));
        gates.extend(cx_gates.into_iter().rev());
    }
    (cost, gates)
}

/// The cheapest replacement of a group, if it lowers the estimated CX count, two per control
/// qubit. The control states are covered by a disjoint cover, or, when most of them are
/// present, the gate is applied unconditionally and undone on a cover of the missing ones.
/// Groups with more than `MAX_CTRL_QUBITS` control qubits, or covering more than `max_patterns`
/// control states, are skipped.
fn simplify_group(
    group: &[&ControlledInstruction],
    tol: f64,
    max_patterns: u64,
) -> Option<Vec<ReplacementGate>> {
    let target_qubits = &group[0].target_qubits;
    let ctrl_qubits = group
        .iter()
        .flat_map(|inst| &inst.ctrl_qubits)
        .copied()
        .sorted()
        .dedup()
        .collect_vec();
    let num_bits = ctrl_qubits.len();
    if num_bits > MAX_CTRL_QUBITS {
        return None;
    }
    let num_states = 1u64 << num_bits;
    let num_expanded: u64 = group
        .iter()
        .map(|inst| num_states >> inst.ctrl_qubits.len())
        .sum();
    if num_expanded > max_patterns {
        return None;
    }
    let split = split_by_base_gate(group, &ctrl_qubits, tol)?;

    let mut total = 0;
    let mut replacement = Vec::new();
    for (base_gate, states) in &split {
        let (mut cost, mut gates) = replacement_gates(
            base_gate,
            disjoint_cover(states, num_bits),
            &ctrl_qubits,
            target_qubits,
        );
        // The complement: the gate unconditionally, undone on the missing states.
        if split.len() == 1
            && 2 * states.len() as u64 >= num_states
            && let Some(inverse) = base_gate.0.inverse(&base_gate.1)
        {
            let missing = (0..num_states)
                .filter(|s| !states.contains(s))
                .collect_vec();
            let cubes = disjoint_cover(&missing, num_bits);
            let (inverse_cost, inverse_gates) =
                replacement_gates(&inverse, cubes, &ctrl_qubits, target_qubits);
            if inverse_cost < cost {
                cost = inverse_cost;
                gates = vec![(base_gate.clone(), vec![], target_qubits.clone())];
                gates.extend(inverse_gates);
            }
        }
        total += cost;
        replacement.extend(gates);
    }
    let original: u32 = group
        .iter()
        .map(|inst| 2 * inst.ctrl_qubits.len() as u32)
        .sum();
    (total < original).then_some(replacement)
}

/// Append a replacement gate to `dag`. Controlled gates other than CX are built in Python.
fn apply_replacement_gate(
    py: Python,
    dag: &mut DAGCircuit,
    (base_gate, controls, target_qubits): &ReplacementGate,
) -> PyResult<()> {
    let (op, params): (PackedOperation, Params) = match controls.as_slice() {
        [] => (base_gate.0.into(), base_gate.1.clone()),
        [(_, true)] if base_gate.0 == StandardGate::X => (StandardGate::CX.into(), smallvec![]),
        _ => {
            // Gates with an arbitrary control state only exist in Python.
            let ctrl_state = controls
                .iter()
                .rev()
                .fold(0u64, |acc, (_, value)| acc << 1 | *value as u64);
            let kwargs = PyDict::new(py);
            kwargs.set_item("ctrl_state", ctrl_state)?;
            kwargs.set_item("annotated", false)?;
            let gate = base_gate
                .0
                .create_py_op(py, Some(base_gate.1.clone()), None)?;
            let gate = gate
                .bind(py)
                .call_method("control", (controls.len(),), Some(&kwargs))?;
            let op: OperationFromPython<NoBlocks> = gate.extract()?;
            (
                op.operation,
                op.params.map(|p| p.unwrap_params()).unwrap_or_default(),
            )
        }
    };
    let qubits = controls
        .iter()
        .map(|(qubit, _)| qubit)
        .chain(target_qubits)
        .copied()
        .collect_vec();
    dag.apply_operation_back(
        op,
        &qubits,
        &[],
        (!params.is_empty()).then_some(Parameters::Params(params)),
        None,
        #[cfg(feature = "cache_pygates")]
        None,
    )?;
    Ok(())
}

/// Run the ControlPatternSimplification pass on `dag`.
///
/// Args:
///     dag: the DAG to be optimized.
///     approximation_degree: the threshold of the average gate fidelity used to decide
///         whether two rotation angles are equal, and whether a rotation can be removed. The
///         default, 1, means no approximation, as in `CommutativeOptimization`: the tolerance is
///         then `MINIMUM_TOL`.
///     max_patterns: groups of gates covering more control states than this are skipped. The
///         default, 1024, keeps the time spent on a group to a few hundred milliseconds. The
///         Python class passes its own `_MAX_PATTERNS`, which has the same value.
///
/// Returns:
///     The optimized DAG, or `None` if the pass did not change the circuit.
pub fn run_control_pattern_simplification(
    py: Python,
    dag: &DAGCircuit,
    approximation_degree: f64,
    max_patterns: u64,
) -> PyResult<Option<DAGCircuit>> {
    let tol = MINIMUM_TOL.max(1. - approximation_degree);
    let nodes = dag.topological_op_nodes(false).collect_vec();
    let instructions: Vec<ControlledInstruction> = nodes
        .iter()
        .map(|node| controlled_instruction(py, dag, *node))
        .try_collect()?;

    let mut replacements: HashMap<usize, Vec<ReplacementGate>> = HashMap::new();
    let mut replaced: HashSet<usize> = HashSet::new();
    for nested in [true, false] {
        for indices in collect_groups(&instructions, &replaced, nested, tol) {
            let group = indices.iter().map(|k| &instructions[*k]).collect_vec();
            if let Some(replacement) = simplify_group(&group, tol, max_patterns) {
                replaced.extend(&indices);
                replacements.extend(indices[1..].iter().map(|k| (*k, vec![])));
                replacements.insert(indices[0], replacement);
            }
        }
    }
    if replacements.is_empty() {
        return Ok(None);
    }
    let mut new_dag = dag.copy_empty_like(VarsMode::Alike, BlocksMode::Keep);
    for (pos, node) in nodes.iter().enumerate() {
        match replacements.get(&pos) {
            Some(replacement) => {
                for gate in replacement {
                    apply_replacement_gate(py, &mut new_dag, gate)?;
                }
            }
            None => drop(new_dag.push_back(dag[*node].unwrap_operation().clone())?),
        }
    }
    Ok(Some(new_dag))
}

#[pyfunction(name = "run_control_pattern_simplification")]
#[pyo3(signature = (dag, approximation_degree=1.0, max_patterns=1024))]
pub fn py_run_control_pattern_simplification(
    py: Python,
    dag: &PyDAGCircuit,
    approximation_degree: f64,
    max_patterns: u64,
) -> PyResult<Option<PyDAGCircuit>> {
    let new_dag = run_control_pattern_simplification(
        py,
        dag.try_read()?,
        approximation_degree,
        max_patterns,
    )?;
    Ok(new_dag.map(|new_dag| PyDAGCircuit::from_dagcircuit_with_cloned_metadata(new_dag, dag)))
}

pub fn control_pattern_simplification_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_run_control_pattern_simplification))?;
    Ok(())
}

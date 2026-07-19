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

use crate::QiskitError;
use crate::linear::pmh::synth_pmh;
use crate::linear::utils::_row_op;
use fixedbitset::FixedBitSet;
use ndarray::Array2;
use numpy::PyReadonlyArray2;
use pyo3::{prelude::*, types::PyList};
use qiskit_circuit::Qubit;
use qiskit_circuit::circuit_data::{CircuitData, PyCircuitData};
use qiskit_circuit::operations::{Param, StandardGate};
use smallvec::{SmallVec, smallvec};
use std::f64::consts::PI;
type Instruction = (StandardGate, SmallVec<[Param; 3]>, SmallVec<[Qubit; 2]>);

fn get_instr(angle: AngleSpec, qubit_idx: usize) -> Instruction {
    let sm_vec = smallvec![];
    let qubit = smallvec![Qubit(qubit_idx as u32)];
    match angle {
        AngleSpec::Gate(gate) => (gate, sm_vec, qubit),
        AngleSpec::Phase(angle) => (
            StandardGate::Phase,
            smallvec![Param::Float(angle % PI)],
            qubit,
        ),
    }
}

#[derive(Clone, Copy)]
enum AngleSpec {
    Gate(StandardGate), // t, tdg, s, sdg, z
    Phase(f64),         // numeric angle
}

// A parity is a vector in `F_2^n`: the set of qubits that participate in one
// phase term. We store it as a `FixedBitSet` of exactly `n` bits, where bit `k`
// is set iff qubit `k` is in the parity.

// A frame on the algorithm's explicit stack: `(S, I, target)`.
type Data = (FixedBitSet, AngleSpec);
type Frame = (Vec<Data>, Vec<usize>, Option<usize>);

/// Implements `GraySynth` algorithm by Amy, Azimzadeh, and Mosca, described in the paper
///`arXiv:1712.01859 <https://arxiv.org/abs/1712.01859>`_
#[pyfunction]
#[pyo3(signature = (cnots, angles, section_size=2))]
pub fn synth_cnot_phase_aam(
    cnots: PyReadonlyArray2<bool>,
    angles: &Bound<PyList>,
    section_size: Option<usize>,
) -> PyResult<PyCircuitData> {
    // converting to Option<usize>
    let cnots = cnots.as_array().to_owned();
    let num_qubits = cnots.nrows();
    let num_parities = cnots.ncols();

    if num_parities != angles.len() {
        return Err(QiskitError::new_err(
            "Size of \"cnots\" and \"angles\" do not match.",
        ));
    }

    let mut angle_specs: Vec<AngleSpec> = Vec::with_capacity(angles.len());
    for data in angles.iter() {
        let spec = if let Ok(label) = data.extract::<String>() {
            match label.as_str() {
                "" => return Err(QiskitError::new_err("angle must not be an empty string")),
                "t" => AngleSpec::Gate(StandardGate::T),
                "tdg" => AngleSpec::Gate(StandardGate::Tdg),
                "s" => AngleSpec::Gate(StandardGate::S),
                "sdg" => AngleSpec::Gate(StandardGate::Sdg),
                "z" => AngleSpec::Gate(StandardGate::Z),
                other => AngleSpec::Phase(
                    other
                        .parse::<f64>()
                        .map_err(|_| QiskitError::new_err(format!("invalid angle: {other:?}")))?,
                ),
            }
        } else if let Ok(theta) = data.extract::<f64>() {
            AngleSpec::Phase(theta)
        } else {
            return Err(QiskitError::new_err(
                "each angle must be a gate label (t, tdg, s, sdg, z) or a number",
            ));
        };
        angle_specs.push(spec);
    }
    let angles = angle_specs;

    let mut s: Vec<Data> = Vec::with_capacity(num_qubits);

    for j in 0..num_parities {
        let mut p = FixedBitSet::with_capacity(num_qubits);
        for i in 0..num_qubits {
            if cnots[[i, j]] {
                p.insert(i);
            }
        }
        if !p.is_clear() {
            s.push((p, angles[j]));
        }
    }

    let mut tranx_matrix = Array2::<bool>::from_shape_fn((num_qubits, num_qubits), |(i, j)| i == j);

    let mut circuit: Vec<Instruction> = Vec::new();

    // Schedule phase gates that can be applied immediately, before any CNOT.
    // The initial linear function is the identity, so at the very start of the
    // circuit qubit `k` already carries exactly the parity x_k. Any parity in `s`
    // that is a single unit vector e_k (exactly one set bit, at position k) can
    // therefore have its phase gate placed directly on qubit `k` with zero CNOTs.
    // Apply those phase gates now, and drop them from `s`, so the recursion below
    // never routes them.

    s.retain(|(parity, angle)| {
        let mut ones = parity.ones();
        match (ones.next(), ones.next()) {
            // Exactly one set bit `k` => parity == e_k: emit on qubit k, drop it.
            (Some(k), None) => {
                circuit.push(get_instr(*angle, k));
                false
            }
            // Empty or weight >= 2: keep for the recursion.
            _ => true,
        }
    });

    // `common` will hold the bitwise AND of all parities in the current S.
    // Bit `k` of `common` is 1 iff every y ∈ S has bit k set, that is, bit
    // k is shared by all remaining parities.
    let mut common = FixedBitSet::with_capacity(num_qubits);

    // `counts[k]` will hold the number of parities in the current S whose
    // bit k is 1. Used to pick the best split bit.
    let mut counts: Vec<usize> = vec![0usize; num_qubits];

    // We pre-allocate space for ~2n+4 frames, enough to avoid most
    // reallocations: the recursion tree has depth ≤ n, and each level
    // pushes two children.
    let mut stack: Vec<Frame> = Vec::with_capacity(2 * num_qubits + 4);
    stack.push((s, (0..num_qubits).collect(), None));

    while let Some((mut s, indices, target_opt)) = stack.pop() {
        // Skip empty branches.
        if s.is_empty() {
            continue;
        }

        // While every remaining y in S shares a `1` in some bit other than
        // the target, we can emit ONE CNOT to fold that bit into the
        // target. Each such CNOT advances every parity in S simultaneously.
        if let Some(t) = target_opt {
            loop {
                // AND all parities in S into `common`. Seed with the first
                // parity (`clone_from` reuses `common`'s allocation), then
                // intersect the rest in place.
                common.clone_from(&s[0].0);

                // `s.iter().skip(1)` walks `s` starting from index 1; we've
                // already used `s[0]` for the seed. If `common` collapses to
                // all-zero we can stop early and further ANDs stay zero.
                for (y, _) in s.iter().skip(1) {
                    common.intersect_with(y);
                    if common.is_clear() {
                        break; // all-zero already: further ANDs can't restore bits
                    }
                }

                // Never pick j == t (control == target would be invalid).
                common.set(t, false);

                // Lowest remaining shared bit, or None. Length is exactly
                // num_qubits, so no `j < num_qubits` guard is needed.
                match common.ones().next() {
                    Some(j) => {
                        // Emit the CNOT (control = j, target = t).
                        circuit.push((
                            StandardGate::CX,
                            smallvec![],
                            smallvec![Qubit(j as u32), Qubit(t as u32)],
                        ));

                        // In the paper, Lemma 4.1 says After CNOT(j,t), every parity in every
                        // frame on the stack AND in our local `s` must be updated by
                        // y_j = y_j XOR y_t.
                        //
                        // Note: at this point `s` has been moved OUT of the stack.
                        // So `stack` and `s` are disjoint we have to update each separately.
                        apply_row_op_stack(&mut stack, t, j);
                        apply_row_op_set(&mut s, t, j);
                        _row_op(tranx_matrix.view_mut(), j, t);
                    }
                    None => break, // No shared 1-bit anywhere means we are done.
                }
            }
        }

        if indices.is_empty() {
            if let Some(t) = target_opt {
                for (_, angle) in &s {
                    circuit.push(get_instr(*angle, t));
                }
            }
            continue;
        }

        // We want j in indices maximizing the larger half:
        //     j = argmax_{j ∈ indices} max(|{y : y_j = 0}|, |{y : y_j = 1}|)
        // Equivalently: pick the most lopsided bit.
        // Reset `counts` to zero without reallocating.
        counts.fill(0);

        // Count how many parities have each bit set. `ones()` yields exactly
        // the set-bit indices (increasing, all < num_qubits).
        for (y, _) in &s {
            for idx in y.ones() {
                counts[idx] += 1;
            }
        }

        let total = s.len();

        let (mut best_j, mut best_max) = (indices[0], 0usize);
        for &cand in &indices {
            let ones = counts[cand];
            let m = ones.max(total - ones);
            if m > best_max {
                best_max = m;
                best_j = cand;
            }
        }
        let j = best_j;

        let mut s0: Vec<Data> = Vec::with_capacity(s.len());
        let mut s1: Vec<Data> = Vec::with_capacity(s.len());
        for (y, a) in s {
            if y.contains(j) {
                s1.push((y, a));
            } else {
                s0.push((y, a));
            }
        }

        let new_indices: Vec<usize> = indices.iter().copied().filter(|&i| i != j).collect();

        // Push S_1 first, then S_0, so S_0 ends up on TOP of the stack and
        // is processed first, matching the paper's recursion order.
        let s1_target = target_opt.or(Some(j));

        if !s1.is_empty() {
            stack.push((s1, new_indices.clone(), s1_target));
        }
        if !s0.is_empty() {
            stack.push((s0, new_indices, target_opt));
        }
    }

    circuit.extend(synth_pmh(tranx_matrix, section_size).rev());

    Ok(CircuitData::from_standard_gates(num_qubits as u32, circuit, Param::Float(0.0))?.into())
}

/// Apply  y_j = y_j XOR y_t  to every parity in every frame on the stack.
#[inline]
fn apply_row_op_stack(stack: &mut [Frame], control: usize, target: usize) {
    for frame in stack.iter_mut() {
        apply_row_op_set(&mut frame.0, control, target);
    }
}

/// Apply  y_j = y_j XOR y_t  to every parity in the given slice.
/// For each parity, if bit `i` is set we flip bit `j` (that is exactly XOR).
#[inline]
fn apply_row_op_set(s: &mut [Data], i: usize, j: usize) {
    for (y, _) in s.iter_mut() {
        if y.contains(i) {
            y.toggle(j);
        }
    }
}

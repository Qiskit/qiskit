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

use crate::linear::pmh::synth_pmh;
use ndarray::Array2;
use numpy::PyReadonlyArray2;
use pyo3::{prelude::*, types::PyList};
use qiskit_circuit::Qubit;
use qiskit_circuit::circuit_data::{CircuitData, PyCircuitData};
use qiskit_circuit::operations::{Param, StandardGate};
use smallvec::{SmallVec, smallvec};
use std::f64::consts::PI;
type Instruction = (StandardGate, SmallVec<[Param; 3]>, SmallVec<[Qubit; 2]>);

fn get_instr(angle: String, qubit_idx: usize) -> Option<Instruction> {
    Some(match angle.as_str() {
        "t" => (
            StandardGate::T,
            smallvec![],
            smallvec![Qubit(qubit_idx as u32)],
        ),
        "tdg" => (
            StandardGate::Tdg,
            smallvec![],
            smallvec![Qubit(qubit_idx as u32)],
        ),
        "s" => (
            StandardGate::S,
            smallvec![],
            smallvec![Qubit(qubit_idx as u32)],
        ),
        "sdg" => (
            StandardGate::Sdg,
            smallvec![],
            smallvec![Qubit(qubit_idx as u32)],
        ),
        "z" => (
            StandardGate::Z,
            smallvec![],
            smallvec![Qubit(qubit_idx as u32)],
        ),
        angles_in_pi => (
            StandardGate::Phase,
            smallvec![Param::Float((angles_in_pi.parse::<f64>().ok()?) % PI)],
            smallvec![Qubit(qubit_idx as u32)],
        ),
    })
}

// A vector in `F_2^n`, packed into 64-bit "word" for fast bitwise work.
// Bits are stored LSB-first within each `u64` bit 0 lives in the lowest
// bit of/ word 0, bit 1 in the next-lowest of word 0, … bit 64 in the
// lowest bit of word 1, and so on.
// We compare `Parity` values word-by-word, so two `Parity`s with different
// word counts compare unequal even if their logical bits are the same.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Parity {
    word_u64: Vec<u64>,
}

impl Parity {
    // Construct an all-zero parity with enough storage for at least `n` bits.
    #[inline]
    fn zeros(n: usize) -> Self {
        Parity {
            word_u64: vec![0u64; n.div_ceil(64)],
        }
    }

    // Read bit `i` as a `bool`.
    #[inline]
    fn get(&self, i: usize) -> bool {
        (self.word_u64[i / 64] >> (i % 64)) & 1 == 1
    }

    // Set bit `i` to 1.
    #[inline]
    fn set(&mut self, i: usize) {
        self.word_u64[i / 64] |= 1u64 << (i % 64);
    }

    // Is this the zero vector?
    fn is_zero(&self) -> bool {
        self.word_u64.iter().all(|&w| w == 0)
    }
}

// A frame on the algorithm's explicit stack: `(S, I, target)`.
type Data = (Parity, String);
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

    let angles = angles
        .iter()
        .filter_map(|data| {
            data.extract::<String>()
                .or_else(|_| data.extract::<f64>().map(|f| f.to_string()))
                .ok()
        })
        .collect::<Vec<String>>();

    let mut s: Vec<Data> = Vec::with_capacity(num_qubits);

    for j in 0..num_parities {
        let mut p = Parity::zeros(num_qubits);
        for i in 0..num_qubits {
            if cnots[[i, j]] {
                p.set(i);
            }
        }
        if !p.is_zero() && !angles[j].is_empty() {
            s.push((p, angles[j].clone()));
        }
    }

    let mut tranx_matrix = Array2::<bool>::default((num_qubits, num_qubits));
    for k in 0..num_qubits {
        tranx_matrix[[k, k]] = true;
    }

    let mut circuit: Vec<Instruction> = Vec::new();

    // Number of 64-bit word we need to store one n-bit parity.
    let word_u64_per = num_qubits.div_ceil(64);

    // `common` will hold the bitwise AND of all parities in the current S.
    // Bit `k` of `common` is 1 iff every y ∈ S has bit k set, that is, bit
    // k is shared by all remaining parities.
    let mut common: Vec<u64> = vec![0u64; word_u64_per];

    // `counts[k]` will hold the number of parities in the current S whose
    // bit k is 1. Used to pick the best split bit.
    let mut counts: Vec<usize> = vec![0usize; num_qubits];

    // We pre-allocate space for ~2n+4 frames — enough to avoid most
    // reallocations: the recursion tree has depth ≤ n, and each level
    // pushes two children.
    let mut stack: Vec<Frame> = Vec::with_capacity(2 * num_qubits + 4);
    stack.push((s, (0..num_qubits).collect(), None));

    while let Some((mut s, indices, target_opt)) = stack.pop() {
        // Skip empty branches.
        if s.is_empty() {
            continue;
        }

        // While every remaining y ∈ S shares a `1` in some bit other than
        // the target, we can emit ONE CNOT to fold that bit into the
        // target. Each such CNOT advances every parity in S simultaneously.
        if let Some(t) = target_opt {
            // Pre-compute (word-index, bit-mask) for the target once.
            let t_word_u64 = t / 64;
            let t_mask = 1u64 << (t % 64);

            loop {
                // AND all parities in S into `common`. Seed with the first
                // parity's word, then AND in the rest. `copy_from_slice`
                // copies a slice into another slice
                common.copy_from_slice(&s[0].0.word_u64);

                // This is an optimization step, where we track whether
                // `common` still has any 1-bit. If it collapses to
                // all-zero, we can stop applying AND because further
                // AND would leave it zero.
                let mut any_nonzero = common.iter().any(|&w| w != 0);

                // `s.iter().skip(1)` walks `s` starting from index 1; we've
                // already used `s[0]` for the seed.
                for (y, _) in s.iter().skip(1) {
                    if !any_nonzero {
                        break;
                    }

                    // Re-evaluate `any_nonzero` while ANDing, so we don't
                    // need a separate pass.
                    any_nonzero = false;
                    for (idx, common_val) in common.iter_mut().enumerate().take(word_u64_per) {
                        *common_val &= y.word_u64[idx];
                        if *common_val != 0 {
                            any_nonzero = true;
                        }
                    }
                }

                // We never want to pick j == t because it will cause control and
                // target qubit for CNOT to be the same.
                // Clearing bit t guarantees the next step won't pick it.
                common[t_word_u64] &= !t_mask;

                // `Option<usize>` represents "either a found index or
                // nothing." Starts as None.
                let mut found: Option<usize> = None;

                // `.enumerate()` yields `(word_index, &word_value)` pairs.
                for (wor, &w) in common.iter().enumerate() {
                    if w != 0 {
                        // `trailing_zeros()` returns the number of trailing zero bits that is
                        // the position of the lowest set bit. This is typically a single
                        // hardware instruction
                        let j = wor * 64 + (w.trailing_zeros() as usize);

                        // Guard against picking a bit past `n` the top word may have unused
                        // high bits if n is not a multiple of 64.
                        if j < num_qubits {
                            found = Some(j);
                            break;
                        }
                    }
                }

                match found {
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

                        for i in 0..num_qubits {
                            let bit_c = tranx_matrix[[j, i]];
                            tranx_matrix[[t, i]] ^= bit_c;
                        }
                    }
                    None => break, // No shared 1-bit anywhere means we are done.
                }
            }
        }

        if indices.is_empty() {
            if let Some(t) = target_opt {
                for (_, angle) in &s {
                    circuit.push(get_instr(angle.clone(), t).unwrap());
                }
            }
            continue;
        }

        // We want j ∈ indices maximizing the larger half:
        //     j = argmax_{j ∈ indices} max(|{y : y_j = 0}|, |{y : y_j = 1}|)
        // Equivalently: pick the most lopsided bit.
        // Reset `counts` to zero without reallocating.
        for c in counts.iter_mut() {
            *c = 0;
        }

        // Count how many parities have each bit set.
        // We iterate over *set bits only*, by repeatedly stripping the
        // lowest set bit with `w &= w - 1`. This is asymptotically faster
        // than checking each bit position one-by-one when parities are
        // sparse (few `1`s).
        for (y, _) in &s {
            for (wor, &word) in y.word_u64.iter().enumerate() {
                let mut w = word;
                let base = wor * 64;
                while w != 0 {
                    let b = w.trailing_zeros() as usize;
                    let idx = base + b;
                    if idx < num_qubits {
                        counts[idx] += 1;
                    }
                    // Strip the lowest set bit. Classical bit-twiddling
                    // identity: `w & (w - 1)` clears the lowest 1 of `w`.
                    w &= w - 1;
                }
            }
        }

        let total = s.len();

        let (mut best_j, mut best_max) = (indices[0], 0usize);
        for &cand in &indices {
            let ones = counts[cand];
            let m = if ones > total - ones {
                ones
            } else {
                total - ones
            };
            if m > best_max {
                best_max = m;
                best_j = cand;
            }
        }
        let j = best_j;

        let mut s0: Vec<Data> = Vec::with_capacity(s.len());
        let mut s1: Vec<Data> = Vec::with_capacity(s.len());
        for (y, a) in s {
            if y.get(j) {
                s1.push((y, a));
            } else {
                s0.push((y, a));
            }
        }

        let new_indices: Vec<usize> = indices.iter().copied().filter(|&i| i != j).collect();

        // Push S_1 first, then S_0, so S_0 ends up on TOP of the stack and
        // is processed first — matching the paper's recursion order.
        let s1_target = if target_opt.is_none() {
            Some(j)
        } else {
            target_opt
        };

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
#[inline]
fn apply_row_op_set(s: &mut [Data], i: usize, j: usize) {
    // Pre-compute word indices and masks once saves doing `i / 64`
    // and `1 << (i % 64)` for every parity in the slice.
    let i_word = i / 64;
    let i_mask = 1u64 << (i % 64);
    let j_word = j / 64;
    let j_mask = 1u64 << (j % 64);

    for (y, _) in s.iter_mut() {
        if (y.word_u64[i_word] & i_mask) != 0 {
            y.word_u64[j_word] ^= j_mask;
        }
    }
}

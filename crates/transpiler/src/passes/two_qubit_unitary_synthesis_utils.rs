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

use hashbrown::HashSet;
use ndarray::prelude::*;
use num_complex::Complex64;
use smallvec::SmallVec;

use pyo3::prelude::*;

use qiskit_circuit::operations::{Operation, Param};
use qiskit_circuit::packed_instruction::PackedOperation;

use crate::target::Qargs;
use crate::target::Target;
use crate::QiskitError;
use qiskit_circuit::PhysicalQubit;
use qiskit_synthesis::two_qubit_decompose::{
    TwoQubitBasisDecomposer, TwoQubitControlledUDecomposer, TwoQubitGateSequence,
};

#[derive(Clone, Debug)]
pub(crate) enum DecomposerType {
    TwoQubitBasis(Box<TwoQubitBasisDecomposer>),
    TwoQubitControlledU(Box<TwoQubitControlledUDecomposer>),
    XX(PyObject),
}

#[derive(Clone, Debug)]
pub(crate) struct DecomposerElement {
    pub(crate) decomposer: DecomposerType,
    pub(crate) packed_op: PackedOperation,
    pub(crate) params: SmallVec<[Param; 3]>,
}

#[derive(Clone, Debug)]
pub(crate) struct TwoQubitUnitarySequence {
    pub(crate) gate_sequence: TwoQubitGateSequence,
    pub(crate) decomp_op: PackedOperation,
    pub(crate) decomp_params: SmallVec<[Param; 3]>,
}

/// Function to evaluate hardware-native direction, this allows to correct
/// the synthesis output to match the target constraints.
/// Returns:
///     * `true` if gate qubits are in the hardware-native direction
///     * `false` if gate qubits must be flipped to match hardware-native direction
#[inline]
pub(crate) fn preferred_direction(
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
#[inline]
pub(crate) fn synth_su4_sequence(
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
        unreachable!("synth_su4_sequence should only be called for TwoQubitBasisDecomposer or TwoQubitControlledUDecomposer.")
    };
    let sequence = TwoQubitUnitarySequence {
        gate_sequence: synth,
        decomp_op: decomposer_2q.packed_op.clone(),
        decomp_params: decomposer_2q.params.clone(),
    };
    match preferred_direction {
        None => Ok(sequence),
        Some(preferred_dir) => {
            let mut synth_direction: Option<SmallVec<[u8; 2]>> = None;
            // if the gates in synthesis are in the opposite direction of the preferred direction
            // resynthesize a new operator which is the original conjugated by swaps.
            // this new operator is doubly mirrored from the original and is locally equivalent.
            for (gate, _, qubits) in sequence.gate_sequence.gates() {
                if gate.is_none() || gate.unwrap().name() == "cx" {
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
        reversed_gates.push((*gate, params.clone(), new_qubit_ids.clone()));
    }
    let mut reversed_synth: TwoQubitGateSequence = TwoQubitGateSequence::new();
    reversed_synth.set_state((reversed_gates, synth.global_phase()));
    let sequence = TwoQubitUnitarySequence {
        gate_sequence: reversed_synth,
        decomp_op: decomposer_2q.packed_op.clone(),
        decomp_params: decomposer_2q.params.clone(),
    };
    Ok(sequence)
}

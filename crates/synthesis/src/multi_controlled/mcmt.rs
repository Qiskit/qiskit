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

use crate::QiskitError;
use itertools::Itertools;
use pyo3::prelude::*;
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::circuit_instruction::OperationFromPython;
use qiskit_circuit::instruction::{IntoInstructionView, Parameters};
use qiskit_circuit::operations::{Param, StandardGate};
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::{Clbit, Qubit};
use smallvec::{smallvec, SmallVec};

type CCXChainItem = PyResult<(
    PackedOperation,
    SmallVec<[Param; 3]>,
    Vec<Qubit>,
    Vec<Clbit>,
)>;

/// A Toffoli chain, implementing a multi-control condition on all controls using
/// ``controls.len() - 1`` auxiliary qubits.
///
/// For example, for 4 controls we require 3 auxiliaries and create the circuit
///
///     control_0: ──■──────────────
///                  │
///     control_1: ──■──────────────
///                  │
///     control_2: ──┼────■─────────
///                  │    │
///     control_3: ──┼────┼────■────
///                ┌─┴─┐  │    │
///         aux_0: ┤ X ├──■────┼────
///                └───┘┌─┴─┐  │
///         aux_1: ─────┤ X ├──■────
///                     └───┘┌─┴─┐         "master control" qubit: controlling on this
///         aux_2: ──────────┤ X ├──  <--  implements a controlled operation on all qubits
///                          └───┘         in the "control" register
fn ccx_chain<'a>(
    controls: &'a [usize],
    auxiliaries: &'a [usize],
) -> impl DoubleEndedIterator<Item = CCXChainItem> + 'a {
    let n = controls.len() - 1; // number of chain elements
    std::iter::once((controls[0], controls[1], auxiliaries[0]))
        .chain((0..n - 1).map(|i| (controls[i + 2], auxiliaries[i], auxiliaries[i + 1])))
        .map(|(ctrl1, ctrl2, target)| {
            Ok((
                StandardGate::CCX.into(),
                smallvec![],
                vec![Qubit::new(ctrl1), Qubit::new(ctrl2), Qubit::new(target)],
                vec![],
            ))
        })
}

/// Implement multi-control, multi-target of a single-qubit gate using a V-chain with
/// (num_ctrl_qubits - 1) auxiliary qubits.
/// ``controlled_gate`` here must already be the controlled operation, e.g. if we
/// call MCMT of X, then it must be a CX gate. This is because I currently don't know how to
/// nicely map the single-qubit gate to it's controlled version.
///
/// For example, 4 controls and 2 target qubits for the Hadamard gate, generates
///
///     q_0: ──■──────────────────────────────────■──
///            │                                  │
///     q_1: ──■──────────────────────────────────■──
///            │                                  │
///     q_2: ──┼────■────────────────────────■────┼──
///            │    │                        │    │
///     q_3: ──┼────┼────■──────────────■────┼────┼──
///            │    │    │  ┌───┐       │    │    │
///     q_4: ──┼────┼────┼──┤ H ├───────┼────┼────┼──
///            │    │    │  └─┬─┘┌───┐  │    │    │
///     q_5: ──┼────┼────┼────┼──┤ H ├──┼────┼────┼──
///          ┌─┴─┐  │    │    │  └─┬─┘  │    │  ┌─┴─┐
///     q_6: ┤ X ├──■────┼────┼────┼────┼────■──┤ X ├
///          └───┘┌─┴─┐  │    │    │    │  ┌─┴─┐└───┘
///     q_7: ─────┤ X ├──■────┼────┼────■──┤ X ├─────
///               └───┘┌─┴─┐  │    │  ┌─┴─┐└───┘
///     q_8: ──────────┤ X ├──■────■──┤ X ├──────────
///                    └───┘          └───┘
///
#[pyfunction]
#[pyo3(signature = (controlled_gate, num_ctrl_qubits, num_target_qubits, control_state=None))]
pub fn mcmt_v_chain(
    py: Python,
    controlled_gate: OperationFromPython,
    num_ctrl_qubits: usize,
    num_target_qubits: usize,
    control_state: Option<usize>,
) -> PyResult<CircuitData> {
    if num_ctrl_qubits < 1 {
        return Err(QiskitError::new_err("Need at least 1 control qubit."));
    }

    let gate_params: SmallVec<[Param; 3]> = controlled_gate
        .try_legacy_params()
        .unwrap()
        .into_iter()
        .cloned()
        .collect();
    let packed_controlled_gate = controlled_gate.operation;
    let num_qubits = if num_ctrl_qubits > 1 {
        2 * num_ctrl_qubits - 1 + num_target_qubits
    } else {
        1 + num_target_qubits // we can have 1 control and multiple targets
    };

    let control_state = control_state.unwrap_or(usize::pow(2, num_ctrl_qubits as u32) - 1);

    // First, we handle bitflips in case of open controls.
    let flip_control_state = (0..num_ctrl_qubits)
        .filter(|index| control_state & (1 << index) == 0)
        .map(|index| {
            Ok((
                PackedOperation::from_standard_gate(StandardGate::X),
                smallvec![] as SmallVec<[Param; 3]>,
                vec![Qubit::new(index)],
                vec![] as Vec<Clbit>,
            ))
        });

    // Then, we create the operations that apply the controlled base gate.
    // That's because we only add the V-chain of CCX gates, if the number of controls
    // is larger than 1, otherwise we're already done here.
    let master_control = if num_ctrl_qubits > 1 {
        num_qubits - 1
    } else {
        0
    };
    let targets = (0..num_target_qubits).map(|i| {
        Ok((
            packed_controlled_gate.clone(),
            gate_params.clone(),
            vec![Qubit::new(master_control), Qubit::new(num_ctrl_qubits + i)],
            vec![] as Vec<Clbit>,
        ))
    });

    // Finally we add the V-chain (or return in case of 1 control).
    if num_ctrl_qubits == 1 {
        CircuitData::from_packed_operations(
            py,
            num_qubits as u32,
            0,
            flip_control_state
                .clone()
                .chain(targets)
                .chain(flip_control_state)
                .map_ok(|(op, params, qubits, clbits)| {
                    (op, Some(Parameters::Params(params)), qubits, clbits)
                }),
            Param::Float(0.0),
        )
    } else {
        // If the number of controls is larger than 1, and we need to apply the V-chain,
        // create it here and sandwich the targets in-between.
        let controls: Vec<usize> = (0..num_ctrl_qubits).collect();
        let auxiliaries: Vec<usize> = (num_ctrl_qubits + num_target_qubits..num_qubits).collect();
        let down_chain = ccx_chain(&controls, &auxiliaries);
        let up_chain = ccx_chain(&controls, &auxiliaries).rev();

        CircuitData::from_packed_operations(
            py,
            num_qubits as u32,
            0,
            flip_control_state
                .clone()
                .chain(down_chain)
                .chain(targets)
                .chain(up_chain)
                .chain(flip_control_state)
                .map_ok(|(op, params, qubits, clbits)| {
                    (op, Some(Parameters::Params(params)), qubits, clbits)
                }),
            Param::Float(0.0),
        )
    }
}

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

use pyo3::prelude::*;
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::circuit_instruction::OperationFromPython;
use qiskit_circuit::operations::{Param, StandardGate};
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::{Clbit, Qubit};
use smallvec::{smallvec, SmallVec};

/// A Toffoli chain, implementing a multi-control condition on all controls using
/// ``controls.len() - 1`` auxiliary qubits.
fn ccx_chain<'a>(
    controls: &'a [usize],
    auxiliaries: &'a [usize],
) -> impl DoubleEndedIterator<
    Item = (
        PackedOperation,
        SmallVec<[Param; 3]>,
        Vec<Qubit>,
        Vec<Clbit>,
    ),
> + 'a {
    let n = controls.len() - 1; // number of chain elements
    let indices = std::iter::once((controls[0], controls[1], auxiliaries[0]))
        .chain((0..n - 1).map(|i| (controls[i + 2], auxiliaries[i], auxiliaries[i + 1])));
    indices.map(|(ctrl1, ctrl2, target)| {
        (
            StandardGate::CCXGate.into(),
            smallvec![],
            vec![
                Qubit(ctrl1 as u32),
                Qubit(ctrl2 as u32),
                Qubit(target as u32),
            ],
            vec![],
        )
    })
}

/// Implement multi-control, multi-target of a single-qubit gate using a V-chain with
/// (num_ctrl_qubits - 1) auxiliary qubits.
/// ``controlled_gate`` here must already be the controlled operation, e.g. if we
/// call MCMT of X, then it must be a CX gate. This is because I currently don't know how to
/// nicely map the single-qubit gate to it's controlled version.
#[pyfunction]
#[pyo3(signature = (controlled_gate, num_ctrl_qubits, num_target_qubits))]
pub fn mcmt_v_chain(
    py: Python,
    controlled_gate: OperationFromPython,
    num_ctrl_qubits: usize,
    num_target_qubits: usize,
) -> PyResult<CircuitData> {
    let packed_controlled_gate = controlled_gate.operation;
    let num_qubits = if num_ctrl_qubits > 1 {
        2 * num_ctrl_qubits - 1 + num_target_qubits
    } else {
        1 + num_target_qubits // we can have 1 control and multiple targets
    };

    // First, we create the operations that apply the controlled base gate.
    // That's because we only add the V-chain of CCX gates, if the number of controls
    // is larger than 1, otherwise we're already done here.
    let master_control = if num_ctrl_qubits > 1 {
        num_qubits - 1
    } else {
        0
    };
    let targets = (0..num_target_qubits).map(|i| {
        (
            packed_controlled_gate.clone(),
            smallvec![] as SmallVec<[Param; 3]>,
            vec![
                Qubit(master_control as u32),
                Qubit((num_ctrl_qubits + i) as u32),
            ],
            vec![] as Vec<Clbit>,
        )
    });

    if num_ctrl_qubits == 1 {
        CircuitData::from_packed_operations(py, num_qubits as u32, 0, targets, Param::Float(0.0))
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
            down_chain.chain(targets).chain(up_chain),
            Param::Float(0.0),
        )
    }
}

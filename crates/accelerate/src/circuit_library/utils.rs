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
use qiskit_circuit::{
    imports,
    operations::{Param, PyInstruction, StandardGate},
    packed_instruction::PackedOperation,
    Clbit, Qubit,
};
use smallvec::{smallvec, SmallVec};

pub(super) type StandardInstruction = (StandardGate, SmallVec<[Param; 3]>, SmallVec<[Qubit; 2]>);
pub(super) type Instruction = (
    PackedOperation,
    SmallVec<[Param; 3]>,
    Vec<Qubit>,
    Vec<Clbit>,
);

/// Get an iterator that returns a barrier or an empty element.
pub fn maybe_barrier(
    py: Python,
    num_qubits: u32,
    insert_barriers: bool,
) -> Box<dyn Iterator<Item = PyResult<Instruction>>> {
    // TODO could speed this up by only defining the barrier class once
    if !insert_barriers {
        Box::new(std::iter::empty())
    } else {
        let barrier_cls = imports::BARRIER.get_bound(py);
        let barrier = barrier_cls
            .call1((num_qubits,))
            .expect("Could not create Barrier Python-side");
        let barrier_inst = PyInstruction {
            qubits: num_qubits,
            clbits: 0,
            params: 0,
            op_name: "barrier".to_string(),
            control_flow: false,
            instruction: barrier.into(),
        };
        Box::new(std::iter::once(Ok((
            barrier_inst.into(),
            smallvec![],
            (0..num_qubits).map(Qubit).collect(),
            vec![] as Vec<Clbit>,
        ))))
    }
}

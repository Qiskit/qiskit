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

use qiskit_circuit::{
    operations::{Param, StandardGate},
    packed_instruction::PackedOperation,
    Clbit, Qubit,
};
use smallvec::SmallVec;

pub(super) type StandardInstruction = (StandardGate, SmallVec<[Param; 3]>, SmallVec<[Qubit; 2]>);
pub(super) type Instruction = (
    PackedOperation,
    SmallVec<[Param; 3]>,
    Vec<Qubit>,
    Vec<Clbit>,
);

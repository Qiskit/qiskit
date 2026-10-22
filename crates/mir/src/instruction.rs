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

//! Definition of Instruction and InstructionList used in Qiskit Middle IR.

use crate::ops::{MirOp, MirOp::PauliRotation};
use crate::qid::QidList;
use crate::symbol_table::SymbolIdList;
use crate::verifier::Verifier;
use crate::{
    line_graph,
    line_graph::{LineGraph, iter::IterMain},
};
use std::fmt;
use thiserror::Error;

/// Instruction in Qiskit Middle IR
///
/// This represents an instruction in MIR. Each instruction consists of an
/// operation, a list of Quantum IDs, a list of SymbolIds of arguments and a
/// list of SymbolIds for returns. Any of these lists may be empty, indicating
/// that the specific field is not required for the given operation. For
/// example, a `PauliRotation` does not "return" any values and thus the
/// `returns` field should be an empty vector. Similarly, a branch operation
/// does not operate on, or effect any QuantumIDs and thus the `qids` list
/// should be empty.
#[derive(Clone, Debug, PartialEq)]
pub struct MirInstruction {
    op: MirOp,
    qids: QidList,
    args: SymbolIdList,
    returns: SymbolIdList, // Or just SymbolId
}

impl MirInstruction {
    pub fn new(op: MirOp, qids: QidList, args: SymbolIdList, returns: SymbolIdList) -> Self {
        Self {
            op,
            qids,
            args,
            returns,
        }
    }

    pub fn op(&self) -> &MirOp {
        &self.op
    }

    pub fn qids(&self) -> &QidList {
        &self.qids
    }

    pub fn args(&self) -> &SymbolIdList {
        &self.args
    }
    pub fn returns(&self) -> &SymbolIdList {
        &self.returns
    }
}

impl fmt::Display for MirInstruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "op: {:?}, qids: {:?}, args: {:?}, returns: {:?}",
            self.op, self.qids, self.args, self.returns
        )
    }
}

#[derive(Error, Debug)]
pub enum MirVerifierError {
    #[error("Mismatch operand lengths: {} != {}", len1, len2)]
    MismatchedOperands { len1: usize, len2: usize },
}

impl Verifier for MirInstruction {
    type Error = MirVerifierError;

    fn verify(&self) -> Result<(), Self::Error> {
        match &self.op {
            PauliRotation(pauli_rotation) => {
                // Number of Qids on the instruction must match the number of
                // paulis on the PauliRotation operation.
                if self.qids().len() != pauli_rotation.paulis().len() {
                    return Err(MirVerifierError::MismatchedOperands {
                        len1: self.qids().len(),
                        len2: pauli_rotation.paulis().len(),
                    });
                }
            }
        }
        Ok(())
    }
}

pub struct MirInstructionIndex(line_graph::Index);

impl MirInstructionIndex {
    pub fn inner(&self) -> line_graph::Index {
        self.0
    }
}

#[derive(Debug)]
pub struct MirInstructionList {
    instr_list: LineGraph<MirInstruction>,
}

impl MirInstructionList {
    pub fn new() -> Self {
        Self {
            instr_list: LineGraph::<MirInstruction>::new(),
        }
    }

    /// Add instruction to the end of the list
    pub fn push_back(&mut self, instr: MirInstruction) -> MirInstructionIndex {
        MirInstructionIndex(self.instr_list.push_back(instr))
    }

    /// Add instruction to the beginning of the list
    pub fn push_front(&mut self, instr: MirInstruction) -> MirInstructionIndex {
        MirInstructionIndex(self.instr_list.push_front(instr))
    }

    /// Iterator to walk over the list
    pub fn iter(&self) -> MirInstructionListIter<'_> {
        MirInstructionListIter {
            inner: self.instr_list.iter_main(),
        }
    }
}

impl<'a> IntoIterator for &'a MirInstructionList {
    type Item = &'a MirInstruction;
    type IntoIter = MirInstructionListIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl fmt::Display for MirInstructionList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.instr_list)
    }
}

pub struct MirInstructionListIter<'a> {
    inner: IterMain<'a, MirInstruction>,
}

impl<'a> Iterator for MirInstructionListIter<'a> {
    type Item = &'a MirInstruction;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|s| s.weight())
    }
}

impl Verifier for MirInstructionList {
    type Error = MirVerifierError;

    fn verify(&self) -> Result<(), Self::Error> {
        self.iter().map(|instr| instr.verify()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::pbc::{AngleType, Pauli, PauliList, PauliRotation};
    use crate::qid::Qid;
    use crate::symbol_table::SymbolIdList;

    fn create_pauli_rotation_op(pauli: Pauli, size: usize, angle: f64) -> MirOp {
        MirOp::PauliRotation(PauliRotation::new(
            PauliList::new(vec![pauli; size]),
            AngleType::new(angle),
        ))
    }

    #[test]
    fn test_create_instruction() {
        let instr = MirInstruction {
            op: create_pauli_rotation_op(Pauli::I, 8, 0.5),
            qids: QidList::new(vec![Qid::virt(0); 8]),
            args: SymbolIdList::default(),
            returns: SymbolIdList::default(),
        };

        assert_eq!(*instr.op(), create_pauli_rotation_op(Pauli::I, 8, 0.5));
        assert_eq!(instr.qids().len(), 8);
        assert_eq!(instr.args().len(), 0);
        assert_eq!(instr.returns().len(), 0);
    }

    #[test]
    fn test_instruction_list() {
        let mut instr_list = MirInstructionList::new();

        let instr1 = MirInstruction {
            op: create_pauli_rotation_op(Pauli::X, 8, 0.5),
            qids: QidList::new(vec![Qid::virt(0); 8]),
            args: SymbolIdList::default(),
            returns: SymbolIdList::default(),
        };

        let instr2 = MirInstruction {
            op: create_pauli_rotation_op(Pauli::Y, 8, 0.5),
            qids: QidList::new(vec![Qid::virt(0); 8]),
            args: SymbolIdList::default(),
            returns: SymbolIdList::default(),
        };

        let instr3 = MirInstruction {
            op: create_pauli_rotation_op(Pauli::Z, 8, 0.5),
            qids: QidList::new(vec![Qid::virt(0); 8]),
            args: SymbolIdList::default(),
            returns: SymbolIdList::default(),
        };

        let check_list = vec![instr3.clone(), instr1.clone(), instr2.clone()];

        instr_list.push_back(instr1);
        instr_list.push_back(instr2);
        instr_list.push_front(instr3);

        // Iterate through the instruction list.
        // Expect to find instr3, instr1, instr2
        for (index, instr) in instr_list.iter().enumerate() {
            assert_eq!(*instr, check_list[index])
        }

        instr_list.verify().unwrap();
    }

    #[test]
    fn test_invalid_pauli_instruction() {
        let instr = MirInstruction {
            op: create_pauli_rotation_op(Pauli::X, 4, 0.5),
            qids: QidList::new(vec![Qid::virt(0); 8]),
            args: SymbolIdList::default(),
            returns: SymbolIdList::default(),
        };

        let result = instr.verify();

        assert!(matches!(
            result,
            Err(MirVerifierError::MismatchedOperands { len1: 8, len2: 4 })
        ));
    }
}

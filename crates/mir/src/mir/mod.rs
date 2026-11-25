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

//! Definition of the Qiskit Middle IR.

use crate::instruction::{InstructionVerificationError, MirInstruction};
use crate::line_graph::LineGraph;
use crate::qid::{Qid, QidSpace, ResourceTable};
use crate::symbol_table::SymbolTable;

use hashbrown::HashMap;
use thiserror::Error;

/// The Qiskit Middle IR
///
/// This represents a quantum Mir compilation unit. It consists of an instruction list, a resource
/// table and a symbol table.
#[derive(Debug)]
pub struct Mir {
    instructions: LineGraph<MirInstruction>,
    resource_table: ResourceTable,
    symbol_table: SymbolTable,
}

impl Mir {
    /// Create an empty `Mir` with virtual qubits; use [`Mir::with_space`] for physical qubits.
    pub fn new() -> Self {
        Self::with_space(QidSpace::Virtual)
    }

    /// Create an empty `Mir` that uses qubits from the given space.
    pub fn with_space(space: QidSpace) -> Self {
        Self {
            instructions: LineGraph::new(),
            resource_table: ResourceTable::new(space),
            symbol_table: SymbolTable::default(),
        }
    }

    /// The instruction list of this `Mir`.
    pub fn instructions(&self) -> &LineGraph<MirInstruction> {
        &self.instructions
    }

    /// Mutable access to the instruction list of this `Mir`.
    pub fn instructions_mut(&mut self) -> &mut LineGraph<MirInstruction> {
        &mut self.instructions
    }

    /// The resource table of this `Mir`.
    pub fn resource_table(&self) -> &ResourceTable {
        &self.resource_table
    }

    /// Mutable access to the resource table of this `Mir`.
    pub fn resource_table_mut(&mut self) -> &mut ResourceTable {
        &mut self.resource_table
    }

    /// The symbol table of this `Mir`.
    pub fn symbol_table(&self) -> &SymbolTable {
        &self.symbol_table
    }

    /// Mutable access to the symbol table of this `Mir`.
    pub fn symbol_table_mut(&mut self) -> &mut SymbolTable {
        &mut self.symbol_table
    }

    /// Replace every occurrence of a qid in `mapping` with its mapped qid in all instructions
    /// of the main sequence.
    pub fn replace_instructions_qids(&mut self, mapping: &HashMap<Qid, Qid>) {
        let mut cur = self.instructions.head_index();
        while let Some(idx) = cur {
            let node = &mut self.instructions[idx];
            for qid in node.weight_mut().qids_mut().iter_mut() {
                if let Some(&mapped_qid) = mapping.get(qid) {
                    *qid = mapped_qid;
                }
            }

            cur = node.next();
        }
    }

    /// Run verification on components that make up Mir.
    ///
    /// This walks the data structures for Mir and runs verification on them.
    /// Some of these verifications will require access to additional data
    /// structures managed by Mir (e.g., [`MirInstruction::verify()`] requires
    /// access to the [`ResourceTable`]), thus providing some amount of
    /// cross-structure consistency and correctness.
    ///
    /// Current verification walks the instructions list, and for each
    /// [`MirInstruction`] verifies that it is correct. This should be extended
    /// to run similar verification on the [`SymbolTable`], and possibly the
    /// [`ResourceTable`], ensuring that assumptions for each of those data
    /// structures are met.
    ///
    /// This can be run after individual passes, or after collections of passes,
    /// to ensure Mir remains consistent throughout the compilation.
    ///
    /// This is not intended to be run *during* a pass to ensure correctness.
    pub fn verify(&self) -> Result<(), MirVerificationError> {
        Ok(self.instructions.iter_main().try_for_each(|instr| {
            instr
                .weight()
                .verify(&self.resource_table, &self.symbol_table)
        })?)
    }

    /// Get a displayable view of Mir using the given options.
    pub fn display_with(&self, options: MirDisplayOptions) -> MirDisplay<'_> {
        MirDisplay { mir: self, options }
    }
}

impl Default for Mir {
    fn default() -> Self {
        Self::new()
    }
}
#[cfg(feature = "passmanager")]
mod passmanager {
    use super::Mir;
    qiskit_util::dyn_types::static_dyn_typed!(Mir);
    impl qiskit_passmanager::IR for Mir {}
}

#[derive(Debug, Error)]
pub enum MirVerificationError {
    #[error("Instruction List Error: {0}")]
    InstructionList(#[from] InstructionVerificationError),
    // This can be expanded to include other types of verification errors (e.g.,
    // SymbolTable, ResourceTable, etc)
}

/// Options controlling how a Mir is printed.
#[derive(Clone, Debug)]
pub struct MirDisplayOptions {
    /// The header line to print before everything else; None omits the header.
    pub header: Option<String>,
    /// Whether to print the resource table (qids).
    pub resource_table: bool,
    /// Whether to print the symbol table (classical symbols).
    pub symbol_table: bool,
    /// Whether to print the instructions.
    pub instructions: bool,
}

impl Default for MirDisplayOptions {
    fn default() -> Self {
        Self {
            header: Some("mir, @circuit".to_string()),
            resource_table: true,
            symbol_table: true,
            instructions: true,
        }
    }
}

pub struct MirDisplay<'a> {
    mir: &'a Mir,
    options: MirDisplayOptions,
}

impl std::fmt::Display for MirDisplay<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(header) = &self.options.header {
            writeln!(f, "{header}")?;
        }

        if self.options.resource_table {
            for (qid, info) in self.mir.resource_table.iter_qids() {
                writeln!(f, "{qid}, {}", info.layout())?;
            }
        }

        if self.options.symbol_table {
            for sym in self.mir.symbol_table.iter() {
                writeln!(f, "{sym}")?;
            }
        }

        if self.options.instructions {
            for node in self.mir.instructions.iter_main() {
                writeln!(f, "{}", node.weight())?;
            }
        }
        Ok(())
    }
}

impl std::fmt::Display for Mir {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.display_with(MirDisplayOptions::default()).fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instruction::InstructionVerificationError;
    use crate::ops::MirOp;
    use crate::ops::pbc::{AngleType, Pauli, PauliRotation, PauliString};
    use crate::qid::{BitLayout, QidInfo, QidList};
    use crate::symbol_table::{Symbol, SymbolIdList, SymbolType};

    #[test]
    fn test_populate_mir() {
        let mut mir = Mir::new();

        // Add a qubit, and an instruction using it to `Mir`
        let qid_info = QidInfo::new(BitLayout::Base(1));
        let qid = mir.resource_table_mut().add_qid(qid_info.clone()).unwrap();

        let op = MirOp::PauliRotation(PauliRotation::new(
            PauliString::new(vec![Pauli::X]),
            AngleType::new(0.5),
        ));

        let instruction = MirInstruction::new(
            op,
            QidList::new(vec![qid]),
            SymbolIdList::default(),
            SymbolIdList::default(),
        );
        mir.instructions_mut().push_back(instruction.clone());

        assert_eq!(mir.resource_table().iter_qids().len(), 1);
        assert_eq!(mir.resource_table().info(qid), Some(&qid_info));

        assert_eq!(mir.instructions().node_count(), 1);
        assert_eq!(
            *mir.instructions().iter_main().next().unwrap().weight(),
            instruction
        );

        // Add a symbol to `Mir`
        let sym_id = mir
            .symbol_table_mut()
            .add(Symbol::new("a".to_string(), SymbolType::Bool))
            .unwrap();

        let symbol = mir.symbol_table().get(sym_id).unwrap();
        assert_eq!(symbol.name(), "a");
        assert_eq!(symbol.id(), Some(sym_id));
    }

    #[test]
    fn test_create_mir_with_qid_space() {
        let mut virtual_mir = Mir::new();
        let qid = virtual_mir
            .resource_table_mut()
            .add_qid(QidInfo::new(BitLayout::Base(1)))
            .unwrap();
        assert_eq!(qid.space(), QidSpace::Virtual);

        let mut physical_mir = Mir::with_space(QidSpace::Physical);
        let qid = physical_mir
            .resource_table_mut()
            .add_qid(QidInfo::new(BitLayout::Base(1)))
            .unwrap();
        assert_eq!(qid.space(), QidSpace::Physical);
    }

    #[test]
    fn test_bad_instruction() {
        let mut mir = Mir::new();
        let qid = mir
            .resource_table_mut()
            .add_qid(QidInfo::new(BitLayout::Base(1)))
            .unwrap();
        mir.instructions_mut().push_back(MirInstruction::new(
            MirOp::PauliRotation(PauliRotation::new(
                PauliString::new(vec![Pauli::X; 2]),
                AngleType::new(0.5),
            )),
            QidList::new(vec![qid; 2]), // passing same Qid to a pauli rotation is illegal
            SymbolIdList::default(),
            SymbolIdList::default(),
        ));

        //mir.verify().unwrap();
        assert!(matches!(
            mir.verify(),
            Err(MirVerificationError::InstructionList(
                InstructionVerificationError::AliasedQids
            ))
        ));
    }

    #[test]
    fn test_replace_instructions_qids() {
        let mut mir = Mir::new();
        for _ in 0..3 {
            mir.resource_table_mut()
                .add_qid(QidInfo::new(BitLayout::Base(1)))
                .unwrap();
        }
        // add instruction on [0,1]
        mir.instructions_mut().push_back(MirInstruction::new(
            MirOp::PauliRotation(PauliRotation::new(
                PauliString::new(vec![Pauli::X; 2]),
                AngleType::new(0.5),
            )),
            QidList::new(vec![Qid::virt(0), Qid::virt(1)]),
            SymbolIdList::default(),
            SymbolIdList::default(),
        ));
        // add instruction on [2,0]
        mir.instructions_mut().push_back(MirInstruction::new(
            MirOp::PauliRotation(PauliRotation::new(
                PauliString::new(vec![Pauli::X; 2]),
                AngleType::new(0.5),
            )),
            QidList::new(vec![Qid::virt(2), Qid::virt(0)]),
            SymbolIdList::default(),
            SymbolIdList::default(),
        ));

        // map qids 0 and 1 to 10, 11; 2 is not mapped.
        let mapping = HashMap::from([(Qid::virt(0), Qid::virt(10)), (Qid::virt(1), Qid::virt(11))]);

        // call replace
        mir.replace_instructions_qids(&mapping);

        let mut instrs = mir.instructions().iter_main();

        assert_eq!(
            instrs.next().unwrap().weight().qids().inner(),
            &vec![Qid::virt(10), Qid::virt(11)]
        );
        assert_eq!(
            instrs.next().unwrap().weight().qids().inner(),
            &vec![Qid::virt(2), Qid::virt(10)]
        );
    }
}

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

use crate::instruction::MirInstruction;
use crate::line_graph::LineGraph;
use crate::qid::{QidSpace, ResourceTable};
use crate::symbol_table::SymbolTable;

/// The Qiskit Middle IR
///
/// This represents a quantum MIR compilation unit. It consists of an instruction list, a resource
/// table and a symbol table.
#[derive(Debug)]
pub struct Mir {
    instructions: LineGraph<MirInstruction>,
    resource_table: ResourceTable,
    symbol_table: SymbolTable,
}

impl Mir {
    /// Create an empty Mir.
    pub fn new() -> Self {
        Self {
            instructions: LineGraph::new(),
            resource_table: ResourceTable::new(QidSpace::Virtual),
            symbol_table: SymbolTable::default(),
        }
    }

    /// The instruction list of this MIR.
    pub fn instructions(&self) -> &LineGraph<MirInstruction> {
        &self.instructions
    }

    /// Mutable access to the instruction list of this MIR.
    pub fn instructions_mut(&mut self) -> &mut LineGraph<MirInstruction> {
        &mut self.instructions
    }

    /// The resource table of this MIR.
    pub fn resource_table(&self) -> &ResourceTable {
        &self.resource_table
    }

    /// Mutable access to the resource table of this MIR.
    pub fn resource_table_mut(&mut self) -> &mut ResourceTable {
        &mut self.resource_table
    }

    /// The symbol table of this MIR.
    pub fn symbol_table(&self) -> &SymbolTable {
        &self.symbol_table
    }

    /// Mutable access to the symbol table of this MIR.
    pub fn symbol_table_mut(&mut self) -> &mut SymbolTable {
        &mut self.symbol_table
    }
}

impl Default for Mir {
    fn default() -> Self {
        Self::new()
    }
}

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

//! Validate memory analysis of Mir structs using mem_dbg feature works.   

use mem_dbg::{DbgFlags, MemDbg, MemSize, SizeFlags};
use qiskit_mir::instruction::MirInstruction;
use qiskit_mir::line_graph::LineGraph;
use qiskit_mir::ops::MirOp;
use qiskit_mir::ops::pbc::{AngleType, Pauli, PauliRotation, PauliString};
use qiskit_mir::qid::{Qid, QidList};
use qiskit_mir::symbol_table::SymbolIdList;

#[test]
fn validate_mem_dbg() {
    let qids = QidList::new(vec![Qid::virt(0), Qid::virt(1)]);

    let instr = MirInstruction::new(
        MirOp::PauliRotation(PauliRotation::new(
            PauliString::new(vec![Pauli::X; 2]),
            AngleType::new(0.5),
        )),
        qids,
        SymbolIdList::default(),
        SymbolIdList::default(),
    );

    let mut graph = LineGraph::new();
    graph.push_back(instr);

    // Verify mem_size
    let used = graph.mem_size(SizeFlags::default());
    assert!(used > 0, "Expected positive memory allocation");

    // Verify mem_dbg. Write into a buffer insteaf of stdout.
    let mut buffer = String::new();
    graph
        .mem_dbg_on(&mut buffer, DbgFlags::default())
        .expect("failed to render mem_dbg tree");
}

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

//! The following integration tests build MIR by hand to exercise every construct that MIR can
//! currently express, and check the textual (debug) output for it.
//!
//! # Running the tests and writing the output to files
//!
//! ```text
//! QISKIT_DEBUG_DIR=/tmp/mir_debug cargo test -p qiskit-mir-passes --test constructs
//! ```

use qiskit_mir::instruction::MirInstruction;
use qiskit_mir::mir::{Mir, MirDisplayOptions};
use qiskit_mir::ops::MirOp;
use qiskit_mir::ops::pbc::{AngleType, Pauli, PauliMeasure, PauliRotation, PauliString};
use qiskit_mir::qid::{BitLayout, Qid, QidInfo, QidList};
use qiskit_mir::symbol_table::{Symbol, SymbolId, SymbolIdList, SymbolType};

fn rotation(paulis: &[Pauli], angle: f64, qids: &[Qid]) -> MirInstruction {
    MirInstruction::new(
        MirOp::PauliRotation(PauliRotation::new(
            PauliString::new(paulis.to_vec()),
            AngleType::new(angle),
        )),
        QidList::new(qids.to_vec()),
        SymbolIdList::default(),
        SymbolIdList::default(),
    )
}

fn measure(paulis: &[Pauli], qids: &[Qid], result: SymbolId) -> MirInstruction {
    MirInstruction::new(
        MirOp::PauliMeasure(PauliMeasure::new(PauliString::new(paulis.to_vec()))),
        QidList::new(qids.to_vec()),
        SymbolIdList::default(),
        SymbolIdList::new(vec![result]),
    )
}

fn bool_not(src: SymbolId, dest: SymbolId) -> MirInstruction {
    MirInstruction::new(
        MirOp::BoolNot,
        QidList::new(vec![]),
        SymbolIdList::new(vec![src]),
        SymbolIdList::new(vec![dest]),
    )
}

fn add_base(mir: &mut Mir, width: u32) -> Qid {
    mir.resource_table_mut()
        .add_qid(QidInfo::new(BitLayout::Base(width)))
        .unwrap()
}

fn add_symbol(mir: &mut Mir, name: &str) -> SymbolId {
    mir.symbol_table_mut()
        .add(Symbol::new(name.to_string(), SymbolType::Bool))
        .unwrap()
}

/// Build a (virtual) `Mir` that uses every layout and every operation.
fn build_constructs_mir() -> Mir {
    let mut mir = Mir::new();

    // Resource table: Base, Abstract and Alias layouts.
    let v0 = add_base(&mut mir, 1);
    let v1 = add_base(&mut mir, 1);
    let v2 = add_base(&mut mir, 2);
    let _v3 = mir
        .resource_table_mut()
        .add_qid(QidInfo::widthless())
        .unwrap();
    // An alias of the (complete) bits of two other qids.
    let v4 = mir
        .resource_table_mut()
        .add_qid(QidInfo::new(BitLayout::Alias(
            vec![Some(v0.bit(0)), Some(v1.bit(0))].into_boxed_slice(),
        )))
        .unwrap();
    // An alias where one of the bits does not alias anything.
    let v5 = mir
        .resource_table_mut()
        .add_qid(QidInfo::new(BitLayout::Alias(
            vec![Some(v0.bit(0)), None].into_boxed_slice(),
        )))
        .unwrap();

    // Symbol table.
    let c0 = add_symbol(&mut mir, "c0");
    let c_not = add_symbol(&mut mir, "c_not");

    let instructions = mir.instructions_mut();

    // PauliRotation: single-qubit rotations with Pauli and various angles.
    instructions.push_back(rotation(&[Pauli::I], 1.0, &[v0]));
    instructions.push_back(rotation(&[Pauli::X], -0.5, &[v1]));
    // PauliRotation: joint rotation over several qids.
    instructions.push_back(rotation(&[Pauli::Z, Pauli::X], 0.5, &[v0, v1]));
    // PauliRotation: a qid that is wider than one bit, alone and together with others.
    instructions.push_back(rotation(&[Pauli::X, Pauli::X], 0.5, &[v2]));
    instructions.push_back(rotation(
        &[Pauli::X, Pauli::Y, Pauli::Z, Pauli::X],
        0.25,
        &[v0, v1, v2],
    ));
    // PauliRotation: on alias qids.
    instructions.push_back(rotation(&[Pauli::Z, Pauli::Z], 0.5, &[v4]));
    instructions.push_back(rotation(&[Pauli::X, Pauli::Y], 0.25, &[v5]));

    // PauliMeasure: single-qubit, and a joint parity measurement.
    instructions.push_back(measure(&[Pauli::Z], &[v0], c0));
    instructions.push_back(measure(&[Pauli::Z, Pauli::Z], &[v0, v1], c0));

    // BoolNot: out-of-place.
    instructions.push_back(bool_not(c0, c_not));

    mir
}

const ALL_HEADER: &str = "mir, @constructs_circuit\n";
const ALL_RESOURCE_TABLE: &str = concat!(
    "v0, Base(1)\n",
    "v1, Base(1)\n",
    "v2, Base(2)\n",
    "v3, Abstract\n",
    "v4, Alias([(v0, 0), (v1, 0)])\n",
    "v5, Alias([(v0, 0), _])\n",
);
const ALL_SYMBOL_TABLE: &str = concat!("%0, c0, Bool\n", "%1, c_not, Bool\n");
const ALL_INSTRUCTIONS: &str = concat!(
    "[] = PauliRotation(I, 1.0), [v0]\n",
    "[] = PauliRotation(X, -0.5), [v1]\n",
    "[] = PauliRotation(ZX, 0.5), [v0,v1]\n",
    "[] = PauliRotation(XX, 0.5), [v2]\n",
    "[] = PauliRotation(XYZX, 0.25), [v0,v1,v2]\n",
    "[] = PauliRotation(ZZ, 0.5), [v4]\n",
    "[] = PauliRotation(XY, 0.25), [v5]\n",
    "[%0] = PauliMeasure(Z), [v0]\n",
    "[%0] = PauliMeasure(ZZ), [v0,v1]\n",
    "[%1] = BoolNot(), [%0]\n",
);

/// Write `text` to `file_name` inside the directory named by the `QISKIT_DEBUG_DIR` environment
/// variable. Creates the directory if it does not exist.
fn dump_debug(file_name: &str, text: &str) {
    if let Some(dir) = std::env::var_os("QISKIT_DEBUG_DIR") {
        let dir = std::path::Path::new(&dir);
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join(file_name), text).unwrap();
    }
}

#[test]
fn constructs_display_full() {
    let mir = build_constructs_mir();
    mir.verify().unwrap();

    // A custom header, with every section enabled.
    let options = MirDisplayOptions {
        header: Some("mir, @constructs_circuit".to_string()),
        ..Default::default()
    };

    let output = mir.display_with(options).to_string();
    dump_debug("constructs_full_log.txt", &output);
    let expected = format!("{ALL_HEADER}{ALL_RESOURCE_TABLE}{ALL_SYMBOL_TABLE}{ALL_INSTRUCTIONS}");
    assert_eq!(output, expected);
}

#[test]
fn constructs_display_instructions() {
    let mir = build_constructs_mir();

    // Only the instructions are printed.
    let options = MirDisplayOptions {
        header: None,
        resource_table: false,
        symbol_table: false,
        instructions: true,
    };

    let output = mir.display_with(options).to_string();
    dump_debug("constructs_instructions_log.txt", &output);
    assert_eq!(output, ALL_INSTRUCTIONS);
}

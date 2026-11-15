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

//! Trivial layout pass of MIR to given physical qids.
use hashbrown::HashMap;
use qiskit_mir::{
    mir::Mir,
    qid::{BitLayout, Qid, QidSpace, ResourceTable},
};
use qiskit_passmanager::{PassContext, StaticPass};

/// Layout a Mir on physical qids with a trivial mapping of each virtual qid to physical qid with same value.
/// Assumes given Mir contains only base virtual qids of width 1.
/// Assumes given physical resource table contains only base physical qids of width 1.
/// Maps all virtual qids to physical by value. Does not check if qids are used in instructions, i.e. also maps idle qids if they exist.
/// Modifies the Mir in place by replacing all the instructions' qids with the physical
/// ones. Mir's original resource table is left unchanged.
/// After layout the Mir's instructions use physical qids that are not in its resource table, so the current Mir::verify() cannot run on it.
///
/// # Errors
///
/// Returns an error if a virtual qid has no physical qid with the same value in the physical table,
/// if input Mir qids are not virtual or input physical table qids are not physical or if
/// input virtual qids or physical qids are not base qids of width 1.
pub fn trivial_mir_layout(
    mir: &mut Mir,
    physical_resource_table: &ResourceTable,
) -> anyhow::Result<()> {
    // Verify input Mir qids are virtual, input resource table is physical
    // and all qids are base qids of width 1.
    verify_single_bit_base_qids(mir.resource_table(), QidSpace::Virtual)?;
    verify_single_bit_base_qids(physical_resource_table, QidSpace::Physical)?;

    // Create a trivial mapping
    let mapping = create_trivial_virtual_to_physical_mapping(mir, physical_resource_table)?;

    mir.replace_instructions_qids(&mapping);

    Ok(())
}

/// Check that every qid of `table` is in `space` and is a base qid of width 1.
fn verify_single_bit_base_qids(table: &ResourceTable, space: QidSpace) -> anyhow::Result<()> {
    for (qid, info) in table.iter_qids() {
        anyhow::ensure!(
            qid.space() == space,
            "expected a {space:?} resource table, found {qid}"
        );
        anyhow::ensure!(
            matches!(info.layout(), BitLayout::Base(1)),
            "{qid} must be a base qid of width 1, found layout {:?}",
            info.layout()
        );
    }
    Ok(())
}

/// Create trivial mapping from Mir with virtual qids to physical qids by value.
/// Returns an error if a virtual qid has no physical qid with the same value in the physical table
fn create_trivial_virtual_to_physical_mapping(
    mir: &Mir,
    physical_resource_table: &ResourceTable,
) -> anyhow::Result<HashMap<Qid, Qid>> {
    let mut mapping = HashMap::new();
    for (virt, _) in mir.resource_table().iter_qids() {
        let phys = Qid::phys(virt.value());
        anyhow::ensure!(
            physical_resource_table.info(phys).is_some(),
            "virtual qid {virt} has no matching physical qid {phys}"
        );
        mapping.insert(virt, phys);
    }
    Ok(mapping)
}

/// `StaticPass` that lays out a Mir on physical qids with a trivial allocation
/// of each virtual qid to a physical qid.
///
/// The physical resource table is a field of the pass for now. It is expected to come from the
/// backend target (via `PassContext`) once that exists.
pub struct TrivialMirLayout {
    physical_resource_table: ResourceTable,
}
impl TrivialMirLayout {
    /// Initialize the layout with a given physical resource table.
    pub fn new(physical_resource_table: ResourceTable) -> Self {
        Self {
            physical_resource_table,
        }
    }
}

impl StaticPass<Mir> for TrivialMirLayout {
    fn run(&self, mut mir: Box<Mir>, _context: &mut PassContext) -> anyhow::Result<Box<Mir>> {
        trivial_mir_layout(&mut mir, &self.physical_resource_table)?;
        Ok(mir)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use qiskit_mir::{
        instruction::MirInstruction,
        ops::{
            MirOp,
            pbc::{AngleType, Pauli, PauliRotation, PauliString},
        },
        qid::{QidBit, QidInfo, QidList},
        symbol_table::SymbolIdList,
    };

    /// Create resource table helper.
    fn create_test_resource_table(space: QidSpace, n: u32) -> ResourceTable {
        let mut table = ResourceTable::new(space);
        for _ in 0..n {
            table.add_qid(QidInfo::new(BitLayout::Base(1))).unwrap();
        }
        table
    }

    /// Create a Mir helper.
    /// A Mir with one instruction over all qids in the given resource table
    fn create_simplet_test_mir(table: ResourceTable) -> Mir {
        let mut mir = Mir::new();
        let qids: Vec<Qid> = table.iter_qids().map(|(q, _)| q).collect();

        let op = MirOp::PauliRotation(PauliRotation::new(
            PauliString::new(vec![Pauli::Z; qids.len()]),
            AngleType::new(0.5),
        ));
        mir.instructions_mut().push_back(MirInstruction::new(
            op,
            QidList::new(qids),
            SymbolIdList::default(),
            SymbolIdList::default(),
        ));
        *mir.resource_table_mut() = table;

        mir
    }

    #[test]
    fn layout_replaces_qids_to_physical() {
        let virt = create_test_resource_table(QidSpace::Virtual, 3);
        let mut mir = create_simplet_test_mir(virt);
        trivial_mir_layout(&mut mir, &create_test_resource_table(QidSpace::Physical, 4)).unwrap();

        let instr = mir.instructions().iter_main().next().unwrap().weight();
        assert_eq!(
            instr.qids().inner(),
            &vec![Qid::phys(0), Qid::phys(1), Qid::phys(2)]
        );
    }

    #[test]
    fn layout_errors_on_missing_qid_value_in_physical_table() {
        let mut mir = create_simplet_test_mir(create_test_resource_table(QidSpace::Virtual, 10));
        assert!(
            trivial_mir_layout(&mut mir, &create_test_resource_table(QidSpace::Physical, 8))
                .is_err()
        );
    }

    #[test]
    fn layout_rejects_invalid_input_resource_tables() {
        // Invalid Mir resource table cases
        let dummy_physical_table = create_test_resource_table(QidSpace::Physical, 3);

        // Physical qids in the Mir's resource table.
        let mut mir = create_simplet_test_mir(create_test_resource_table(QidSpace::Physical, 3));
        assert!(trivial_mir_layout(&mut mir, &dummy_physical_table).is_err());

        // An alias in the Mir's resource table.
        let mut virtual_table = create_test_resource_table(QidSpace::Virtual, 2);
        let bits = [0, 1].map(|i| Some(QidBit::new(Qid::virt(i), 0)));
        virtual_table
            .add_qid(QidInfo::new(BitLayout::Alias(Box::new(bits))))
            .unwrap();
        let mut mir = create_simplet_test_mir(virtual_table);
        assert!(trivial_mir_layout(&mut mir, &dummy_physical_table).is_err());

        // Abstract qid in the Mir's resource table.
        let mut virtual_table = create_test_resource_table(QidSpace::Virtual, 2);
        virtual_table
            .add_qid(QidInfo::new(BitLayout::Abstract))
            .unwrap();
        let mut mir = create_simplet_test_mir(virtual_table);
        assert!(trivial_mir_layout(&mut mir, &dummy_physical_table).is_err());

        // Base qid with width>1 in the Mir's resource table.
        let mut virtual_table = create_test_resource_table(QidSpace::Virtual, 2);
        virtual_table
            .add_qid(QidInfo::new(BitLayout::Base(2)))
            .unwrap();
        let mut mir = create_simplet_test_mir(virtual_table);
        assert!(trivial_mir_layout(&mut mir, &dummy_physical_table).is_err());

        // Invalid physical resource table cases
        let dummy_virtual_table = create_test_resource_table(QidSpace::Virtual, 3);

        // Virtual qids in the input physical table.
        let mut mir = create_simplet_test_mir(dummy_virtual_table.clone());
        assert!(
            trivial_mir_layout(&mut mir, &create_test_resource_table(QidSpace::Virtual, 3))
                .is_err()
        );

        // An alias in the input physical table.
        let mut physical_table = create_test_resource_table(QidSpace::Physical, 2);
        let bits = [0, 1].map(|i| Some(QidBit::new(Qid::phys(i), 0)));
        physical_table
            .add_qid(QidInfo::new(BitLayout::Alias(Box::new(bits))))
            .unwrap();
        let mut mir = create_simplet_test_mir(dummy_virtual_table.clone());
        assert!(trivial_mir_layout(&mut mir, &physical_table).is_err());

        // An abstract qid in the input physical table.
        let mut physical_table = create_test_resource_table(QidSpace::Physical, 2);
        physical_table
            .add_qid(QidInfo::new(BitLayout::Abstract))
            .unwrap();
        let mut mir = create_simplet_test_mir(dummy_virtual_table.clone());
        assert!(trivial_mir_layout(&mut mir, &physical_table).is_err());

        // A base qid with width>1 in the input physical table.
        let mut physical_table = create_test_resource_table(QidSpace::Physical, 2);
        physical_table
            .add_qid(QidInfo::new(BitLayout::Base(2)))
            .unwrap();
        let mut mir = create_simplet_test_mir(dummy_virtual_table.clone());
        assert!(trivial_mir_layout(&mut mir, &physical_table).is_err());
    }
}

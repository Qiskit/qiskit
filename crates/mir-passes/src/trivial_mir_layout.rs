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
    qid::{BitLayout, Qid, QidList, QidSpace, ResourceTable},
};
use qiskit_passmanager::{PassContext, StaticPass};

/// Layout a Mir on physical qids with a trivial mapping of each virtual qid to physical qid in the same position.
/// Assumes given Mir contains only base virtual qids (no aliases).
/// Assumes given physical resource table contains only base physical qids and is at least the size of the Mir resource table.
/// Maps all virtual qids to physical by position. Does not check if qids are used in instructions, i.e. also maps idle qids if they exist.
/// Modifies the Mir in place by replacing all the instructions' qids with the physical
/// ones and replacing the Mir resource table with the given physical table.
/// # Errors
///
/// Returns an error if the physical table has fewer qids than the Mir's resource table,
/// if input Mir qids are not virtual or input physical table qids are not physical or if
/// input virtual qids or physical qids are aliases.
pub fn trivial_mir_layout(
    mir: &mut Mir,
    physical_resource_table: ResourceTable,
) -> anyhow::Result<()> {
    // Verify input Mir qids are virtual, input resource table is physical
    // and all qids are base qids.
    verify_base_qids(mir.resource_table(), QidSpace::Virtual)?;
    verify_base_qids(&physical_resource_table, QidSpace::Physical)?;

    // Verify given resource table is large enough
    let num_physical = physical_resource_table.iter_qids().len();
    let num_virtual = mir.resource_table().iter_qids().len();
    anyhow::ensure!(
        num_physical >= num_virtual,
        "physical resource table ({num_physical} qids) is smaller than the Mir's resource table ({num_virtual} qids)"
    );

    // Create a trivial mapping
    let mapping: HashMap<Qid, Qid> =
        create_trivial_virtual_to_physical_mapping(mir, &physical_resource_table);

    // Collect all instruction indices
    let indices: Vec<_> = mir
        .instructions()
        .iter_main()
        .indexed()
        .map(|(i, _)| i)
        .collect();

    // Replace qids in each instruction by applying the mapping
    for idx in indices {
        let instr = mir.instructions_mut().data_unwrap_mut(idx).weight_mut();
        let new_qids_vec = instr.qids().inner().iter().map(|q| mapping[q]).collect();
        instr.set_qids(QidList::new(new_qids_vec));
    }

    // Swap the resource table
    *mir.resource_table_mut() = physical_resource_table;
    Ok(())
}

/// Check that every qid of `table` is in `space` and is a base qid.
fn verify_base_qids(table: &ResourceTable, space: QidSpace) -> anyhow::Result<()> {
    for (qid, info) in table.iter_qids() {
        anyhow::ensure!(
            qid.space() == space,
            "expected a {space:?} resource table, found {qid}"
        );
        anyhow::ensure!(
            matches!(info.layout(), BitLayout::Base(_)),
            "{qid} is not a base qid, aliases are not supported"
        );
    }
    Ok(())
}

/// Create trivial mapping from Mir with virtual qids to physical qids.
/// The i-th virtual qid of `Mir` is mapped to the i-th qid of `physical`.
fn create_trivial_virtual_to_physical_mapping(
    mir: &Mir,
    physical_resource_table: &ResourceTable,
) -> HashMap<Qid, Qid> {
    mir.resource_table()
        .iter_qids()
        .zip(physical_resource_table.iter_qids())
        .map(|((virt, _), (phys, _))| (virt, phys))
        .collect()
}

/// `StaticPass` that lays out a Mir on physical qids with a trivial allocation
/// of each virtual qid to a physical qid.
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
        trivial_mir_layout(&mut mir, self.physical_resource_table.clone())?;
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
        qid::{QidBit, QidInfo},
        symbol_table::SymbolIdList,
    };

    /// Create resource table helper.
    fn create_table(space: QidSpace, n: u32) -> ResourceTable {
        let mut table = ResourceTable::new(space);
        for _ in 0..n {
            table.add_qid(QidInfo::new(BitLayout::Base(1))).unwrap();
        }
        table
    }

    /// Create a Mir helper.
    /// A Mir with one instruction over all qids in the given resource table
    fn create_simple_mir(table: ResourceTable) -> Mir {
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
        let virt = create_table(QidSpace::Virtual, 3);
        let mut mir = create_simple_mir(virt);
        trivial_mir_layout(&mut mir, create_table(QidSpace::Physical, 4)).unwrap();

        let instr = mir.instructions().iter_main().next().unwrap().weight();
        assert_eq!(
            instr.qids().inner(),
            &vec![Qid::phys(0), Qid::phys(1), Qid::phys(2)]
        );
        assert_eq!(mir.resource_table().iter_qids().len(), 4);
        assert!(
            mir.resource_table()
                .iter_qids()
                .all(|(q, _)| q.space() == QidSpace::Physical)
        );
    }

    #[test]
    fn layout_errors_on_smaller_physical_table() {
        let mut mir = create_simple_mir(create_table(QidSpace::Virtual, 10));
        assert!(trivial_mir_layout(&mut mir, create_table(QidSpace::Physical, 8)).is_err());
    }

    #[test]
    fn layout_rejects_invalid_input_resource_tables() {
        // Physical qids in the Mir's resource table.
        let mut mir = create_simple_mir(create_table(QidSpace::Physical, 2));
        assert!(trivial_mir_layout(&mut mir, create_table(QidSpace::Physical, 2)).is_err());

        // Virtual qids in the input physical table.
        let mut mir = create_simple_mir(create_table(QidSpace::Virtual, 2));
        assert!(trivial_mir_layout(&mut mir, create_table(QidSpace::Virtual, 2)).is_err());

        // An alias in the Mir's resource table.
        let mut virtual_table = create_table(QidSpace::Virtual, 2);
        let bits = [0, 1].map(|i| Some(QidBit::new(Qid::virt(i), 0)));
        virtual_table
            .add_qid(QidInfo::new(BitLayout::Alias(Box::new(bits))))
            .unwrap();
        let mut mir = create_simple_mir(virtual_table);
        assert!(trivial_mir_layout(&mut mir, create_table(QidSpace::Physical, 3)).is_err());

        // An alias in the input physical table.
        let mut physical_table = create_table(QidSpace::Physical, 2);
        let bits = [0, 1].map(|i| Some(QidBit::new(Qid::phys(i), 0)));
        physical_table
            .add_qid(QidInfo::new(BitLayout::Alias(Box::new(bits))))
            .unwrap();
        let mut mir = create_simple_mir(create_table(QidSpace::Virtual, 2));
        assert!(trivial_mir_layout(&mut mir, physical_table).is_err());
    }
}

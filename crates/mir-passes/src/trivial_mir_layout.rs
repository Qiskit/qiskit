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
    qid::{Qid, QidList, ResourceTable},
};
use qiskit_passmanager::{PassContext, StaticPass};

/// Layout a Mir on physical qids with a trivial mapping of each virtual qid to physical qid in the same position.
/// Assumes given physical resource table is at least the size of the Mir
/// resource table, and maps all qids including idle ones if exist (without
/// checking they are used in the instructions).
/// Modify the Mir in place by replacing all the instructions' qids with the physical
/// ones and replacing the resource table with the given physical table.
/// # Errors
/// Returns an error if the physical table has fewer qids than the Mir's resource table.
pub fn trivial_mir_layout(
    mir: &mut Mir,
    physical_resource_table: &ResourceTable,
) -> anyhow::Result<()> {
    // Verify given resource table is large enough
    let num_physical = physical_resource_table.iter_qids().len();
    let num_virtual = mir.resource_table().iter_qids().len();
    anyhow::ensure!(
        num_physical >= num_virtual,
        "physical resource table ({num_physical} qids) is smaller than the Mir's resource table ({num_virtual} qids)"
    );

    // Create a trivial mapping
    let mapping: HashMap<Qid, Qid> =
        create_trivial_virtual_to_physical_mapping(mir, physical_resource_table);

    // Collect all instruction indices
    let indices: Vec<_> = mir
        .instructions()
        .iter_main()
        .indexed()
        .map(|(i, _)| i)
        .collect();

    //Replace qids in each instruction by applying the mapping
    for idx in indices {
        let instr = mir.instructions_mut().data_unwrap_mut(idx).weight_mut();
        let new_qids_vec = instr.qids().inner().iter().map(|q| mapping[q]).collect();
        instr.set_qids(QidList::new(new_qids_vec));
    }

    //Swap the resource table
    *mir.resource_table_mut() = physical_resource_table.clone();
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
    ///Initialize the layout with a given phydical resource table.
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

// This code is part of Qiskit.
//
// (C) Copyright IBM 2023
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use hashbrown::HashMap;
use std::sync::Arc;

pub type IndexType = u32;
pub type BitType = u32;

#[derive(Clone, Debug)]
pub struct InternContext {
    slots: Vec<Arc<Vec<BitType>>>,
    slot_lookup: HashMap<Arc<Vec<BitType>>, IndexType>,
}

impl InternContext {
    pub fn intern(&mut self, args: Vec<BitType>) -> IndexType {
        if let Some(slot_idx) = self.slot_lookup.get(&args) {
            return *slot_idx;
        }

        let args = Arc::new(args);
        let slot_idx = self.slots.len() as IndexType;
        self.slots.push(args.clone());
        self.slot_lookup.insert_unique_unchecked(args, slot_idx);
        slot_idx
    }

    pub fn lookup(&self, slot_idx: IndexType) -> &Vec<BitType> {
        self.slots.get(slot_idx as usize).unwrap()
    }
}

impl InternContext {
    pub fn new() -> Self {
        InternContext {
            slots: Vec::new(),
            slot_lookup: HashMap::new(),
        }
    }
}

impl Default for InternContext {
    fn default() -> Self {
        Self::new()
    }
}

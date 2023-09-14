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
use pyo3::prelude::*;
use std::collections::VecDeque;
use std::mem::take;
use std::sync::Arc;

pub type IndexType = u32;
pub type BitType = u32;

#[derive(Clone, Debug)]
struct SharedOperandList {
    operands: Arc<Vec<BitType>>,
    use_count: usize,
}

#[pyclass(module = "qiskit._accelerate.quantum_circuit")]
#[derive(Clone, Debug)]
pub struct InternContext {
    slots: Vec<Option<SharedOperandList>>,
    free_slots: VecDeque<IndexType>,
    slot_lookup: HashMap<Arc<Vec<BitType>>, IndexType>,
}

impl InternContext {
    pub fn intern(&mut self, args: Vec<BitType>) -> Option<IndexType> {
        let args = Arc::new(args);
        if !self.slot_lookup.contains_key(&args)
            && self.free_slots.is_empty()
            && IndexType::MAX == self.slots.len().try_into().unwrap()
        {
            return None;
        }

        let slot_idx = self.slot_lookup.entry(args.clone()).or_insert_with(|| {
            if !self.free_slots.is_empty() {
                self.free_slots.pop_front().unwrap()
            } else {
                self.slots.push(None);
                self.slots.len().try_into().unwrap()
            }
        });
        let shared_args = self
            .slots
            .get_mut(*slot_idx as usize)
            .unwrap()
            .get_or_insert_with(|| SharedOperandList {
                operands: args,
                use_count: 0,
            });
        shared_args.use_count += 1;
        Some(*slot_idx)
    }

    pub fn lookup(&self, slot_idx: IndexType) -> &Vec<BitType> {
        let slot = self.slots.get(slot_idx as usize).unwrap();
        &slot.as_ref().unwrap().operands
    }

    pub fn drop_use(&mut self, slot_idx: IndexType) {
        let mut shared = take(&mut self.slots[slot_idx as usize]).unwrap();
        if let SharedOperandList {
            operands,
            use_count: 1,
        } = shared
        {
            self.slot_lookup.remove(&operands);
            self.free_slots.push_back(slot_idx);
            return;
        };
        shared.use_count -= 1;
        self.slots[slot_idx as usize] = Some(shared);
    }
}

#[pymethods]
impl InternContext {
    #[new]
    pub fn new() -> Self {
        InternContext {
            slots: Vec::new(),
            free_slots: VecDeque::new(),
            slot_lookup: HashMap::new(),
        }
    }
}

impl Default for InternContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod test {
    use super::InternContext;

    #[test]
    fn intern_existing() {
        let mut context = InternContext::new();
        let slot_idx = context.intern(vec![1, 2, 3]).unwrap();
        assert_eq!(slot_idx, context.intern(vec![1, 2, 3]).unwrap());
        assert!(context.free_slots.is_empty());

        let interned_args = context.lookup(slot_idx);
        assert_eq!(interned_args, &vec![1, 2, 3]);
    }

    #[test]
    fn slot_reused() {
        let mut context = InternContext::new();
        let slot_idx_0 = context.intern(vec![1, 2, 3]).unwrap();
        let slot_idx_1 = context.intern(vec![4, 5, 6]).unwrap();
        assert_ne!(slot_idx_0, slot_idx_1);

        context.drop_use(slot_idx_0);
        assert_eq!(slot_idx_0, context.intern(vec![7, 8, 9]).unwrap());
    }
}

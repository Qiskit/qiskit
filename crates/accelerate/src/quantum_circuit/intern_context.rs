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

macro_rules! println {
    ($($rest:tt)*) => {
        #[cfg(debug_interner)]
        std::println!($($rest)*)
    }
}

#[cfg(debug_interner)]
fn unique_id() -> u64 {
    use std::sync::atomic::AtomicU64;
    use std::sync::atomic::Ordering::SeqCst;
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    COUNTER.fetch_add(1, SeqCst)
}

pub type IndexType = u16;
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
    #[cfg(debug_interner)]
    id: u64,
}

impl InternContext {
    pub fn intern(&mut self, args: Vec<BitType>) -> Option<IndexType> {
        let args = Arc::new(args);
        if !self.slot_lookup.contains_key(&args)
            && self.free_slots.is_empty()
            && self.slots.len() == IndexType::MAX.into()
        {
            return None;
        }

        let slot_idx = self.slot_lookup.entry(args.clone()).or_insert_with(|| {
            if !self.free_slots.is_empty() {
                let slot = self.free_slots.pop_front().unwrap();
                println!("{:?}| Reusing empty slot {slot}", self.id);
                slot
            } else {
                let slot = self.slots.len();
                println!("{:?}| Using new empty slot {slot}", self.id);
                self.slots.push(None);
                slot.try_into().unwrap()
            }
        });
        let shared_args = self
            .slots
            .get_mut(*slot_idx as usize)
            .unwrap()
            .get_or_insert_with(|| {
                println!("{:?}| Initializing slot {slot_idx} for:", self.id);
                println!("{:?}|    {:?}: {args_hash}", self.id, args);
                SharedOperandList {
                    operands: args,
                    use_count: 0,
                }
            });
        shared_args.use_count += 1;
        println!(
            "{:?}| Incrementing uses for slot {slot_idx}. Use count: {:?}",
            self.id, shared_args.use_count
        );
        Some(*slot_idx)
    }

    pub fn lookup(&self, slot_idx: IndexType) -> &Vec<BitType> {
        let slot = self.slots.get(slot_idx as usize).unwrap();
        let operands = &slot.as_ref().unwrap().operands;
        println!("{:?}| Got slot {slot_idx}:", self.id);
        println!("{:?}|    {:?}", self.id, operands);
        operands
    }

    pub fn drop_use(&mut self, slot_idx: IndexType) -> () {
        let mut shared = take(&mut self.slots[slot_idx as usize]).unwrap();
        if let SharedOperandList {
            operands,
            use_count: 1,
        } = shared
        {
            println!("{:?}| Unallocating slot {slot_idx}.", self.id);
            self.slot_lookup.remove(&operands);
            self.free_slots.push_back(slot_idx);
            return;
        };

        shared.use_count -= 1;
        println!(
            "{:?}| Decremented uses for slot {slot_idx}. Use count: {:?}",
            self.id, shared.use_count
        );
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
            #[cfg(debug_interner)]
            id: unique_id(),
        }
    }
}

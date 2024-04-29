// This code is part of Qiskit.
//
// (C) Copyright IBM 2023, 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use hashbrown::HashMap;
use std::hash::Hash;
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub struct CacheSlot(u32);

pub struct CacheFullError;

/// An append-only data structure for caching generic
/// Rust types.
///
/// Takes ownership of inserted values and returns a [CacheSlot]
/// that can be used to get a reference to an equivalent value
/// from this cache later on.
#[derive(Clone, Debug)]
pub struct SlottedCache<T> {
    slots: Vec<Arc<T>>,
    slot_lookup: HashMap<Arc<T>, CacheSlot>,
}

impl<T: Eq + Hash> SlottedCache<T> {
    pub fn new() -> Self {
        SlottedCache {
            slots: Vec::new(),
            slot_lookup: HashMap::new(),
        }
    }

    /// Consumes `value` and returns a [CacheSlot] that can be used
    /// to obtain a reference to an equivalent value by
    /// calling [SlottedCache.get].
    pub fn insert(&mut self, value: T) -> Result<CacheSlot, CacheFullError> {
        if let Some(slot_idx) = self.slot_lookup.get(&value) {
            return Ok(*slot_idx);
        }

        let args = Arc::new(value);
        let slot: CacheSlot = CacheSlot(self.slots.len().try_into().map_err(|_| CacheFullError)?);
        self.slots.push(args.clone());
        self.slot_lookup.insert_unique_unchecked(args, slot);
        Ok(slot)
    }

    /// Returns a reference to the value held at the provided [CacheSlot],
    /// which must be valid in this cache.
    pub fn get(&self, slot: CacheSlot) -> &T {
        self.slots.get(slot.0 as usize).unwrap()
    }
}

impl<T: Eq + Hash> Default for SlottedCache<T> {
    fn default() -> Self {
        Self::new()
    }
}

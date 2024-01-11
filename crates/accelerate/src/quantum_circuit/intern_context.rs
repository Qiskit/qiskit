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
use pyo3::exceptions::PyRuntimeError;
use pyo3::PyResult;
use std::sync::Arc;

pub type IndexType = u32;
pub type BitType = u32;

/// A Rust-only data structure (not a pyclass!) for interning
/// `Vec<BitType>`.
///
/// Takes ownership of vectors given to [InternContext.intern]
/// and returns an [IndexType] index that can be used to look up
/// an _equivalent_ sequence by reference via [InternContext.lookup].
#[derive(Clone, Debug)]
pub struct InternContext {
    slots: Vec<Arc<Vec<BitType>>>,
    slot_lookup: HashMap<Arc<Vec<BitType>>, IndexType>,
}

impl InternContext {
    pub fn new() -> Self {
        InternContext {
            slots: Vec::new(),
            slot_lookup: HashMap::new(),
        }
    }

    /// Takes `args` by reference and returns an index that can be used
    /// to obtain a reference to an equivalent sequence of `BitType` by
    /// calling [CircuitData.lookup].
    pub fn intern(&mut self, args: Vec<BitType>) -> PyResult<IndexType> {
        if let Some(slot_idx) = self.slot_lookup.get(&args) {
            return Ok(*slot_idx);
        }

        let args = Arc::new(args);
        let slot_idx: IndexType = self
            .slots
            .len()
            .try_into()
            .map_err(|_| PyRuntimeError::new_err("InternContext capacity exceeded!"))?;
        self.slots.push(args.clone());
        self.slot_lookup.insert_unique_unchecked(args, slot_idx);
        Ok(slot_idx)
    }

    /// Returns the sequence corresponding to `slot_idx`, which must
    /// be a value returned by [InternContext.intern].
    pub fn lookup(&self, slot_idx: IndexType) -> &[BitType] {
        self.slots.get(slot_idx as usize).unwrap()
    }
}

impl Default for InternContext {
    fn default() -> Self {
        Self::new()
    }
}

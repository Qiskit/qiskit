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

use std::hash::Hash;
use std::sync::Arc;

use hashbrown::HashMap;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Index(u32);

impl IntoPy<PyObject> for Index {
    fn into_py(self, py: Python<'_>) -> PyObject {
        self.0.into_py(py)
    }
}

/// An append-only data structure for interning generic
/// Rust types.
#[derive(Clone, Debug)]
pub struct IndexedInterner<T> {
    entries: Vec<Arc<T>>,
    index_lookup: HashMap<Arc<T>, Index>,
}

pub trait Interner<K> {
    type Output;

    /// Takes ownership of the provided key and returns the interned
    /// type.
    fn intern(self, value: K) -> Self::Output;
}

impl<'a, T> Interner<Index> for &'a IndexedInterner<T> {
    type Output = &'a T;

    fn intern(self, index: Index) -> Self::Output {
        let value = self.entries.get(index.0 as usize).unwrap();
        value.as_ref()
    }
}

impl<'a, T> Interner<T> for &'a mut IndexedInterner<T>
where
    T: Eq + Hash,
{
    type Output = PyResult<Index>;

    fn intern(self, key: T) -> Self::Output {
        if let Some(index) = self.index_lookup.get(&key).copied() {
            Ok(index)
        } else {
            let args = Arc::new(key);
            let index: Index = Index(self.entries.len().try_into().map_err(|_| {
                PyRuntimeError::new_err("The interner has run out of indices (cache is full)!")
            })?);
            self.entries.push(args.clone());
            self.index_lookup.insert_unique_unchecked(args, index);
            Ok(index)
        }
    }
}

impl<T: Eq + Hash> IndexedInterner<T> {
    pub fn new() -> Self {
        IndexedInterner {
            entries: Vec::new(),
            index_lookup: HashMap::new(),
        }
    }
}

impl<T: Eq + Hash> Default for IndexedInterner<T> {
    fn default() -> Self {
        Self::new()
    }
}

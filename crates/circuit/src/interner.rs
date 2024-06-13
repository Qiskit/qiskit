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

#[derive(Clone, Copy, Debug)]
pub struct Index(u32);

pub enum InternerKey<T> {
    Index(Index),
    Value(T),
}

impl<T> From<Index> for InternerKey<T> {
    fn from(value: Index) -> Self {
        InternerKey::Index(value)
    }
}

pub struct InternerValue<'a, T> {
    pub index: Index,
    pub value: &'a T,
}

impl IntoPy<PyObject> for Index {
    fn into_py(self, py: Python<'_>) -> PyObject {
        self.0.into_py(py)
    }
}

pub struct CacheFullError;

impl From<CacheFullError> for PyErr {
    fn from(_: CacheFullError) -> Self {
        PyRuntimeError::new_err("The bit operands cache is full!")
    }
}

/// An append-only data structure for interning generic
/// Rust types.
#[derive(Clone, Debug)]
pub struct IndexedInterner<T> {
    entries: Vec<Arc<T>>,
    index_lookup: HashMap<Arc<T>, Index>,
}

pub trait Interner<T> {
    type Key;
    type Output;

    /// Takes ownership of the provided key and returns the interned
    /// type.
    fn intern(self, value: Self::Key) -> Self::Output;
}

impl<'a, T> Interner<T> for &'a IndexedInterner<T> {
    type Key = Index;
    type Output = InternerValue<'a, T>;

    fn intern(self, index: Index) -> Self::Output {
        let value = self.entries.get(index.0 as usize).unwrap();
        InternerValue {
            index,
            value: value.as_ref(),
        }
    }
}

impl<'a, T> Interner<T> for &'a mut IndexedInterner<T>
where
    T: Eq + Hash,
{
    type Key = InternerKey<T>;
    type Output = Result<InternerValue<'a, T>, CacheFullError>;

    fn intern(self, key: Self::Key) -> Self::Output {
        match key {
            InternerKey::Index(index) => {
                let value = self.entries.get(index.0 as usize).unwrap();
                Ok(InternerValue {
                    index,
                    value: value.as_ref(),
                })
            }
            InternerKey::Value(value) => {
                if let Some(index) = self.index_lookup.get(&value).copied() {
                    Ok(InternerValue {
                        index,
                        value: self.entries.get(index.0 as usize).unwrap(),
                    })
                } else {
                    let args = Arc::new(value);
                    let index: Index =
                        Index(self.entries.len().try_into().map_err(|_| CacheFullError)?);
                    self.entries.push(args.clone());
                    Ok(InternerValue {
                        index,
                        value: self.index_lookup.insert_unique_unchecked(args, index).0,
                    })
                }
            }
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

// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use std::{fmt::Debug, hash::Hash, sync::OnceLock};

use crate::{bit::BitLocations, register::Register};
use indexmap::IndexMap;
use pyo3::{
    prelude::*,
    types::{IntoPyDict, PyDict},
};

pub type BitIndexType<B, R> = IndexMap<B, BitLocations<R>>;

/// Structure that keeps a mapping of bits and their locations within
/// the circuit.
#[derive(Debug)]
pub struct BitLocator<B, R: Register> {
    bit_locations: BitIndexType<B, R>,
    cached: OnceLock<Py<PyDict>>,
}

/// Custom implementation of BitLocator to skip copying the cache.
impl<B, R> Clone for BitLocator<B, R>
where
    B: Debug + Clone + Hash + Eq,
    R: Register + Debug + Clone,
{
    fn clone(&self) -> Self {
        Self {
            bit_locations: self.bit_locations.clone(),
            cached: OnceLock::new(),
        }
    }
}

impl<B, R> Default for BitLocator<B, R>
where
    B: Debug + Clone + Hash + Eq,
    R: Register + Debug + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<B, R> BitLocator<B, R>
where
    B: Debug + Clone + Hash + Eq,
    R: Register + Debug + Clone,
{
    pub fn new() -> Self {
        Self {
            bit_locations: BitIndexType::new(),
            cached: OnceLock::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            bit_locations: BitIndexType::with_capacity(capacity),
            cached: OnceLock::new(),
        }
    }

    pub fn insert(&mut self, bit: B, location: BitLocations<R>) -> Option<BitLocations<R>> {
        self.cached.take();
        self.bit_locations.insert(bit, location)
    }

    pub fn get(&self, bit: &B) -> Option<&BitLocations<R>> {
        self.bit_locations.get(bit)
    }

    pub fn get_mut(&mut self, bit: &B) -> Option<&mut BitLocations<R>> {
        self.cached.take();
        self.bit_locations.get_mut(bit)
    }

    pub fn contains_key(&self, bit: &B) -> bool {
        self.bit_locations.contains_key(bit)
    }
}

impl<B, R> BitLocator<B, R>
where
    B: Debug + Clone + Hash + Eq + for<'py> IntoPyObject<'py> + for<'py> FromPyObject<'py>,
    R: Register + Debug + Clone + for<'py> IntoPyObject<'py> + for<'py> FromPyObject<'py>,
{
    pub fn cached(&self, py: Python) -> &Py<PyDict> {
        self.cached.get_or_init(|| {
            self.bit_locations
                .iter()
                .map(|(bit, loc)| (bit.clone(), loc.clone()))
                .into_py_dict(py)
                .unwrap()
                .into()
        })
    }

    pub fn from_py_dict(dict: &Bound<PyDict>) -> PyResult<Self> {
        let mut locator = Self::new();
        for (key, value) in dict {
            locator.insert(key.extract()?, value.extract()?);
        }
        Ok(locator)
    }
}

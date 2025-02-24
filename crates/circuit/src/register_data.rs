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

use std::fmt::Debug;
use std::sync::OnceLock;

use indexmap::IndexMap;
use pyo3::{
    prelude::*,
    types::{IntoPyDict, PyDict, PyList},
};

use crate::{circuit_data::CircuitError, register::Register};

#[derive(Debug, Clone)]
pub struct RegisterData<R: Register> {
    registers: IndexMap<String, R>,
    cached_registers: OnceLock<Py<PyDict>>,
}

impl<R> Default for RegisterData<R>
where
    R: Debug + Clone + Register,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<R> RegisterData<R>
where
    R: Debug + Clone + Register,
{
    /// Creates an empty instance of [RegisterData]
    pub fn new() -> Self {
        Self {
            registers: IndexMap::new(),
            cached_registers: OnceLock::new(),
        }
    }

    /// Creates an empty instance of [RegisterData] with enough capacity
    /// for the specified registers.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            registers: IndexMap::with_capacity(capacity),
            cached_registers: OnceLock::new(),
        }
    }

    /// Inserts a [Register] into the circuit. If it exists, the index of where
    /// it sits will be returned.
    ///
    /// __**Note** if `strict` is passed, the insertion will fail.
    pub fn add_register(&mut self, register: R, strict: bool) -> PyResult<bool> {
        if self
            .registers
            .insert(register.name().to_string(), register.clone())
            .is_none()
        {
            self.cached_registers.take();
            Ok(true)
        } else if strict {
            return Err(CircuitError::new_err(format!(
                "register name \"{}\" already exists",
                register.name()
            )));
        } else {
            return Ok(false);
        }
    }

    /// Return the number of registers stored
    pub fn len(&self) -> usize {
        self.registers.len()
    }

    /// Checks if a [Register] exists within the circuit
    pub fn contains(&self, register: &R) -> bool {
        self.registers.contains_key(register.name())
    }

    /// Checks if a [Register] exists within the circuit by name
    pub fn contains_key(&self, key: &str) -> bool {
        self.registers.contains_key(key)
    }

    /// Retrieves register by name, returns none if not found.
    pub fn get(&self, register: &str) -> Option<&R> {
        self.registers.get(register)
    }

    /// Removes register by name and returns it, returns `None` if it
    /// doesn't exist.
    pub fn remove(&mut self, register: &str) -> Option<R> {
        self.registers.shift_remove(register).inspect(|reg| {
            self.cached_registers.take();
        })
    }

    /// Returns a slice of all the [Register] instances.
    pub fn registers(&self) -> impl ExactSizeIterator<Item = &R> {
        self.registers.values()
    }

    /// Called during Python garbage collection, only!.
    /// Note: INVALIDATES THIS INSTANCE.
    pub fn dispose(&mut self) {
        self.registers.clear();
        self.cached_registers.take();
    }

    /// Create an instance of [RegisterData] from an existing mapping
    /// of `String` and the specified register type `R`.
    pub fn from_mapping<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = (String, R)>,
    {
        Self {
            registers: iter.into_iter().collect(),
            cached_registers: OnceLock::new(),
        }
    }
}

impl<R> RegisterData<R>
where
    R: Debug + Clone + Register + for<'py> IntoPyObject<'py>,
{
    /// Returns the dictionary mapping the register names and the register instances.
    pub fn cached(&self, py: Python) -> &Py<PyDict> {
        self.cached_registers
            .get_or_init(|| self.registers.clone().into_py_dict(py).unwrap().into())
    }

    /// Return list of registers in the circuit.
    pub fn cached_list<'py>(&'py self, py: Python<'py>) -> Bound<'py, PyList> {
        self.cached_registers
            .get_or_init(|| self.registers.clone().into_py_dict(py).unwrap().into())
            .bind(py)
            .values()
    }

    pub fn cached_raw(&self) -> Option<&Py<PyDict>> {
        self.cached_registers.get()
    }
}

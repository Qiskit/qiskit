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

use hashbrown::HashMap;
use pyo3::{
    prelude::*,
    types::{IntoPyDict, PyDict, PyList},
};

use crate::{circuit_data::CircuitError, register::Register};

#[derive(Debug, Clone)]
pub struct RegisterData<R: Register> {
    registers: Vec<R>,
    registers_indices: HashMap<String, usize>,
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
            registers: Vec::new(),
            registers_indices: HashMap::new(),
            cached_registers: OnceLock::new(),
        }
    }

    /// Creates an empty instance of [RegisterData] with enough capacity
    /// for the specified registers.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            registers: Vec::with_capacity(capacity),
            registers_indices: HashMap::with_capacity(capacity),
            cached_registers: OnceLock::new(),
        }
    }

    /// Inserts a [Register] into the circuit. If it exists, the index of where
    /// it sits will be returned.
    ///
    /// __**Note** if `strict` is passed, the insertion will fail.
    pub fn add_register(&mut self, register: R, strict: bool) -> PyResult<usize> {
        if let Ok(idx) = self
            .registers_indices
            .try_insert(register.name().to_string(), self.registers.len())
        {
            self.registers.push(register);
            self.cached_registers.take();
            Ok(*idx)
        } else if strict {
            return Err(CircuitError::new_err(format!(
                "register name \"{}\" already exists",
                register.name()
            )));
        } else {
            return Ok(self
                .registers_indices
                .get(register.name())
                .copied()
                .unwrap());
        }
    }

    /// Checks if a [Register] exists within the circuit
    pub fn contains(&self, register: &R) -> bool {
        self.registers_indices.contains_key(register.name())
    }

    /// Retrieves register by name, returns none if not found.
    pub fn get(&self, register: &str) -> Option<&R> {
        self.registers_indices
            .get(register)
            .map(|idx| &self.registers[*idx])
    }

    /// Returns a slice of all the [Register] instances.
    pub fn registers(&self) -> &Vec<R> {
        &self.registers
    }

    /// Called during Python garbage collection, only!.
    /// Note: INVALIDATES THIS INSTANCE.
    pub fn dispose(&mut self) {
        self.registers.clear();
        self.registers_indices.clear();
        self.cached_registers.take();
    }
}

impl<R> RegisterData<R>
where
    R: Debug + Clone + Register + for<'py> IntoPyObject<'py>,
{
    /// Returns the dictionary mapping the register names and the register instances.
    pub fn cached(&self, py: Python) -> &Py<PyDict> {
        self.cached_registers.get_or_init(|| {
            self.registers
                .iter()
                .cloned()
                .map(|reg| (reg.name().to_string(), reg))
                .into_py_dict(py)
                .unwrap()
                .into()
        })
    }

    /// Return list of registers in the circuit.
    pub fn cached_list<'py>(&'py self, py: Python<'py>) -> Bound<'py, PyList> {
        self.cached_registers
            .get_or_init(|| {
                self.registers
                    .iter()
                    .cloned()
                    .map(|reg| (reg.name().to_string(), reg))
                    .into_py_dict(py)
                    .unwrap()
                    .into()
            })
            .bind(py)
            .values()
    }

    pub fn cached_raw(&self) -> Option<&Py<PyDict>> {
        self.cached_registers.get()
    }
}

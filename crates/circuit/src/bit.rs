use std::hash::{DefaultHasher, Hash, Hasher};

use pyo3::prelude::*;

use crate::{
    interner::Interned,
    register::{Register, RegisterAsKey},
};

/// Object representing a Python bit, that allows us to keep backwards compatibility
/// with the previous structure.
#[pyclass(name = "Bit")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PyBit {
    register: Option<RegisterAsKey>, // Register identifier
    index: Option<u32>,              // Index within Register
}

#[pymethods]
impl PyBit {
    #[new]
    #[pyo3(signature=(register=None, index=None))]
    pub fn new(register: Option<RegisterAsKey>, index: Option<u32>) -> Self {
        Self {
            register,
            index,
        }
    }

    fn __eq__<'py>(slf: Bound<'py, Self>, other: Bound<'py, Self>) -> bool {
        let borrowed = slf.borrow();
        let other_borrowed = other.borrow();
        if borrowed.register.is_some() && borrowed.index.is_some() {
            return borrowed.register == other_borrowed.register
                && borrowed.index == other_borrowed.index;
        }

        slf.is(&other)
    }

    fn __hash__(slf: Bound<'_, Self>) -> PyResult<isize> {
        let borrowed = slf.borrow();
        let mut hasher = DefaultHasher::new();
        borrowed.hash(&mut hasher);
        Ok(hasher.finish() as isize)
    }
}

/// Keeps information about where a qubit is located within the circuit.
#[derive(Debug, Clone)]
pub struct BitInfo<T: Register + Hash + Eq> {
    register_idx: Interned<T>,
    index: u32,
}

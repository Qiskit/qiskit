use pyo3::prelude::*;

use crate::register::{RegisterAsKey, RegistryIndex};

#[pyclass(name = "Qubit")]
pub struct PyBit {
    reg_ref: RegisterAsKey, // Register identifier
    reg_idx: u32,           // Index within Register
}

#[pymethods]
impl PyBit {
    #[new]
    pub fn new(reg_ref: RegisterAsKey, index: u32) -> Self {
        Self {
            reg_ref,
            reg_idx: index,
        }
    }

    
}

pub struct BitInfo {
    register_idx: RegistryIndex,
    index: u32,
}

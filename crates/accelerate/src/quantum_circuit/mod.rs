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

pub mod circuit_data;
pub mod circuit_instruction;
pub mod intern_context;
mod py_ext;

use pyo3::prelude::*;

#[pymodule]
pub fn quantum_circuit(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<circuit_data::CircuitData>()?;
    m.add_class::<circuit_instruction::CircuitInstruction>()?;
    Ok(())
}

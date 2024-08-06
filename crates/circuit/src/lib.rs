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

pub mod bit_data;
pub mod circuit_data;
pub mod circuit_instruction;
pub mod dag_node;
pub mod gate_matrix;
pub mod imports;
pub mod operations;
pub mod packed_instruction;
pub mod parameter_table;
pub mod slice;
pub mod util;

mod interner;

use pyo3::prelude::*;

pub type BitType = u32;
#[derive(Copy, Clone, Debug, Hash, Ord, PartialOrd, Eq, PartialEq)]
pub struct Qubit(pub BitType);
#[derive(Copy, Clone, Debug, Hash, Ord, PartialOrd, Eq, PartialEq)]
pub struct Clbit(pub BitType);

impl From<BitType> for Qubit {
    fn from(value: BitType) -> Self {
        Qubit(value)
    }
}

impl From<Qubit> for BitType {
    fn from(value: Qubit) -> Self {
        value.0
    }
}

impl From<BitType> for Clbit {
    fn from(value: BitType) -> Self {
        Clbit(value)
    }
}

impl From<Clbit> for BitType {
    fn from(value: Clbit) -> Self {
        value.0
    }
}

#[pymodule]
pub fn circuit(m: Bound<PyModule>) -> PyResult<()> {
    m.add_class::<circuit_data::CircuitData>()?;
    m.add_class::<dag_node::DAGNode>()?;
    m.add_class::<dag_node::DAGInNode>()?;
    m.add_class::<dag_node::DAGOutNode>()?;
    m.add_class::<dag_node::DAGOpNode>()?;
    m.add_class::<circuit_instruction::CircuitInstruction>()?;
    m.add_class::<operations::StandardGate>()?;
    Ok(())
}

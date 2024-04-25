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

mod bit_packing;
pub mod circuit_data;
pub mod circuit_instruction;
pub mod dag_circuit;
pub mod dag_node;
pub mod intern_context;

use pyo3::prelude::*;

#[pymodule]
pub fn quantum_circuit(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<circuit_data::CircuitData>()?;
    m.add_class::<dag_circuit::DAGCircuit>()?;
    m.add_class::<dag_node::DAGNode>()?;
    m.add_class::<dag_node::DAGInNode>()?;
    m.add_class::<dag_node::DAGOutNode>()?;
    m.add_class::<dag_node::DAGOpNode>()?;
    m.add_class::<circuit_instruction::CircuitInstruction>()?;
    Ok(())
}

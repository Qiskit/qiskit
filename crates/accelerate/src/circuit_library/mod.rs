// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use pyo3::prelude::*;
use pyo3::types::PyTuple;

mod entanglement;

#[pyfunction]
#[pyo3(signature = (block_size, num_qubits, entanglement, offset=0))]
pub fn get_entangler_map<'py>(
    py: Python<'py>,
    block_size: u32,
    num_qubits: u32,
    entanglement: &Bound<PyAny>,
    offset: usize,
) -> PyResult<Vec<Bound<'py, PyTuple>>> {
    // The entanglement is Result<impl Iterator<Item = Result<Vec<u32>>>>, so there's two
    // levels of errors we must handle: the outer error is handled by the outer match statement,
    // and the inner (Result<Vec<u32>>) is handled upon the PyTuple creation.
    match entanglement::get_entanglement(num_qubits, block_size, entanglement, offset) {
        Ok(entanglement) => entanglement
            .into_iter()
            .map(|vec| match vec {
                Ok(vec) => Ok(PyTuple::new_bound(py, vec)),
                Err(e) => Err(e),
            })
            .collect::<Result<Vec<_>, _>>(),
        Err(e) => Err(e),
    }
}

#[pymodule]
pub fn circuit_library(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(get_entangler_map))?;
    Ok(())
}

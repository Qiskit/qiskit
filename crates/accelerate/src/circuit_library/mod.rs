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

/// Get the entanglement for given number of qubits and block size.
///
/// Args:
///     num_qubits: The number of qubits to entangle.
///     block_size: The entanglement block size (e.g. 2 for CX or 3 for CCX).
///     entanglement: The entanglement strategy. This can be one of:
///
///         * string: Available options are ``"linear"``, ``"reverse_linear"``, ``"circular"``,
///             ``"pairwise"`` or ``"sca"``.
///         * list of tuples: A list of entanglements given as tuple, e.g. [(0, 1), (1, 2)].
///         * callable: A callable that takes as input an offset as ``int`` (usually the layer
///             in the variational circuit) and returns a string or list of tuples to use as
///             entanglement in this layer.
///
///     offset: An offset used by certain entanglement strategies (e.g. ``"sca"``) or if the
///         entanglement is given as callable. This is typically used to have different
///         entanglement structures in different layers of variational quantum circuits.
///
/// Returns:
///     The entanglement as list of tuples.
///
/// Raises:
///     QiskitError: In case the entanglement is invalid.
#[pyfunction]
#[pyo3(signature = (num_qubits, block_size, entanglement, offset=0))]
pub fn get_entangler_map<'py>(
    py: Python<'py>,
    num_qubits: u32,
    block_size: u32,
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

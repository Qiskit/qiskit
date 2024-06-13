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

mod cnot_synthesis;

use ndarray::ArrayView2;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pyfunction]
#[pyo3(signature = (matrix, section_size))]
fn synth_cnot_count_full_pmh(
    py: Python,
    matrix: PyReadonlyArray2<bool>,
    section_size: i64,
) -> PyResult<PyObject> {
    let view: ArrayView2<bool> = matrix.as_array();
    let result = cnot_synthesis::pmh_synth(view, &(section_size as usize));
    Ok(result.to_object(py))
}

#[pymodule]
pub fn synthesis(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(synth_cnot_count_full_pmh, m)?)?;
    Ok(())
}

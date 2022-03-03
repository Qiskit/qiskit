// This code is part of Qiskit.
//
// (C) Copyright IBM 2022
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::Python;

use hashbrown::HashMap;
use ndarray::{ArrayView1, Axis};
use numpy::{IntoPyArray, PyReadonlyArray2};

#[pyfunction]
#[pyo3(text_signature = "(array, /)")]
pub fn unique(array: PyReadonlyArray2<u16>, py: Python) -> (PyObject, PyObject) {
    let array = array.as_array();
    let shape = array.shape();
    let mut table = HashMap::<ArrayView1<u16>, usize>::with_capacity(shape[0]);
    let mut indexes = Vec::new();
    let mut inverses = vec![0; shape[0]];
    for (i, v) in array.axis_iter(Axis(0)).enumerate() {
        match table.get(&v) {
            Some(id) => inverses[i] = *id,
            None => {
                let new_id = table.len();
                table.insert(v, new_id);
                inverses[i] = new_id;
                indexes.push(i);
            }
        }
    }
    (
        indexes.into_pyarray(py).into(),
        inverses.into_pyarray(py).into(),
    )
}

#[pymodule]
pub fn array_unique(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(unique))?;
    Ok(())
}

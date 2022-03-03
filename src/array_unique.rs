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

use hashbrown::HashMap;
use numpy::PyReadonlyArray2;

#[pyfunction]
#[pyo3(text_signature = "(array)")]
pub fn unique(array: PyReadonlyArray2<u16>) -> PyResult<(Vec<usize>, Vec<usize>)> {
    let array = array.as_array();
    let shape = array.shape();
    let mut table = HashMap::<Vec<u16>, usize>::new();
    let mut indexes = vec![0; 0];
    let mut inverses = vec![0; shape[0]];
    for i in 0..shape[0] {
        let mut v = vec![0u16; shape[1]];
        for j in 0..shape[1] {
            v[j] = array[(i, j)];
        }
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
    Ok((indexes, inverses))
}

#[pymodule]
pub fn array_unique(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(unique))?;
    Ok(())
}

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

use ndarray::ArrayBase;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::Python;

use num_complex::Complex64;
use numpy::ndarray::linalg::kron;
use numpy::ndarray::{array, Array, Dim, OwnedRepr};
use numpy::{PyArray, PyReadonlyArray, ToPyArray};


#[pyfunction]
#[pyo3(text_signature = "(matrix, q_list, /)")]
pub fn block_to_matrix_2q_rust(
    py: Python,
    matrix: PyReadonlyArray<Complex64, Dim<[usize; 2]>>,
    q_list: Vec<usize>,
) -> PyResult<Py<PyArray<Complex64, Dim<[usize; 2]>>>> {
    let matrix_arr = matrix.as_array().to_owned();
    let swap_gate = array![
        [
            Complex64::new(1., 0.),
            Complex64::new(0., 0.),
            Complex64::new(0., 0.),
            Complex64::new(0., 0.)
        ],
        [
            Complex64::new(0., 0.),
            Complex64::new(0., 0.),
            Complex64::new(1., 0.),
            Complex64::new(0., 0.)
        ],
        [
            Complex64::new(0., 0.),
            Complex64::new(1., 0.),
            Complex64::new(0., 0.),
            Complex64::new(0., 0.)
        ],
        [
            Complex64::new(0., 0.),
            Complex64::new(0., 0.),
            Complex64::new(0., 0.),
            Complex64::new(1., 0.)
        ]
    ];
    let identity: ArrayBase<OwnedRepr<Complex64>, Dim<[usize; 2]>> = Array::eye(2);

    let mut basis_change = false;
    let current: Array<Complex64, Dim<[usize; 2]>>;
    if q_list.len() < 2 {
        if q_list[0] == 1 {
            current = kron(&matrix_arr, &identity);
        } else {
            current = kron(&identity, &matrix_arr);
        }
    } else {
        if q_list[0] > q_list[1] && matrix_arr != swap_gate {
            basis_change = true;
        }
        current = matrix_arr;
    }

    if basis_change {
        Ok((swap_gate.dot(&current)).dot(&swap_gate).to_pyarray(py).to_owned())
    } else {
        Ok(current.to_pyarray(py).to_owned())
    }
}

#[pymodule]
pub fn convert_2q_block_matrix(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(block_to_matrix_2q_rust))?;
    Ok(())
}

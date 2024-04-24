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

use num_complex::Complex64;
use numpy::ndarray::{aview2, Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use smallvec::SmallVec;

use faer::modules::core::mul::matmul;
use faer::perm::swap_rows;
use faer::prelude::*;
use faer::{Mat, Parallelism};
use faer_ext::{IntoFaerComplex, IntoNdarrayComplex};

static ONE_QUBIT_IDENTITY: [[Complex64; 2]; 2] = [
    [Complex64::new(1., 0.), Complex64::new(0., 0.)],
    [Complex64::new(0., 0.), Complex64::new(1., 0.)],
];

/// Return the matrix Operator resulting from a block of Instructions.
#[pyfunction]
#[pyo3(text_signature = "(op_list, /")]
pub fn blocks_to_matrix(
    py: Python,
    op_list: Vec<(PyReadonlyArray2<Complex64>, SmallVec<[u8; 2]>)>,
) -> PyResult<Py<PyArray2<Complex64>>> {
    let identity = aview2(&ONE_QUBIT_IDENTITY).into_faer_complex();
    let input_matrix = op_list[0].0.as_array().into_faer_complex();

    let mut matrix = match op_list[0].1.as_slice() {
        [0] => identity.kron(input_matrix.as_ref()),
        [1] => input_matrix.kron(identity.as_ref()),
        [0, 1] => input_matrix.to_owned(),
        [1, 0] => change_basis_faer(input_matrix),
        [] => Mat::<c64>::identity(4, 4),
        _ => unreachable!(),
    };
    for (op_matrix, q_list) in op_list.into_iter().skip(1) {
        let op_matrix = op_matrix.as_array().into_faer_complex();

        let result = match q_list.as_slice() {
            [0] => Some(identity.kron(op_matrix.as_ref())),
            [1] => Some(op_matrix.kron(identity.as_ref())),
            [1, 0] => Some(change_basis_faer(op_matrix)),
            [] => Some(Mat::<c64>::identity(4, 4)),
            _ => None,
        };

        let aux = matrix.clone();
        match result {
            Some(x) => {
                matmul(
                    matrix.as_mut(),
                    x.as_ref(),
                    aux.as_ref(),
                    None,
                    c64::new(1., 0.),
                    Parallelism::None,
                );
            },
            None => {
                matmul(
                    matrix.as_mut(),
                    op_matrix,
                    aux.as_ref(),
                    None,
                    c64::new(1., 0.),
                    Parallelism::None,
                );
            }
        };
    }

    Ok(matrix
        .as_ref()
        .into_ndarray_complex()
        .to_owned()
        .into_pyarray_bound(py)
        .unbind())
}

/// Switches the order of qubits in a two qubit operation.
/// This function will substitue `change_basis` once the
/// `two_qubit_decompose.rs` uses Mat<c64> instead of ArrayView2
#[inline]
pub fn change_basis_faer(matrix: MatRef<c64>) -> Mat<c64> {
    let mut trans_matrix: Mat<c64> = matrix.transpose().to_owned();
    let (row1, row2) = trans_matrix.as_mut().two_rows_mut(1, 2);
    swap_rows(row1, row2);

    trans_matrix = trans_matrix.transpose().to_owned();
    let (row1, row2) = trans_matrix.as_mut().two_rows_mut(1, 2);
    swap_rows(row1, row2);

    trans_matrix
}

/// Switches the order of qubits in a two qubit operation.
#[inline]
pub fn change_basis(matrix: ArrayView2<Complex64>) -> Array2<Complex64> {
    let mut trans_matrix: Array2<Complex64> = matrix.reversed_axes().to_owned();
    for index in 0..trans_matrix.ncols() {
        trans_matrix.swap([1, index], [2, index]);
    }
    trans_matrix = trans_matrix.reversed_axes();
    for index in 0..trans_matrix.ncols() {
        trans_matrix.swap([1, index], [2, index]);
    }
    trans_matrix
}

#[pymodule]
pub fn convert_2q_block_matrix(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(blocks_to_matrix))?;
    Ok(())
}

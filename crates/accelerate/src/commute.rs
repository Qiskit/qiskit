use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::Python;

use approx::AbsDiffEq;
use num_complex::Complex64;
use numpy::ndarray::linalg::kron;
use numpy::ndarray::{Array, Array2};
use numpy::PyReadonlyArray2;

/// Return the matrix Operator resulting from a block of Instructions.
#[pyfunction]
#[pyo3(text_signature = "(op_1, qargs_1, op_2, qargs_2, /)")]
pub fn commute_check(
    op_1: PyReadonlyArray2<Complex64>,
    qargs_1: Vec<usize>,
    op_2: PyReadonlyArray2<Complex64>,
    qargs_2: Vec<usize>,
) -> PyResult<bool> {
    let mut op_1: Array2<Complex64> = op_1.as_array().to_owned();
    let mut op_2: Array2<Complex64> = op_2.as_array().to_owned();
    let lenq1 = qargs_1.len();
    let lenq2 = qargs_2.len();
    if lenq1 != lenq2 {
        let diff: u32 = lenq1.abs_diff(lenq2) as u32;
        let id_op: Array2<Complex64> = Array::eye(2_usize.pow(diff));
        if lenq2 < lenq1 {
            op_2 = kron(&op_2, &id_op);
        } else {
            op_1 = kron(&id_op, &op_1);
        }
    }
    let op12 = op_1.dot(&op_2);
    let op21 = op_2.dot(&op_1);
    Ok(allclose(&op12, &op21, 1e-8))
}

fn allclose(op_1: &Array2<Complex64>, op_2: &Array2<Complex64>, epsilon: f64) -> bool {
    let mut result = true;
    for row in 0..op_1.nrows() {
        for col in 0..op_1.ncols() {
            let reals = op_1[[row, col]]
                .re
                .abs_diff_eq(&op_2[[row, col]].re, epsilon);
            let imags = op_1[[row, col]]
                .im
                .abs_diff_eq(&op_2[[row, col]].im, epsilon);
            result &= reals && imags;
        }
    }
    result
}

#[pymodule]
pub fn commute(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(commute_check))?;
    Ok(())
}

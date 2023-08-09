use numpy::{PyArray2, ToPyArray};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::Python;

use num_complex::Complex64;
use numpy::ndarray::linalg::kron;
use numpy::ndarray::{Array, Array2};
use numpy::PyReadonlyArray2;

/// Return the matrix Operator resulting from a block of Instructions.
#[pyfunction]
#[pyo3(text_signature = "(op_1, qargs_1, op_2, qargs_2, num_qubits /)")]
pub fn commute_check(
    py: Python,
    op_1: PyReadonlyArray2<Complex64>,
    qargs_1: Vec<usize>,
    op_2: PyReadonlyArray2<Complex64>,
    qargs_2: Vec<usize>,
    num_qubits: usize,
) -> PyResult<(Py<PyArray2<Complex64>>, Py<PyArray2<Complex64>>)> {
    let mut op_1: Array2<Complex64> = op_1.as_array().into_owned();
    let op_2: Array2<Complex64> = op_2.as_array().into_owned();
    let lenq1 = qargs_1.len();
    let lenq2 = qargs_2.len();
    if lenq1 != lenq2 {
        let extra_qarg2: u32 = (num_qubits - lenq1) as u32;
        if extra_qarg2 > 0 {
            let id_op: Array2<Complex64> = Array::eye(2_usize.pow(extra_qarg2));
            op_1 = kron(&id_op, &op_1);
        }
    }
    let op12 = op_1.dot(&op_2);
    let op21 = op_2.dot(&op_1);
    Ok((
        op12.to_pyarray(py).to_owned(),
        op21.to_pyarray(py).to_owned(),
    ))
}

#[pymodule]
pub fn commute(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(commute_check))?;
    Ok(())
}

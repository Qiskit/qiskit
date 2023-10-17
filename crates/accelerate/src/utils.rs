use pyo3::prelude::*;

use faer::Faer;
use faer::IntoFaerComplex;
use num_complex::Complex;
use numpy::PyReadonlyArray2;

/// Return indices that sort data.
/// If `data` contains two elements that are incomparable,
/// an error will be thrown.
pub fn argsort<T: PartialOrd>(data: &[T]) -> Vec<usize> {
    let mut indices = (0..data.len()).collect::<Vec<_>>();
    indices.sort_by(|&a, &b| data[a].partial_cmp(&data[b]).unwrap());
    indices
}

// TODO: Use traits and parameters
/// Modulo operation. This should give the same result
/// as `numpy.mod`.
pub fn modulo(a: f64, b: f64) -> f64 {
    ((a % b) + b) % b
}

/// Return the eigenvalues of `unitary` as a Python `list`.
#[pyfunction]
#[pyo3(text_signature = "(unitary, /")]
pub fn eigenvalues(unitary: PyReadonlyArray2<Complex<f64>>) -> Vec<Complex<f64>> {
    unitary
        .as_array()
        .into_faer_complex()
        .complex_eigenvalues()
        .into_iter()
        .map(|x| Complex::<f64>::new(x.re, x.im))
        .collect()
}

#[pymodule]
pub fn utils(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(eigenvalues))?;
    Ok(())
}

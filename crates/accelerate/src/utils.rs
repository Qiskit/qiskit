use pyo3::prelude::*;
use pyo3::types::PySlice;

use faer::prelude::*;
use faer::IntoFaerComplex;
use num_complex::Complex;
use numpy::{IntoPyArray, PyReadonlyArray2};

/// A private enumeration type used to extract arguments to pymethod
/// that may be either an index or a slice
#[derive(FromPyObject)]
pub enum SliceOrInt<'a> {
    // The order here defines the order the variants are tried in the FromPyObject` derivation.
    // `Int` is _much_ more common, so that should be first.
    Int(isize),
    Slice(&'a PySlice),
}

/// Return indices that sort partially ordered data.
/// If `data` contains two elements that are incomparable,
/// an error will be thrown.
pub fn arg_sort<T: PartialOrd>(data: &[T]) -> Vec<usize> {
    let mut indices = (0..data.len()).collect::<Vec<_>>();
    indices.sort_by(|&a, &b| data[a].partial_cmp(&data[b]).unwrap());
    indices
}

/// Return the eigenvalues of `unitary` as a one-dimensional `numpy.ndarray`
/// with `dtype(complex128)`.
#[pyfunction]
#[pyo3(text_signature = "(unitary, /")]
pub fn eigenvalues(py: Python, unitary: PyReadonlyArray2<Complex<f64>>) -> PyObject {
    unitary
        .as_array()
        .into_faer_complex()
        .complex_eigenvalues()
        .into_iter()
        .map(|x| Complex::<f64>::new(x.re, x.im))
        .collect::<Vec<_>>()
        .into_pyarray(py)
        .into()
}

#[pymodule]
pub fn utils(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(eigenvalues))?;
    Ok(())
}

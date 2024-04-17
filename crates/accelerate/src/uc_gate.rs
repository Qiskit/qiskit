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

use num_complex::{Complex64, ComplexFloat};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::Python;
use std::f64::consts::PI;

use ndarray::prelude::*;
use numpy::{IntoPyArray, PyReadonlyArray2};
use faer_ext::{IntoFaerComplex, IntoNdarrayComplex};

use crate::euler_one_qubit_decomposer::det_one_qubit;

const PI2: f64 = PI / 2.;
const EPS: f64 = 1e-10;

/// This method implements the decomposition given in equation (3) in
/// https://arxiv.org/pdf/quant-ph/0410066.pdf.
///
/// The decomposition is used recursively to decompose uniformly controlled gates.
///
/// a,b = single qubit unitaries
/// v,u,r = outcome of the decomposition given in the reference mentioned above
///
/// (see there for the details).
#[pyfunction]
pub fn demultiplex_single_uc(
    py: Python,
    a: PyReadonlyArray2<Complex64>,
    b: PyReadonlyArray2<Complex64>,
) -> (PyObject, PyObject, PyObject) {
    let a = a.as_array();
    let b = b.as_array();
    let x = a.dot(&b.mapv(|x| x.conj()).t());
    let det_x = det_one_qubit(x.view());
    let x11 = x[[0, 0]] / det_x.sqrt();
    let phi = det_x.arg();

    let r1 = (Complex64::new(0., 1.) / 2. * (PI2 - phi / 2. - x11.arg())).exp();
    let r2 = (Complex64::new(0., 1.) / 2. * (PI2 - phi / 2. + x11.arg() + PI)).exp();

    let r = array![
        [r1, Complex64::new(0., 0.)],
        [Complex64::new(0., 0.), r2],
    ];

    let decomp = r.dot(&x).dot(&r).view().into_faer_complex().complex_eigendecomposition();
    let mut u: Array2<Complex64> = decomp.u().into_ndarray_complex().to_owned();
    let s = decomp.s().column_vector();
    let mut diag: Array1<Complex64> = Array1::from_shape_fn(u.shape()[0], |i| {
        s[i].to_num_complex()
    });

    // If d is not equal to diag(i,-i), then we put it into this "standard" form
    // (see eq. (13) in https://arxiv.org/pdf/quant-ph/0410066.pdf) by interchanging
    // the eigenvalues and eigenvectors
    if (diag[0] + Complex64::new(0., 1.)).abs() < EPS {
        diag = diag.slice(s![..;-1]).to_owned();
        u = u.slice(s![.., ..;-1]).to_owned();
//        d = np.flip(d, 0);
//        u = np.flip(u, 1);
    }
    diag.mapv_inplace(|x| x.sqrt());
    let d = Array2::from_diag(&diag);
    let v = d.dot(&u.mapv(|x| x.conj()).t()).dot(&r.mapv(|x| x.conj()).t()).dot(&b);
    (v.into_pyarray_bound(py).into(), u.into_pyarray_bound(py).into(), r.into_pyarray_bound(py).into())
}


#[pymodule]
pub fn uc_gate(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(demultiplex_single_uc))?;
    Ok(())
}

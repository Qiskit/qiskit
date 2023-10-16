// In numpy matrices real and imaginary components are adjacent:
//   np.array([1,2,3], dtype='complex').view('float64')
//   array([1., 0., 2., 0., 3., 0.])
// The matrix faer::Mat<c64> has this layout.
// faer::Mat<num_complex::Complex<f64>> instead stores a matrix
// of real components and one of imaginary components.
// In order to avoid copying we want to use `MatRef<c64>` or `MatMut<c64>`.

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::Python;
use num_complex::{Complex};
use std::f64::consts::PI;

//use numpy::{ToPyArray, PyArray};
use numpy::PyReadonlyArray2;
use faer::{Mat, MatRef, mat, Scale};
use faer::Faer;
use faer_core::c64;
//use faer_core::ComplexField; // for access to fields of struct
use faer::IntoFaerComplex;

use crate::utils;

// FIXME: zero and one exist but I cant find the right incantation
const C0 : c64 = c64 {re: 0.0, im: 0.0};
const C1 : c64 = c64 {re: 1.0, im: 0.0};
const C1IM : c64 = c64 { re: 0.0, im: 1.0};

// FIXME: Find a way to compute these matrices at compile time.
fn transform_from_magic_basis(unitary : Mat<c64>) -> Mat<c64> {
    let _b_nonnormalized : Mat<c64> = mat![[C1, C1IM, C0, C0],
                                           [C0, C0, C1IM, C1],
                                           [C0, C0, C1IM, -C1],
                                           [C1, -C1IM, C0, C0]];
    let _b_nonnormalized_dagger = Scale(c64 {re: 0.5, im: 0.0}) * _b_nonnormalized.adjoint();
    _b_nonnormalized_dagger * unitary * _b_nonnormalized
}

// fn transform_to_magic_basis(unitary : Mat<c64>) -> Mat<c64> {
//     let _b_nonnormalized : Mat<c64> = mat![[C1, C1IM, C0, C0],
//                                            [C0, C0, C1IM, C1],
//                                            [C0, C0, C1IM, -C1],
//                                            [C1, -C1IM, C0, C0]];
//     let _b_nonnormalized_dagger = Scale(c64 {re: 0.5, im: 0.0}) * _b_nonnormalized.adjoint();
//     _b_nonnormalized * unitary * _b_nonnormalized_dagger
// }

// faer::c64 and num_complex::Complex<f64> are both structs
// holding two f64's. But several functions are not defined for
// c64. So we implement them here. These things should be contribute
// upstream.

pub trait PowF {
    fn powf(self, pow : f64) -> c64;
}

impl PowF for c64 {
    fn powf(self, pow : f64) -> c64 {
        c64::from(Complex::<f64>::from(self).powf(pow))
    }
}

pub trait Arg {
    fn arg(self) -> f64;
}

impl Arg for c64 {
    fn arg(self) -> f64 {
        let c = Complex::<f64>::from(self);
        c.arg()
    }
}

// FIXME: full of ineffciencies
fn __weyl_coordinates(unitary : MatRef<c64>) -> (f64, f64, f64) {
    let pi = PI;
    let pi2 = PI / 2.0;
    let pi4 = PI / 4.0;
    let uscaled = Scale(C1 / unitary.determinant().powf(0.25)) * unitary;
    let uup = transform_from_magic_basis(uscaled);
    let uad = uup.transpose();
    let d : Vec<c64> = (uad * &uup).eigenvalues();
    let mut darg : Vec<_> = d.iter().map(|x| - x.arg() / 2.0).collect();
    darg[3] = -darg[0] - darg[1] - darg[2];
    let mut cs : Vec<_> = (0..3).map(|i| utils::modulo((darg[i] + darg[3]) / 2.0, 2.0 * pi)).collect();
    let cstemp : Vec<f64> = cs.iter()
        .map(|x| utils::modulo(*x, pi2))
        .map(|x| x.min(pi2 - x))
        .collect();
    let mut order = utils::argsort(&cstemp);
    (order[0], order[1], order[2]) = (order[1], order[2], order[0]);
    (cs[0], cs[1], cs[2]) = (cs[order[0]], cs[order[1]], cs[order[2]]);

    // Flip into Weyl chamber
    let pi32 = 3.0 * pi2;
    if cs[0] > pi2 {
        cs[0] -= pi32;
    }
    if cs[1] > pi2 {
        cs[1] -= pi32;
    }
    let mut conjs = 0;
    if cs[0] > pi4 {
        cs[0] = pi2 - cs[0];
        conjs += 1;
    }
    if cs[1] > pi4 {
        cs[1] = pi2 - cs[1];
        conjs += 1;
    }
    if cs[2] > pi2 {
        cs[2] -= pi32;
    }
    if conjs == 1 {
        cs[2] = pi2 - cs[2];
    }
    if cs[2] > pi4 {
        cs[2] -= pi2;
    }
    (cs[1], cs[0], cs[2])
}


// For debugging. We can remove this later
#[pyfunction]
#[pyo3(text_signature = "(unitary, /")]
pub fn _weyl_coordinates(unitary : PyReadonlyArray2<Complex<f64>>) -> (f64, f64, f64) {
    let u = unitary.as_array().into_faer_complex();
    __weyl_coordinates(u)
}

#[pyfunction]
#[pyo3(text_signature = "(basis_b, basis_fidelity, unitary, /")]
pub fn _num_basis_gates(basis_b : f64, basis_fidelity : f64, unitary : PyReadonlyArray2<Complex<f64>>) -> usize {
    let u = unitary.as_array().into_faer_complex();
    __num_basis_gates(basis_b, basis_fidelity, u)
}

fn __num_basis_gates(basis_b : f64, basis_fidelity : f64, unitary : MatRef<c64>) -> usize {
    let (a, b, c) = __weyl_coordinates(unitary);
    let pi4 = PI / 4.0;
    let traces = [
        c64::new(4.0 * (a.cos() * b.cos() * c.cos()), 4.0 * (a.sin() * b.sin() * c.sin())),
        c64::new(4.0  * (pi4 - a).cos() * (basis_b - b).cos() * c.cos(),
                 4.0 * (pi4 - a).sin() * (basis_b - b).sin() * c.sin()),
        c64::new(4.0 * c.cos(), 0.0),
        c64::new(4.0, 0.0)
    ];
    let mut imax : usize = 0;
    let mut max_fid = 0.0;
//    for i in 0..4 {
    for (i, trace) in traces.iter().enumerate() {
        let fid = trace_to_fid(*trace) * basis_fidelity.powi(i as i32);
        if fid > max_fid {
            max_fid = fid;
            imax = i
        }
    }
    imax
}

// The interface for abs2 is in flux. So use this for now.
fn myabs2(z: c64) -> f64 {
    z.re * z.re + z.im * z.im
}

/// Average gate fidelity is :math:`Fbar = (d + |Tr (Utarget \\cdot U^dag)|^2) / d(d+1)`
/// M. Horodecki, P. Horodecki and R. Horodecki, PRA 60, 1888 (1999)
fn trace_to_fid(trace : c64) -> f64 {
    (4.0 + myabs2(trace)) / 20.0
}

#[pymodule]
pub fn two_qubit_decompose(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(_num_basis_gates))?;
    m.add_wrapped(wrap_pyfunction!(_weyl_coordinates))?;
    Ok(())
}

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::Python;
use num_complex::Complex;

use numpy::PyReadonlyArray2;
//use faer::complex_native::c64;
use faer::{Mat, mat, Scale};
use faer::Faer;
use faer_core::c64;

// FIXME: zero and one exist but I cant find the right incantation
const c0 : c64 = c64 {re: 0.0, im: 0.0};
const c1 : c64 = c64 {re: 1.0, im: 0.0};
const c1im : c64 = c64 { re: 0.0, im: 1.0};

fn transform_to_magic_basis(U : Mat<c64>) -> Mat<c64> {
    let _B_nonnormalized : Mat<c64> = mat![[c1, c1im, c0, c0],
                                           [c0, c0, c1im, c1],
                                           [c0, c0, c1im, -c1],
                                           [c1, -c1im, c0, c0]];
    let _B_nonnormalized_dagger = Scale(c64 {re: 0.5, im: 0.0}) * _B_nonnormalized.conjugate();
    return _B_nonnormalized * U * _B_nonnormalized_dagger;
}

fn transform_from_magic_basis(U : Mat<c64>) -> Mat<c64> {
    let _B_nonnormalized : Mat<c64> = mat![[c1, c1im, c0, c0],
                                           [c0, c0, c1im, c1],
                                           [c0, c0, c1im, -c1],
                                           [c1, -c1im, c0, c0]];
    let _B_nonnormalized_dagger = Scale(c64 {re: 0.5, im: 0.0}) * _B_nonnormalized.conjugate();
    return _B_nonnormalized_dagger * U * _B_nonnormalized;
}

// FIXME: make some kind of trait
fn powf(base: c64, pow : f64) -> c64 {
    return c64::from(Complex::<f64>::from(base).powf(pow));
}

pub trait Arg {
    fn arg(self) -> f64;
}

impl Arg for c64 {
    fn arg(self) -> f64 {
        let c = Complex::<f64>::from(self);
        return c.arg();
    }
}

// FIXME: full of ineffciencies
fn weyl_coordinates(umat : Mat<c64>) {
    let pi = std::f64::consts::PI;
    let pi2 = std::f64::consts::PI / 2.0;
    let pi4 = std::f64::consts::PI / 4.0;
    let uscaled = Scale(c1 / powf(umat.determinant(), 0.25)) * umat;
    let a = c0.arg();
    let uup = transform_from_magic_basis(uscaled);
    let uad = uup.adjoint();
    let d : Vec<c64> = (&uad * &uup).eigenvalues();
    let mut darg : Vec<_> = d.iter().map(|x| - x.arg() / 2.0).collect();
    darg[3] = -darg[0] - darg[1] - darg[2];
    let mut cs = vec![0.0; 3];
    for i in 0..2 {
        cs[i] = (darg[i] + darg[3]) / 2.0 % (2.0 * pi);
    }
    let mut cstemp : Vec<f64> = cs.iter().map(|x| x % pi2).collect();
    for i in 0..3 {
        cstemp[i] = cstemp[i].min(cstemp[i] - pi2);
    }
}

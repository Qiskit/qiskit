// This code is part of Qiskit.
//
// (C) Copyright IBM 2023
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

// In numpy matrices real and imaginary components are adjacent:
//   np.array([1,2,3], dtype='complex').view('float64')
//   array([1., 0., 2., 0., 3., 0.])
// The matrix faer::Mat<c64> has this layout.
// faer::Mat<num_complex::Complex<f64>> instead stores a matrix
// of real components and one of imaginary components.
// In order to avoid copying we want to use `MatRef<c64>` or `MatMut<c64>`.

use approx::abs_diff_eq;
use num_complex::{Complex, Complex64, ComplexFloat};
use pyo3::exceptions::PyIndexError;
use pyo3::import_exception;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::Python;
use smallvec::{smallvec, SmallVec};
use std::f64::consts::PI;

use faer::IntoFaerComplex;
use faer::IntoNdarray;
use faer::IntoNdarrayComplex;
use faer::Side::Lower;
use faer::{prelude::*, scale, Mat, MatRef};
use faer_core::{c64, ComplexField};
use ndarray::linalg::kron;
use ndarray::prelude::*;
use ndarray::Zip;
use numpy::PyReadonlyArray2;
use numpy::ToPyArray;

use crate::euler_one_qubit_decomposer::{
    angles_from_unitary, det_one_qubit, unitary_to_gate_sequence_inner, EulerBasis,
    ANGLE_ZERO_EPSILON,
};
use crate::utils;

use rand::prelude::*;
use rand_distr::StandardNormal;
use rand_pcg::Pcg64Mcg;

const PI2: f64 = PI / 2.0;
const PI4: f64 = PI / 4.0;
const PI32: f64 = 3.0 * PI2;
const TWO_PI: f64 = 2.0 * PI;

const C1: c64 = c64 { re: 1.0, im: 0.0 };

const ONE_QUBIT_IDENTITY: [[Complex64; 2]; 2] = [
    [Complex64::new(1., 0.), Complex64::new(0., 0.)],
    [Complex64::new(0., 0.), Complex64::new(1., 0.)],
];

const B_NON_NORMALIZED: [[Complex64; 4]; 4] = [
    [
        Complex64::new(1.0, 0.),
        Complex64::new(0., 1.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
    ],
    [
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 1.),
        Complex64::new(1.0, 0.0),
    ],
    [
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 1.),
        Complex64::new(-1., 0.),
    ],
    [
        Complex64::new(1., 0.),
        Complex64::new(0., -1.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
    ],
];

const B_NON_NORMALIZED_DAGGER: [[Complex64; 4]; 4] = [
    [
        Complex64::new(0.5, 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0.5, 0.0),
    ],
    [
        Complex64::new(0., -0.5),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.5),
    ],
    [
        Complex64::new(0., 0.),
        Complex64::new(0., -0.5),
        Complex64::new(0., -0.5),
        Complex64::new(0., 0.),
    ],
    [
        Complex64::new(0., 0.),
        Complex64::new(0.5, 0.),
        Complex64::new(-0.5, 0.),
        Complex64::new(0., 0.),
    ],
];

enum MagicBasisTransform {
    Into,
    OutOf,
}

fn magic_basis_transform(
    unitary: ArrayView2<Complex64>,
    direction: MagicBasisTransform,
) -> Array2<Complex64> {
    let _b_nonnormalized = aview2(&B_NON_NORMALIZED);
    let _b_nonnormalized_dagger = aview2(&B_NON_NORMALIZED_DAGGER);
    match direction {
        MagicBasisTransform::OutOf => _b_nonnormalized_dagger.dot(&unitary).dot(&_b_nonnormalized),
        MagicBasisTransform::Into => _b_nonnormalized.dot(&unitary).dot(&_b_nonnormalized_dagger),
    }
}

fn transform_from_magic_basis(u: Mat<c64>) -> Mat<c64> {
    let unitary: ArrayView2<Complex64> = u.as_ref().into_ndarray_complex();
    magic_basis_transform(unitary, MagicBasisTransform::OutOf)
        .view()
        .into_faer_complex()
        .to_owned()
}

// faer::c64 and num_complex::Complex<f64> are both structs
// holding two f64's. But several functions are not defined for
// c64. So we implement them here. These things should be contribute
// upstream.

pub trait PowF {
    fn powf(self, pow: f64) -> c64;
}

impl PowF for c64 {
    fn powf(self, pow: f64) -> c64 {
        c64::from(self.to_num_complex().powf(pow))
    }
}

pub trait Arg {
    fn arg(self) -> f64;
}

impl Arg for c64 {
    fn arg(self) -> f64 {
        self.to_num_complex().arg()
    }
}

pub trait TraceToFidelity {
    /// Average gate fidelity is :math:`Fbar = (d + |Tr (Utarget \\cdot U^dag)|^2) / d(d+1)`
    /// M. Horodecki, P. Horodecki and R. Horodecki, PRA 60, 1888 (1999)
    fn trace_to_fid(self) -> f64;
}

impl TraceToFidelity for Complex64 {
    fn trace_to_fid(self) -> f64 {
        (4.0 + self.abs().powi(2)) / 20.0
    }
}

impl TraceToFidelity for c64 {
    fn trace_to_fid(self) -> f64 {
        (4.0 + self.faer_abs2()) / 20.0
    }
}

fn decompose_two_qubit_product_gate(
    special_unitary: ArrayView2<Complex64>,
) -> PyResult<(Array2<Complex64>, Array2<Complex64>, f64)> {
    let mut r: Array2<Complex64> = special_unitary.slice(s![..2, ..2]).to_owned();
    let mut det_r = det_one_qubit(r.view());
    if det_r.abs() < 0.1 {
        r = special_unitary.slice(s![2.., ..2]).to_owned();
        det_r = det_one_qubit(r.view());
    }
    if det_r.abs() < 0.1 {
        import_exception!(qiskit, QiskitError);
        return Err(QiskitError::new_err(
            "decompose_two_qubit_product_gate: unable to decompose: detR < 0.1",
        ));
    }
    r.mapv_inplace(|x| x / det_r.sqrt());
    let r_t_conj: Array2<Complex64> = r.t().mapv(|x| x.conj());
    let eye = aview2(&ONE_QUBIT_IDENTITY);
    let mut temp = kron(&eye, &r_t_conj);
    temp = special_unitary.dot(&temp);
    let mut l = temp.slice(s![..;2, ..;2]).to_owned();
    let det_l = det_one_qubit(l.view());
    if det_l.abs() < 0.9 {
        import_exception!(qiskit, QiskitError);
        return Err(QiskitError::new_err(
            "decompose_two_qubit_product_gate: unable to decompose: detL < 0.9",
        ));
    }
    l.mapv_inplace(|x| x / det_l.sqrt());
    let phase = det_l.arg() / 2.;
    Ok((l, r, phase))
}

fn __weyl_coordinates(unitary: MatRef<c64>) -> [f64; 3] {
    let uscaled = scale(C1 / unitary.determinant().powf(0.25)) * unitary;
    let uup = transform_from_magic_basis(uscaled);
    let mut darg: Vec<_> = (uup.transpose() * &uup)
        .complex_eigenvalues()
        .into_iter()
        .map(|x: c64| -x.arg() / 2.0)
        .collect();
    darg[3] = -darg[0] - darg[1] - darg[2];
    let mut cs: Vec<_> = (0..3)
        .map(|i| ((darg[i] + darg[3]) / 2.0).rem_euclid(2.0 * PI))
        .collect();
    let cstemp: Vec<f64> = cs
        .iter()
        .map(|x| x.rem_euclid(PI2))
        .map(|x| x.min(PI2 - x))
        .collect();
    let mut order = utils::arg_sort(&cstemp);
    (order[0], order[1], order[2]) = (order[1], order[2], order[0]);
    (cs[0], cs[1], cs[2]) = (cs[order[0]], cs[order[1]], cs[order[2]]);

    // Flip into Weyl chamber
    if cs[0] > PI2 {
        cs[0] -= PI32;
    }
    if cs[1] > PI2 {
        cs[1] -= PI32;
    }
    let mut conjs = 0;
    if cs[0] > PI4 {
        cs[0] = PI2 - cs[0];
        conjs += 1;
    }
    if cs[1] > PI4 {
        cs[1] = PI2 - cs[1];
        conjs += 1;
    }
    if cs[2] > PI2 {
        cs[2] -= PI32;
    }
    if conjs == 1 {
        cs[2] = PI2 - cs[2];
    }
    if cs[2] > PI4 {
        cs[2] -= PI2;
    }
    [cs[1], cs[0], cs[2]]
}

#[pyfunction]
#[pyo3(text_signature = "(basis_b, basis_fidelity, unitary, /")]
pub fn _num_basis_gates(
    basis_b: f64,
    basis_fidelity: f64,
    unitary: PyReadonlyArray2<Complex<f64>>,
) -> usize {
    let u = unitary.as_array().into_faer_complex();
    __num_basis_gates(basis_b, basis_fidelity, u)
}

fn __num_basis_gates(basis_b: f64, basis_fidelity: f64, unitary: MatRef<c64>) -> usize {
    let [a, b, c] = __weyl_coordinates(unitary);
    let traces = [
        c64::new(
            4.0 * (a.cos() * b.cos() * c.cos()),
            4.0 * (a.sin() * b.sin() * c.sin()),
        ),
        c64::new(
            4.0 * (PI4 - a).cos() * (basis_b - b).cos() * c.cos(),
            4.0 * (PI4 - a).sin() * (basis_b - b).sin() * c.sin(),
        ),
        c64::new(4.0 * c.cos(), 0.0),
        c64::new(4.0, 0.0),
    ];
    // The originial Python had `np.argmax`, which returns the lowest index in case two or more
    // values have a common maximum value.
    // `max_by` and `min_by` return the highest and lowest indices respectively, in case of ties.
    // So to reproduce `np.argmax`, we use `min_by` and switch the order of the
    // arguments in the comparison.
    traces
        .into_iter()
        .enumerate()
        .map(|(idx, trace)| (idx, trace.trace_to_fid() * basis_fidelity.powi(idx as i32)))
        .min_by(|(_idx1, fid1), (_idx2, fid2)| fid2.partial_cmp(fid1).unwrap())
        .unwrap()
        .0
}

/// A good approximation to the best value x to get the minimum
/// trace distance for :math:`U_d(x, x, x)` from :math:`U_d(a, b, c)`.
fn closest_partial_swap(a: f64, b: f64, c: f64) -> f64 {
    let m = (a + b + c) / 3.;
    let [am, bm, cm] = [a - m, b - m, c - m];
    let [ab, bc, ca] = [a - b, b - c, c - a];
    m + am * bm * cm * (6. + ab * ab + bc * bc + ca * ca) / 18.
}

fn rx_matrix(theta: f64) -> Array2<Complex64> {
    let half_theta = theta / 2.;
    let cos = Complex64::new(half_theta.cos(), 0.);
    let isin = Complex64::new(0., -half_theta.sin());
    array![[cos, isin], [isin, cos]]
}

fn ry_matrix(theta: f64) -> Array2<Complex64> {
    let half_theta = theta / 2.;
    let cos = Complex64::new(half_theta.cos(), 0.);
    let sin = Complex64::new(half_theta.sin(), 0.);
    array![[cos, -sin], [sin, cos]]
}

fn rz_matrix(theta: f64) -> Array2<Complex64> {
    let ilam2 = Complex64::new(0., 0.5 * theta);
    array![
        [(-ilam2).exp(), Complex64::new(0., 0.)],
        [Complex64::new(0., 0.), ilam2.exp()]
    ]
}

const DEFAULT_FIDELITY: f64 = 1.0 - 1.0e-9;
const C1_IM: Complex64 = Complex64::new(0.0, 1.0);

#[derive(Clone, Debug, Copy)]
#[pyclass(module = "qiskit._accelerate.two_qubit_decompose")]
enum Specialization {
    General,
    IdEquiv,
    SWAPEquiv,
    PartialSWAPEquiv,
    PartialSWAPFlipEquiv,
    ControlledEquiv,
    MirrorControlledEquiv,
    // These next 3 gates use the definition of fSim from eq (1) in:
    // https://arxiv.org/pdf/2001.08343.pdf
    #[allow(non_camel_case_types)]
    fSimaabEquiv,
    #[allow(non_camel_case_types)]
    fSimabbEquiv,
    #[allow(non_camel_case_types)]
    fSimabmbEquiv,
}

impl Specialization {
    fn to_u8(self) -> u8 {
        match self {
            Specialization::General => 0,
            Specialization::IdEquiv => 1,
            Specialization::SWAPEquiv => 2,
            Specialization::PartialSWAPEquiv => 3,
            Specialization::PartialSWAPFlipEquiv => 4,
            Specialization::ControlledEquiv => 5,
            Specialization::MirrorControlledEquiv => 6,
            Specialization::fSimaabEquiv => 7,
            Specialization::fSimabbEquiv => 8,
            Specialization::fSimabmbEquiv => 9,
        }
    }

    fn from_u8(val: u8) -> Self {
        match val {
            0 => Specialization::General,
            1 => Specialization::IdEquiv,
            2 => Specialization::SWAPEquiv,
            3 => Specialization::PartialSWAPEquiv,
            4 => Specialization::PartialSWAPFlipEquiv,
            5 => Specialization::ControlledEquiv,
            6 => Specialization::MirrorControlledEquiv,
            7 => Specialization::fSimaabEquiv,
            8 => Specialization::fSimabbEquiv,
            9 => Specialization::fSimabmbEquiv,
            _ => unreachable!("Invalid specialization value"),
        }
    }
}

#[derive(Clone, Debug)]
#[allow(non_snake_case)]
#[pyclass(module = "qiskit._accelerate.two_qubit_decompose", subclass)]
pub struct TwoQubitWeylDecomposition {
    #[pyo3(get)]
    a: f64,
    #[pyo3(get)]
    b: f64,
    #[pyo3(get)]
    c: f64,
    #[pyo3(get)]
    global_phase: f64,
    K1l: Array2<Complex64>,
    K2l: Array2<Complex64>,
    K1r: Array2<Complex64>,
    K2r: Array2<Complex64>,
    #[pyo3(get)]
    specialization: Specialization,
    default_euler_basis: EulerBasis,
    #[pyo3(get)]
    requested_fidelity: Option<f64>,
    #[pyo3(get)]
    calculated_fidelity: f64,
    unitary_matrix: Array2<Complex64>,
}

impl TwoQubitWeylDecomposition {
    fn weyl_gate(
        &self,
        simplify: bool,
        sequence: &mut TwoQubitSequenceVec,
        atol: f64,
        global_phase: &mut f64,
    ) {
        match self.specialization {
            Specialization::MirrorControlledEquiv => {
                sequence.push(("swap".to_string(), SmallVec::new(), smallvec![0, 1]));
                sequence.push((
                    "rzz".to_string(),
                    smallvec![(PI4 - self.c) * 2.],
                    smallvec![0, 1],
                ));
                *global_phase += PI4
            }
            Specialization::SWAPEquiv => {
                sequence.push(("swap".to_string(), SmallVec::new(), smallvec![0, 1]));
                *global_phase -= 3. * PI / 4.
            }
            _ => {
                if !simplify || self.a.abs() > atol {
                    sequence.push(("rxx".to_string(), smallvec![-self.a * 2.], smallvec![0, 1]));
                }
                if !simplify || self.b.abs() > atol {
                    sequence.push(("ryy".to_string(), smallvec![-self.b * 2.], smallvec![0, 1]));
                }
                if !simplify || self.c.abs() > atol {
                    sequence.push(("rzz".to_string(), smallvec![-self.c * 2.], smallvec![0, 1]));
                }
            }
        }
    }
}

const IPZ: [[Complex64; 2]; 2] = [
    [C1_IM, Complex64::new(0., 0.)],
    [Complex64::new(0., 0.), Complex64::new(0., -1.)],
];
const IPY: [[Complex64; 2]; 2] = [
    [Complex64::new(0., 0.), Complex64::new(1., 0.)],
    [Complex64::new(-1., 0.), Complex64::new(0., 0.)],
];
const IPX: [[Complex64; 2]; 2] = [
    [Complex64::new(0., 0.), C1_IM],
    [C1_IM, Complex64::new(0., 0.)],
];

#[pymethods]
impl TwoQubitWeylDecomposition {
    fn __getstate__(&self, py: Python) -> ([f64; 5], [PyObject; 5], u8, String, Option<f64>) {
        let specialization = self.specialization.to_u8();
        (
            [
                self.a,
                self.b,
                self.c,
                self.global_phase,
                self.calculated_fidelity,
            ],
            [
                self.K1l.to_pyarray(py).into(),
                self.K1r.to_pyarray(py).into(),
                self.K2l.to_pyarray(py).into(),
                self.K2r.to_pyarray(py).into(),
                self.unitary_matrix.to_pyarray(py).into(),
            ],
            specialization,
            self.default_euler_basis.to_str(),
            self.requested_fidelity,
        )
    }

    fn __setstate__(
        &mut self,
        state: (
            [f64; 5],
            [PyReadonlyArray2<Complex64>; 5],
            u8,
            String,
            Option<f64>,
        ),
    ) {
        self.a = state.0[0];
        self.b = state.0[1];
        self.c = state.0[2];
        self.global_phase = state.0[3];
        self.calculated_fidelity = state.0[4];
        self.K1l = state.1[0].as_array().to_owned();
        self.K1r = state.1[1].as_array().to_owned();
        self.K2l = state.1[2].as_array().to_owned();
        self.K2r = state.1[3].as_array().to_owned();
        self.unitary_matrix = state.1[4].as_array().to_owned();
        self.default_euler_basis = EulerBasis::from_string(&state.3).unwrap();
        self.requested_fidelity = state.4;
        self.specialization = Specialization::from_u8(state.2);
    }

    fn __getnewargs__(&self, py: Python) -> (PyObject, Option<f64>, Option<Specialization>, bool) {
        (
            self.unitary_matrix.to_pyarray(py).into(),
            self.requested_fidelity,
            None,
            true,
        )
    }

    #[new]
    #[pyo3(signature=(unitary_matrix, fidelity=DEFAULT_FIDELITY, _specialization=None, _pickle_context=false))]
    fn new(
        unitary_matrix: PyReadonlyArray2<Complex64>,
        fidelity: Option<f64>,
        _specialization: Option<Specialization>,
        _pickle_context: bool,
    ) -> PyResult<Self> {
        // If we're in a pickle context just make the closest to an empty
        // object as we can with minimal allocations and effort. All the
        // data will be filled in during deserialization from __setstate__.
        if _pickle_context {
            return Ok(TwoQubitWeylDecomposition {
                a: 0.,
                b: 0.,
                c: 0.,
                global_phase: 0.,
                K1l: Array2::zeros((0, 0)),
                K2l: Array2::zeros((0, 0)),
                K1r: Array2::zeros((0, 0)),
                K2r: Array2::zeros((0, 0)),
                specialization: Specialization::General,
                default_euler_basis: EulerBasis::ZYZ,
                requested_fidelity: fidelity,
                calculated_fidelity: 0.,
                unitary_matrix: Array2::zeros((0, 0)),
            });
        }
        let ipz: ArrayView2<Complex64> = aview2(&IPZ);
        let ipy: ArrayView2<Complex64> = aview2(&IPY);
        let ipx: ArrayView2<Complex64> = aview2(&IPX);

        let mut u = unitary_matrix.as_array().to_owned();
        let unitary_matrix = unitary_matrix.as_array().to_owned();
        let det_u = u.view().into_faer_complex().determinant().to_num_complex();
        let det_pow = det_u.powf(-0.25);
        u.mapv_inplace(|x| x * det_pow);
        let mut global_phase = det_u.arg() / 4.;
        let u_p = magic_basis_transform(u.view(), MagicBasisTransform::OutOf);
        let m2 = u_p.t().dot(&u_p);
        let default_euler_basis = EulerBasis::ZYZ;

        // M2 is a symmetric complex matrix. We need to decompose it as M2 = P D P^T where
        // P ∈ SO(4), D is diagonal with unit-magnitude elements.
        //
        // We can't use raw `eig` directly because it isn't guaranteed to give us real or othogonal
        // eigenvectors. Instead, since `M2` is complex-symmetric,
        //   M2 = A + iB
        // for real-symmetric `A` and `B`, and as
        //   M2^+ @ M2 = A^2 + B^2 + i [A, B] = 1
        // we must have `A` and `B` commute, and consequently they are simultaneously diagonalizable.
        // Mixing them together _should_ account for any degeneracy problems, but it's not
        // guaranteed, so we repeat it a little bit.  The fixed seed is to make failures
        // deterministic; the value is not important.
        let mut state = Pcg64Mcg::seed_from_u64(2023);
        let mut found = false;
        let mut d: Array1<Complex64> = Array1::zeros(0);
        let mut p: Array2<Complex64> = Array2::zeros((0, 0));
        for i in 0..100 {
            let rand_a: f64;
            let rand_b: f64;
            // For debugging the algorithm use the same RNG values from the
            // previous Python implementation for the first random trial.
            // In most cases this loop only executes a single iteration and
            // using the same rng values rules out possible RNG differences
            // as the root cause of a test failure
            if i == 0 {
                rand_a = 1.2602066112249388;
                rand_b = 0.22317849046722027;
            } else {
                rand_a = state.sample(StandardNormal);
                rand_b = state.sample(StandardNormal);
            }
            let m2_real = m2.mapv(|val| rand_a * val.re + rand_b * val.im);
            let p_inner = m2_real
                .view()
                .into_faer()
                .selfadjoint_eigendecomposition(Lower)
                .u()
                .into_ndarray()
                .mapv(Complex64::from);
            let d_inner = p_inner.t().dot(&m2).dot(&p_inner).diag().to_owned();
            let mut diag_d: Array2<Complex64> = Array2::zeros((4, 4));
            diag_d
                .diag_mut()
                .iter_mut()
                .enumerate()
                .for_each(|(index, x)| *x = d_inner[index]);

            let compare = p_inner.dot(&diag_d).dot(&p_inner.t());
            found = abs_diff_eq!(compare.view(), m2, epsilon = 1.0e-13);
            if found {
                p = p_inner;
                d = d_inner;
                break;
            }
        }
        if !found {
            import_exception!(qiskit, QiskitError);
            return Err(QiskitError::new_err(format!(
                "TwoQubitWeylDecomposition: failed to diagonalize M2. Please report this at https://github.com/Qiskit/qiskit-terra/issues/4159. Input: {:?}", unitary_matrix
            )));
        }
        let mut d = -d.map(|x| x.arg() / 2.);
        d[3] = -d[0] - d[1] - d[2];
        let mut cs: SmallVec<[f64; 3]> = (0..3)
            .map(|i| ((d[i] + d[3]) / 2.0).rem_euclid(TWO_PI))
            .collect();
        let cstemp: SmallVec<[f64; 3]> = cs
            .iter()
            .map(|x| x.rem_euclid(PI2))
            .map(|x| x.min(PI2 - x))
            .collect();
        let mut order = utils::arg_sort(&cstemp);
        (order[0], order[1], order[2]) = (order[1], order[2], order[0]);
        (cs[0], cs[1], cs[2]) = (cs[order[0]], cs[order[1]], cs[order[2]]);
        (d[0], d[1], d[2]) = (d[order[0]], d[order[1]], d[order[2]]);
        let mut p_orig = p.clone();
        for (i, item) in order.iter().enumerate().take(3) {
            let slice_a = p.slice_mut(s![.., i]);
            let slice_b = p_orig.slice_mut(s![.., *item]);
            Zip::from(slice_a).and(slice_b).for_each(::std::mem::swap);
        }
        if p.view().into_faer_complex().determinant().re < 0. {
            p.slice_mut(s![.., -1]).mapv_inplace(|x| -x);
        }
        let mut temp: Array2<Complex64> = Array2::zeros((4, 4));
        temp.diag_mut()
            .iter_mut()
            .enumerate()
            .for_each(|(index, x)| *x = (C1_IM * d[index]).exp());
        let k1 = magic_basis_transform(u_p.dot(&p).dot(&temp).view(), MagicBasisTransform::Into);
        let k2 = magic_basis_transform(p.t(), MagicBasisTransform::Into);

        #[allow(non_snake_case)]
        let (mut K1l, mut K1r, phase_l) = decompose_two_qubit_product_gate(k1.view())?;
        #[allow(non_snake_case)]
        let (K2l, mut K2r, phase_r) = decompose_two_qubit_product_gate(k2.view())?;
        global_phase += phase_l + phase_r;

        // Flip into Weyl chamber
        if cs[0] > PI2 {
            cs[0] -= PI32;
            K1l = K1l.dot(&ipy);
            K1r = K1r.dot(&ipy);
            global_phase += PI2;
        }
        if cs[1] > PI2 {
            cs[1] -= PI32;
            K1l = K1l.dot(&ipx);
            K1r = K1r.dot(&ipx);
            global_phase += PI2;
        }
        let mut conjs = 0;
        if cs[0] > PI4 {
            cs[0] = PI2 - cs[0];
            K1l = K1l.dot(&ipy);
            K2r = ipy.dot(&K2r);
            conjs += 1;
            global_phase -= PI2;
        }
        if cs[1] > PI4 {
            cs[1] = PI2 - cs[1];
            K1l = K1l.dot(&ipx);
            K2r = ipx.dot(&K2r);
            conjs += 1;
            global_phase += PI2;
            if conjs == 1 {
                global_phase -= PI;
            }
        }
        if cs[2] > PI2 {
            cs[2] -= PI32;
            K1l = K1l.dot(&ipz);
            K1r = K1r.dot(&ipz);
            global_phase += PI2;
            if conjs == 1 {
                global_phase -= PI;
            }
        }
        if conjs == 1 {
            cs[2] = PI2 - cs[2];
            K1l = K1l.dot(&ipz);
            K2r = ipz.dot(&K2r);
            global_phase += PI2;
        }
        if cs[2] > PI4 {
            cs[2] -= PI2;
            K1l = K1l.dot(&ipz);
            K1r = K1r.dot(&ipz);
            global_phase -= PI2;
        }
        let [a, b, c] = [cs[1], cs[0], cs[2]];
        let is_close = |ap: f64, bp: f64, cp: f64| -> bool {
            let [da, db, dc] = [a - ap, b - bp, c - cp];
            let tr = 4.
                * Complex64::new(
                    da.cos() * db.cos() * dc.cos(),
                    da.sin() * db.sin() * dc.sin(),
                );
            match fidelity {
                Some(fid) => tr.trace_to_fid() >= fid,
                // Set to false here to default to general specialization in the absence of a
                // fidelity and provided specialization.
                None => false,
            }
        };

        let closest_abc = closest_partial_swap(a, b, c);
        let closest_ab_minus_c = closest_partial_swap(a, b, -c);
        let mut flipped_from_original = false;
        let specialization = match _specialization {
            Some(specialization) => specialization,
            None => {
                if is_close(0., 0., 0.) {
                    Specialization::IdEquiv
                } else if is_close(PI4, PI4, PI4) || is_close(PI4, PI4, -PI4) {
                    Specialization::SWAPEquiv
                } else if is_close(closest_abc, closest_abc, closest_abc) {
                    Specialization::PartialSWAPEquiv
                } else if is_close(closest_ab_minus_c, closest_ab_minus_c, -closest_ab_minus_c) {
                    Specialization::PartialSWAPFlipEquiv
                } else if is_close(a, 0., 0.) {
                    Specialization::ControlledEquiv
                } else if is_close(PI4, PI4, c) {
                    Specialization::MirrorControlledEquiv
                } else if is_close((a + b) / 2., (a + b) / 2., c) {
                    Specialization::fSimaabEquiv
                } else if is_close(a, (b + c) / 2., (b + c) / 2.) {
                    Specialization::fSimabbEquiv
                } else if is_close(a, (b - c) / 2., (c - b) / 2.) {
                    Specialization::fSimabmbEquiv
                } else {
                    Specialization::General
                }
            }
        };
        let general = TwoQubitWeylDecomposition {
            a,
            b,
            c,
            global_phase,
            K1l,
            K1r,
            K2l,
            K2r,
            specialization: Specialization::General,
            default_euler_basis,
            requested_fidelity: fidelity,
            calculated_fidelity: -1.0,
            unitary_matrix,
        };
        let mut specialized: TwoQubitWeylDecomposition = match specialization {
            // :math:`U \sim U_d(0,0,0) \sim Id`
            //
            // This gate binds 0 parameters, we make it canonical by setting
            // :math:`K2_l = Id` , :math:`K2_r = Id`.
            Specialization::IdEquiv => TwoQubitWeylDecomposition {
                specialization,
                a: 0.,
                b: 0.,
                c: 0.,
                K1l: general.K1l.dot(&general.K2l),
                K1r: general.K1r.dot(&general.K2r),
                K2l: Array2::eye(2),
                K2r: Array2::eye(2),
                ..general
            },
            // :math:`U \sim U_d(\pi/4, \pi/4, \pi/4) \sim U(\pi/4, \pi/4, -\pi/4) \sim \text{SWAP}`
            //
            // This gate binds 0 parameters, we make it canonical by setting
            // :math:`K2_l = Id` , :math:`K2_r = Id`.
            Specialization::SWAPEquiv => {
                if c > 0. {
                    TwoQubitWeylDecomposition {
                        specialization,
                        a: PI4,
                        b: PI4,
                        c: PI4,
                        K1l: general.K1l.dot(&general.K2r),
                        K1r: general.K1r.dot(&general.K2l),
                        K2l: Array2::eye(2),
                        K2r: Array2::eye(2),
                        ..general
                    }
                } else {
                    flipped_from_original = true;
                    TwoQubitWeylDecomposition {
                        specialization,
                        a: PI4,
                        b: PI4,
                        c: PI4,
                        global_phase: global_phase + PI2,
                        K1l: general.K1l.dot(&ipz).dot(&general.K2r),
                        K1r: general.K1r.dot(&ipz).dot(&general.K2l),
                        K2l: Array2::eye(2),
                        K2r: Array2::eye(2),
                        ..general
                    }
                }
            }
            // :math:`U \sim U_d(\alpha\pi/4, \alpha\pi/4, \alpha\pi/4) \sim \text{SWAP}^\alpha`
            //
            // This gate binds 3 parameters, we make it canonical by setting:
            //
            // :math:`K2_l = Id`.
            Specialization::PartialSWAPEquiv => {
                let closest = closest_partial_swap(a, b, c);
                let mut k2l_dag = general.K2l.t().to_owned();
                k2l_dag.view_mut().mapv_inplace(|x| x.conj());
                TwoQubitWeylDecomposition {
                    specialization,
                    a: closest,
                    b: closest,
                    c: closest,
                    K1l: general.K1l.dot(&general.K2l),
                    K1r: general.K1r.dot(&general.K2l),
                    K2r: k2l_dag.dot(&general.K2r),
                    K2l: Array2::eye(2),
                    ..general
                }
            }
            // :math:`U \sim U_d(\alpha\pi/4, \alpha\pi/4, -\alpha\pi/4) \sim \text{SWAP}^\alpha`
            //
            // (a non-equivalent root of SWAP from the TwoQubitWeylPartialSWAPEquiv
            // similar to how :math:`x = (\pm \sqrt(x))^2`)
            //
            // This gate binds 3 parameters, we make it canonical by setting:
            //
            // :math:`K2_l = Id`
            Specialization::PartialSWAPFlipEquiv => {
                let closest = closest_partial_swap(a, b, -c);
                let mut k2l_dag = general.K2l.t().to_owned();
                k2l_dag.mapv_inplace(|x| x.conj());
                TwoQubitWeylDecomposition {
                    specialization,
                    a: closest,
                    b: closest,
                    c: -closest,
                    K1l: general.K1l.dot(&general.K2l),
                    K1r: general.K1r.dot(&ipz).dot(&general.K2l).dot(&ipz),
                    K2r: ipz.dot(&k2l_dag).dot(&ipz).dot(&general.K2r),
                    K2l: Array2::eye(2),
                    ..general
                }
            }
            // :math:`U \sim U_d(\alpha, 0, 0) \sim \text{Ctrl-U}`
            //
            // This gate binds 4 parameters, we make it canonical by setting:
            //
            //      :math:`K2_l = Ry(\theta_l) Rx(\lambda_l)` ,
            //      :math:`K2_r = Ry(\theta_r) Rx(\lambda_r)` .
            Specialization::ControlledEquiv => {
                let euler_basis = EulerBasis::XYX;
                let [k2ltheta, k2lphi, k2llambda, k2lphase] =
                    angles_from_unitary(general.K2l.view(), euler_basis);
                let [k2rtheta, k2rphi, k2rlambda, k2rphase] =
                    angles_from_unitary(general.K2r.view(), euler_basis);
                TwoQubitWeylDecomposition {
                    specialization,
                    a,
                    b: 0.,
                    c: 0.,
                    global_phase: global_phase + k2lphase + k2rphase,
                    K1l: general.K1l.dot(&rx_matrix(k2lphi)),
                    K1r: general.K1r.dot(&rx_matrix(k2rphi)),
                    K2l: ry_matrix(k2ltheta).dot(&rx_matrix(k2llambda)),
                    K2r: ry_matrix(k2rtheta).dot(&rx_matrix(k2rlambda)),
                    default_euler_basis: euler_basis,
                    ..general
                }
            }
            // :math:`U \sim U_d(\pi/4, \pi/4, \alpha) \sim \text{SWAP} \cdot \text{Ctrl-U}`
            //
            // This gate binds 4 parameters, we make it canonical by setting:
            //
            // :math:`K2_l = Ry(\theta_l)\cdot Rz(\lambda_l)` , :math:`K2_r = Ry(\theta_r)\cdot Rz(\lambda_r)`
            Specialization::MirrorControlledEquiv => {
                let [k2ltheta, k2lphi, k2llambda, k2lphase] =
                    angles_from_unitary(general.K2l.view(), EulerBasis::ZYZ);
                let [k2rtheta, k2rphi, k2rlambda, k2rphase] =
                    angles_from_unitary(general.K2r.view(), EulerBasis::ZYZ);
                TwoQubitWeylDecomposition {
                    specialization,
                    a: PI4,
                    b: PI4,
                    c,
                    global_phase: global_phase + k2lphase + k2rphase,
                    K1l: general.K1l.dot(&rz_matrix(k2rphi)),
                    K1r: general.K1r.dot(&rz_matrix(k2lphi)),
                    K2l: ry_matrix(k2ltheta).dot(&rz_matrix(k2llambda)),
                    K2r: ry_matrix(k2rtheta).dot(&rz_matrix(k2rlambda)),
                    ..general
                }
            }
            // :math:`U \sim U_d(\alpha, \alpha, \beta), \alpha \geq |\beta|`
            //
            // This gate binds 5 parameters, we make it canonical by setting:
            //
            // :math:`K2_l = Ry(\theta_l)\cdot Rz(\lambda_l)`.
            Specialization::fSimaabEquiv => {
                let [k2ltheta, k2lphi, k2llambda, k2lphase] =
                    angles_from_unitary(general.K2l.view(), EulerBasis::ZYZ);
                TwoQubitWeylDecomposition {
                    specialization,
                    a: (a + b) / 2.,
                    b: (a + b) / 2.,
                    c,
                    global_phase: global_phase + k2lphase,
                    K1r: general.K1r.dot(&rz_matrix(k2lphi)),
                    K1l: general.K1l.dot(&rz_matrix(k2lphi)),
                    K2l: ry_matrix(k2ltheta).dot(&rz_matrix(k2llambda)),
                    K2r: rz_matrix(-k2lphi).dot(&general.K2r),
                    ..general
                }
            }
            // :math:`U \sim U_d(\alpha, \beta, -\beta), \alpha \geq \beta \geq 0`
            //
            // This gate binds 5 parameters, we make it canonical by setting:
            //
            // :math:`K2_l = Ry(\theta_l)Rx(\lambda_l)`
            Specialization::fSimabbEquiv => {
                let euler_basis = EulerBasis::XYX;
                let [k2ltheta, k2lphi, k2llambda, k2lphase] =
                    angles_from_unitary(general.K2l.view(), euler_basis);
                TwoQubitWeylDecomposition {
                    specialization,
                    a,
                    b: (b + c) / 2.,
                    c: (b + c) / 2.,
                    global_phase: global_phase + k2lphase,
                    K1r: general.K1r.dot(&rx_matrix(k2lphi)),
                    K1l: general.K1l.dot(&rx_matrix(k2lphi)),
                    K2l: ry_matrix(k2ltheta).dot(&rx_matrix(k2llambda)),
                    K2r: rx_matrix(-k2lphi).dot(&general.K2r),
                    default_euler_basis: euler_basis,
                    ..general
                }
            }
            // :math:`U \sim U_d(\alpha, \beta, -\beta), \alpha \geq \beta \geq 0`
            //
            // This gate binds 5 parameters, we make it canonical by setting:
            //
            // :math:`K2_l = Ry(\theta_l)Rx(\lambda_l)`
            Specialization::fSimabmbEquiv => {
                let euler_basis = EulerBasis::XYX;
                let [k2ltheta, k2lphi, k2llambda, k2lphase] =
                    angles_from_unitary(general.K2l.view(), euler_basis);
                TwoQubitWeylDecomposition {
                    specialization,
                    a,
                    b: (b - c) / 2.,
                    c: -((b - c) / 2.),
                    global_phase: global_phase + k2lphase,
                    K1l: general.K1l.dot(&rx_matrix(k2lphi)),
                    K1r: general.K1r.dot(&ipz).dot(&rx_matrix(k2lphi)).dot(&ipz),
                    K2l: ry_matrix(k2ltheta).dot(&rx_matrix(k2llambda)),
                    K2r: ipz.dot(&rx_matrix(-k2lphi)).dot(&ipz).dot(&general.K2r),
                    default_euler_basis: euler_basis,
                    ..general
                }
            }
            // U has no special symmetry.
            //
            // This gate binds all 6 possible parameters, so there is no need to make the single-qubit
            // pre-/post-gates canonical.
            Specialization::General => general,
        };

        let tr = if flipped_from_original {
            let [da, db, dc] = [
                PI2 - a - specialized.a,
                b - specialized.b,
                -c - specialized.c,
            ];
            4. * Complex64::new(
                da.cos() * db.cos() * dc.cos(),
                da.sin() * db.sin() * dc.sin(),
            )
        } else {
            let [da, db, dc] = [a - specialized.a, b - specialized.b, c - specialized.c];
            4. * Complex64::new(
                da.cos() * db.cos() * dc.cos(),
                da.sin() * db.sin() * dc.sin(),
            )
        };
        specialized.calculated_fidelity = tr.trace_to_fid();
        if let Some(fid) = specialized.requested_fidelity {
            if specialized.calculated_fidelity + 1.0e-13 < fid {
                import_exception!(qiskit, QiskitError);
                return Err(QiskitError::new_err(format!(
                    "Specialization: {:?} calculated fidelity: {} is worse than requested fidelity: {}",
                    specialized.specialization,
                    specialized.calculated_fidelity,
                    fid
                )));
            }
        }
        specialized.global_phase += tr.arg();
        Ok(specialized)
    }

    #[allow(non_snake_case)]
    #[getter]
    fn K1l(&self, py: Python) -> PyObject {
        self.K1l.to_pyarray(py).into()
    }

    #[allow(non_snake_case)]
    #[getter]
    fn K1r(&self, py: Python) -> PyObject {
        self.K1r.to_pyarray(py).into()
    }

    #[allow(non_snake_case)]
    #[getter]
    fn K2l(&self, py: Python) -> PyObject {
        self.K2l.to_pyarray(py).into()
    }

    #[allow(non_snake_case)]
    #[getter]
    fn K2r(&self, py: Python) -> PyObject {
        self.K2r.to_pyarray(py).into()
    }

    #[getter]
    fn unitary_matrix(&self, py: Python) -> PyObject {
        self.unitary_matrix.to_pyarray(py).into()
    }

    #[pyo3(signature = (euler_basis=None, simplify=false, atol=None))]
    fn circuit(
        &self,
        euler_basis: Option<&str>,
        simplify: bool,
        atol: Option<f64>,
    ) -> PyResult<TwoQubitGateSequence> {
        let euler_basis: EulerBasis = match euler_basis {
            Some(basis) => EulerBasis::from_string(basis)?,
            None => self.default_euler_basis,
        };
        let target_1q_basis_list: Vec<EulerBasis> = vec![euler_basis];

        let mut gate_sequence = Vec::new();
        let mut global_phase: f64 = self.global_phase;

        let c2r = unitary_to_gate_sequence_inner(
            self.K2r.view(),
            &target_1q_basis_list,
            0,
            None,
            simplify,
            atol,
        )
        .unwrap();
        for gate in c2r.gates {
            gate_sequence.push((gate.0, gate.1, smallvec![0]))
        }
        global_phase += c2r.global_phase;
        let c2l = unitary_to_gate_sequence_inner(
            self.K2l.view(),
            &target_1q_basis_list,
            1,
            None,
            simplify,
            atol,
        )
        .unwrap();
        for gate in c2l.gates {
            gate_sequence.push((gate.0, gate.1, smallvec![1]))
        }
        global_phase += c2l.global_phase;
        self.weyl_gate(
            simplify,
            &mut gate_sequence,
            atol.unwrap_or(ANGLE_ZERO_EPSILON),
            &mut global_phase,
        );
        let c1r = unitary_to_gate_sequence_inner(
            self.K1r.view(),
            &target_1q_basis_list,
            0,
            None,
            simplify,
            atol,
        )
        .unwrap();
        for gate in c1r.gates {
            gate_sequence.push((gate.0, gate.1, smallvec![0]))
        }
        global_phase += c2r.global_phase;
        let c1l = unitary_to_gate_sequence_inner(
            self.K1l.view(),
            &target_1q_basis_list,
            1,
            None,
            simplify,
            atol,
        )
        .unwrap();
        for gate in c1l.gates {
            gate_sequence.push((gate.0, gate.1, smallvec![1]))
        }
        Ok(TwoQubitGateSequence {
            gates: gate_sequence,
            global_phase,
        })
    }
}

type TwoQubitSequenceVec = Vec<(String, SmallVec<[f64; 3]>, SmallVec<[u8; 2]>)>;

#[pyclass(sequence)]
pub struct TwoQubitGateSequence {
    gates: TwoQubitSequenceVec,
    #[pyo3(get)]
    global_phase: f64,
}

#[pymethods]
impl TwoQubitGateSequence {
    #[new]
    fn new() -> Self {
        TwoQubitGateSequence {
            gates: Vec::new(),
            global_phase: 0.,
        }
    }

    fn __getstate__(&self) -> (TwoQubitSequenceVec, f64) {
        (self.gates.clone(), self.global_phase)
    }

    fn __setstate__(&mut self, state: (TwoQubitSequenceVec, f64)) {
        self.gates = state.0;
        self.global_phase = state.1;
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.gates.len())
    }

    fn __getitem__(&self, py: Python, idx: utils::SliceOrInt) -> PyResult<PyObject> {
        match idx {
            utils::SliceOrInt::Slice(slc) => {
                let len = self.gates.len().try_into().unwrap();
                let indices = slc.indices(len)?;
                let mut out_vec: TwoQubitSequenceVec = Vec::new();
                // Start and stop will always be positive the slice api converts
                // negatives to the index for example:
                // list(range(5))[-1:-3:-1]
                // will return start=4, stop=2, and step=-
                let mut pos: isize = indices.start;
                let mut cond = if indices.step < 0 {
                    pos > indices.stop
                } else {
                    pos < indices.stop
                };
                while cond {
                    if pos < len as isize {
                        out_vec.push(self.gates[pos as usize].clone());
                    }
                    pos += indices.step;
                    if indices.step < 0 {
                        cond = pos > indices.stop;
                    } else {
                        cond = pos < indices.stop;
                    }
                }
                Ok(out_vec.into_py(py))
            }
            utils::SliceOrInt::Int(idx) => {
                let len = self.gates.len() as isize;
                if idx >= len || idx < -len {
                    Err(PyIndexError::new_err(format!("Invalid index, {idx}")))
                } else if idx < 0 {
                    let len = self.gates.len();
                    Ok(self.gates[len - idx.unsigned_abs()].to_object(py))
                } else {
                    Ok(self.gates[idx as usize].to_object(py))
                }
            }
        }
    }
}

#[pymodule]
pub fn two_qubit_decompose(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(_num_basis_gates))?;
    m.add_class::<TwoQubitGateSequence>()?;
    m.add_class::<TwoQubitWeylDecomposition>()?;
    m.add_class::<Specialization>()?;
    Ok(())
}

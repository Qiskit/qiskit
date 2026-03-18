// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
use approx::abs_diff_eq;
use num_complex::{Complex, Complex64, ComplexFloat};
use smallvec::SmallVec;
use std::f64::consts::PI;
use std::ops::Deref;

use faer::Side::Lower;
use faer::{Mat, MatRef, Scale, prelude::*};
use ndarray::Zip;
use ndarray::linalg::kron;
use ndarray::prelude::*;
use numpy::ToPyArray;
use numpy::{IntoPyArray, PyArrayLike2, PyReadonlyArray2};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;

use crate::QiskitError;
use crate::euler_one_qubit_decomposer::{
    ANGLE_ZERO_EPSILON, EulerBasis, EulerBasisSet, angles_from_unitary, det_one_qubit,
    unitary_to_gate_sequence_inner,
};
use crate::linalg::{faer_to_ndarray, ndarray_to_faer};

use rand::prelude::*;
use rand_distr::StandardNormal;
use rand_pcg::Pcg64Mcg;

use qiskit_circuit::circuit_data::{CircuitData, PyCircuitData};
use qiskit_circuit::gate_matrix::ONE_QUBIT_IDENTITY;
use qiskit_circuit::operations::{Param, StandardGate};
use qiskit_circuit::util::{C_M_ONE, C_ONE, C_ZERO, GateArray1Q, GateArray2Q, IM, M_IM, c64};
use qiskit_circuit::{Qubit, impl_intopyobject_for_copy_pyclass};

use super::common::{
    DEFAULT_FIDELITY, TraceToFidelity, closest_partial_swap, rx_matrix, ry_matrix, rz_matrix,
    transpose_conjugate,
};

const PI2: f64 = PI / 2.;
const PI4: f64 = PI / 4.;
const PI32: f64 = 3.0 * PI2;
const TWO_PI: f64 = 2.0 * PI;
const C1: c64 = c64 { re: 1.0, im: 0.0 };

static B_NON_NORMALIZED: GateArray2Q = [
    [C_ONE, IM, C_ZERO, C_ZERO],
    [C_ZERO, C_ZERO, IM, C_ONE],
    [C_ZERO, C_ZERO, IM, C_M_ONE],
    [C_ONE, M_IM, C_ZERO, C_ZERO],
];

static B_NON_NORMALIZED_DAGGER: GateArray2Q = [
    [c64(0.5, 0.), C_ZERO, C_ZERO, c64(0.5, 0.)],
    [c64(0., -0.5), C_ZERO, C_ZERO, c64(0., 0.5)],
    [C_ZERO, c64(0., -0.5), c64(0., -0.5), C_ZERO],
    [C_ZERO, c64(0.5, 0.), c64(-0.5, 0.), C_ZERO],
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
    let unitary: ArrayView2<Complex64> = faer_to_ndarray(u.as_ref());
    ndarray_to_faer(magic_basis_transform(unitary, MagicBasisTransform::OutOf).view()).to_owned()
}

/// Return indices that sort partially ordered data.
/// If `data` contains two elements that are incomparable,
/// an error will be thrown.
fn arg_sort<T: PartialOrd>(data: &[T]) -> Vec<usize> {
    let mut indices = (0..data.len()).collect::<Vec<_>>();
    indices.sort_by(|&a, &b| data[a].partial_cmp(&data[b]).unwrap());
    indices
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
        return Err(QiskitError::new_err(
            "decompose_two_qubit_product_gate: unable to decompose: detR < 0.1",
        ));
    }
    r.mapv_inplace(|x| x / det_r.sqrt());
    let r_t_conj: Array2<Complex64> = transpose_conjugate(r.view());
    let eye = aview2(&ONE_QUBIT_IDENTITY);
    let mut temp = kron(&eye, &r_t_conj);
    temp = special_unitary.dot(&temp);
    let mut l = temp.slice(s![..;2, ..;2]).to_owned();
    let det_l = det_one_qubit(l.view());
    if det_l.abs() < 0.9 {
        return Err(QiskitError::new_err(
            "decompose_two_qubit_product_gate: unable to decompose: detL < 0.9",
        ));
    }
    l.mapv_inplace(|x| x / det_l.sqrt());
    let phase = det_l.arg() / 2.;

    Ok((l, r, phase))
}

#[pyfunction]
#[pyo3(name = "decompose_two_qubit_product_gate")]
/// Decompose :math:`U = U_l \otimes U_r` where :math:`U \in SU(4)`,
/// and :math:`U_l,~U_r \in SU(2)`.
/// Args:
///     special_unitary_matrix: special unitary matrix to decompose
/// Raises:
///     QiskitError: if decomposition isn't possible.
pub fn py_decompose_two_qubit_product_gate(
    py: Python,
    special_unitary: PyArrayLike2<Complex64, numpy::AllowTypeChange>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, f64)> {
    let view = special_unitary.as_array();
    let (l, r, phase) = decompose_two_qubit_product_gate(view)?;
    Ok((
        l.into_pyarray(py).into_any().unbind(),
        r.into_pyarray(py).into_any().unbind(),
        phase,
    ))
}

/// Computes the Weyl coordinates for a given two-qubit unitary matrix.
///
/// Args:
///     U (np.ndarray): Input two-qubit unitary.
///
/// Returns:
///     np.ndarray: Array of the 3 Weyl coordinates.
#[pyfunction]
pub fn weyl_coordinates(py: Python, unitary: PyReadonlyArray2<Complex64>) -> PyResult<Py<PyAny>> {
    let array = unitary.as_array();
    Ok(__weyl_coordinates(ndarray_to_faer(array))?
        .to_vec()
        .into_pyarray(py)
        .into_any()
        .unbind())
}

fn __weyl_coordinates(unitary: MatRef<c64>) -> PyResult<[f64; 3]> {
    let uscaled = Scale(C1 / unitary.determinant().powf(0.25)) * unitary;
    let uup = transform_from_magic_basis(uscaled);
    let mut darg: Vec<_> = (uup.transpose() * &uup)
        .eigenvalues()
        .map_err(|e| QiskitError::new_err(format!("{e:?}")))?
        .into_iter()
        .map(|x| -x.arg() / 2.0)
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
    let mut order = arg_sort(&cstemp);
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
    Ok([cs[1], cs[0], cs[2]])
}

#[pyfunction]
#[pyo3(text_signature = "(basis_b, basis_fidelity, unitary, /")]
pub fn _num_basis_gates(
    basis_b: f64,
    basis_fidelity: f64,
    unitary: PyReadonlyArray2<Complex<f64>>,
) -> PyResult<usize> {
    let u = ndarray_to_faer(unitary.as_array());
    __num_basis_gates(basis_b, basis_fidelity, u)
}

pub(super) fn __num_basis_gates(
    basis_b: f64,
    basis_fidelity: f64,
    unitary: MatRef<c64>,
) -> PyResult<usize> {
    let [a, b, c] = __weyl_coordinates(unitary)?;
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
    // The original Python had `np.argmax`, which returns the lowest index in case two or more
    // values have a common maximum value.
    // `max_by` and `min_by` return the highest and lowest indices respectively, in case of ties.
    // So to reproduce `np.argmax`, we use `min_by` and switch the order of the
    // arguments in the comparison.
    Ok(traces
        .into_iter()
        .enumerate()
        .map(|(idx, trace)| (idx, trace.trace_to_fid() * basis_fidelity.powi(idx as i32)))
        .min_by(|(_idx1, fid1), (_idx2, fid2)| fid2.partial_cmp(fid1).unwrap())
        .unwrap()
        .0)
}

#[derive(Clone, Debug, Copy, PartialEq, Eq)]
#[pyclass(module = "qiskit._accelerate.two_qubit_decompose", eq, from_py_object)]
pub enum Specialization {
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
impl_intopyobject_for_copy_pyclass!(Specialization);

#[pymethods]
impl Specialization {
    fn __reduce__(&self, py: Python) -> PyResult<Py<PyAny>> {
        // Ideally we'd use the string-only form of `__reduce__` for simplicity, but PyO3 enums
        // don't produce Python singletons, and pickle doesn't like that.
        let val: u8 = match self {
            Self::General => 0,
            Self::IdEquiv => 1,
            Self::SWAPEquiv => 2,
            Self::PartialSWAPEquiv => 3,
            Self::PartialSWAPFlipEquiv => 4,
            Self::ControlledEquiv => 5,
            Self::MirrorControlledEquiv => 6,
            Self::fSimaabEquiv => 7,
            Self::fSimabbEquiv => 8,
            Self::fSimabmbEquiv => 9,
        };
        (py.get_type::<Self>().getattr("_from_u8")?, (val,)).into_py_any(py)
    }

    #[staticmethod]
    fn _from_u8(val: u8) -> PyResult<Self> {
        match val {
            0 => Ok(Self::General),
            1 => Ok(Self::IdEquiv),
            2 => Ok(Self::SWAPEquiv),
            3 => Ok(Self::PartialSWAPEquiv),
            4 => Ok(Self::PartialSWAPFlipEquiv),
            5 => Ok(Self::ControlledEquiv),
            6 => Ok(Self::MirrorControlledEquiv),
            7 => Ok(Self::fSimaabEquiv),
            8 => Ok(Self::fSimabbEquiv),
            9 => Ok(Self::fSimabmbEquiv),
            x => Err(PyValueError::new_err(format!(
                "unknown specialization discriminant '{x}'"
            ))),
        }
    }
}

#[derive(Clone, Debug)]
#[allow(non_snake_case)]
#[pyclass(
    module = "qiskit._accelerate.two_qubit_decompose",
    subclass,
    skip_from_py_object
)]
pub struct TwoQubitWeylDecomposition {
    #[pyo3(get)]
    pub(super) a: f64,
    #[pyo3(get)]
    pub(super) b: f64,
    #[pyo3(get)]
    pub(super) c: f64,
    #[pyo3(get)]
    pub global_phase: f64,
    pub(super) K1l: Array2<Complex64>,
    pub(super) K2l: Array2<Complex64>,
    pub(super) K1r: Array2<Complex64>,
    pub(super) K2r: Array2<Complex64>,
    #[pyo3(get)]
    pub specialization: Specialization,
    default_euler_basis: EulerBasis,
    #[pyo3(get)]
    requested_fidelity: Option<f64>,
    #[pyo3(get)]
    calculated_fidelity: f64,
    pub(super) unitary_matrix: Array2<Complex64>,
}

impl TwoQubitWeylDecomposition {
    pub fn a(&self) -> f64 {
        self.a
    }
    pub fn b(&self) -> f64 {
        self.b
    }
    pub fn c(&self) -> f64 {
        self.c
    }

    pub fn k1l_view(&self) -> ArrayView2<'_, Complex64> {
        self.K1l.view()
    }

    pub fn k2l_view(&self) -> ArrayView2<'_, Complex64> {
        self.K2l.view()
    }

    pub fn k1r_view(&self) -> ArrayView2<'_, Complex64> {
        self.K1r.view()
    }

    pub fn k2r_view(&self) -> ArrayView2<'_, Complex64> {
        self.K2r.view()
    }

    fn weyl_gate(
        &self,
        simplify: bool,
        sequence: &mut CircuitData,
        atol: f64,
        global_phase: &mut f64,
    ) -> PyResult<()> {
        match self.specialization {
            Specialization::MirrorControlledEquiv => {
                sequence.push_standard_gate(StandardGate::Swap, &[], &[Qubit(0), Qubit(1)])?;
                sequence.push_standard_gate(
                    StandardGate::RZZ,
                    &[Param::Float((PI4 - self.c) * 2.)],
                    &[Qubit(0), Qubit(1)],
                )?;
                *global_phase += PI4
            }
            Specialization::SWAPEquiv => {
                sequence.push_standard_gate(StandardGate::Swap, &[], &[Qubit(0), Qubit(1)])?;
                *global_phase -= 3. * PI / 4.
            }
            _ => {
                if !simplify || self.a.abs() > atol {
                    sequence.push_standard_gate(
                        StandardGate::RXX,
                        &[Param::Float(-self.a * 2.)],
                        &[Qubit(0), Qubit(1)],
                    )?;
                }
                if !simplify || self.b.abs() > atol {
                    sequence.push_standard_gate(
                        StandardGate::RYY,
                        &[Param::Float(-self.b * 2.)],
                        &[Qubit(0), Qubit(1)],
                    )?;
                }
                if !simplify || self.c.abs() > atol {
                    sequence.push_standard_gate(
                        StandardGate::RZZ,
                        &[Param::Float(-self.c * 2.)],
                        &[Qubit(0), Qubit(1)],
                    )?;
                }
            }
        }
        Ok(())
    }

    /// Instantiate a new TwoQubitWeylDecomposition with rust native
    /// data structures
    pub fn new_inner(
        unitary_matrix: ArrayView2<Complex64>,

        fidelity: Option<f64>,
        _specialization: Option<Specialization>,
    ) -> PyResult<Self> {
        let ipz: ArrayView2<Complex64> = aview2(&IPZ);
        let ipy: ArrayView2<Complex64> = aview2(&IPY);
        let ipx: ArrayView2<Complex64> = aview2(&IPX);

        let mut u = unitary_matrix.to_owned();
        let unitary_matrix = unitary_matrix.to_owned();
        let det_u = ndarray_to_faer(u.view()).determinant();
        let det_pow = det_u.powf(-0.25);
        u.mapv_inplace(|x| x * det_pow);
        let mut global_phase = det_u.arg() / 4.;
        let u_p = magic_basis_transform(u.view(), MagicBasisTransform::OutOf);
        let m2 = u_p.t().dot(&u_p);
        let default_euler_basis = EulerBasis::ZYZ;

        // M2 is a symmetric complex matrix. We need to decompose it as M2 = P D P^T where
        // P ∈ SO(4), D is diagonal with unit-magnitude elements.
        //
        // We can't use raw `eig` directly because it isn't guaranteed to give us real or orthogonal
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
            let p_inner = faer_to_ndarray(
                ndarray_to_faer(m2_real.view())
                    .self_adjoint_eigen(Lower)
                    .map_err(|e| QiskitError::new_err(format!("{e:?}")))?
                    .U(),
            )
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
            return Err(QiskitError::new_err(format!(
                "TwoQubitWeylDecomposition: failed to diagonalize M2. Please report this at https://github.com/Qiskit/qiskit-terra/issues/4159. Input: {unitary_matrix:?}"
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
        let mut order = arg_sort(&cstemp);
        (order[0], order[1], order[2]) = (order[1], order[2], order[0]);
        (cs[0], cs[1], cs[2]) = (cs[order[0]], cs[order[1]], cs[order[2]]);
        (d[0], d[1], d[2]) = (d[order[0]], d[order[1]], d[order[2]]);
        let mut p_orig = p.clone();
        for (i, item) in order.iter().enumerate().take(3) {
            let slice_a = p.slice_mut(s![.., i]);
            let slice_b = p_orig.slice_mut(s![.., *item]);
            Zip::from(slice_a).and(slice_b).for_each(::std::mem::swap);
        }
        if ndarray_to_faer(p.view()).determinant().re < 0. {
            p.slice_mut(s![.., -1]).mapv_inplace(|x| -x);
        }
        let mut temp: Array2<Complex64> = Array2::zeros((4, 4));
        temp.diag_mut()
            .iter_mut()
            .enumerate()
            .for_each(|(index, x)| *x = (IM * d[index]).exp());
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
                * c64(
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
            4. * c64(
                da.cos() * db.cos() * dc.cos(),
                da.sin() * db.sin() * dc.sin(),
            )
        } else {
            let [da, db, dc] = [a - specialized.a, b - specialized.b, c - specialized.c];
            4. * c64(
                da.cos() * db.cos() * dc.cos(),
                da.sin() * db.sin() * dc.sin(),
            )
        };
        specialized.calculated_fidelity = tr.trace_to_fid();
        if let Some(fid) = specialized.requested_fidelity {
            if specialized.calculated_fidelity + 1.0e-13 < fid {
                return Err(QiskitError::new_err(format!(
                    "Specialization: {:?} calculated fidelity: {} is worse than requested fidelity: {}",
                    specialized.specialization, specialized.calculated_fidelity, fid
                )));
            }
        }
        specialized.global_phase += tr.arg();
        Ok(specialized)
    }
}

static IPZ: GateArray1Q = [[IM, C_ZERO], [C_ZERO, M_IM]];
static IPY: GateArray1Q = [[C_ZERO, C_ONE], [C_M_ONE, C_ZERO]];
static IPX: GateArray1Q = [[C_ZERO, IM], [IM, C_ZERO]];

#[pymethods]
impl TwoQubitWeylDecomposition {
    #[staticmethod]
    #[pyo3(signature=(angles, matrices, specialization, default_euler_basis, calculated_fidelity, requested_fidelity=None))]
    fn _from_state(
        angles: [f64; 4],
        matrices: [PyReadonlyArray2<Complex64>; 5],
        specialization: Specialization,
        default_euler_basis: EulerBasis,
        calculated_fidelity: f64,
        requested_fidelity: Option<f64>,
    ) -> Self {
        let [a, b, c, global_phase] = angles;
        Self {
            a,
            b,
            c,
            global_phase,
            K1l: matrices[0].as_array().to_owned(),
            K1r: matrices[1].as_array().to_owned(),
            K2l: matrices[2].as_array().to_owned(),
            K2r: matrices[3].as_array().to_owned(),
            specialization,
            default_euler_basis,
            calculated_fidelity,
            requested_fidelity,
            unitary_matrix: matrices[4].as_array().to_owned(),
        }
    }

    fn __reduce__(&self, py: Python) -> PyResult<Py<PyAny>> {
        (
            py.get_type::<Self>().getattr("_from_state")?,
            (
                [self.a, self.b, self.c, self.global_phase],
                [
                    self.K1l.to_pyarray(py),
                    self.K1r.to_pyarray(py),
                    self.K2l.to_pyarray(py),
                    self.K2r.to_pyarray(py),
                    self.unitary_matrix.to_pyarray(py),
                ],
                self.specialization,
                self.default_euler_basis,
                self.calculated_fidelity,
                self.requested_fidelity,
            ),
        )
            .into_py_any(py)
    }

    #[new]
    #[pyo3(signature=(unitary_matrix, fidelity=DEFAULT_FIDELITY, _specialization=None))]
    pub fn new(
        unitary_matrix: PyReadonlyArray2<Complex64>,
        fidelity: Option<f64>,
        _specialization: Option<Specialization>,
    ) -> PyResult<Self> {
        TwoQubitWeylDecomposition::new_inner(unitary_matrix.as_array(), fidelity, _specialization)
    }

    #[allow(non_snake_case)]
    #[getter]
    pub fn K1l(&self, py: Python) -> Py<PyAny> {
        self.K1l.to_pyarray(py).into_any().unbind()
    }

    #[allow(non_snake_case)]
    #[getter]
    pub fn K1r(&self, py: Python) -> Py<PyAny> {
        self.K1r.to_pyarray(py).into_any().unbind()
    }

    #[allow(non_snake_case)]
    #[getter]
    pub fn K2l(&self, py: Python) -> Py<PyAny> {
        self.K2l.to_pyarray(py).into_any().unbind()
    }

    #[allow(non_snake_case)]
    #[getter]
    pub fn K2r(&self, py: Python) -> Py<PyAny> {
        self.K2r.to_pyarray(py).into_any().unbind()
    }

    #[getter]
    pub fn unitary_matrix(&self, py: Python) -> Py<PyAny> {
        self.unitary_matrix.to_pyarray(py).into_any().unbind()
    }

    #[pyo3(signature = (euler_basis=None, simplify=false, atol=None))]
    fn circuit(
        &self,
        euler_basis: Option<PyBackedStr>,
        simplify: bool,
        atol: Option<f64>,
    ) -> PyResult<PyCircuitData> {
        let euler_basis: EulerBasis = match euler_basis {
            Some(basis) => EulerBasis::__new__(basis.deref())?,
            None => self.default_euler_basis,
        };
        let mut target_1q_basis_list = EulerBasisSet::new();
        target_1q_basis_list.add_basis(euler_basis);

        let mut gate_sequence = CircuitData::with_capacity(2, 0, 21, Param::Float(0.))?;
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
            gate_sequence.push_standard_gate(
                gate.0,
                &gate.1.into_iter().map(Param::Float).collect::<Vec<_>>(),
                &[Qubit(0)],
            )?;
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
            gate_sequence.push_standard_gate(
                gate.0,
                &gate.1.into_iter().map(Param::Float).collect::<Vec<_>>(),
                &[Qubit(1)],
            )?;
        }
        global_phase += c2l.global_phase;
        self.weyl_gate(
            simplify,
            &mut gate_sequence,
            atol.unwrap_or(ANGLE_ZERO_EPSILON),
            &mut global_phase,
        )?;
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
            gate_sequence.push_standard_gate(
                gate.0,
                &gate.1.into_iter().map(Param::Float).collect::<Vec<_>>(),
                &[Qubit(0)],
            )?;
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
            gate_sequence.push_standard_gate(
                gate.0,
                &gate.1.into_iter().map(Param::Float).collect::<Vec<_>>(),
                &[Qubit(1)],
            )?;
        }
        gate_sequence.set_global_phase_f64(global_phase);
        Ok(gate_sequence.into())
    }
}

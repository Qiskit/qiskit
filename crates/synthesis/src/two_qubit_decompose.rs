// This code is part of Qiskit.
//
// (C) Copyright IBM 2023-2025
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

use approx::{abs_diff_eq, relative_eq};
use num_complex::{Complex, Complex64, ComplexFloat};
use num_traits::Zero;
use smallvec::{SmallVec, smallvec};
use std::f64::consts::{FRAC_1_SQRT_2, PI};
use std::ops::Deref;

use faer::Side::Lower;
use faer::{Mat, MatRef, prelude::*};
use faer_ext::IntoFaer;
use nalgebra::{Dim, Matrix2, Matrix4, MatrixView, MatrixView2, MatrixView4, Vector4};
use ndarray::linalg::kron;
use ndarray::prelude::*;
use numpy::{IntoPyArray, ToPyArray};
use numpy::{PyArray2, PyArrayLike2, PyReadonlyArray1, PyReadonlyArray2};

use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::PyType;

use crate::euler_one_qubit_decomposer::{
    ANGLE_ZERO_EPSILON, EulerBasis, EulerBasisSet, OneQubitGateSequence, angles_from_unitary,
    unitary_to_gate_sequence_inner,
};

use crate::QiskitError;
use qiskit_quantum_info::convert_2q_block_matrix::change_basis;

use rand::prelude::*;
use rand_distr::StandardNormal;
use rand_pcg::Pcg64Mcg;

use qiskit_circuit::bit::ShareableQubit;
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::circuit_instruction::OperationFromPython;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::gate_matrix::{CX_GATE, H_GATE, ONE_QUBIT_IDENTITY, S_GATE, SDG_GATE};
use qiskit_circuit::operations::{Operation, OperationRef, Param, StandardGate};
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::util::{C_M_ONE, C_ONE, C_ZERO, IM, M_IM, c64};
use qiskit_circuit::{Qubit, impl_intopyobject_for_copy_pyclass};

const PI2: f64 = PI / 2.;
const PI4: f64 = PI / 4.;
const PI32: f64 = 3.0 * PI2;
const TWO_PI: f64 = 2.0 * PI;
const C1: c64 = c64 { re: 1.0, im: 0.0 };
// Worst case length is 5x 1q gates for each 1q decomposition + 1x 2q gate
// We might overallocate a bit if the euler basis is different but
// the worst case is just 16 extra elements with just a String and 2 smallvecs
// each. This is only transient though as the circuit sequences aren't long lived
// and are just used to create a QuantumCircuit or DAGCircuit when we return to
// Python space.
const TWO_QUBIT_SEQUENCE_DEFAULT_CAPACITY: usize = 21;

static B_NON_NORMALIZED: Matrix4<Complex64> = Matrix4::new(
    C_ONE, IM, C_ZERO, C_ZERO, C_ZERO, C_ZERO, IM, C_ONE, C_ZERO, C_ZERO, IM, C_M_ONE, C_ONE, M_IM,
    C_ZERO, C_ZERO,
);

static B_NON_NORMALIZED_DAGGER: Matrix4<Complex64> = Matrix4::new(
    c64(0.5, 0.),
    C_ZERO,
    C_ZERO,
    c64(0.5, 0.),
    c64(0., -0.5),
    C_ZERO,
    C_ZERO,
    c64(0., 0.5),
    C_ZERO,
    c64(0., -0.5),
    c64(0., -0.5),
    C_ZERO,
    C_ZERO,
    c64(0.5, 0.),
    c64(-0.5, 0.),
    C_ZERO,
);

enum MagicBasisTransform {
    Into,
    OutOf,
}

pub fn ndarray_to_matrix4(mat_view: ArrayView2<Complex64>) -> PyResult<Matrix4<Complex64>> {
    if mat_view.shape() != [4, 4] {
        return Err(QiskitError::new_err(
            "decompose_two_qubit_product_gate: expected a 4x4 matrix",
        ));
    };
    Ok(Matrix4::from_fn(|i, j| mat_view[(i, j)]))
}

fn ndarray_to_matrix2(mat_view: ArrayView2<Complex64>) -> PyResult<Matrix2<Complex64>> {
    if mat_view.shape() != [2, 2] {
        return Err(QiskitError::new_err(
            "decompose_two_qubit_product_gate: expected a 2x2 matrix",
        ));
    };
    Ok(Matrix2::from_fn(|i, j| mat_view[(i, j)]))
}

fn faer_to_matrix4<T>(mat_view: MatRef<T>) -> PyResult<Matrix4<Complex64>>
where
    T: Copy + Into<Complex64>,
{
    debug_assert!(
        mat_view.ncols() != 4 || mat_view.nrows() != 4,
        "decompose_two_qubit_product_gate: expected a 4x4 matrix",
    );
    Ok(Matrix4::from_fn(|i, j| mat_view[(i, j)].into()))
}

fn faer_from_matrix4(mat: MatrixView4<Complex64>) -> MatRef<c64> {
    nalgebra_to_ndarray(mat).into_faer()
}

fn nalgebra_to_ndarray<R: Dim, C: Dim, RStride: Dim, CStride: Dim>(
    mat: MatrixView<Complex64, R, C, RStride, CStride>,
) -> ArrayView2<Complex64> {
    let dim = ::ndarray::Dim(mat.shape());
    let strides = ::ndarray::Dim(mat.strides());

    // SAFETY: We know the array is a 2d array in a contiguous block (DMatrix uses a vec for backing storage) so we
    // don't need to check for invalid format
    unsafe { ArrayView2::from_shape_ptr(dim.strides(strides), mat.get_unchecked(0)) }
}

fn magic_basis_transform(
    unitary: Matrix4<Complex64>,
    direction: MagicBasisTransform,
) -> Matrix4<Complex64> {
    match direction {
        MagicBasisTransform::OutOf => B_NON_NORMALIZED_DAGGER * unitary * B_NON_NORMALIZED,
        MagicBasisTransform::Into => B_NON_NORMALIZED * unitary * B_NON_NORMALIZED_DAGGER,
    }
}

// TODO: test, generalize
fn kron_2q(a: Matrix2<Complex64>, b: Matrix2<Complex64>) -> Matrix4<Complex64> {
    Matrix4::from_fn(|i, j| {
        let ai = i / 2;
        let aj = j / 2;
        let bi = i % 2;
        let bj = j % 2;
        a[(ai, aj)] * b[(bi, bj)]
    })
}

// TODO: we can remove this?
// fn transform_from_magic_basis(u: Matrix4<Complex64>) -> Matrix4<Complex64> {
//     let unitary: ArrayView2<Complex64> = u.as_ref().into_ndarray_complex();
//     magic_basis_transform(unitary, MagicBasisTransform::OutOf)
//         .view()
//         .into_faer_complex()
//         .to_owned()
// }

// faer::c64 and num_complex::Complex<f64> are both structs
// holding two f64's. But several functions are not defined for
// c64. So we implement them here. These things should be contribute
// upstream.

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

#[pyfunction]
#[pyo3(name = "trace_to_fid")]
/// Average gate fidelity is :math:`Fbar = (d + |Tr (Utarget \\cdot U^dag)|^2) / d(d+1)`
/// M. Horodecki, P. Horodecki and R. Horodecki, PRA 60, 1888 (1999)
fn py_trace_to_fid(trace: Complex64) -> PyResult<f64> {
    let fid = trace.trace_to_fid();
    Ok(fid)
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
    special_unitary: Matrix4<Complex64>,
) -> PyResult<(Matrix2<Complex64>, Matrix2<Complex64>, f64)> {
    let mut r = special_unitary.fixed_view::<2, 2>(0, 0);
    let mut det_r = r.determinant();
    if det_r.abs() < 0.1 {
        r = special_unitary.fixed_view::<2, 2>(2, 0);
        det_r = r.determinant();
    }
    if det_r.abs() < 0.1 {
        return Err(QiskitError::new_err(
            "decompose_two_qubit_product_gate: unable to decompose: detR < 0.1",
        ));
    }
    let normalized_r: Matrix2<Complex64> = r.map(|x| x / det_r.sqrt());
    let n_r_adjoint = normalized_r.adjoint();

    let eye = Matrix2::identity();
    let mut temp = kron_2q(eye, n_r_adjoint);
    temp = special_unitary * temp;
    let l = Matrix2::from_fn(|i, j| temp[(i * 2, j * 2)]);
    let det_l = l.determinant();
    if det_l.abs() < 0.9 {
        return Err(QiskitError::new_err(
            "decompose_two_qubit_product_gate: unable to decompose: detL < 0.9",
        ));
    }
    let normalized_l = l.map(|x| x / det_l.sqrt());
    let phase = det_l.arg() / 2.;

    Ok((normalized_l, normalized_r, phase))
}

#[pyfunction]
#[pyo3(name = "decompose_two_qubit_product_gate")]
/// Decompose :math:`U = U_l \otimes U_r` where :math:`U \in SU(4)`,
/// and :math:`U_l,~U_r \in SU(2)`.
/// Args:
///     special_unitary_matrix: special unitary matrix to decompose
/// Raises:
///     QiskitError: if decomposition isn't possible.
fn py_decompose_two_qubit_product_gate(
    py: Python,
    special_unitary: PyArrayLike2<Complex64>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, f64)> {
    // let view = special_unitary.as_array();
    // let (l, r, phase) = decompose_two_qubit_product_gate(view)?;
    let (l, r, phase) =
        decompose_two_qubit_product_gate(ndarray_to_matrix4(special_unitary.as_array())?)?;
    Ok((
        l.to_pyarray(py).into_any().unbind(),
        r.to_pyarray(py).into_any().unbind(),
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
fn weyl_coordinates(py: Python, unitary: PyReadonlyArray2<Complex64>) -> PyResult<Py<PyAny>> {
    let array = ndarray_to_matrix4(unitary.as_array()).unwrap();
    Ok(__weyl_coordinates(array)?
        .to_vec()
        .into_pyarray(py)
        .into_any()
        .unbind())
}

fn __weyl_coordinates(unitary: Matrix4<Complex64>) -> PyResult<[f64; 3]> {
    let unitary = faer_to_matrix4(faer_from_matrix4(unitary.as_view()))?;
    let unscaled = unitary * (C1 / unitary.determinant().powf(0.25));
    let uup = magic_basis_transform(unscaled, MagicBasisTransform::OutOf);
    let uup = faer_from_matrix4(uup.as_view());
    let mut darg: Vec<_> = (uup.transpose() * uup)
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
    let u = ndarray_to_matrix4(unitary.as_array()).unwrap();
    __num_basis_gates(basis_b, basis_fidelity, u)
}

fn __num_basis_gates(
    basis_b: f64,
    basis_fidelity: f64,
    unitary: Matrix4<Complex64>,
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

/// A good approximation to the best value x to get the minimum
/// trace distance for :math:`U_d(x, x, x)` from :math:`U_d(a, b, c)`.
fn closest_partial_swap(a: f64, b: f64, c: f64) -> f64 {
    let m = (a + b + c) / 3.;
    let [am, bm, cm] = [a - m, b - m, c - m];
    let [ab, bc, ca] = [a - b, b - c, c - a];
    m + am * bm * cm * (6. + ab * ab + bc * bc + ca * ca) / 18.
}

fn rx_matrix(theta: f64) -> Matrix2<Complex64> {
    let half_theta = theta / 2.;
    let cos = c64(half_theta.cos(), 0.);
    let isin = c64(0., -half_theta.sin());
    Matrix2::from_row_slice(&[cos, isin, isin, cos])
}

fn ry_matrix(theta: f64) -> Matrix2<Complex64> {
    let half_theta = theta / 2.;
    let cos = c64(half_theta.cos(), 0.);
    let sin = c64(half_theta.sin(), 0.);
    Matrix2::from_row_slice(&[cos, -sin, sin, cos])
}

fn rz_matrix(theta: f64) -> Matrix2<Complex64> {
    let ilam2 = c64(0., 0.5 * theta);
    Matrix2::from_row_slice(&[(-ilam2).exp(), C_ZERO, C_ZERO, ilam2.exp()])
}

/// Generates the array :math:`e^{(i a XX + i b YY + i c ZZ)}`
fn ud(a: f64, b: f64, c: f64) -> Array2<Complex64> {
    array![
        [
            (IM * c).exp() * (a - b).cos(),
            C_ZERO,
            C_ZERO,
            IM * (IM * c).exp() * (a - b).sin()
        ],
        [
            C_ZERO,
            (M_IM * c).exp() * (a + b).cos(),
            IM * (M_IM * c).exp() * (a + b).sin(),
            C_ZERO
        ],
        [
            C_ZERO,
            IM * (M_IM * c).exp() * (a + b).sin(),
            (M_IM * c).exp() * (a + b).cos(),
            C_ZERO
        ],
        [
            IM * (IM * c).exp() * (a - b).sin(),
            C_ZERO,
            C_ZERO,
            (IM * c).exp() * (a - b).cos()
        ]
    ]
}

#[pyfunction]
#[pyo3(name = "Ud")]
fn py_ud(py: Python, a: f64, b: f64, c: f64) -> Py<PyArray2<Complex64>> {
    let ud_mat = ud(a, b, c);
    ud_mat.into_pyarray(py).unbind()
}

fn compute_unitary(sequence: &TwoQubitSequenceVec, global_phase: f64) -> Array2<Complex64> {
    let identity = aview2(&ONE_QUBIT_IDENTITY);
    let phase = c64(0., global_phase).exp();
    let mut matrix = Array2::from_diag(&arr1(&[phase, phase, phase, phase]));
    sequence
        .iter()
        .map(|inst| {
            // This only gets called by get_sx_vz_3cx_efficient_euler()
            // which only uses sx, x, rz, and cx gates for the circuit
            // sequence. If we get a different gate this is getting called
            // by something else and is invalid.
            let gate_matrix = inst
                .0
                .matrix(&inst.1.iter().map(|x| Param::Float(*x)).collect::<Vec<_>>())
                .unwrap();
            (gate_matrix, &inst.2)
        })
        .for_each(|(op_matrix, q_list)| {
            let result = match q_list.as_slice() {
                [0] => Some(kron(&identity, &op_matrix)),
                [1] => Some(kron(&op_matrix, &identity)),
                [1, 0] => Some(change_basis(op_matrix.view())),
                [] => Some(Array2::eye(4)),
                _ => None,
            };
            matrix = match result {
                Some(result) => result.dot(&matrix),
                None => op_matrix.dot(&matrix),
            }
        });
    matrix
}

const DEFAULT_FIDELITY: f64 = 1.0 - 1.0e-9;

#[derive(Clone, Debug, Copy, PartialEq, Eq)]
#[pyclass(module = "qiskit._accelerate.two_qubit_decompose", eq)]
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
#[pyclass(module = "qiskit._accelerate.two_qubit_decompose", subclass)]
pub struct TwoQubitWeylDecomposition {
    #[pyo3(get)]
    a: f64,
    #[pyo3(get)]
    b: f64,
    #[pyo3(get)]
    c: f64,
    #[pyo3(get)]
    pub global_phase: f64,
    K1l: Matrix2<Complex64>,
    K2l: Matrix2<Complex64>,
    K1r: Matrix2<Complex64>,
    K2r: Matrix2<Complex64>,
    #[pyo3(get)]
    pub specialization: Specialization,
    default_euler_basis: EulerBasis,
    #[pyo3(get)]
    requested_fidelity: Option<f64>,
    #[pyo3(get)]
    calculated_fidelity: f64,
    unitary_matrix: Matrix4<Complex64>,
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

    pub fn k1l(&self) -> Matrix2<Complex64> {
        self.K1l
    }

    pub fn k2l(&self) -> Matrix2<Complex64> {
        self.K2l
    }

    pub fn k1r(&self) -> Matrix2<Complex64> {
        self.K1r
    }

    pub fn k2r(&self) -> Matrix2<Complex64> {
        self.K2r
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
        unitary_matrix: Matrix4<Complex64>,

        fidelity: Option<f64>,
        _specialization: Option<Specialization>,
    ) -> PyResult<Self> {
        let det_u = unitary_matrix.determinant();
        let det_pow = det_u.powf(-0.25);
        let u = unitary_matrix.map(|x| x * det_pow);
        let mut global_phase = det_u.arg() / 4.;
        let u_p = magic_basis_transform(u, MagicBasisTransform::OutOf);
        let m2 = u_p.tr_mul(&u_p);
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
        let mut d = Vector4::<Complex64>::zeros();
        let mut p = Matrix4::<Complex64>::zeros();
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
            let m2_real = m2.map(|val| rand_a * val.re + rand_b * val.im);
            // TODO: get rid of faer if possible
            let faer_mat: Mat<f64> = Mat::from_fn(4, 4, |i, j| m2_real[(i, j)]);
            let decomposition = faer_mat
                .self_adjoint_eigen(Lower)
                .map_err(|e| QiskitError::new_err(format!("{e:?}")))?;
            let p_inner_faer = decomposition.U();
            // let p_inner_faer = faer_mat
            //     .selfadjoint_eigendecomposition(Lower)
            //     .u();
            let p_inner = faer_to_matrix4(p_inner_faer)?;
            let d_inner = (p_inner.tr_mul(&m2) * p_inner).diagonal();
            // compare = p_inner * d_inner * p_inner.T
            let compare =
                Matrix4::from_fn(|i, j| p_inner[(i, j)] * d_inner[j]) * p_inner.transpose();
            found = abs_diff_eq!(compare, m2, epsilon = 1.0e-13);
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

        // let mut p_orig = p.clone();
        // for (i, item) in order.iter().enumerate().take(3) {
        //     let slice_a = p.slice_mut(s![.., i]);
        //     let slice_b = p_orig.slice_mut(s![.., *item]);
        //     Zip::from(slice_a).and(slice_b).for_each(::std::mem::swap);
        // }
        let mut permuted_p = Matrix4::from_columns(&[
            Vector4::from(p.column(order[0])),
            Vector4::from(p.column(order[1])),
            Vector4::from(p.column(order[2])),
            Vector4::from(p.column(3)),
        ]);
        if permuted_p.determinant().re < 0. {
            permuted_p.column_mut(3).scale_mut(-1.);
        }
        let temp = Matrix4::from_diagonal(&Vector4::from_fn(|index, _| (IM * d[index]).exp()));
        // let mut temp: Array2<Complex64> = Array2::zeros((4, 4));
        // temp.diag_mut()
        //     .iter_mut()
        //     .enumerate()
        //     .for_each(|(index, x)| *x = (IM * d[index]).exp());
        let k1 = magic_basis_transform(u_p * permuted_p * temp, MagicBasisTransform::Into);
        let k2 = magic_basis_transform(permuted_p.transpose(), MagicBasisTransform::Into);

        #[allow(non_snake_case)]
        let (mut K1l, mut K1r, phase_l) = decompose_two_qubit_product_gate(k1)?;
        #[allow(non_snake_case)]
        let (K2l, mut K2r, phase_r) = decompose_two_qubit_product_gate(k2)?;
        global_phase += phase_l + phase_r;

        // Flip into Weyl chamber
        if cs[0] > PI2 {
            cs[0] -= PI32;
            K1l *= IPY;
            K1r *= IPY;
            global_phase += PI2;
        }
        if cs[1] > PI2 {
            cs[1] -= PI32;
            K1l *= IPX;
            K1r *= IPX;
            global_phase += PI2;
        }
        let mut conjs = 0;
        if cs[0] > PI4 {
            cs[0] = PI2 - cs[0];
            K1l *= IPY;
            K2r = IPY * K2r;
            conjs += 1;
            global_phase -= PI2;
        }
        if cs[1] > PI4 {
            cs[1] = PI2 - cs[1];
            K1l *= IPX;
            K2r = IPX * K2r;
            conjs += 1;
            global_phase += PI2;
            if conjs == 1 {
                global_phase -= PI;
            }
        }
        if cs[2] > PI2 {
            cs[2] -= PI32;
            K1l *= IPZ;
            K1r *= IPZ;
            global_phase += PI2;
            if conjs == 1 {
                global_phase -= PI;
            }
        }
        if conjs == 1 {
            cs[2] = PI2 - cs[2];
            K1l *= IPZ;
            K2r = IPZ * K2r;
            global_phase += PI2;
        }
        if cs[2] > PI4 {
            cs[2] -= PI2;
            K1l *= IPZ;
            K1r *= IPZ;
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
                K1l: general.K1l * general.K2l,
                K1r: general.K1r * general.K2r,
                K2l: Matrix2::identity(),
                K2r: Matrix2::identity(),
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
                        K1l: general.K1l * general.K2r,
                        K1r: general.K1r * general.K2l,
                        K2l: Matrix2::identity(),
                        K2r: Matrix2::identity(),
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
                        K1l: general.K1l * IPZ * general.K2r,
                        K1r: general.K1r * IPZ * general.K2l,
                        K2l: Matrix2::identity(),
                        K2r: Matrix2::identity(),
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
                TwoQubitWeylDecomposition {
                    specialization,
                    a: closest,
                    b: closest,
                    c: closest,
                    K1l: general.K1l * general.K2l,
                    K1r: general.K1r * general.K2l,
                    K2r: general.K2l.adjoint() * general.K2r,
                    K2l: Matrix2::identity(),
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
                TwoQubitWeylDecomposition {
                    specialization,
                    a: closest,
                    b: closest,
                    c: -closest,
                    K1l: general.K1l * general.K2l,
                    K1r: general.K1r * IPZ * general.K2l * IPZ,
                    K2r: IPZ * general.K2l.adjoint() * IPZ * general.K2r,
                    K2l: Matrix2::identity(),
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
                let k2l_view: MatrixView2<Complex64> = general.K2l.as_view();
                let [k2ltheta, k2lphi, k2llambda, k2lphase] =
                    angles_from_unitary(nalgebra_to_ndarray(k2l_view), euler_basis);
                let k2r_view: MatrixView2<Complex64> = general.K2r.as_view();
                let [k2rtheta, k2rphi, k2rlambda, k2rphase] =
                    angles_from_unitary(nalgebra_to_ndarray(k2r_view), euler_basis);
                TwoQubitWeylDecomposition {
                    specialization,
                    a,
                    b: 0.,
                    c: 0.,
                    global_phase: global_phase + k2lphase + k2rphase,
                    K1l: general.K1l * rx_matrix(k2lphi),
                    K1r: general.K1r * rx_matrix(k2rphi),
                    K2l: ry_matrix(k2ltheta) * rx_matrix(k2llambda),
                    K2r: ry_matrix(k2rtheta) * rx_matrix(k2rlambda),
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
                let k2l_view: MatrixView2<Complex64> = general.K2l.as_view();
                let [k2ltheta, k2lphi, k2llambda, k2lphase] =
                    angles_from_unitary(nalgebra_to_ndarray(k2l_view), EulerBasis::ZYZ);
                let k2r_view: MatrixView2<Complex64> = general.K2r.as_view();
                let [k2rtheta, k2rphi, k2rlambda, k2rphase] =
                    angles_from_unitary(nalgebra_to_ndarray(k2r_view), EulerBasis::ZYZ);
                TwoQubitWeylDecomposition {
                    specialization,
                    a: PI4,
                    b: PI4,
                    c,
                    global_phase: global_phase + k2lphase + k2rphase,
                    K1l: general.K1l * rz_matrix(k2rphi),
                    K1r: general.K1r * rz_matrix(k2lphi),
                    K2l: ry_matrix(k2ltheta) * rz_matrix(k2llambda),
                    K2r: ry_matrix(k2rtheta) * rz_matrix(k2rlambda),
                    ..general
                }
            }
            // :math:`U \sim U_d(\alpha, \alpha, \beta), \alpha \geq |\beta|`
            //
            // This gate binds 5 parameters, we make it canonical by setting:
            //
            // :math:`K2_l = Ry(\theta_l)\cdot Rz(\lambda_l)`.
            Specialization::fSimaabEquiv => {
                let k2l_view: MatrixView2<Complex64> = general.K2l.as_view();
                let [k2ltheta, k2lphi, k2llambda, k2lphase] =
                    angles_from_unitary(nalgebra_to_ndarray(k2l_view), EulerBasis::ZYZ);
                TwoQubitWeylDecomposition {
                    specialization,
                    a: (a + b) / 2.,
                    b: (a + b) / 2.,
                    c,
                    global_phase: global_phase + k2lphase,
                    K1r: general.K1r * rz_matrix(k2lphi),
                    K1l: general.K1l * rz_matrix(k2lphi),
                    K2l: ry_matrix(k2ltheta) * rz_matrix(k2llambda),
                    K2r: rz_matrix(-k2lphi) * general.K2r,
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
                let k2l_view: MatrixView2<Complex64> = general.K2l.as_view();
                let [k2ltheta, k2lphi, k2llambda, k2lphase] =
                    angles_from_unitary(nalgebra_to_ndarray(k2l_view), euler_basis);
                TwoQubitWeylDecomposition {
                    specialization,
                    a,
                    b: (b + c) / 2.,
                    c: (b + c) / 2.,
                    global_phase: global_phase + k2lphase,
                    K1r: general.K1r * rx_matrix(k2lphi),
                    K1l: general.K1l * rx_matrix(k2lphi),
                    K2l: ry_matrix(k2ltheta) * rx_matrix(k2llambda),
                    K2r: rx_matrix(-k2lphi) * general.K2r,
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
                let k2l_view: MatrixView2<Complex64> = general.K2l.as_view();
                let [k2ltheta, k2lphi, k2llambda, k2lphase] =
                    angles_from_unitary(nalgebra_to_ndarray(k2l_view), euler_basis);
                TwoQubitWeylDecomposition {
                    specialization,
                    a,
                    b: (b - c) / 2.,
                    c: -((b - c) / 2.),
                    global_phase: global_phase + k2lphase,
                    K1l: general.K1l * rx_matrix(k2lphi),
                    K1r: general.K1r * IPZ * rx_matrix(k2lphi) * IPZ,
                    K2l: ry_matrix(k2ltheta) * rx_matrix(k2llambda),
                    K2r: IPZ * rx_matrix(-k2lphi) * IPZ * general.K2r,
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

static IPZ: Matrix2<Complex64> = Matrix2::new(IM, C_ZERO, C_ZERO, M_IM);
static IPY: Matrix2<Complex64> = Matrix2::new(C_ZERO, C_ONE, C_M_ONE, C_ZERO);
static IPX: Matrix2<Complex64> = Matrix2::new(C_ZERO, IM, IM, C_ZERO);
static H_GATE_MATRIX: Matrix2<Complex64> =
    Matrix2::new(H_GATE[0][0], H_GATE[0][1], H_GATE[1][0], H_GATE[1][1]);

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
            K1l: ndarray_to_matrix2(matrices[0].as_array()).unwrap(),
            K1r: ndarray_to_matrix2(matrices[1].as_array()).unwrap(),
            K2l: ndarray_to_matrix2(matrices[2].as_array()).unwrap(),
            K2r: ndarray_to_matrix2(matrices[3].as_array()).unwrap(),
            specialization,
            default_euler_basis,
            calculated_fidelity,
            requested_fidelity,
            unitary_matrix: ndarray_to_matrix4(matrices[4].as_array()).unwrap(),
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
    fn new(
        unitary_matrix: PyReadonlyArray2<Complex64>,
        fidelity: Option<f64>,
        _specialization: Option<Specialization>,
    ) -> PyResult<Self> {
        TwoQubitWeylDecomposition::new_inner(
            ndarray_to_matrix4(unitary_matrix.as_array())?,
            fidelity,
            _specialization,
        )
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
    fn K2l(&self, py: Python) -> Py<PyAny> {
        self.K2l.to_pyarray(py).into_any().unbind()
    }

    #[allow(non_snake_case)]
    #[getter]
    fn K2r(&self, py: Python) -> Py<PyAny> {
        self.K2r.to_pyarray(py).into_any().unbind()
    }

    #[getter]
    fn unitary_matrix(&self, py: Python) -> Py<PyAny> {
        self.unitary_matrix.to_pyarray(py).into_any().unbind()
    }

    #[pyo3(signature = (euler_basis=None, simplify=false, atol=None))]
    fn circuit(
        &self,
        euler_basis: Option<PyBackedStr>,
        simplify: bool,
        atol: Option<f64>,
    ) -> PyResult<CircuitData> {
        let euler_basis: EulerBasis = match euler_basis {
            Some(basis) => EulerBasis::__new__(basis.deref())?,
            None => self.default_euler_basis,
        };
        let mut target_1q_basis_list = EulerBasisSet::new();
        target_1q_basis_list.add_basis(euler_basis);

        let mut gate_sequence = CircuitData::with_capacity(2, 0, 21, Param::Float(0.))?;
        let mut global_phase: f64 = self.global_phase;

        let k2r_view: MatrixView2<Complex64> = self.K2r.as_view();
        let c2r = unitary_to_gate_sequence_inner(
            nalgebra_to_ndarray(k2r_view),
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
        let k2l_view: MatrixView2<Complex64> = self.K2l.as_view();
        let c2l = unitary_to_gate_sequence_inner(
            nalgebra_to_ndarray(k2l_view),
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
        let k1r_view: MatrixView2<Complex64> = self.K1r.as_view();
        let c1r = unitary_to_gate_sequence_inner(
            nalgebra_to_ndarray(k1r_view),
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
        let k1l_view: MatrixView2<Complex64> = self.K1l.as_view();
        let c1l = unitary_to_gate_sequence_inner(
            nalgebra_to_ndarray(k1l_view),
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
        gate_sequence.set_global_phase(Param::Float(global_phase))?;
        Ok(gate_sequence)
    }
}

type TwoQubitSequenceVec = Vec<(PackedOperation, SmallVec<[f64; 3]>, SmallVec<[u8; 2]>)>;

#[derive(Clone, Debug)]
pub struct TwoQubitGateSequence {
    gates: TwoQubitSequenceVec,
    global_phase: f64,
}

impl TwoQubitGateSequence {
    pub fn gates(&self) -> &TwoQubitSequenceVec {
        &self.gates
    }

    pub fn into_gates(self) -> TwoQubitSequenceVec {
        self.gates
    }

    pub fn global_phase(&self) -> f64 {
        self.global_phase
    }

    pub fn set_state(&mut self, state: (TwoQubitSequenceVec, f64)) {
        self.gates = state.0;
        self.global_phase = state.1;
    }

    pub fn new() -> Self {
        TwoQubitGateSequence {
            gates: Vec::new(),
            global_phase: 0.,
        }
    }
}

impl Default for TwoQubitGateSequence {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug)]
#[allow(non_snake_case)]
#[pyclass(module = "qiskit._accelerate.two_qubit_decompose", subclass)]
pub struct TwoQubitBasisDecomposer {
    gate: PackedOperation,
    gate_params: SmallVec<[f64; 3]>,
    basis_fidelity: f64,
    euler_basis: EulerBasis,
    pulse_optimize: Option<bool>,
    basis_decomposer: TwoQubitWeylDecomposition,
    #[pyo3(get)]
    super_controlled: bool,
    u0l: Matrix2<Complex64>,
    u0r: Matrix2<Complex64>,
    u1l: Matrix2<Complex64>,
    u1ra: Matrix2<Complex64>,
    u1rb: Matrix2<Complex64>,
    u2la: Matrix2<Complex64>,
    u2lb: Matrix2<Complex64>,
    u2ra: Matrix2<Complex64>,
    u2rb: Matrix2<Complex64>,
    u3l: Matrix2<Complex64>,
    u3r: Matrix2<Complex64>,
    q0l: Matrix2<Complex64>,
    q0r: Matrix2<Complex64>,
    q1la: Matrix2<Complex64>,
    q1lb: Matrix2<Complex64>,
    q1ra: Matrix2<Complex64>,
    q1rb: Matrix2<Complex64>,
    q2l: Matrix2<Complex64>,
    q2r: Matrix2<Complex64>,
}
impl TwoQubitBasisDecomposer {
    /// Return the KAK gate name
    pub fn gate_name(&self) -> &str {
        self.gate.name()
    }

    /// Compute the number of basis gates needed for a given unitary
    pub fn num_basis_gates_inner(&self, unitary: Matrix4<Complex64>) -> PyResult<usize> {
        __num_basis_gates(self.basis_decomposer.b, self.basis_fidelity, unitary)
    }

    fn decomp1_inner(
        &self,
        target: &TwoQubitWeylDecomposition,
    ) -> SmallVec<[Matrix2<Complex64>; 8]> {
        // FIXME: fix for z!=0 and c!=0 using closest reflection (not always in the Weyl chamber)
        smallvec![
            self.basis_decomposer.K2r.ad_mul(&target.K2r),
            self.basis_decomposer.K2l.ad_mul(&target.K2l),
            target.K1r * self.basis_decomposer.K1r.adjoint(),
            target.K1l * self.basis_decomposer.K1l.adjoint()
        ]
    }

    fn decomp2_supercontrolled_inner(
        &self,
        target: &TwoQubitWeylDecomposition,
    ) -> SmallVec<[Matrix2<Complex64>; 8]> {
        smallvec![
            self.q2r * target.K2r,
            self.q2l * target.K2l,
            self.q1ra * rz_matrix(2. * target.b) * self.q1rb,
            self.q1la * rz_matrix(-2. * target.a) * self.q1lb,
            target.K1r * self.q0r,
            target.K1l * self.q0l,
        ]
    }

    fn decomp3_supercontrolled_inner(
        &self,
        target: &TwoQubitWeylDecomposition,
    ) -> SmallVec<[Matrix2<Complex64>; 8]> {
        smallvec![
            self.u3r * target.K2r,
            self.u3l * target.K2l,
            self.u2ra * rz_matrix(2. * target.b) * self.u2rb,
            self.u2la * rz_matrix(-2. * target.a) * self.u2lb,
            self.u1ra * rz_matrix(-2. * target.c) * self.u1rb,
            self.u1l,
            target.K1r * self.u0r,
            target.K1l * self.u0l,
        ]
    }

    /// Decomposition of SU(4) gate for device with SX, virtual RZ, and CNOT gates assuming
    /// two CNOT gates are needed.
    ///
    /// This first decomposes each unitary from the KAK decomposition into ZXZ on the source
    /// qubit of the CNOTs and XZX on the targets in order to commute operators to beginning and
    /// end of decomposition. The beginning and ending single qubit gates are then
    /// collapsed and re-decomposed with the single qubit decomposer. This last step could be avoided
    /// if performance is a concern.
    fn get_sx_vz_2cx_efficient_euler(
        &self,
        decomposition: &SmallVec<[Matrix2<Complex64>; 8]>,
        target_decomposed: &TwoQubitWeylDecomposition,
        use_xgate: bool,
    ) -> Option<TwoQubitGateSequence> {
        let mut gates = Vec::new();
        let mut global_phase = target_decomposed.global_phase;
        global_phase -= 2. * self.basis_decomposer.global_phase;
        let euler_q0: Vec<[f64; 3]> = decomposition
            .iter()
            .step_by(2)
            .map(|decomp| {
                let decomp_view: MatrixView2<Complex64> = decomp.as_view();
                let euler_angles =
                    angles_from_unitary(nalgebra_to_ndarray(decomp_view), EulerBasis::ZXZ);
                global_phase += euler_angles[3];
                [euler_angles[2], euler_angles[0], euler_angles[1]]
            })
            .collect();
        let euler_q1: Vec<[f64; 3]> = decomposition
            .iter()
            .skip(1)
            .step_by(2)
            .map(|decomp| {
                let decomp_view: MatrixView2<Complex64> = decomp.as_view();
                let euler_angles =
                    angles_from_unitary(nalgebra_to_ndarray(decomp_view), EulerBasis::XZX);
                global_phase += euler_angles[3];
                [euler_angles[2], euler_angles[0], euler_angles[1]]
            })
            .collect();
        let mut euler_matrix_q0 = rx_matrix(euler_q0[0][1]) * rz_matrix(euler_q0[0][0]);
        euler_matrix_q0 = rz_matrix(euler_q0[0][2] + euler_q0[1][0] + PI2) * euler_matrix_q0;
        self.append_1q_sequence(&mut gates, &mut global_phase, euler_matrix_q0, 0);
        let mut euler_matrix_q1 = rz_matrix(euler_q1[0][1]) * rx_matrix(euler_q1[0][0]);
        euler_matrix_q1 = rx_matrix(euler_q1[0][2] + euler_q1[1][0]) * euler_matrix_q1;
        self.append_1q_sequence(&mut gates, &mut global_phase, euler_matrix_q1, 1);
        gates.push((StandardGate::CX.into(), smallvec![], smallvec![0, 1]));
        if (euler_q0[1][1] - PI).abs() < ANGLE_ZERO_EPSILON {
            if use_xgate {
                gates.push((StandardGate::X.into(), smallvec![], smallvec![0]));
            } else {
                gates.push((StandardGate::SX.into(), smallvec![], smallvec![0]));
                gates.push((StandardGate::SX.into(), smallvec![], smallvec![0]));
            }
        } else {
            gates.push((StandardGate::SX.into(), smallvec![], smallvec![0]));
            gates.push((
                StandardGate::RZ.into(),
                smallvec![euler_q0[1][1] - PI],
                smallvec![0],
            ));
            gates.push((StandardGate::SX.into(), smallvec![], smallvec![0]));
        }
        gates.push((
            StandardGate::RZ.into(),
            smallvec![euler_q1[1][1]],
            smallvec![1],
        ));
        global_phase += PI2;
        gates.push((StandardGate::CX.into(), smallvec![], smallvec![0, 1]));
        let mut euler_matrix_q0 =
            rx_matrix(euler_q0[2][1]) * rz_matrix(euler_q0[1][2] + euler_q0[2][0] + PI2);
        euler_matrix_q0 = rz_matrix(euler_q0[2][2]) * euler_matrix_q0;
        self.append_1q_sequence(&mut gates, &mut global_phase, euler_matrix_q0, 0);
        let mut euler_matrix_q1 =
            rz_matrix(euler_q1[2][1]) * rx_matrix(euler_q1[1][2] + euler_q1[2][0]);
        euler_matrix_q1 = rx_matrix(euler_q1[2][2]) * euler_matrix_q1;
        self.append_1q_sequence(&mut gates, &mut global_phase, euler_matrix_q1, 1);
        Some(TwoQubitGateSequence {
            gates,
            global_phase,
        })
    }

    /// Decomposition of SU(4) gate for device with SX, virtual RZ, and CNOT gates assuming
    /// three CNOT gates are needed.
    ///
    /// This first decomposes each unitary from the KAK decomposition into ZXZ on the source
    /// qubit of the CNOTs and XZX on the targets in order commute operators to beginning and
    /// end of decomposition. Inserting Hadamards reverses the direction of the CNOTs and transforms
    /// a variable Rx -> variable virtual Rz. The beginning and ending single qubit gates are then
    /// collapsed and re-decomposed with the single qubit decomposer. This last step could be avoided
    /// if performance is a concern.
    fn get_sx_vz_3cx_efficient_euler(
        &self,
        decomposition: &SmallVec<[Matrix2<Complex64>; 8]>,
        target_decomposed: &TwoQubitWeylDecomposition,
    ) -> Option<TwoQubitGateSequence> {
        let mut gates = Vec::new();
        let mut global_phase = target_decomposed.global_phase;
        global_phase -= 3. * self.basis_decomposer.global_phase;
        global_phase = global_phase.rem_euclid(TWO_PI);
        let atol = 1e-10; // absolute tolerance for floats
        // Decompose source unitaries to zxz
        let euler_q0: Vec<[f64; 3]> = decomposition
            .iter()
            .step_by(2)
            .map(|decomp| {
                let decomp_view: MatrixView2<Complex64> = decomp.as_view();
                let euler_angles =
                    angles_from_unitary(nalgebra_to_ndarray(decomp_view), EulerBasis::ZXZ);
                global_phase += euler_angles[3];
                [euler_angles[2], euler_angles[0], euler_angles[1]]
            })
            .collect();
        // Decompose target unitaries to xzx
        let euler_q1: Vec<[f64; 3]> = decomposition
            .iter()
            .skip(1)
            .step_by(2)
            .map(|decomp| {
                let decomp_view: MatrixView2<Complex64> = decomp.as_view();
                let euler_angles =
                    angles_from_unitary(nalgebra_to_ndarray(decomp_view), EulerBasis::XZX);
                global_phase += euler_angles[3];
                [euler_angles[2], euler_angles[0], euler_angles[1]]
            })
            .collect();

        let x12 = euler_q0[1][2] + euler_q0[2][0];
        let x12_is_non_zero = !abs_diff_eq!(x12, 0., epsilon = atol);
        let mut x12_is_old_mult = None;
        let mut x12_phase = 0.;
        let x12_is_pi_mult = abs_diff_eq!(x12.sin(), 0., epsilon = atol);
        if x12_is_pi_mult {
            x12_is_old_mult = Some(abs_diff_eq!(x12.cos(), -1., epsilon = atol));
            x12_phase = PI * x12.cos();
        }
        let x02_add = x12 - euler_q0[1][0];
        let x12_is_half_pi = abs_diff_eq!(x12, PI2, epsilon = atol);

        let mut euler_matrix_q0 = rx_matrix(euler_q0[0][1]) * rz_matrix(euler_q0[0][0]);
        if x12_is_non_zero && x12_is_pi_mult {
            euler_matrix_q0 = rz_matrix(euler_q0[0][2] - x02_add) * euler_matrix_q0;
        } else {
            euler_matrix_q0 = rz_matrix(euler_q0[0][2] + euler_q0[1][0]) * euler_matrix_q0;
        }
        euler_matrix_q0 = H_GATE_MATRIX * euler_matrix_q0;
        self.append_1q_sequence(&mut gates, &mut global_phase, euler_matrix_q0, 0);

        let rx_0 = rx_matrix(euler_q1[0][0]);
        let rz = rz_matrix(euler_q1[0][1]);
        let rx_1 = rx_matrix(euler_q1[0][2] + euler_q1[1][0]);
        let mut euler_matrix_q1 = rz * rx_0;
        euler_matrix_q1 = rx_1 * euler_matrix_q1;
        euler_matrix_q1 = H_GATE_MATRIX * euler_matrix_q1;
        self.append_1q_sequence(&mut gates, &mut global_phase, euler_matrix_q1, 1);

        gates.push((StandardGate::CX.into(), smallvec![], smallvec![1, 0]));

        if x12_is_pi_mult {
            // even or odd multiple
            if x12_is_non_zero {
                global_phase += x12_phase;
            }
            if x12_is_non_zero && x12_is_old_mult.unwrap() {
                gates.push((
                    StandardGate::RZ.into(),
                    smallvec![-euler_q0[1][1]],
                    smallvec![0],
                ));
            } else {
                gates.push((
                    StandardGate::RZ.into(),
                    smallvec![euler_q0[1][1]],
                    smallvec![0],
                ));
                global_phase += PI;
            }
        }
        if x12_is_half_pi {
            gates.push((StandardGate::SX.into(), smallvec![], smallvec![0]));
            global_phase -= PI4;
        } else if x12_is_non_zero && !x12_is_pi_mult {
            if self.pulse_optimize.is_none() {
                self.append_1q_sequence(&mut gates, &mut global_phase, rx_matrix(x12), 0);
            } else {
                return None;
            }
        }
        if abs_diff_eq!(euler_q1[1][1], PI2, epsilon = atol) {
            gates.push((StandardGate::SX.into(), smallvec![], smallvec![1]));
            global_phase -= PI4
        } else if self.pulse_optimize.is_none() {
            self.append_1q_sequence(&mut gates, &mut global_phase, rx_matrix(euler_q1[1][1]), 1);
        } else {
            return None;
        }
        gates.push((
            StandardGate::RZ.into(),
            smallvec![euler_q1[1][2] + euler_q1[2][0]],
            smallvec![1],
        ));
        gates.push((StandardGate::CX.into(), smallvec![], smallvec![1, 0]));
        gates.push((
            StandardGate::RZ.into(),
            smallvec![euler_q0[2][1]],
            smallvec![0],
        ));
        if abs_diff_eq!(euler_q1[2][1], PI2, epsilon = atol) {
            gates.push((StandardGate::SX.into(), smallvec![], smallvec![1]));
            global_phase -= PI4;
        } else if self.pulse_optimize.is_none() {
            self.append_1q_sequence(&mut gates, &mut global_phase, rx_matrix(euler_q1[2][1]), 1);
        } else {
            return None;
        }
        gates.push((StandardGate::CX.into(), smallvec![], smallvec![1, 0]));
        let mut euler_matrix = rz_matrix(euler_q0[2][2] + euler_q0[3][0]) * H_GATE_MATRIX;
        euler_matrix = rx_matrix(euler_q0[3][1]) * euler_matrix;
        euler_matrix = rz_matrix(euler_q0[3][2]) * euler_matrix;
        self.append_1q_sequence(&mut gates, &mut global_phase, euler_matrix, 0);

        let mut euler_matrix = rx_matrix(euler_q1[2][2] + euler_q1[3][0]) * H_GATE_MATRIX;
        euler_matrix = rz_matrix(euler_q1[3][1]) * euler_matrix;
        euler_matrix = rx_matrix(euler_q1[3][2]) * euler_matrix;
        self.append_1q_sequence(&mut gates, &mut global_phase, euler_matrix, 1);

        let out_unitary = compute_unitary(&gates, global_phase);
        // TODO: fix the sign problem to avoid correction here
        if abs_diff_eq!(
            target_decomposed.unitary_matrix[(0, 0)],
            -out_unitary[[0, 0]],
            epsilon = atol
        ) {
            global_phase += PI;
        }
        Some(TwoQubitGateSequence {
            gates,
            global_phase,
        })
    }

    fn append_1q_sequence(
        &self,
        gates: &mut TwoQubitSequenceVec,
        global_phase: &mut f64,
        unitary: Matrix2<Complex64>,
        qubit: u8,
    ) {
        let mut target_1q_basis_list = EulerBasisSet::new();
        target_1q_basis_list.add_basis(self.euler_basis);
        let unitary_view: MatrixView2<Complex64> = unitary.as_view();
        let sequence = unitary_to_gate_sequence_inner(
            nalgebra_to_ndarray(unitary_view),
            &target_1q_basis_list,
            qubit as usize,
            None,
            true,
            None,
        );
        if let Some(sequence) = sequence {
            *global_phase += sequence.global_phase;
            for gate in sequence.gates {
                gates.push((gate.0.into(), gate.1, smallvec![qubit]));
            }
        }
    }

    fn pulse_optimal_chooser(
        &self,
        best_nbasis: u8,
        decomposition: &SmallVec<[Matrix2<Complex64>; 8]>,
        target_decomposed: &TwoQubitWeylDecomposition,
    ) -> PyResult<Option<TwoQubitGateSequence>> {
        if self.pulse_optimize.is_some()
            && (best_nbasis == 0 || best_nbasis == 1 || best_nbasis > 3)
        {
            return Ok(None);
        }
        let mut use_xgate = false;
        match self.euler_basis {
            EulerBasis::ZSX => (),
            EulerBasis::ZSXX => {
                use_xgate = true;
            }
            _ => {
                if self.pulse_optimize.is_some() {
                    return Err(QiskitError::new_err(format!(
                        "'pulse_optimize' currently only works with ZSX basis ({} used)",
                        self.euler_basis.as_str()
                    )));
                } else {
                    return Ok(None);
                }
            }
        }
        if !matches!(
            self.gate.view(),
            OperationRef::StandardGate(StandardGate::CX)
        ) {
            if self.pulse_optimize.is_some() {
                return Err(QiskitError::new_err(
                    "pulse_optimizer currently only works with CNOT entangling gate",
                ));
            } else {
                return Ok(None);
            }
        }
        let res = if best_nbasis == 3 {
            self.get_sx_vz_3cx_efficient_euler(decomposition, target_decomposed)
        } else if best_nbasis == 2 {
            self.get_sx_vz_2cx_efficient_euler(decomposition, target_decomposed, use_xgate)
        } else {
            None
        };
        if self.pulse_optimize.is_some() && res.is_none() {
            return Err(QiskitError::new_err(
                "Failed to compute requested pulse optimal decomposition",
            ));
        }
        Ok(res)
    }

    pub fn new_inner(
        gate: PackedOperation,
        gate_params: SmallVec<[f64; 3]>,
        gate_matrix: ArrayView2<Complex64>,
        basis_fidelity: f64,
        euler_basis: &str,
        pulse_optimize: Option<bool>,
    ) -> PyResult<Self> {
        let basis_decomposer = TwoQubitWeylDecomposition::new_inner(
            ndarray_to_matrix4(gate_matrix)?,
            Some(DEFAULT_FIDELITY),
            None,
        )?;
        let super_controlled = relative_eq!(basis_decomposer.a, PI4, max_relative = 1e-09)
            && relative_eq!(basis_decomposer.c, 0.0, max_relative = 1e-09);

        // Create some useful matrices U1, U2, U3 are equivalent to the basis,
        // expand as Ui = Ki1.Ubasis.Ki2
        let b = basis_decomposer.b;
        let temp = c64(0.5, -0.5);
        let k11l = Matrix2::from_row_slice(&[
            temp * (M_IM * c64(0., -b).exp()),
            temp * c64(0., -b).exp(),
            temp * (M_IM * c64(0., b).exp()),
            temp * -(c64(0., b).exp()),
        ]);
        let k11r = Matrix2::from_row_slice(&[
            FRAC_1_SQRT_2 * (IM * c64(0., -b).exp()),
            FRAC_1_SQRT_2 * -c64(0., -b).exp(),
            FRAC_1_SQRT_2 * c64(0., b).exp(),
            FRAC_1_SQRT_2 * (M_IM * c64(0., b).exp()),
        ]);
        let k32l_k21l = Matrix2::from_row_slice(&[
            FRAC_1_SQRT_2 * c64(1., (2. * b).cos()),
            FRAC_1_SQRT_2 * (IM * (2. * b).sin()),
            FRAC_1_SQRT_2 * (IM * (2. * b).sin()),
            FRAC_1_SQRT_2 * c64(1., -(2. * b).cos()),
        ]);
        let temp = c64(0.5, 0.5);
        let k21r = Matrix2::from_row_slice(&[
            temp * (M_IM * c64(0., -2. * b).exp()),
            temp * c64(0., -2. * b).exp(),
            temp * (IM * c64(0., 2. * b).exp()),
            temp * c64(0., 2. * b).exp(),
        ]);
        let k22l = Matrix2::new(
            c64(FRAC_1_SQRT_2, 0.),
            c64(-FRAC_1_SQRT_2, 0.),
            c64(FRAC_1_SQRT_2, 0.),
            c64(FRAC_1_SQRT_2, 0.),
        );
        let k22r = Matrix2::new(Complex64::zero(), C_ONE, C_M_ONE, Complex64::zero());
        let k31l = Matrix2::from_row_slice(&[
            FRAC_1_SQRT_2 * c64(0., -b).exp(),
            FRAC_1_SQRT_2 * c64(0., -b).exp(),
            FRAC_1_SQRT_2 * -c64(0., b).exp(),
            FRAC_1_SQRT_2 * c64(0., b).exp(),
        ]);
        let k31r = Matrix2::from_row_slice(&[
            IM * c64(0., b).exp(),
            Complex64::zero(),
            Complex64::zero(),
            M_IM * c64(0., -b).exp(),
        ]);
        let temp = c64(0.5, 0.5);
        let k32r = Matrix2::from_row_slice(&[
            temp * c64(0., b).exp(),
            temp * -c64(0., -b).exp(),
            temp * (M_IM * c64(0., b).exp()),
            temp * (M_IM * c64(0., -b).exp()),
        ]);
        let k1ld = basis_decomposer.K1l.adjoint();
        let k1rd = basis_decomposer.K1r.adjoint();
        let k2ld = basis_decomposer.K2l.adjoint();
        let k2rd = basis_decomposer.K2r.adjoint();
        // Pre-build the fixed parts of the matrices used in 3-part decomposition
        let u0l = k31l * k1ld;
        let u0r = k31r * k1rd;
        let u1l = k2ld * k32l_k21l * k1ld;
        let u1ra = k2rd * k32r;
        let u1rb = k21r * k1rd;
        let u2la = k2ld * k22l;
        let u2lb = k11l * k1ld;
        let u2ra = k2rd * k22r;
        let u2rb = k11r * k1rd;
        let u3l = k2ld * K12L_ARR;
        let u3r = k2rd * K12R_ARR;
        // Pre-build the fixed parts of the matrices used in the 2-part decomposition
        let q0l = K12L_ARR.adjoint() * k1ld;
        let q0r = K12R_ARR.adjoint() * IPZ * k1rd;
        let q1la = k2ld * k11l.adjoint();
        let q1lb = k11l * k1ld;
        let q1ra = k2rd * IPZ * k11r.adjoint();
        let q1rb = k11r * k1rd;
        let q2l = k2ld * K12L_ARR;
        let q2r = k2rd * K12R_ARR;

        Ok(TwoQubitBasisDecomposer {
            gate,
            gate_params,
            basis_fidelity,
            euler_basis: EulerBasis::__new__(euler_basis)?,
            pulse_optimize,
            basis_decomposer,
            super_controlled,
            u0l,
            u0r,
            u1l,
            u1ra,
            u1rb,
            u2la,
            u2lb,
            u2ra,
            u2rb,
            u3l,
            u3r,
            q0l,
            q0r,
            q1la,
            q1lb,
            q1ra,
            q1rb,
            q2l,
            q2r,
        })
    }

    pub fn call_inner(
        &self,
        unitary: Matrix4<Complex64>,
        basis_fidelity: Option<f64>,
        approximate: bool,
        _num_basis_uses: Option<u8>,
    ) -> PyResult<TwoQubitGateSequence> {
        let basis_fidelity = if !approximate {
            1.0
        } else {
            basis_fidelity.unwrap_or(self.basis_fidelity)
        };
        let target_decomposed =
            TwoQubitWeylDecomposition::new_inner(unitary, Some(DEFAULT_FIDELITY), None)?;
        let traces = self.traces(&target_decomposed);
        let best_nbasis = _num_basis_uses.unwrap_or_else(|| {
            traces
                .into_iter()
                .enumerate()
                .map(|(idx, trace)| (idx, trace.trace_to_fid() * basis_fidelity.powi(idx as i32)))
                .min_by(|(_idx1, fid1), (_idx2, fid2)| fid2.partial_cmp(fid1).unwrap())
                .unwrap()
                .0 as u8
        });
        let decomposition = match best_nbasis {
            0 => decomp0_inner(&target_decomposed),
            1 => self.decomp1_inner(&target_decomposed),
            2 => self.decomp2_supercontrolled_inner(&target_decomposed),
            3 => self.decomp3_supercontrolled_inner(&target_decomposed),
            _ => unreachable!("Invalid basis to use"),
        };
        let pulse_optimize = self.pulse_optimize.unwrap_or(true);
        let sequence = if pulse_optimize {
            self.pulse_optimal_chooser(best_nbasis, &decomposition, &target_decomposed)?
        } else {
            None
        };
        if let Some(seq) = sequence {
            return Ok(seq);
        }
        let mut target_1q_basis_list = EulerBasisSet::new();
        target_1q_basis_list.add_basis(self.euler_basis);
        let euler_decompositions: SmallVec<[Option<OneQubitGateSequence>; 8]> = decomposition
            .iter()
            .map(|decomp| {
                let decomp_view: MatrixView2<Complex64> = decomp.as_view();
                unitary_to_gate_sequence_inner(
                    nalgebra_to_ndarray(decomp_view),
                    &target_1q_basis_list,
                    0,
                    None,
                    true,
                    None,
                )
            })
            .collect();
        let mut gates = Vec::with_capacity(TWO_QUBIT_SEQUENCE_DEFAULT_CAPACITY);
        let mut global_phase = target_decomposed.global_phase;
        global_phase -= best_nbasis as f64 * self.basis_decomposer.global_phase;
        if best_nbasis == 2 {
            global_phase += PI;
        }
        for i in 0..best_nbasis as usize {
            if let Some(euler_decomp) = &euler_decompositions[2 * i] {
                for gate in &euler_decomp.gates {
                    gates.push((gate.0.into(), gate.1.clone(), smallvec![0]));
                }
                global_phase += euler_decomp.global_phase
            }
            if let Some(euler_decomp) = &euler_decompositions[2 * i + 1] {
                for gate in &euler_decomp.gates {
                    gates.push((gate.0.into(), gate.1.clone(), smallvec![1]));
                }
                global_phase += euler_decomp.global_phase
            }
            gates.push((self.gate.clone(), self.gate_params.clone(), smallvec![0, 1]));
        }
        if let Some(euler_decomp) = &euler_decompositions[2 * best_nbasis as usize] {
            for gate in &euler_decomp.gates {
                gates.push((gate.0.into(), gate.1.clone(), smallvec![0]));
            }
            global_phase += euler_decomp.global_phase
        }
        if let Some(euler_decomp) = &euler_decompositions[2 * best_nbasis as usize + 1] {
            for gate in &euler_decomp.gates {
                gates.push((gate.0.into(), gate.1.clone(), smallvec![1]));
            }
            global_phase += euler_decomp.global_phase
        }
        Ok(TwoQubitGateSequence {
            gates,
            global_phase,
        })
    }
    /// Decompose a two-qubit ``unitary`` over fixed basis and :math:`SU(2)` using the best
    /// approximation given that each basis application has a finite ``basis_fidelity``.
    fn generate_sequence(
        &self,
        unitary: PyReadonlyArray2<Complex64>,
        basis_fidelity: Option<f64>,
        approximate: bool,
        _num_basis_uses: Option<u8>,
    ) -> PyResult<TwoQubitGateSequence> {
        let basis_fidelity = if !approximate {
            1.0
        } else {
            basis_fidelity.unwrap_or(self.basis_fidelity)
        };
        let target_decomposed =
            TwoQubitWeylDecomposition::new(unitary, Some(DEFAULT_FIDELITY), None)?;
        let traces = self.traces(&target_decomposed);
        let best_nbasis = traces
            .into_iter()
            .enumerate()
            .map(|(idx, trace)| (idx, trace.trace_to_fid() * basis_fidelity.powi(idx as i32)))
            .min_by(|(_idx1, fid1), (_idx2, fid2)| fid2.partial_cmp(fid1).unwrap())
            .unwrap()
            .0;
        let best_nbasis = _num_basis_uses.unwrap_or(best_nbasis as u8);
        let decomposition = match best_nbasis {
            0 => decomp0_inner(&target_decomposed),
            1 => self.decomp1_inner(&target_decomposed),
            2 => self.decomp2_supercontrolled_inner(&target_decomposed),
            3 => self.decomp3_supercontrolled_inner(&target_decomposed),
            _ => unreachable!("Invalid basis to use"),
        };
        let pulse_optimize = self.pulse_optimize.unwrap_or(true);
        let sequence = if pulse_optimize {
            self.pulse_optimal_chooser(best_nbasis, &decomposition, &target_decomposed)?
        } else {
            None
        };
        if let Some(seq) = sequence {
            return Ok(seq);
        }
        let mut target_1q_basis_list = EulerBasisSet::new();
        target_1q_basis_list.add_basis(self.euler_basis);
        let euler_decompositions: SmallVec<[Option<OneQubitGateSequence>; 8]> = decomposition
            .iter()
            .map(|decomp| {
                let decomp_view: MatrixView2<Complex64> = decomp.as_view();
                unitary_to_gate_sequence_inner(
                    nalgebra_to_ndarray(decomp_view),
                    &target_1q_basis_list,
                    0,
                    None,
                    true,
                    None,
                )
            })
            .collect();
        let mut gates = Vec::with_capacity(TWO_QUBIT_SEQUENCE_DEFAULT_CAPACITY);
        let mut global_phase = target_decomposed.global_phase;
        global_phase -= best_nbasis as f64 * self.basis_decomposer.global_phase;
        if best_nbasis == 2 {
            global_phase += PI;
        }
        for i in 0..best_nbasis as usize {
            if let Some(euler_decomp) = &euler_decompositions[2 * i] {
                for gate in &euler_decomp.gates {
                    gates.push((gate.0.into(), gate.1.clone(), smallvec![0]));
                }
                global_phase += euler_decomp.global_phase
            }
            if let Some(euler_decomp) = &euler_decompositions[2 * i + 1] {
                for gate in &euler_decomp.gates {
                    gates.push((gate.0.into(), gate.1.clone(), smallvec![1]));
                }
                global_phase += euler_decomp.global_phase
            }
            gates.push((self.gate.clone(), self.gate_params.clone(), smallvec![0, 1]));
        }
        if let Some(euler_decomp) = &euler_decompositions[2 * best_nbasis as usize] {
            for gate in &euler_decomp.gates {
                gates.push((gate.0.into(), gate.1.clone(), smallvec![0]));
            }
            global_phase += euler_decomp.global_phase
        }
        if let Some(euler_decomp) = &euler_decompositions[2 * best_nbasis as usize + 1] {
            for gate in &euler_decomp.gates {
                gates.push((gate.0.into(), gate.1.clone(), smallvec![1]));
            }
            global_phase += euler_decomp.global_phase
        }
        Ok(TwoQubitGateSequence {
            gates,
            global_phase,
        })
    }
}

static K12R_ARR: Matrix2<Complex64> = Matrix2::new(
    c64(0., FRAC_1_SQRT_2),
    c64(FRAC_1_SQRT_2, 0.),
    c64(-FRAC_1_SQRT_2, 0.),
    c64(0., -FRAC_1_SQRT_2),
);

static K12L_ARR: Matrix2<Complex64> =
    Matrix2::new(c64(0.5, 0.5), c64(0.5, 0.5), c64(-0.5, 0.5), c64(0.5, -0.5));

fn decomp0_inner(target: &TwoQubitWeylDecomposition) -> SmallVec<[Matrix2<Complex64>; 8]> {
    smallvec![target.K1r * target.K2r, target.K1l * target.K2l,]
}

type PickleNewArgs<'a> = (Py<PyAny>, Py<PyAny>, f64, &'a str, Option<bool>);

#[pymethods]
impl TwoQubitBasisDecomposer {
    fn __getnewargs__(&self, py: Python) -> PyResult<PickleNewArgs<'_>> {
        let params: Vec<Param> = self.gate_params.iter().map(|x| Param::Float(*x)).collect();
        Ok((
            match self.gate.view() {
                OperationRef::StandardGate(standard) => {
                    standard.create_py_op(py, Some(&params), None)?.into_any()
                }
                OperationRef::Gate(gate) => gate.gate.clone_ref(py),
                OperationRef::Unitary(unitary) => unitary.create_py_op(py, None)?.into_any(),
                _ => unreachable!("decomposer gate must be a gate"),
            },
            self.basis_decomposer
                .unitary_matrix
                .to_pyarray(py)
                .into_any()
                .unbind(),
            self.basis_fidelity,
            self.euler_basis.as_str(),
            self.pulse_optimize,
        ))
    }

    #[new]
    #[pyo3(signature=(gate, gate_matrix, basis_fidelity=1.0, euler_basis="U", pulse_optimize=None))]
    fn new(
        gate: OperationFromPython,
        gate_matrix: PyReadonlyArray2<Complex64>,
        basis_fidelity: f64,
        euler_basis: &str,
        pulse_optimize: Option<bool>,
    ) -> PyResult<Self> {
        let gate_params: PyResult<SmallVec<[f64; 3]>> = gate
            .params
            .iter()
            .map(|x| match x {
                Param::Float(val) => Ok(*val),
                _ => Err(PyValueError::new_err(
                    "Only unparameterized gates are supported as KAK gate",
                )),
            })
            .collect();
        TwoQubitBasisDecomposer::new_inner(
            gate.operation,
            gate_params?,
            gate_matrix.as_array(),
            basis_fidelity,
            euler_basis,
            pulse_optimize,
        )
    }

    fn traces(&self, target: &TwoQubitWeylDecomposition) -> [Complex64; 4] {
        [
            4. * c64(
                target.a.cos() * target.b.cos() * target.c.cos(),
                target.a.sin() * target.b.sin() * target.c.sin(),
            ),
            4. * c64(
                (PI4 - target.a).cos()
                    * (self.basis_decomposer.b - target.b).cos()
                    * target.c.cos(),
                (PI4 - target.a).sin()
                    * (self.basis_decomposer.b - target.b).sin()
                    * target.c.sin(),
            ),
            c64(4. * target.c.cos(), 0.),
            c64(4., 0.),
        ]
    }

    /// Decompose target :math:`\sim U_d(x, y, z)` with :math:`0` uses of the basis gate.
    /// Result :math:`U_r` has trace:
    ///
    /// .. math::
    ///
    ///     \Big\vert\text{Tr}(U_r\cdot U_\text{target}^{\dag})\Big\vert =
    ///     4\Big\vert (\cos(x)\cos(y)\cos(z)+ j \sin(x)\sin(y)\sin(z)\Big\vert
    ///
    /// which is optimal for all targets and bases
    #[staticmethod]
    fn decomp0(py: Python, target: &TwoQubitWeylDecomposition) -> SmallVec<[Py<PyAny>; 2]> {
        decomp0_inner(target)
            .into_iter()
            .map(|x| x.to_pyarray(py).into_any().unbind())
            .collect()
    }

    /// Decompose target :math:`\sim U_d(x, y, z)` with :math:`1` use of the basis gate
    /// math:`\sim U_d(a, b, c)`.
    /// Result :math:`U_r` has trace:
    ///
    /// .. math::
    ///
    ///     \Big\vert\text{Tr}(U_r \cdot U_\text{target}^{\dag})\Big\vert =
    ///     4\Big\vert \cos(x-a)\cos(y-b)\cos(z-c) + j \sin(x-a)\sin(y-b)\sin(z-c)\Big\vert
    ///
    /// which is optimal for all targets and bases with ``z==0`` or ``c==0``.
    fn decomp1(&self, py: Python, target: &TwoQubitWeylDecomposition) -> SmallVec<[Py<PyAny>; 4]> {
        self.decomp1_inner(target)
            .into_iter()
            .map(|x| x.to_pyarray(py).into_any().unbind())
            .collect()
    }

    /// Decompose target :math:`\sim U_d(x, y, z)` with :math:`2` uses of the basis gate.
    ///
    /// For supercontrolled basis :math:`\sim U_d(\pi/4, b, 0)`, all b, result :math:`U_r` has trace
    ///
    /// .. math::
    ///
    ///     \Big\vert\text{Tr}(U_r \cdot U_\text{target}^\dag) \Big\vert = 4\cos(z)
    ///
    /// which is the optimal approximation for basis of CNOT-class :math:`\sim U_d(\pi/4, 0, 0)`
    /// or DCNOT-class :math:`\sim U_d(\pi/4, \pi/4, 0)` and any target. It may
    /// be sub-optimal for :math:`b \neq 0` (i.e. there exists an exact decomposition for any target
    /// using :math:`B \sim U_d(\pi/4, \pi/8, 0)`, but it may not be this decomposition).
    /// This is an exact decomposition for supercontrolled basis and target :math:`\sim U_d(x, y, 0)`.
    /// No guarantees for non-supercontrolled basis.
    fn decomp2_supercontrolled(
        &self,
        py: Python,
        target: &TwoQubitWeylDecomposition,
    ) -> SmallVec<[Py<PyAny>; 6]> {
        self.decomp2_supercontrolled_inner(target)
            .into_iter()
            .map(|x| x.to_pyarray(py).into_any().unbind())
            .collect()
    }

    /// Decompose target with :math:`3` uses of the basis.
    ///
    /// This is an exact decomposition for supercontrolled basis :math:`\sim U_d(\pi/4, b, 0)`, all b,
    /// and any target. No guarantees for non-supercontrolled basis.
    fn decomp3_supercontrolled(
        &self,
        py: Python,
        target: &TwoQubitWeylDecomposition,
    ) -> SmallVec<[Py<PyAny>; 8]> {
        self.decomp3_supercontrolled_inner(target)
            .into_iter()
            .map(|x| x.to_pyarray(py).into_any().unbind())
            .collect()
    }

    /// Synthesizes a two qubit unitary matrix into a :class:`.DAGCircuit` object
    ///
    /// Args:
    ///     unitary (ndarray): A 4x4 unitary matrix in the form of a numpy complex array
    ///         representing the gate to synthesize
    ///     basis_fidelity (float): The target fidelity of the synthesis. This is a floating point
    ///         value between 1.0 and 0.0.
    ///     approximate (bool): Whether to enable approximation. If set to false this is equivalent
    ///         to setting basis_fidelity to 1.0.
    ///
    /// Returns:
    ///     DAGCircuit: The decomposed circuit for the given unitary.
    #[pyo3(signature = (unitary, basis_fidelity=None, approximate=true, _num_basis_uses=None))]
    fn to_dag(
        &self,
        unitary: PyReadonlyArray2<Complex64>,
        basis_fidelity: Option<f64>,
        approximate: bool,
        _num_basis_uses: Option<u8>,
    ) -> PyResult<DAGCircuit> {
        let sequence =
            self.generate_sequence(unitary, basis_fidelity, approximate, _num_basis_uses)?;
        let mut dag = DAGCircuit::with_capacity(2, 0, None, Some(sequence.gates.len()), None, None);
        dag.set_global_phase(Param::Float(sequence.global_phase))?;
        dag.add_qubit_unchecked(ShareableQubit::new_anonymous())?;
        dag.add_qubit_unchecked(ShareableQubit::new_anonymous())?;
        let mut builder = dag.into_builder();
        for (gate, params, qubits) in sequence.gates {
            let qubits: Vec<Qubit> = qubits.iter().map(|x| Qubit(*x as u32)).collect();
            let params = params.iter().map(|x| Param::Float(*x)).collect();
            builder.apply_operation_back(
                gate,
                &qubits,
                &[],
                Some(params),
                None,
                #[cfg(feature = "cache_pygates")]
                None,
            )?;
        }
        Ok(builder.build())
    }

    /// Synthesizes a two qubit unitary matrix into a :class:`.CircuitData` object
    ///
    /// Args:
    ///     unitary (ndarray): A 4x4 unitary matrix in the form of a numply complex array
    ///         representing the gate to synthesize
    ///     basis_fidelity (float): The target fidelity of the synthesis. This is a floating point
    ///         value between 1.0 and 0.0.
    ///     approximate (bool): Whether to enable approximation. If set to false this is equivalent
    ///         to setting basis_fidelity to 1.0.
    ///
    /// Returns:
    ///     CircuitData: The decomposed circuit for the given unitary.
    #[pyo3(signature = (unitary, basis_fidelity=None, approximate=true, _num_basis_uses=None))]
    fn to_circuit(
        &self,
        unitary: PyReadonlyArray2<Complex64>,
        basis_fidelity: Option<f64>,
        approximate: bool,
        _num_basis_uses: Option<u8>,
    ) -> PyResult<CircuitData> {
        let sequence =
            self.generate_sequence(unitary, basis_fidelity, approximate, _num_basis_uses)?;
        CircuitData::from_packed_operations(
            2,
            0,
            sequence.gates.into_iter().map(|(gate, params, qubits)| {
                Ok((
                    gate,
                    params.iter().map(|x| Param::Float(*x)).collect(),
                    qubits.iter().map(|q| Qubit(*q as u32)).collect(),
                    vec![],
                ))
            }),
            Param::Float(sequence.global_phase),
        )
    }

    fn num_basis_gates(&self, unitary: PyReadonlyArray2<Complex64>) -> PyResult<usize> {
        _num_basis_gates(self.basis_decomposer.b, self.basis_fidelity, unitary)
    }
}

fn u4_to_su4(u4: Matrix4<Complex64>) -> (Matrix4<Complex64>, f64) {
    let phase_factor = u4.determinant().powf(-0.25).conj();
    let su4 = u4.map(|x| x / phase_factor);
    (su4, phase_factor.arg())
}

fn real_trace_transform(mat: &Matrix4<Complex64>) -> Matrix4<Complex64> {
    let a1 = -mat[(1, 3)] * mat[(2, 0)] + mat[(1, 2)] * mat[(2, 1)] + mat[(1, 1)] * mat[(2, 2)]
        - mat[(1, 0)] * mat[(2, 3)];
    let a2 = mat[(0, 3)] * mat[(3, 0)] - mat[(0, 2)] * mat[(3, 1)] - mat[(0, 1)] * mat[(3, 2)]
        + mat[(0, 0)] * mat[(3, 3)];
    let theta = 0.; // Arbitrary!
    let phi = 0.; // This is extra arbitrary!
    let psi = f64::atan2(a1.im + a2.im, a1.re - a2.re) - phi;
    let im = Complex64::new(0., -1.);
    let temp = Vector4::new(
        (theta * im).exp(),
        (phi * im).exp(),
        (psi * im).exp(),
        (-(theta + phi + psi) * im).exp(),
    );
    Matrix4::from_diagonal(&temp)
}

#[pyfunction]
#[pyo3(name = "two_qubit_decompose_up_to_diagonal")]
fn py_two_qubit_decompose_up_to_diagonal(
    py: Python,
    mat: PyReadonlyArray2<Complex64>,
) -> PyResult<(Py<PyAny>, CircuitData)> {
    let mat_arr: ArrayView2<Complex64> = mat.as_array();
    let (real_map, circ) = two_qubit_decompose_up_to_diagonal(mat_arr)?;
    Ok((real_map.into_pyarray(py).into_any().unbind(), circ))
}

pub fn two_qubit_decompose_up_to_diagonal(
    mat: ArrayView2<Complex64>,
) -> PyResult<(Array2<Complex64>, CircuitData)> {
    let mat = ndarray_to_matrix4(mat)?;
    let (su4, phase) = u4_to_su4(mat);
    let real_map = real_trace_transform(&su4);
    let mapped_su4 = real_map * su4;
    let decomp = TwoQubitBasisDecomposer::new_inner(
        StandardGate::CX.into(),
        smallvec![],
        aview2(&CX_GATE),
        1.0,
        "U",
        None,
    )?;

    let circ_seq = decomp.call_inner(mapped_su4, None, true, None)?;
    let circ = CircuitData::from_packed_operations(
        2,
        0,
        circ_seq
            .gates
            .into_iter()
            .map(|(gate, param_floats, qubit_index)| {
                let params: SmallVec<[Param; 3]> =
                    param_floats.into_iter().map(Param::Float).collect();
                let qubits = qubit_index.into_iter().map(|x| Qubit(x as u32)).collect();
                Ok((gate, params, qubits, vec![]))
            }),
        Param::Float(circ_seq.global_phase + phase),
    )?;
    let real_map_array = Array2::from_shape_fn((4, 4), |(i, j)| real_map[(i, j)].conj());
    Ok((real_map_array, circ))
}

static MAGIC: Matrix4<Complex64> = Matrix4::new(
    c64(FRAC_1_SQRT_2, 0.),
    C_ZERO,
    C_ZERO,
    c64(0., FRAC_1_SQRT_2),
    C_ZERO,
    c64(0., FRAC_1_SQRT_2),
    c64(FRAC_1_SQRT_2, 0.),
    C_ZERO,
    C_ZERO,
    c64(0., FRAC_1_SQRT_2),
    c64(-FRAC_1_SQRT_2, 0.),
    C_ZERO,
    c64(FRAC_1_SQRT_2, 0.),
    C_ZERO,
    C_ZERO,
    c64(0., -FRAC_1_SQRT_2),
);

static MAGIC_DAGGER: Matrix4<Complex64> = Matrix4::new(
    c64(FRAC_1_SQRT_2, 0.),
    C_ZERO,
    C_ZERO,
    c64(FRAC_1_SQRT_2, 0.),
    C_ZERO,
    c64(0., -FRAC_1_SQRT_2),
    c64(0., -FRAC_1_SQRT_2),
    C_ZERO,
    C_ZERO,
    c64(FRAC_1_SQRT_2, 0.),
    c64(-FRAC_1_SQRT_2, 0.),
    C_ZERO,
    c64(0., -FRAC_1_SQRT_2),
    C_ZERO,
    C_ZERO,
    c64(0., FRAC_1_SQRT_2),
);

/// Computes the local invariants for a two-qubit unitary.
///
/// Based on:
///
/// Y. Makhlin, Quant. Info. Proc. 1, 243-252 (2002).
///
/// Zhang et al., Phys Rev A. 67, 042313 (2003).
#[pyfunction]
pub fn two_qubit_local_invariants(unitary: PyReadonlyArray2<Complex64>) -> [f64; 3] {
    let mat = ndarray_to_matrix4(unitary.as_array()).unwrap();
    // Transform to bell basis
    let bell_basis_unitary = MAGIC_DAGGER * (mat * MAGIC);
    // Get determinate since +- one is allowed.
    let det_bell_basis = bell_basis_unitary.determinant();
    let m = bell_basis_unitary.transpose() * bell_basis_unitary;
    let mut m_tr2 = m.diagonal().sum();
    m_tr2 *= m_tr2;
    // Table II of Ref. 1 or Eq. 28 of Ref. 2.
    let g1 = m_tr2 / (16. * det_bell_basis);
    let g2 = (m_tr2 - (m * m).diagonal().sum()) / (4. * det_bell_basis);

    // Here we split the real and imag pieces of G1 into two so as
    // to better equate to the Weyl chamber coordinates (c0,c1,c2)
    // and explore the parameter space.
    // Also do a FP trick -0.0 + 0.0 = 0.0
    [g1.re + 0., g1.im + 0., g2.re + 0.]
}

#[pyfunction]
pub fn local_equivalence(weyl: PyReadonlyArray1<f64>) -> PyResult<[f64; 3]> {
    let weyl = weyl.as_array();
    let weyl_2_cos_squared_product: f64 = weyl.iter().map(|x| (x * 2.).cos().powi(2)).product();
    let weyl_2_sin_squared_product: f64 = weyl.iter().map(|x| (x * 2.).sin().powi(2)).product();
    let g0_equiv = weyl_2_cos_squared_product - weyl_2_sin_squared_product;
    let g1_equiv = weyl.iter().map(|x| (x * 4.).sin()).product::<f64>() / 4.;
    let g2_equiv = 4. * weyl_2_cos_squared_product
        - 4. * weyl_2_sin_squared_product
        - weyl.iter().map(|x| (4. * x).cos()).product::<f64>();
    Ok([g0_equiv + 0., g1_equiv + 0., g2_equiv + 0.])
}

/// invert 1q gate sequence
fn invert_1q_gate(
    gate: (StandardGate, SmallVec<[f64; 3]>),
) -> (PackedOperation, SmallVec<[f64; 3]>) {
    let gate_params = gate.1.into_iter().map(Param::Float).collect::<Vec<_>>();
    let inv_gate = gate
        .0
        .inverse(&gate_params)
        .expect("An unexpected standard gate was inverted");
    let inv_gate_params = inv_gate
        .1
        .into_iter()
        .map(|param| match param {
            Param::Float(val) => val,
            _ => unreachable!("Parameterized inverse generated from non-parameterized gate."),
        })
        .collect::<SmallVec<_>>();
    (inv_gate.0.into(), inv_gate_params)
}

#[derive(Clone, Debug, FromPyObject)]
pub enum RXXEquivalent {
    Standard(StandardGate),
    CustomPython(Py<PyType>),
}

impl RXXEquivalent {
    fn matrix(&self, param: f64) -> PyResult<Matrix4<Complex64>> {
        match self {
            Self::Standard(gate) => {
                ndarray_to_matrix4(gate.matrix(&[Param::Float(param)]).unwrap().view())
            }
            Self::CustomPython(gate_cls) => Python::attach(|py: Python| {
                let gate_obj = gate_cls.bind(py).call1((param,))?;
                let raw_matrix = gate_obj
                    .call_method0(intern!(py, "to_matrix"))?
                    .extract::<PyReadonlyArray2<Complex64>>()?;
                ndarray_to_matrix4(raw_matrix.as_array())
            }),
        }
    }
}
impl<'a, 'py> IntoPyObject<'py> for &'a RXXEquivalent {
    type Target = PyAny;
    type Output = Borrowed<'a, 'py, Self::Target>;
    type Error = PyErr;
    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            RXXEquivalent::Standard(gate) => Ok(gate.get_gate_class(py)?.bind_borrowed(py)),
            RXXEquivalent::CustomPython(gate) => Ok(gate.as_any().bind_borrowed(py)),
        }
    }
}
impl<'py> IntoPyObject<'py> for RXXEquivalent {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;
    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            RXXEquivalent::Standard(gate) => Ok(gate.get_gate_class(py)?.bind(py).clone()),
            RXXEquivalent::CustomPython(gate) => Ok(gate.bind(py).clone().into_any()),
        }
    }
}

#[derive(Clone, Debug)]
#[pyclass(module = "qiskit._accelerate.two_qubit_decompose", subclass)]
pub struct TwoQubitControlledUDecomposer {
    rxx_equivalent_gate: RXXEquivalent,
    euler_basis: EulerBasis,
    #[pyo3(get)]
    scale: f64,
}

const DEFAULT_ATOL: f64 = 1e-12;
type InverseReturn = (PackedOperation, SmallVec<[f64; 3]>, SmallVec<[u8; 2]>);

///  Decompose two-qubit unitary in terms of a desired
///  :math:`U \sim U_d(\alpha, 0, 0) \sim \text{Ctrl-U}`
///  gate that is locally equivalent to an :class:`.RXXGate`.
impl TwoQubitControlledUDecomposer {
    /// Compute the number of basis gates needed for a given unitary
    pub fn num_basis_gates_inner(&self, unitary: Matrix4<Complex64>) -> PyResult<usize> {
        let target_decomposed =
            TwoQubitWeylDecomposition::new_inner(unitary, Some(DEFAULT_FIDELITY), None)?;
        let num_basis_gates = (((target_decomposed.a).abs() > DEFAULT_ATOL) as usize)
            + (((target_decomposed.b).abs() > DEFAULT_ATOL) as usize)
            + (((target_decomposed.c).abs() > DEFAULT_ATOL) as usize);
        Ok(num_basis_gates)
    }

    /// invert 2q gate sequence
    fn invert_2q_gate(
        &self,
        gate: (PackedOperation, SmallVec<[f64; 3]>, SmallVec<[u8; 2]>),
    ) -> PyResult<InverseReturn> {
        let (gate, params, qubits) = gate;
        let inv_gate = match gate.view() {
            OperationRef::StandardGate(gate) => {
                let res = gate
                    .inverse(&params.into_iter().map(Param::Float).collect::<Vec<_>>())
                    .unwrap();
                (res.0.into(), res.1)
            }
            OperationRef::Gate(gate) => {
                Python::attach(|py: Python| -> PyResult<(PackedOperation, SmallVec<_>)> {
                    let raw_inverse = gate.gate.call_method0(py, intern!(py, "inverse"))?;
                    let inverse: OperationFromPython = raw_inverse.extract(py)?;
                    Ok((inverse.operation, inverse.params))
                })?
            }
            // UnitaryGate isn't applicable here as the 2q gate here is the parameterized
            // ControlledU equivalent used in the decomposition. This precludes UnitaryGate
            _ => panic!("Only 2q gate objects can be inverted in the decomposer"),
        };
        let inv_gate_params = inv_gate
            .1
            .into_iter()
            .map(|param| match param {
                Param::Float(val) => val,
                _ => {
                    unreachable!("Parameterized inverse generated from non-parameterized gate.")
                }
            })
            .collect::<SmallVec<_>>();
        Ok((inv_gate.0, inv_gate_params, qubits))
    }

    ///  Takes an angle and returns the circuit equivalent to an RXXGate with the
    ///  RXX equivalent gate as the two-qubit unitary.
    ///  Args:
    ///      angle: Rotation angle (in this case one of the Weyl parameters a, b, or c)
    ///  Returns:
    ///      Circuit: Circuit equivalent to an RXXGate.
    ///  Raises:
    ///      QiskitError: If the circuit is not equivalent to an RXXGate.
    fn to_rxx_gate(&self, angle: f64) -> PyResult<TwoQubitGateSequence> {
        // The user-provided RXX equivalent gate may be locally equivalent to the RXX gate
        // but with some scaling in the rotation angle. For example, RXX(angle) has Weyl
        // parameters (angle, 0, 0) for angle in [0, pi/2] but the user provided gate, i.e.
        // :code:`self.rxx_equivalent_gate(angle)` might produce the Weyl parameters
        // (scale * angle, 0, 0) where scale != 1. This is the case for the CPhase gate.

        let mat = self.rxx_equivalent_gate.matrix(self.scale * angle)?;
        let decomposer_inv =
            TwoQubitWeylDecomposition::new_inner(mat, Some(DEFAULT_FIDELITY), None)?;

        let mut target_1q_basis_list = EulerBasisSet::new();
        target_1q_basis_list.add_basis(self.euler_basis);

        // Express the RXX in terms of the user-provided RXX equivalent gate.
        let mut gates = Vec::with_capacity(13);
        let mut global_phase = -decomposer_inv.global_phase;

        let k1r_view: MatrixView2<Complex64> = decomposer_inv.K1r.as_view();
        let k2r_view: MatrixView2<Complex64> = decomposer_inv.K2r.as_view();
        let k1l_view: MatrixView2<Complex64> = decomposer_inv.K1l.as_view();
        let k2l_view: MatrixView2<Complex64> = decomposer_inv.K2l.as_view();

        let unitary_k1r = unitary_to_gate_sequence_inner(
            nalgebra_to_ndarray(k1r_view),
            &target_1q_basis_list,
            0,
            None,
            true,
            None,
        );
        let unitary_k2r = unitary_to_gate_sequence_inner(
            nalgebra_to_ndarray(k2r_view),
            &target_1q_basis_list,
            0,
            None,
            true,
            None,
        );
        let unitary_k1l = unitary_to_gate_sequence_inner(
            nalgebra_to_ndarray(k1l_view),
            &target_1q_basis_list,
            0,
            None,
            true,
            None,
        );
        let unitary_k2l = unitary_to_gate_sequence_inner(
            nalgebra_to_ndarray(k2l_view),
            &target_1q_basis_list,
            0,
            None,
            true,
            None,
        );

        if let Some(unitary_k2r) = unitary_k2r {
            global_phase -= unitary_k2r.global_phase;
            for gate in unitary_k2r.gates.into_iter().rev() {
                let (inv_gate_name, inv_gate_params) = invert_1q_gate(gate);
                gates.push((inv_gate_name, inv_gate_params, smallvec![0]));
            }
        }
        if let Some(unitary_k2l) = unitary_k2l {
            global_phase -= unitary_k2l.global_phase;
            for gate in unitary_k2l.gates.into_iter().rev() {
                let (inv_gate_name, inv_gate_params) = invert_1q_gate(gate);
                gates.push((inv_gate_name, inv_gate_params, smallvec![1]));
            }
        }
        let rxx_op = match &self.rxx_equivalent_gate {
            RXXEquivalent::Standard(gate) => PackedOperation::from_standard_gate(*gate),
            RXXEquivalent::CustomPython(gate_cls) => {
                Python::attach(|py| -> PyResult<PackedOperation> {
                    let op: OperationFromPython =
                        gate_cls.bind(py).call1((self.scale * angle,))?.extract()?;
                    Ok(op.operation)
                })?
            }
        };
        gates.push((rxx_op, smallvec![self.scale * angle], smallvec![0, 1]));

        if let Some(unitary_k1r) = unitary_k1r {
            global_phase -= unitary_k1r.global_phase;
            for gate in unitary_k1r.gates.into_iter().rev() {
                let (inv_gate_name, inv_gate_params) = invert_1q_gate(gate);
                gates.push((inv_gate_name, inv_gate_params, smallvec![0]));
            }
        }
        if let Some(unitary_k1l) = unitary_k1l {
            global_phase -= unitary_k1l.global_phase;
            for gate in unitary_k1l.gates.into_iter().rev() {
                let (inv_gate_name, inv_gate_params) = invert_1q_gate(gate);
                gates.push((inv_gate_name, inv_gate_params, smallvec![1]));
            }
        }

        Ok(TwoQubitGateSequence {
            gates,
            global_phase,
        })
    }

    /// Appends U_d(a, b, c) to the circuit.
    fn weyl_gate(
        &self,
        circ: &mut TwoQubitGateSequence,
        target_decomposed: TwoQubitWeylDecomposition,
        atol: f64,
    ) -> PyResult<()> {
        let circ_a = self.to_rxx_gate(-2.0 * target_decomposed.a)?;
        circ.gates.extend(circ_a.gates);
        let mut global_phase = circ_a.global_phase;

        let mut target_1q_basis_list = EulerBasisSet::new();
        target_1q_basis_list.add_basis(self.euler_basis);

        let s_decomp = unitary_to_gate_sequence_inner(
            aview2(&S_GATE),
            &target_1q_basis_list,
            0,
            None,
            true,
            None,
        );
        let sdg_decomp = unitary_to_gate_sequence_inner(
            aview2(&SDG_GATE),
            &target_1q_basis_list,
            0,
            None,
            true,
            None,
        );
        let h_decomp = unitary_to_gate_sequence_inner(
            aview2(&H_GATE),
            &target_1q_basis_list,
            0,
            None,
            true,
            None,
        );

        // translate RYY(b) into a circuit based on the desired Ctrl-U gate.
        if (target_decomposed.b).abs() > atol {
            let circ_b = self.to_rxx_gate(-2.0 * target_decomposed.b)?;
            global_phase += circ_b.global_phase;
            if let Some(sdg_decomp) = sdg_decomp {
                global_phase += 2.0 * sdg_decomp.global_phase;
                for gate in sdg_decomp.gates.into_iter() {
                    let gate_params = gate.1;
                    circ.gates
                        .push((gate.0.into(), gate_params.clone(), smallvec![0]));
                    circ.gates.push((gate.0.into(), gate_params, smallvec![1]));
                }
            }
            circ.gates.extend(circ_b.gates);
            if let Some(s_decomp) = s_decomp {
                global_phase += 2.0 * s_decomp.global_phase;
                for gate in s_decomp.gates.into_iter() {
                    let gate_params = gate.1;
                    circ.gates
                        .push((gate.0.into(), gate_params.clone(), smallvec![0]));
                    circ.gates.push((gate.0.into(), gate_params, smallvec![1]));
                }
            }
        }

        // translate RZZ(c) into a circuit based on the desired Ctrl-U gate.
        if (target_decomposed.c).abs() > atol {
            // Since the Weyl chamber is here defined as a > b > |c| we may have
            // negative c. This will cause issues in _to_rxx_gate
            // as TwoQubitWeylControlledEquiv will map (c, 0, 0) to (|c|, 0, 0).
            // We therefore produce RZZ(|c|) and append its inverse to the
            // circuit if c < 0.
            let mut gamma = -2.0 * target_decomposed.c;
            if gamma <= 0.0 {
                let circ_c = self.to_rxx_gate(gamma)?;
                global_phase += circ_c.global_phase;

                if let Some(ref h_decomp) = h_decomp {
                    global_phase += 2.0 * h_decomp.global_phase;
                    for gate in h_decomp.gates.clone().into_iter() {
                        let gate_params = gate.1;
                        circ.gates
                            .push((gate.0.into(), gate_params.clone(), smallvec![0]));
                        circ.gates.push((gate.0.into(), gate_params, smallvec![1]));
                    }
                }
                circ.gates.extend(circ_c.gates);
                if let Some(ref h_decomp) = h_decomp {
                    global_phase += 2.0 * h_decomp.global_phase;
                    for gate in h_decomp.gates.clone().into_iter() {
                        let gate_params = gate.1;
                        circ.gates
                            .push((gate.0.into(), gate_params.clone(), smallvec![0]));
                        circ.gates.push((gate.0.into(), gate_params, smallvec![1]));
                    }
                }
            } else {
                // invert the circuit above
                gamma *= -1.0;
                let circ_c = self.to_rxx_gate(gamma)?;
                global_phase -= circ_c.global_phase;
                if let Some(ref h_decomp) = h_decomp {
                    global_phase += 2.0 * h_decomp.global_phase;
                    for gate in h_decomp.gates.clone().into_iter() {
                        let gate_params = gate.1;
                        circ.gates
                            .push((gate.0.into(), gate_params.clone(), smallvec![0]));
                        circ.gates.push((gate.0.into(), gate_params, smallvec![1]));
                    }
                }
                for gate in circ_c.gates.into_iter().rev() {
                    let (inv_gate_name, inv_gate_params, inv_gate_qubits) =
                        self.invert_2q_gate(gate)?;
                    circ.gates
                        .push((inv_gate_name, inv_gate_params, inv_gate_qubits));
                }
                if let Some(ref h_decomp) = h_decomp {
                    global_phase += 2.0 * h_decomp.global_phase;
                    for gate in h_decomp.gates.clone().into_iter() {
                        let gate_params = gate.1;
                        circ.gates
                            .push((gate.0.into(), gate_params.clone(), smallvec![0]));
                        circ.gates.push((gate.0.into(), gate_params, smallvec![1]));
                    }
                }
            }
        }

        circ.global_phase = global_phase;
        Ok(())
    }

    ///  Returns the Weyl decomposition in circuit form.
    ///  Note: atol is passed to OneQubitEulerDecomposer.
    pub fn call_inner(
        &self,
        unitary: Matrix4<Complex64>,
        atol: Option<f64>,
    ) -> PyResult<TwoQubitGateSequence> {
        let target_decomposed =
            TwoQubitWeylDecomposition::new_inner(unitary, Some(DEFAULT_FIDELITY), None)?;

        let mut target_1q_basis_list = EulerBasisSet::new();
        target_1q_basis_list.add_basis(self.euler_basis);

        let k1r_view: MatrixView2<Complex64> = target_decomposed.K1r.as_view();
        let k2r_view: MatrixView2<Complex64> = target_decomposed.K2r.as_view();
        let k1l_view: MatrixView2<Complex64> = target_decomposed.K1l.as_view();
        let k2l_view: MatrixView2<Complex64> = target_decomposed.K2l.as_view();

        let unitary_c1r = unitary_to_gate_sequence_inner(
            nalgebra_to_ndarray(k1r_view),
            &target_1q_basis_list,
            0,
            None,
            true,
            None,
        );
        let unitary_c2r = unitary_to_gate_sequence_inner(
            nalgebra_to_ndarray(k2r_view),
            &target_1q_basis_list,
            0,
            None,
            true,
            None,
        );
        let unitary_c1l = unitary_to_gate_sequence_inner(
            nalgebra_to_ndarray(k1l_view),
            &target_1q_basis_list,
            0,
            None,
            true,
            None,
        );
        let unitary_c2l = unitary_to_gate_sequence_inner(
            nalgebra_to_ndarray(k2l_view),
            &target_1q_basis_list,
            0,
            None,
            true,
            None,
        );

        let mut gates = Vec::with_capacity(59);
        let mut global_phase = target_decomposed.global_phase;

        if let Some(unitary_c2r) = unitary_c2r {
            global_phase += unitary_c2r.global_phase;
            for gate in unitary_c2r.gates.into_iter() {
                gates.push((gate.0.into(), gate.1, smallvec![0]));
            }
        }
        if let Some(unitary_c2l) = unitary_c2l {
            global_phase += unitary_c2l.global_phase;
            for gate in unitary_c2l.gates.into_iter() {
                gates.push((gate.0.into(), gate.1, smallvec![1]));
            }
        }
        let mut gates1 = TwoQubitGateSequence {
            gates,
            global_phase,
        };
        self.weyl_gate(&mut gates1, target_decomposed, atol.unwrap_or(DEFAULT_ATOL))?;
        global_phase += gates1.global_phase;

        if let Some(unitary_c1r) = unitary_c1r {
            global_phase += unitary_c1r.global_phase;
            for gate in unitary_c1r.gates.into_iter() {
                gates1.gates.push((gate.0.into(), gate.1, smallvec![0]));
            }
        }
        if let Some(unitary_c1l) = unitary_c1l {
            global_phase += unitary_c1l.global_phase;
            for gate in unitary_c1l.gates.into_iter() {
                gates1.gates.push((gate.0.into(), gate.1, smallvec![1]));
            }
        }

        gates1.global_phase = global_phase;
        Ok(gates1)
    }

    /// Initialize the KAK decomposition.
    pub fn new_inner(rxx_equivalent_gate: RXXEquivalent, euler_basis: &str) -> PyResult<Self> {
        let atol = DEFAULT_ATOL;
        let test_angles = [0.2, 0.3, PI2];

        let scales: PyResult<Vec<f64>> = test_angles
            .into_iter()
            .map(|test_angle| {
                match &rxx_equivalent_gate {
                    RXXEquivalent::Standard(gate) => {
                        if gate.num_params() != 1 {
                            return Err(QiskitError::new_err(
                                "Equivalent gate needs to take exactly 1 angle parameter.",
                            ));
                        }
                    }
                    RXXEquivalent::CustomPython(gate_cls) => {
                        let takes_param = Python::attach(|py: Python| {
                            gate_cls.bind(py).call1((test_angle,)).ok().is_none()
                        });
                        if takes_param {
                            return Err(QiskitError::new_err(
                                "Equivalent gate needs to take exactly 1 angle parameter.",
                            ));
                        }
                    }
                };
                let mat: nalgebra::Matrix<
                    Complex<f64>,
                    nalgebra::Const<4>,
                    nalgebra::Const<4>,
                    nalgebra::ArrayStorage<Complex<f64>, 4, 4>,
                > = rxx_equivalent_gate.matrix(test_angle)?;
                let decomp =
                    TwoQubitWeylDecomposition::new_inner(mat, Some(DEFAULT_FIDELITY), None)?;
                let mat_rxx = ndarray_to_matrix4(
                    StandardGate::RXX
                        .matrix(&[Param::Float(test_angle)])
                        .unwrap()
                        .view(),
                )?;
                let decomposer_rxx = TwoQubitWeylDecomposition::new_inner(
                    mat_rxx,
                    None,
                    Some(Specialization::ControlledEquiv),
                )?;
                let decomposer_equiv = TwoQubitWeylDecomposition::new_inner(
                    mat,
                    Some(DEFAULT_FIDELITY),
                    Some(Specialization::ControlledEquiv),
                )?;
                let scale_a = decomposer_rxx.a / decomposer_equiv.a;
                if (decomp.a * 2.0 - test_angle / scale_a).abs() > atol {
                    return Err(QiskitError::new_err(
                        "The provided gate is not equivalent to an RXX.",
                    ));
                }
                Ok(scale_a)
            })
            .collect();
        let scales = scales?;

        let scale = scales[0];

        // Check that all three tested angles give the same scale
        for scale_val in &scales {
            if !abs_diff_eq!(scale_val, &scale, epsilon = atol) {
                return Err(QiskitError::new_err(
                    "Inconsistent scaling parameters in check.",
                ));
            }
        }

        Ok(TwoQubitControlledUDecomposer {
            scale,
            rxx_equivalent_gate,
            euler_basis: EulerBasis::__new__(euler_basis)?,
        })
    }
}

#[pymethods]
impl TwoQubitControlledUDecomposer {
    fn __getnewargs__(&self) -> (&RXXEquivalent, &str) {
        (&self.rxx_equivalent_gate, self.euler_basis.as_str())
    }

    ///  Initialize the KAK decomposition.
    ///  Args:
    ///      rxx_equivalent_gate: Gate that is locally equivalent to an :class:`.RXXGate`:
    ///      :math:`U \sim U_d(\alpha, 0, 0) \sim \text{Ctrl-U}` gate.
    ///     euler_basis: Basis string to be provided to :class:`.OneQubitEulerDecomposer`
    ///     for 1Q synthesis.
    ///  Raises:
    ///      QiskitError: If the gate is not locally equivalent to an :class:`.RXXGate`.
    #[new]
    #[pyo3(signature=(rxx_equivalent_gate, euler_basis="ZXZ"))]
    pub fn new(rxx_equivalent_gate: RXXEquivalent, euler_basis: &str) -> PyResult<Self> {
        TwoQubitControlledUDecomposer::new_inner(rxx_equivalent_gate, euler_basis)
    }

    #[pyo3(signature=(unitary, atol=None))]
    fn __call__(
        &self,
        unitary: PyReadonlyArray2<Complex64>,
        atol: Option<f64>,
    ) -> PyResult<CircuitData> {
        let unitary = ndarray_to_matrix4(unitary.as_array())?;
        let sequence = self.call_inner(unitary, atol)?;
        CircuitData::from_packed_operations(
            2,
            0,
            sequence.gates.into_iter().map(|(gate, params, qubits)| {
                Ok((
                    gate,
                    params.into_iter().map(Param::Float).collect(),
                    qubits.into_iter().map(|x| Qubit(x as u32)).collect(),
                    vec![],
                ))
            }),
            Param::Float(sequence.global_phase),
        )
    }
}

pub fn two_qubit_decompose(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(_num_basis_gates))?;
    m.add_wrapped(wrap_pyfunction!(py_decompose_two_qubit_product_gate))?;
    m.add_wrapped(wrap_pyfunction!(py_two_qubit_decompose_up_to_diagonal))?;
    m.add_wrapped(wrap_pyfunction!(two_qubit_local_invariants))?;
    m.add_wrapped(wrap_pyfunction!(local_equivalence))?;
    m.add_wrapped(wrap_pyfunction!(py_trace_to_fid))?;
    m.add_wrapped(wrap_pyfunction!(py_ud))?;
    m.add_wrapped(wrap_pyfunction!(weyl_coordinates))?;
    m.add_class::<TwoQubitWeylDecomposition>()?;
    m.add_class::<Specialization>()?;
    m.add_class::<TwoQubitBasisDecomposer>()?;
    m.add_class::<TwoQubitControlledUDecomposer>()?;
    Ok(())
}

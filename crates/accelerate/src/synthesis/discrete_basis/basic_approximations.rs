// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use hashbrown::HashMap;
use nalgebra::{Matrix2, Matrix3};
use ndarray::ArrayView2;
use num_complex::{Complex, ComplexFloat};
use num_traits::FloatConst;
use numpy::Complex64;
use pyo3::{exceptions::PyValueError, prelude::*};
use qiskit_circuit::{
    circuit_data::CircuitData,
    operations::{Operation, Param, StandardGate},
    Qubit,
};
use rstar::{Point, RTree};
use std::f64::consts::FRAC_1_SQRT_2;
use std::{fmt::Debug, ops::Div};
use thiserror::Error;

use super::math;

#[derive(Error, Debug)]
pub enum DiscreteBasisError {
    #[error("Expected discrete (parameter-free) gate.")]
    ExpectDiscreteGate,
    #[error("Expected an initialized gate sequence.")]
    UninitializedSequence,
    #[error("Parameterized gates cannot be decomposed.")]
    ParameterizedGate,
}

impl From<DiscreteBasisError> for PyErr {
    fn from(value: DiscreteBasisError) -> Self {
        PyValueError::new_err(value.to_string())
    }
}

/// A sequence of single qubit gates and their matrix.
///
/// Gates are stored in **circuit order**, not in matrix multiplication order. That means that
/// e.g. [H, T] corresponds to the matrix U = T @ H. The matrix is not stored as U(2), but in
/// a SO(3) representation, which discards the global phase.
#[pyclass]
#[derive(Clone, Debug)]
pub struct GateSequence {
    // The sequence of standard gates. Can be None if the sequence is only specified by the
    // SO(3) matrix and phase, which is useful for lookup of matrices.
    pub gates: Option<Vec<StandardGate>>,
    // The SO(3) representation of the sequence. Note that this is only equal to SU(2) up to a sign.
    pub matrix_so3: Matrix3<f64>,
    // A global phase taking the U(2) representation of the sequence to SU(2).
    pub phase: f64,
    // Optional, the U(2) matrix of the gates, which can be cached for efficiency.
    // This is invalidated upon any operation and can be recomputed via the ``u2`` method.
    pub matrix_u2: Option<Matrix2<Complex<f64>>>,
}

impl GateSequence {
    /// Create a new, empty sequence.
    fn new() -> Self {
        Self {
            gates: Some(vec![]),
            matrix_so3: Matrix3::identity(),
            phase: 0.,
            matrix_u2: None,
        }
    }

    /// Get the gate labels.
    pub fn label(&self) -> String {
        match &self.gates {
            Some(gates) => gates.iter().map(|gate| gate.name()).collect(),
            None => "none".to_string(),
        }
    }

    /// Initialize a from a U(2) matrix. This sequence does not have gates.
    pub fn from_u2(matrix_u2: &Matrix2<Complex64>, do_checks: bool) -> Self {
        // turn into a SU(2) matrix and compute SO(3) representation from it
        let determinant =
            matrix_u2[(0, 0)] * matrix_u2[(1, 1)] - matrix_u2[(1, 0)] * matrix_u2[(0, 1)];
        let matrix_su2 = matrix_u2.div(determinant.sqrt());
        let matrix_so3 = su2_to_so3(&matrix_su2);
        let phase = determinant.sqrt().arg(); // possibly map to [0, 2pi)

        if do_checks {
            math::assert_so3("Matrix generated from U(2)", &matrix_so3);
        }

        Self {
            gates: None,
            matrix_so3,
            phase: (phase),
            matrix_u2: Some(matrix_u2.clone()),
        }
    }

    /// Initialize from a SO(3) matrix. This sequence does not have gates.
    pub fn from_so3(matrix_so3: &Matrix3<f64>, do_checks: bool) -> Self {
        if do_checks {
            math::assert_so3("SO(3) input matrix", matrix_so3);
        }

        Self {
            gates: None,
            matrix_so3: *matrix_so3,
            phase: 0.,
            matrix_u2: None,
        }
    }

    /// Initialize from a [StandardGate] plus parameters.
    pub fn from_std(gate: &StandardGate, params: &[Param]) -> Result<Self, DiscreteBasisError> {
        let (matrix_so3, phase) = standard_gates_to_so3(gate, params)?;
        let matrix_u2 = standard_gates_to_u2(gate, params)?;

        Ok(Self {
            gates: Some(vec![*gate]),
            matrix_so3,
            phase,
            matrix_u2: Some(matrix_u2),
        })
    }

    /// Merge two [GateSequence]s.
    ///
    /// ``self.dot(other)`` results in a sequence where the gates are ``other.gates + self.gates``.
    pub fn dot(&self, other: &GateSequence) -> GateSequence {
        // merge the gates
        let gates = match (&self.gates, &other.gates) {
            (Some(self_gates), Some(other_gates)) => {
                let mut joint_gates = Vec::with_capacity(other_gates.len() + self_gates.len());
                joint_gates.extend_from_slice(other_gates);
                joint_gates.extend_from_slice(self_gates);
                Some(joint_gates)
            }
            (None, None) => None,
            _ => panic!("Incompatible gates in dot."),
        };

        // update the matrices and global phase
        let phase = self.phase + other.phase; // map to [0, 2pi)
        let matrix_so3 = self.matrix_so3 * other.matrix_so3;

        Self {
            gates,
            matrix_so3,
            phase,
            matrix_u2: None, // just invalidate, recompute only when needed
        }
    }

    /// Return the adjoint.
    pub fn adjoint(&self) -> GateSequence {
        // Flip the gate order and invert them
        let gates = self.gates.as_ref().map(|gates| {
            gates
                .iter()
                .rev()
                .map(|&gate| gate.inverse(&[]).unwrap().0)
                .collect()
        });

        // The transpose of an orthogonal matrix is equal to its inverse
        let matrix_so3 = self.matrix_so3.transpose();

        Self {
            gates,
            matrix_so3,
            phase: -self.phase,
            matrix_u2: None, // just invalidate, recompute only when needed
        }
    }

    /// Remove gate-inverse pairs in-place.
    pub fn inverse_cancellation(&mut self) {
        let gates = match &self.gates {
            Some(gates) => gates,
            None => return, // no gates set, nothing to cancel
        };
        if gates.len() < 2 {
            return; // if there is only 1 gate, there is nothing to cancel
        }

        let mut reduced_gates: Vec<StandardGate> = Vec::with_capacity(gates.len());
        let mut index = 0;
        while index < gates.len() - 1 {
            let inverse = gates[index].inverse(&[]).expect("Failed to get inverse").0;
            if inverse == gates[index + 1] {
                index += 2; // skip the gate-inverse-pair
            } else {
                reduced_gates.push(gates[index]);
                index += 1;
            }
        }
        // add the last gate if it was not considered yet
        if index == gates.len() - 1 {
            reduced_gates.push(gates[index]);
        }

        // we managed to cancel something, recurse, since we may have uncovered new cancellations
        if gates.len() > reduced_gates.len() {
            self.gates = Some(reduced_gates);
            self.inverse_cancellation();
        }
    }

    /// Get the U(2) matrix implemented by the gates. Fails if the gates are ``None`` and no U(2)
    /// matrix is cached.
    pub fn u2(&self) -> Result<Matrix2<Complex<f64>>, DiscreteBasisError> {
        if let Some(matrix_u2) = self.matrix_u2 {
            return Ok(matrix_u2);
        }

        let gates = match &self.gates {
            Some(gates) => gates,
            None => return Err(DiscreteBasisError::UninitializedSequence),
        };

        let mut out = Matrix2::identity();
        for gate in gates {
            let matrix = standard_gates_to_u2(gate, &[])?;
            out = math::matmul_bigcomplex(&matrix, &out);
        }
        Ok(out)
    }

    /// Compute the phase the sequence needs to match the target sequence.
    ///
    /// This assumes that [self] is a good approximation to ``target``, otherwise the result
    /// may not make sense.
    pub fn compute_phase(&self, target: &GateSequence) -> Result<f64, DiscreteBasisError> {
        let self_u2 = self.u2()?;
        let target_u2 = target.u2()?;
        let (target_first, self_first) = target_u2
            .iter()
            .zip(self_u2.iter())
            .find(|(&el, _)| el.abs() >= 1. / 2.)
            .expect("At least one element in the unitary must be >= 1/2.");

        // When we convert SU(2) to SO(3) we lose sign information, which translates to a
        // global phase uncertainty of +-1 = exp(i pi). We fix this here by checking which phase
        // is a better match (one should be clearly correct if the algorithm converged).
        let phase_candidate = self.phase - target.phase;
        let coeff_candidate = Complex::new(0., phase_candidate).exp();
        let candidate = self_first * coeff_candidate;

        if (target_first - candidate).abs() < (target_first + candidate).abs() {
            // phase candidate is correct
            Ok(phase_candidate)
        } else {
            // off by a -1 sign, so shift by PI
            Ok(phase_candidate + f64::PI())
        }
    }

    /// Convert the sequence to a circuit.
    ///
    /// If a target sequence is given, match the phase of [self] to the target.
    pub fn to_circuit(
        &self,
        target: Option<&GateSequence>,
    ) -> Result<CircuitData, DiscreteBasisError> {
        let gates = match &self.gates {
            Some(gates) => gates,
            None => return Err(DiscreteBasisError::UninitializedSequence),
        };

        let global_phase = match target {
            Some(target) => Param::Float(self.compute_phase(target)?),
            None => Param::Float(self.phase),
        };

        let mut circuit = CircuitData::with_capacity(1, 0, gates.len(), global_phase).unwrap();
        for gate in gates {
            circuit.push_standard_gate(*gate, &[], &[Qubit(0)]);
        }
        Ok(circuit)
    }

    /// Push a new standard gate onto [self].
    fn push(&mut self, gate: StandardGate) -> Result<(), DiscreteBasisError> {
        // turn into a SU(2) matrix and compute SO(3) representation from it
        let (so3_matrix, phase) = standard_gates_to_so3(&gate, &[])?;

        // update matrix representations and keep track of the gate
        self.matrix_so3 = so3_matrix * self.matrix_so3;
        self.phase += phase;
        match &mut self.gates {
            Some(gates) => gates.push(gate),
            None => return Err(DiscreteBasisError::UninitializedSequence),
        }

        Ok(())
    }

    /// Return an iterator that adds every gate in ``additions`` to the current sequence.
    fn iter_additions<'a>(
        &'a self,
        additions: &'a [StandardGate],
    ) -> impl Iterator<Item = Result<GateSequence, DiscreteBasisError>> + 'a {
        additions.iter().map(|gate| {
            let mut out = self.clone();
            out.push(*gate)?;
            Ok(out)
        })
    }
}

#[inline]
fn array2_to_matrix2<T: Copy>(view: &ArrayView2<T>) -> Matrix2<T> {
    Matrix2::new(view[[0, 0]], view[(0, 1)], view[(1, 0)], view[(1, 1)])
}

fn su2_to_so3(view: &Matrix2<Complex64>) -> Matrix3<f64> {
    let a = view[(0, 0)].re;
    let b = view[(0, 0)].im;
    let c = -view[(0, 1)].re;
    let d = -view[(0, 1)].im;

    Matrix3::new(
        a.powi(2) - b.powi(2) - c.powi(2) + d.powi(2),
        2.0 * (a * b + c * d),
        2.0 * (b * d - a * c),
        2.0 * (c * d - a * b),
        a.powi(2) - b.powi(2) + c.powi(2) - d.powi(2),
        2.0 * (a * d + b * c),
        2.0 * (a * c + b * d),
        2.0 * (b * c - a * d),
        a.powi(2) + b.powi(2) - c.powi(2) - d.powi(2),
    )
}

/// A point in the R* tree. Contains the SO(3) representation of the gate sequence, plus an
/// optional index to retrieve the gate sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct BasicPoint {
    point: [f64; 9],      // SO(3) representation
    index: Option<usize>, // index to a gate sequence -- could explore using GateSequence directly
}

impl BasicPoint {
    pub fn from_sequence(sequence: &GateSequence, index: Option<usize>) -> Self {
        Self {
            point: ::core::array::from_fn(|i| sequence.matrix_so3[(i % 3, i / 3)]),
            index,
        }
    }
}

impl Point for BasicPoint {
    type Scalar = f64;
    const DIMENSIONS: usize = 9;

    fn generate(generator: impl FnMut(usize) -> Self::Scalar) -> Self {
        BasicPoint {
            point: ::core::array::from_fn(generator),
            index: None, // this point has no associated index
        }
    }

    fn nth(&self, index: usize) -> Self::Scalar {
        self.point[index]
    }

    fn nth_mut(&mut self, index: usize) -> &mut Self::Scalar {
        &mut self.point[index]
    }
}

/// Get the U(2) representation of a standard gate.
pub fn standard_gates_to_u2(
    gate: &StandardGate,
    params: &[Param],
) -> Result<Matrix2<Complex<f64>>, DiscreteBasisError> {
    let matrix_c64 = gate.matrix(params).expect("Failed to get matrix.");
    Ok(array2_to_matrix2(&matrix_c64.view()))
}

/// Get the SO(3) representation of a standard gate.
///
/// Attempts to directly construct the matrix using [f64] accuracy, otherwise falls back
/// to matrix construction and conversion.
fn standard_gates_to_so3(
    gate: &StandardGate,
    params: &[Param],
) -> Result<(Matrix3<f64>, f64), DiscreteBasisError> {
    match gate {
        StandardGate::T => {
            let so3 = Matrix3::new(
                FRAC_1_SQRT_2,
                -FRAC_1_SQRT_2,
                0.,
                FRAC_1_SQRT_2,
                FRAC_1_SQRT_2,
                0.,
                0.,
                0.,
                1.,
            );
            let phase = -f64::FRAC_PI_8();
            Ok((so3, phase))
        }
        StandardGate::Tdg => {
            let so3 = Matrix3::new(
                FRAC_1_SQRT_2,
                FRAC_1_SQRT_2,
                0.,
                -FRAC_1_SQRT_2,
                FRAC_1_SQRT_2,
                0.,
                0.,
                0.,
                1.,
            );
            let phase = f64::FRAC_PI_8();
            Ok((so3, phase))
        }
        StandardGate::S => {
            let so3 = Matrix3::new(0., -1., 0., 1., 0., 0., 0., 0., 1.);
            let phase = -f64::FRAC_PI_4();
            Ok((so3, phase))
        }
        StandardGate::Sdg => {
            let so3 = Matrix3::new(0., 1., 0., -1., 0., 0., 0., 0., 1.);
            let phase = f64::FRAC_PI_4();
            Ok((so3, phase))
        }
        StandardGate::H => {
            let so3 = Matrix3::new(0., 0., -1., 0., -1., 0., -1., 0., 0.);
            let phase = f64::FRAC_PI_2();
            Ok((so3, phase))
        }
        StandardGate::RZ => {
            let angle = match params[0] {
                Param::Float(angle) => angle,
                _ => return Err(DiscreteBasisError::ParameterizedGate),
            };
            let cos = (angle / 2.).cos();
            let sin = (angle / 2.).sin();
            let so3_00 = cos.powi(2) - sin.powi(2);
            let so3_10 = 2. * cos * sin;
            let so3 = Matrix3::new(so3_00, -so3_10, 0., so3_10, so3_00, 0., 0., 0., 1.);
            Ok((so3, 0.))
        }
        _ => {
            let array_u2 = gate
                .matrix(params)
                .expect("Failed to get matrix representation.");
            let matrix_u2 = array2_to_matrix2(&array_u2.view());
            let det = matrix_u2[(0, 0)] * matrix_u2[(1, 1)] - matrix_u2[(1, 0)] * matrix_u2[(0, 1)];
            let smatrix_u2 = matrix_u2.div(det.sqrt());
            let so3 = su2_to_so3(&smatrix_u2);

            let phase = det.sqrt().arg();
            Ok((so3, (phase)))
        }
    }
}

/// The basic approximations for Solovay Kitaev.
///
/// This struct allows to construct a tree of basic approximations and to query the closest
/// sequence given an target sequence (or SO(3) matrix).
#[derive(Debug)]
pub struct BasicApproximations {
    /// All points as flattened SO(3) matrix stored in a R* tree. This does not include the
    /// sequence of gates, see ``approximations``.
    pub points: RTree<BasicPoint>,
    /// A map relating the indices in the R* tree to a sequence of gates. This allows to
    /// retrieve the gates implementing a SO(3) "point" in the tree.
    pub approximations: HashMap<usize, GateSequence>,
}

impl BasicApproximations {
    /// Generate a tree of basic approximations from a set of discrete standard gates and a
    /// maximum depth.
    ///
    /// This will compute an SO(3) representation of any sequence of gates in ``basis_gates`` of
    /// length up to ``depth`` and store it in a tree structure, if there is no other sequence
    /// within a radius of ``sqrt(tol)``.
    ///
    /// All gates must be single-qubit, discrete (i.e. take no parameter) gates.
    ///
    /// # Args
    ///
    /// - ``basis_gates`` - A slice of [StandardGate]s to use in basic approximation.
    /// - ``depth`` - The maximum gate depth of the basic approximations.
    /// - ``tol`` - Control the granularity of the tree; new sequences are accepted if they
    ///     are further than ``sqrt(tol)`` from an existing element.
    pub fn generate_from(
        basis_gates: &[StandardGate],
        depth: usize,
        tol: Option<f64>,
    ) -> Result<Self, DiscreteBasisError> {
        let mut points: RTree<BasicPoint> = RTree::new();
        let mut approximations: HashMap<usize, GateSequence> = HashMap::new();

        // identity approximation
        let root = GateSequence::new();
        points.insert(BasicPoint::from_sequence(&root, Some(0)));
        approximations.insert(0, root);
        let mut index = 1;

        let mut this_level: Vec<GateSequence> = vec![GateSequence::new()];
        let mut next_level: Vec<GateSequence> = Vec::new();
        let radius_sq = tol.unwrap_or(1e-14);

        for _ in 0..depth {
            for node in this_level.iter() {
                for candidate in node.iter_additions(basis_gates) {
                    let candidate = candidate?;
                    let point = BasicPoint::from_sequence(&candidate, Some(index));
                    if points
                        .locate_within_distance(point.clone(), radius_sq)
                        .next()
                        .is_none()
                    {
                        // we don't have this point yet
                        points.insert(point);
                        approximations.insert(index, candidate.clone());
                        index += 1;
                        next_level.push(candidate);
                    }
                }
            }
            this_level.clone_from(&next_level);
            next_level.clear();
        }

        Ok(Self {
            points,
            approximations,
        })
    }

    /// Query the closest point to a [GateSequence].
    pub fn query(&self, sequence: &GateSequence) -> Option<&GateSequence> {
        let query_point = BasicPoint::from_sequence(sequence, None);
        let point = self.points.nearest_neighbor(&query_point).map(|point| {
            let index = point
                .index
                .expect("All registered sequences should have an index. Blame a dev.");
            let best = self
                .approximations
                .get(&index)
                .expect("All available indices should have a sequence. Also blame a dev.");
            best
        });
        point
    }
}

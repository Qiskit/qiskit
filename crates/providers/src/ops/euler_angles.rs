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

//! An op that resolves single-qubit unitaries into Euler angles.

use std::f64::consts::PI;

use ndarray::{ArrayBase, ArrayD, Axis, Data, IxDyn};
use num_complex::{Complex, Complex64};

use super::error::MathOpError;
use super::{ProgramOp, QISKIT};
use crate::tensor::{DType, Dim, Tensor, TensorType, fmt_shape};

/// The angle below which two unitaries are treated as the same rotation.
///
/// This matches `ANGLE_ZERO_EPSILON` in Qiskit's one-qubit decomposer.
pub const DEFAULT_ATOL: f64 = 1e-12;

/// How many angles describe a single-qubit unitary.
const NUM_ANGLES: usize = 3;

/// Compute ZYZ Euler angles from 2x2 unitaries.
///
/// This operation transforms `C128[..., 2, 2]` or `C64[..., 2, 2]` into `F64[..., 3]`,
/// where the last axis of the output has order `[lambda, theta, phi]` such that
/// the temporal gate ordering `rz(phi) ry(theta) rz(lambda)` is equal to
/// the associated 2x2 unitary up to a global phase.
///
/// Diagonal and anti-diagonal unitaries have the one-dimensional freedom to arbitrarily
/// distribute phase between `phi` and `lambda`. This implementation guarantees the following
/// convention whenever (anti-)diagonality is met within [`Self::atol`]:
///
/// ```text
/// theta ~ 0:   [lambda + phi, 0,  0]
/// theta ~ pi:  [lambda - phi, pi, 0]
/// ```

#[derive(Clone, Debug)]
pub struct EulerAnglesZYZ {
    atol: f64,
}

impl Default for EulerAnglesZYZ {
    fn default() -> Self {
        EulerAnglesZYZ::new(DEFAULT_ATOL)
    }
}

impl EulerAnglesZYZ {
    /// Construct an `EulerAngles` op resolving degenerate unitaries within `atol`.
    pub fn new(atol: f64) -> Self {
        Self { atol }
    }

    /// Return the angle below which two unitaries are treated as the same rotation.
    pub fn atol(&self) -> f64 {
        self.atol
    }

    /// Return the angles of every unitary in `matrices`.
    fn angles_of<A, S>(&self, matrices: &ArrayBase<S, IxDyn>) -> ArrayD<f64>
    where
        A: Into<f64> + Copy,
        S: Data<Elem = Complex<A>>,
    {
        let shape = matrices.shape();
        let mut angles = Vec::with_capacity(matrices.len() / 4 * NUM_ANGLES);
        match matrices.as_slice() {
            // A row-major operand already holds each matrix as four consecutive elements.
            Some(flat) => {
                for matrix in flat.chunks_exact(4) {
                    angles.extend(
                        self.angles_of_matrix([matrix[0], matrix[1], matrix[2], matrix[3]]),
                    );
                }
            }
            // Otherwise the elements are gathered per matrix. The two trailing axes are the
            // matrix, so folding the leading ones leaves a stack of 2x2 blocks to iterate.
            None => {
                let batch = shape[..shape.len() - 2].iter().product::<usize>();
                let stack = matrices
                    .to_shape(IxDyn(&[batch, 2, 2]))
                    .expect("the operand's element count is its batch size times four");
                for matrix in stack.axis_iter(Axis(0)) {
                    angles.extend(self.angles_of_matrix([
                        matrix[[0, 0]],
                        matrix[[0, 1]],
                        matrix[[1, 0]],
                        matrix[[1, 1]],
                    ]));
                }
            }
        }

        let mut shape = shape[..shape.len() - 2].to_vec();
        shape.push(NUM_ANGLES);
        ArrayD::from_shape_vec(IxDyn(&shape), angles)
            .expect("three angles per unitary in the operand")
    }

    /// Return the angles of one unitary given its elements in row-major order.
    fn angles_of_matrix<A: Into<f64> + Copy>(&self, matrix: [Complex<A>; 4]) -> [f64; NUM_ANGLES] {
        let [m00, m01, m10, m11] = matrix.map(|z| Complex64::new(z.re.into(), z.im.into()));
        let phase = (m00 * m11 - m01 * m10).sqrt().inv();
        let theta = 2.0 * m10.norm().atan2(m00.norm());
        let [lambda, theta, phi] = if theta < self.atol {
            [(phase * m11).arg() * 2.0, 0.0, 0.0]
        } else if (PI - theta) < self.atol {
            [(phase * m10).arg() * -2.0, PI, 0.0]
        } else {
            let sum = (phase * m11).arg();
            let difference = (phase * m10).arg();
            [sum - difference, theta, sum + difference]
        };
        [wrap(lambda), theta, wrap(phi)]
    }
}

/// Return `angle` wrapped into `(-pi, pi]`.
fn wrap(angle: f64) -> f64 {
    PI - (PI - angle).rem_euclid(2.0 * PI)
}

/// Return the leading axes of a shape whose two trailing axes are each of extent two.
fn leading_axes(shape: &[Dim]) -> Option<&[Dim]> {
    match shape.split_last() {
        Some((&Dim::Fixed(2), rest)) => match rest.split_last() {
            Some((&Dim::Fixed(2), batch)) => Some(batch),
            _ => None,
        },
        _ => None,
    }
}

impl ProgramOp for EulerAnglesZYZ {
    type Error = MathOpError;

    fn name(&self) -> &str {
        "euler_angles_zyz"
    }
    fn namespace(&self) -> &str {
        QISKIT
    }
    fn describe(&self) -> Option<String> {
        Some(format!("atol={:e}", self.atol))
    }
    fn arity(&self) -> usize {
        1
    }
    fn has_builtin_eval(&self) -> bool {
        true
    }
    fn infer_output_types(&self, inputs: &[TensorType]) -> Result<Vec<TensorType>, Self::Error> {
        crate::unpack_operands!(self, inputs, [x]);
        if !matches!(x.dtype, DType::C64 | DType::C128) {
            return Err(MathOpError::UnsupportedDType {
                operand: 0,
                dtype: x.dtype,
            });
        }
        let batch = leading_axes(&x.shape).ok_or_else(|| MathOpError::WrongTrailingShape {
            operand: 0,
            expected: "[2, 2]".to_string(),
            actual: fmt_shape(&x.shape),
        })?;
        let mut shape = batch.to_vec();
        shape.push(Dim::Fixed(NUM_ANGLES));
        Ok(vec![TensorType {
            dtype: DType::F64,
            shape,
        }])
    }
    fn eval(&self, args: &[Tensor]) -> Result<Vec<Tensor>, Self::Error> {
        crate::unpack_operands!(self, args, [x]);
        let angles = match x {
            Tensor::C64(matrices) => self.angles_of(matrices),
            Tensor::C128(matrices) => self.angles_of(matrices),
            other => {
                return Err(MathOpError::UnsupportedDType {
                    operand: 0,
                    dtype: other.dtype(),
                });
            }
        };
        Ok(vec![Tensor::from(angles)])
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::{Array2, arr2};

    /// Return a `TensorType` of `dtype` over fixed axes `shape`.
    fn ty(dtype: DType, shape: &[usize]) -> TensorType {
        TensorType {
            dtype,
            shape: shape.iter().copied().map(Dim::Fixed).collect(),
        }
    }

    /// Return `rz(phi) ry(theta) rz(lambda)`, scaled by `phase`.
    fn unitary(lambda: f64, theta: f64, phi: f64, phase: f64) -> Array2<Complex64> {
        let z = |angle: f64| Complex64::from_polar(1.0, angle);
        let (cos, sin) = ((theta / 2.0).cos(), (theta / 2.0).sin());
        arr2(&[
            [
                z(phase) * z(-(phi + lambda) / 2.0) * cos,
                z(phase) * z(-(phi - lambda) / 2.0) * -sin,
            ],
            [
                z(phase) * z((phi - lambda) / 2.0) * sin,
                z(phase) * z((phi + lambda) / 2.0) * cos,
            ],
        ])
    }

    /// Return the three angles the op resolves a single `matrix` into.
    fn resolve_one(matrix: Array2<Complex64>) -> [f64; 3] {
        let operand = Tensor::from(matrix.into_dyn());
        let resolved = EulerAnglesZYZ::default()
            .eval(std::slice::from_ref(&operand))
            .unwrap();
        let Tensor::F64(angles) = &resolved[0] else {
            panic!("the angles are F64")
        };
        let values = angles.as_slice().expect("a contiguous result");
        [values[0], values[1], values[2]]
    }

    /// Assert that `actual` and `expected` agree to within a generous numerical tolerance.
    fn assert_close(actual: [f64; 3], expected: [f64; 3], context: &str) {
        for (index, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (got - want).abs() < 1e-9,
                "angle {index} of {context}: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn test_euler_angles_full_name_and_arity() {
        let op = EulerAnglesZYZ::default();
        assert_eq!(op.full_name(), "qiskit.euler_angles");
        assert_eq!(op.arity(), 1);
        assert_eq!(op.atol(), DEFAULT_ATOL);
        assert!(op.has_builtin_eval());
        assert_eq!(op.describe().unwrap(), "atol=1e-12");
    }

    #[test]
    fn test_the_two_trailing_axes_become_one_of_three() {
        for (operand, result) in [
            (vec![2, 2], vec![3]),
            (vec![4, 2, 2], vec![4, 3]),
            (vec![5, 1, 2, 2], vec![5, 1, 3]),
            (vec![0, 2, 2], vec![0, 3]),
        ] {
            assert_eq!(
                EulerAnglesZYZ::default()
                    .infer_output_types(&[ty(DType::C128, &operand)])
                    .unwrap(),
                vec![ty(DType::F64, &result)],
                "for operand shape {operand:?}"
            );
        }
    }

    #[test]
    fn test_a_bounded_batch_axis_is_carried_through() {
        let bounded = Dim::Bounded { max: 32 };
        assert_eq!(
            EulerAnglesZYZ::default()
                .infer_output_types(&[TensorType {
                    dtype: DType::C128,
                    shape: vec![bounded, Dim::Fixed(2), Dim::Fixed(2)],
                }])
                .unwrap(),
            vec![TensorType {
                dtype: DType::F64,
                shape: vec![bounded, Dim::Fixed(3)],
            }],
            "the batch prefix is opaque, so a bound on it needs no resolving"
        );
    }

    #[test]
    fn test_a_shape_that_is_not_a_stack_of_matrices_is_refused() {
        for shape in [vec![], vec![2], vec![3, 3], vec![2, 3], vec![3, 2]] {
            let operand = ty(DType::C128, &shape);
            assert!(
                matches!(
                    EulerAnglesZYZ::default().infer_output_types(std::slice::from_ref(&operand)).unwrap_err(),
                    MathOpError::WrongTrailingShape { operand: 0, ref expected, .. }
                        if expected == "[2, 2]"
                ),
                "for shape {shape:?}"
            );
        }

        // A bounded matrix axis is refused, even when size 2
        assert!(matches!(
            EulerAnglesZYZ::default()
                .infer_output_types(&[TensorType {
                    dtype: DType::C128,
                    shape: vec![Dim::Bounded { max: 2 }, Dim::Fixed(2)],
                }])
                .unwrap_err(),
            MathOpError::WrongTrailingShape { operand: 0, .. }
        ));
    }

    #[test]
    fn test_a_real_operand_is_refused() {
        for dtype in [DType::F64, DType::F32, DType::I32, DType::Bit] {
            let operand = ty(dtype, &[2, 2]);
            assert!(
                matches!(
                    EulerAnglesZYZ::default().infer_output_types(std::slice::from_ref(&operand)).unwrap_err(),
                    MathOpError::UnsupportedDType { operand: 0, dtype: refused }
                        if refused == dtype
                ),
                "for dtype {dtype}"
            );
        }
        assert!(matches!(
            EulerAnglesZYZ::default()
                .eval(&[Tensor::from([1.0_f64, 2.0, 3.0, 4.0])])
                .unwrap_err(),
            MathOpError::UnsupportedDType {
                operand: 0,
                dtype: DType::F64
            }
        ));
    }

    #[test]
    fn test_the_angles_invert_the_rotation_they_describe() {
        for &lambda in &[-2.5, -0.4, 0.0, 1.1, 3.0] {
            for &theta in &[0.3, 1.0, PI / 2.0, 2.2, 2.9] {
                for &phi in &[-3.0, -1.2, 0.0, 0.7, 2.6] {
                    let expected = [wrap(lambda), theta, wrap(phi)];
                    assert_close(
                        resolve_one(unitary(lambda, theta, phi, 0.0)),
                        expected,
                        &format!("rz({phi}) ry({theta}) rz({lambda})"),
                    );
                }
            }
        }
    }

    #[test]
    fn test_a_global_phase_on_the_operand_is_ignored() {
        let (lambda, theta, phi) = (0.8, 1.3, -2.1);
        let expected = resolve_one(unitary(lambda, theta, phi, 0.0));
        for phase in [0.5, PI / 2.0, PI, -2.0] {
            assert_close(
                resolve_one(unitary(lambda, theta, phi, phase)),
                expected,
                &format!("a global phase of {phase}"),
            );
        }
    }

    #[test]
    fn test_every_angle_is_wrapped_and_theta_is_never_negative() {
        for &lambda in &[-6.0, -3.2, 3.2, 6.0] {
            for &theta in &[0.0, 0.1, PI / 2.0, PI - 0.1, PI] {
                for &phi in &[-6.0, -3.2, 3.2, 6.0] {
                    let [lambda, theta, phi] = resolve_one(unitary(lambda, theta, phi, 0.0));
                    assert!(
                        (0.0..=PI).contains(&theta),
                        "theta {theta} is outside [0, pi]"
                    );
                    for angle in [lambda, phi] {
                        assert!(
                            angle > -PI - 1e-12 && angle <= PI + 1e-12,
                            "angle {angle} is outside (-pi, pi]"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_a_z_rotation_gives_away_its_phi() {
        let identity = unitary(0.0, 0.0, 0.0, 0.0);
        assert_close(resolve_one(identity), [0.0, 0.0, 0.0], "the identity");

        for (lambda, phi) in [(0.9, 0.4), (0.4, 0.9), (1.3, 0.0), (0.0, 1.3)] {
            assert_close(
                resolve_one(unitary(lambda, 0.0, phi, 0.0)),
                [wrap(lambda + phi), 0.0, 0.0],
                &format!("rz({phi}) rz({lambda})"),
            );
        }

        // `s` is a z rotation by pi/2 that no test above spells out directly.
        let s = arr2(&[
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(0.0, 1.0)],
        ]);
        assert_close(resolve_one(s), [PI / 2.0, 0.0, 0.0], "s");
    }

    #[test]
    fn test_a_pi_pulse_gives_away_its_phi() {
        for (lambda, phi) in [(0.9, 0.4), (0.4, 0.9), (1.3, 0.0), (0.0, 1.3)] {
            assert_close(
                resolve_one(unitary(lambda, PI, phi, 0.0)),
                [wrap(lambda - phi), PI, 0.0],
                &format!("rz({phi}) ry(pi) rz({lambda})"),
            );
        }

        // `x` is a pi pulse about x, which is ry(pi) preceded by a z rotation of pi.
        let x = arr2(&[
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        ]);
        let [lambda, theta, _] = resolve_one(x);
        assert!(
            (theta - PI).abs() < 1e-9,
            "x is a pi pulse, got theta {theta}"
        );
        assert!(
            (lambda.abs() - PI).abs() < 1e-9,
            "x differs from ry(pi) by a z rotation of pi, got lambda {lambda}"
        );
    }

    #[test]
    fn test_the_resolved_angles_are_continuous_at_the_degenerate_points() {
        let op = EulerAnglesZYZ::new(1e-6);
        let resolve = |matrix: Array2<Complex64>| {
            let operand = Tensor::from(matrix.into_dyn());
            let resolved = op.eval(&[operand]).unwrap();
            let Tensor::F64(angles) = &resolved[0] else {
                panic!("the angles are F64")
            };
            let values = angles.as_slice().unwrap();
            [values[0], values[1], values[2]]
        };
        let (lambda, phi) = (0.7, -0.4);

        // Near theta = 0 the sum survives, and the general branch splits it between the two slots.
        let near = resolve(unitary(lambda, 1e-4, phi, 0.0));
        let folded = resolve(unitary(lambda, 1e-9, phi, 0.0));
        assert!(
            ((near[0] + near[2]) - folded[0]).abs() < 1e-6,
            "near zero: {near:?} against {folded:?}"
        );

        // Near theta = pi the difference survives instead.
        let near = resolve(unitary(lambda, PI - 1e-4, phi, 0.0));
        let folded = resolve(unitary(lambda, PI - 1e-9, phi, 0.0));
        assert!(
            ((near[0] - near[2]) - folded[0]).abs() < 1e-6,
            "near pi: {near:?} against {folded:?}"
        );
    }

    #[test]
    fn test_atol_decides_which_unitaries_are_degenerate() {
        let (lambda, theta, phi) = (0.6, 1e-3, -0.2);
        let operand = Tensor::from(unitary(lambda, theta, phi, 0.0).into_dyn());

        let resolve = |atol: f64| {
            let resolved = EulerAnglesZYZ::new(atol)
                .eval(std::slice::from_ref(&operand))
                .unwrap();
            let Tensor::F64(angles) = &resolved[0] else {
                panic!("the angles are F64")
            };
            angles.as_slice().unwrap().to_vec()
        };

        assert!(
            (resolve(1e-2)[1]).abs() < 1e-12,
            "a tolerance above the rotation folds it to a z rotation"
        );
        assert!(
            (resolve(1e-6)[1] - theta).abs() < 1e-9,
            "a tolerance below it resolves the rotation in full"
        );
    }

    #[test]
    fn test_a_batch_is_resolved_position_by_position() {
        // Each unitary must be read from its own block, so the batch is given distinct angles and
        // compared against resolving each one alone.
        let cases = [(0.3, 0.9, -1.2), (-2.0, 2.4, 0.5), (1.5, 0.0, 0.8)];
        let mut stacked = ArrayD::zeros(IxDyn(&[cases.len(), 2, 2]));
        for (index, &(lambda, theta, phi)) in cases.iter().enumerate() {
            let matrix = unitary(lambda, theta, phi, 0.0);
            for row in 0..2 {
                for column in 0..2 {
                    stacked[[index, row, column]] = matrix[[row, column]];
                }
            }
        }

        let resolved = EulerAnglesZYZ::default()
            .eval(&[Tensor::from(stacked)])
            .unwrap();
        assert_eq!(resolved[0].shape(), &[cases.len(), 3]);
        let Tensor::F64(angles) = &resolved[0] else {
            panic!("the angles are F64")
        };
        for (index, &(lambda, theta, phi)) in cases.iter().enumerate() {
            let row = [angles[[index, 0]], angles[[index, 1]], angles[[index, 2]]];
            assert_close(
                row,
                resolve_one(unitary(lambda, theta, phi, 0.0)),
                &format!("batch position {index}"),
            );
        }
    }

    #[test]
    fn test_a_strided_operand_is_read_in_logical_order() {
        // `to_shape` on a non-contiguous operand must copy rather than reinterpret, so a
        // transposed batch is compared against the same matrices laid out contiguously.
        let cases = [(0.3, 0.9, -1.2), (-2.0, 2.4, 0.5)];
        let mut stacked = ArrayD::zeros(IxDyn(&[cases.len(), 2, 2]));
        for (index, &(lambda, theta, phi)) in cases.iter().enumerate() {
            let matrix = unitary(lambda, theta, phi, 0.0);
            for row in 0..2 {
                for column in 0..2 {
                    stacked[[index, row, column]] = matrix[[row, column]];
                }
            }
        }

        let mut transposed = stacked.view();
        transposed.swap_axes(1, 2);
        let resolved = EulerAnglesZYZ::default()
            .eval(&[Tensor::from(transposed.to_owned())])
            .unwrap();
        let Tensor::F64(angles) = &resolved[0] else {
            panic!("the angles are F64")
        };

        for (index, &(lambda, theta, phi)) in cases.iter().enumerate() {
            let matrix = unitary(lambda, theta, phi, 0.0);
            let flipped = arr2(&[
                [matrix[[0, 0]], matrix[[1, 0]]],
                [matrix[[0, 1]], matrix[[1, 1]]],
            ]);
            let row = [angles[[index, 0]], angles[[index, 1]], angles[[index, 2]]];
            assert_close(
                row,
                resolve_one(flipped),
                &format!("transposed position {index}"),
            );
        }
    }

    #[test]
    fn test_an_empty_batch_resolves_to_no_angles() {
        let operand = Tensor::from(ArrayD::<Complex64>::zeros(IxDyn(&[0, 2, 2])));
        let resolved = EulerAnglesZYZ::default().eval(&[operand]).unwrap();
        assert_eq!(resolved[0].shape(), &[0, 3]);
        assert_eq!(resolved[0].dtype(), DType::F64);
    }

    #[test]
    fn test_wrap_lands_on_the_closed_end_of_the_interval() {
        assert_eq!(wrap(PI), PI);
        assert_eq!(wrap(-PI), PI);
        assert_eq!(wrap(0.0), 0.0);
        for (angle, expected) in [
            (PI / 2.0, PI / 2.0),
            (-PI / 2.0, -PI / 2.0),
            (3.0 * PI / 2.0, -PI / 2.0),
            (2.0 * PI, 0.0),
        ] {
            assert!(
                (wrap(angle) - expected).abs() < 1e-12,
                "wrap({angle}) was {}, want {expected}",
                wrap(angle)
            );
        }
    }

    #[test]
    fn test_the_angles_are_ordinary_values_in_a_program() {
        use crate::ops::Mean;
        use crate::program::ProgramFunction;

        let mut function = ProgramFunction::new();
        let matrices = function.add_parameter(ty(DType::C128, &[6, 2, 2]));
        let angles = function
            .add_op(EulerAnglesZYZ::default(), &[matrices])
            .unwrap()[0];
        let mean = function.add_op(Mean::new(0), &[angles]).unwrap()[0];

        assert_eq!(function.type_of(angles), Some(&ty(DType::F64, &[6, 3])));
        assert_eq!(function.type_of(mean), Some(&ty(DType::F64, &[3])));
    }
}

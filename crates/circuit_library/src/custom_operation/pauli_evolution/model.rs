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

use std::sync::Arc;

use nalgebra::DMatrix;
use ndarray::Array2;
use num_complex::Complex64;
use qiskit_circuit::operations::Param;
use qiskit_quantum_info::sparse_observable::{MatrixError, SparseObservable};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PauliEvolutionError {
    #[error("time not float or expression")]
    TimeInvalid,
    #[error("time not evaluated")]
    TimeNotEvaluated,
    #[error("matrix error")]
    Matrix(#[from] MatrixError),
}

/// Time-evolution of an operator.
///
/// For an operator **H** and time **t**, this gate represents the unitary
/// **U(t) = e<sup>-itH</sup>**.
#[derive(Debug, Clone, PartialEq)]
pub struct PauliEvolution {
    operator: SparseObservable,
    time: ComparableParam,
}

impl PauliEvolution {
    /// Construct a new [`PauliEvolution`] with an `operator` and `time` parameter.
    ///
    /// # Errors
    ///
    /// Returns an error if `time` isn't a float or expression.
    pub fn new(operator: SparseObservable, time: Param) -> Result<Self, PauliEvolutionError> {
        if matches!(time, Param::Float(_) | Param::ParameterExpression(_)) {
            Ok(Self {
                operator,
                time: ComparableParam(time),
            })
        } else {
            Err(PauliEvolutionError::TimeInvalid)
        }
    }

    /// Returns a reference to the `operator`.
    pub fn operator(&self) -> &SparseObservable {
        &self.operator
    }

    /// Returns a reference to the `time` parameter.
    pub fn time(&self) -> &Param {
        &self.time.0
    }

    /// Inverse the evolution by negating `time` parameter.
    pub fn inverse(&mut self) {
        match &mut self.time.0 {
            Param::ParameterExpression(time) => {
                *time = Arc::new(time.neg());
            }
            Param::Float(time) => {
                *time *= -1.0;
            }
            _ => {
                unreachable!();
            }
        }
    }

    /// Expands `PauliEvolution` into its dense matrix form.
    ///
    /// See [`SparseObservable::to_matrix`].
    ///
    /// # Errors
    ///
    /// Returns an error if `time` isn't evaluated or the `operator`
    /// can't be expanded into its dense matrix form.
    pub fn to_matrix(&self) -> Result<Array2<Complex64>, PauliEvolutionError> {
        if let Param::Float(time) = self.time() {
            let matrix = self
                .operator()
                .to_matrix()
                .map_err(PauliEvolutionError::from)?;

            let matrix = evolve_matrix(&matrix, *time);
            Ok(matrix)
        } else {
            Err(PauliEvolutionError::TimeNotEvaluated)
        }
    }

    /// Decomposes `PauliEvolution` into its owned components.
    pub fn into_parts(self) -> PauliEvolutionParts {
        PauliEvolutionParts {
            operator: self.operator,
            time: self.time.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PauliEvolutionParts {
    pub operator: SparseObservable,
    pub time: Param,
}

#[derive(Debug, Clone)]
struct ComparableParam(Param);

impl PartialEq for ComparableParam {
    fn eq(&self, other: &Self) -> bool {
        let Self(a) = self;
        let Self(b) = other;

        match (a, b) {
            (Param::Float(a), Param::Float(b)) => a == b,
            (Param::ParameterExpression(a), Param::ParameterExpression(b)) => a == b,
            _ => false,
        }
    }
}

fn evolve_matrix(matrix: &Array2<Complex64>, time: f64) -> Array2<Complex64> {
    debug_assert!(matrix.is_square());

    let minus_i_time = Complex64::new(0.0, -time);
    let row_major_elems = matrix.iter().map(|element| element * minus_i_time);

    let dim = matrix.nrows();
    let matrix = DMatrix::from_row_iterator(dim, dim, row_major_elems);

    let matrix = matrix.exp();
    Array2::from_shape_fn((dim, dim), |(row, col)| matrix[(row, col)])
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::ArrayView2;
    use num_complex::c64;
    use qiskit_circuit::{
        operations::StandardGate, parameter::parameter_expression::ParameterExpression,
    };
    use qiskit_quantum_info::sparse_observable::BitTerm;

    use super::*;

    #[test]
    fn test_inverse_float() {
        let obs = xy();

        let mut evolution = PauliEvolution::new(obs, Param::Float(3.0)).unwrap();
        evolution.inverse();

        assert!(matches!(
            evolution.time(),
            Param::Float(time) if *time == -3.0
        ));
    }

    #[test]
    fn test_inverse_param() {
        let obs = xy();

        let time = Arc::new(ParameterExpression::from_f64(3.0));
        let mut evolution = PauliEvolution::new(obs, Param::ParameterExpression(time)).unwrap();
        evolution.inverse();

        let expected = ParameterExpression::from_f64(-3.0);
        assert!(matches!(
            evolution.time(),
            Param::ParameterExpression(expr) if expr.as_ref() == &expected));
    }

    #[test]
    fn test_to_matrix_xy() {
        let obs = xy();

        let gate = PauliEvolution::new(obs, Param::Float(3.0)).unwrap();
        let res = gate.to_matrix().unwrap();

        let data = &[
            // Row 1
            c64(f64::cos(3.0), 0.0),
            c64(0.0, 0.0),
            c64(0.0, 0.0),
            c64(-f64::sin(3.0), 0.0),
            // Row 2
            c64(0.0, 0.0),
            c64(f64::cos(3.0), 0.0),
            c64(f64::sin(3.0), 0.0),
            c64(0.0, 0.0),
            // Row 3
            c64(0.0, 0.0),
            c64(-f64::sin(3.0), 0.0),
            c64(f64::cos(3.0), 0.0),
            c64(0.0, 0.0),
            // Row 4
            c64(f64::sin(3.0), 0.0),
            c64(0.0, 0.0),
            c64(0.0, 0.0),
            c64(f64::cos(3.0), 0.0),
        ];

        let exp = ArrayView2::from_shape((4, 4), data).expect("shape fits data");
        assert_abs_diff_eq!(res, exp, epsilon = 1e-8);
    }

    #[test]
    fn test_to_matrix_ry() {
        let obs = SparseObservable::new(1, vec![1.0.into()], vec![BitTerm::Y], vec![0], vec![0, 1])
            .expect("is coherent");

        for time in [1.0, 3.0, 5.0] {
            let gate = StandardGate::RY;
            let exp = gate.matrix(&[Param::Float(time)]).unwrap();

            let gate = PauliEvolution::new(obs.clone(), Param::Float(time * 0.5)).unwrap();
            let res = gate.to_matrix().unwrap();

            assert_abs_diff_eq!(res, exp, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_to_matrix_rx() {
        let obs = SparseObservable::new(1, vec![1.0.into()], vec![BitTerm::X], vec![0], vec![0, 1])
            .expect("is coherent");

        for time in [1.0, 3.0, 5.0] {
            let gate = StandardGate::RX;
            let exp = gate.matrix(&[Param::Float(time)]).unwrap();

            let gate = PauliEvolution::new(obs.clone(), Param::Float(time * 0.5)).unwrap();
            let res = gate.to_matrix().unwrap();

            assert_abs_diff_eq!(res, exp, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_to_matrix_rz() {
        let obs = SparseObservable::new(1, vec![1.0.into()], vec![BitTerm::Z], vec![0], vec![0, 1])
            .expect("is coherent");

        for time in [1.0, 3.0, 5.0] {
            let gate = StandardGate::RZ;
            let exp = gate.matrix(&[Param::Float(time)]).unwrap();

            let gate = PauliEvolution::new(obs.clone(), Param::Float(time * 0.5)).unwrap();
            let res = gate.to_matrix().unwrap();

            assert_abs_diff_eq!(res, exp, epsilon = 1e-8);
        }
    }

    fn xy() -> SparseObservable {
        SparseObservable::new(
            2,
            vec![1.0.into()],
            vec![BitTerm::Y, BitTerm::X],
            vec![0, 1],
            vec![0, 2],
        )
        .expect("is coherent")
    }
}

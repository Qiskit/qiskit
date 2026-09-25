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

use std::{error, sync::Arc};

use nalgebra::DMatrix;
use ndarray::Array2;
use num_complex::Complex64;
use qiskit_circuit::{
    operations::{CustomOperation, Operation, Param},
    packed_instruction::PackedOperation,
};
use qiskit_quantum_info::sparse_observable::{MatrixError, SparseObservable};
use smallvec::SmallVec;
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

/// Time-evolution of a hermitian operator.
///
/// For a hermitian operator **H** and time **t**, this gate represents the unitary
/// **U(t) = e<sup>-itH</sup>**.
#[derive(Debug, Clone, PartialEq)]
pub struct PauliEvolution {
    hermitian: SparseObservable,
    time: ComparableParam,
}

impl PauliEvolution {
    /// Construct a new [`PauliEvolution`] with a `hermitian` operator and `time` parameter.
    ///
    /// # Errors
    ///
    /// Returns an error if `time` isn't a float value or expression.
    pub fn new(hermitian: SparseObservable, time: Param) -> Result<Self, PauliEvolutionError> {
        if matches!(time, Param::Float(_) | Param::ParameterExpression(_)) {
            Ok(Self {
                hermitian,
                time: ComparableParam(time),
            })
        } else {
            Err(PauliEvolutionError::TimeInvalid)
        }
    }

    /// Returns a reference to the `hermitian` operator.
    pub fn hermitian(&self) -> &SparseObservable {
        &self.hermitian
    }

    /// Returns a reference to the `time` parameter.
    pub fn time(&self) -> &Param {
        &self.time.0
    }

    /// Expands `PauliEvolution` into its approximate dense matrix form.
    ///
    /// See [`SparseObservable::to_matrix`].
    ///
    /// # Errors
    ///
    /// Returns an error if `time` isn't evaluated or the `hermitian` operator
    /// can't be expanded into its dense matrix form.
    pub fn to_matrix(&self) -> Result<Array2<Complex64>, PauliEvolutionError> {
        if let Param::Float(time) = self.time() {
            let matrix = self
                .hermitian()
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
            hermitian: self.hermitian,
            time: self.time.0,
        }
    }
}

impl Operation for PauliEvolution {
    fn name(&self) -> &'static str {
        "PauliEvolution"
    }

    fn num_qubits(&self) -> u32 {
        self.hermitian.num_qubits()
    }

    fn num_clbits(&self) -> u32 {
        0
    }

    fn num_params(&self) -> u32 {
        0
    }

    fn directive(&self) -> bool {
        false
    }
}

impl CustomOperation for PauliEvolution {
    fn is_unitary(&self) -> bool {
        true
    }

    fn inverse(&self, _params: &[Param]) -> Option<(PackedOperation, SmallVec<[Param; 3]>)> {
        let mut inverse = self.clone();

        match &mut inverse.time.0 {
            Param::ParameterExpression(time) => {
                *time = Arc::new(time.neg());
            }
            Param::Float(time) => {
                *time *= -1.0;
            }
            _ => (),
        }

        let inverse = PackedOperation::from_custom_operation(Box::new(inverse));
        Some((inverse, SmallVec::new()))
    }

    fn matrix(
        &self,
        _params: &[Param],
    ) -> Result<Option<Array2<Complex64>>, Box<dyn error::Error>> {
        let matrix = self.to_matrix().map(Some)?;
        Ok(matrix)
    }
}

#[derive(Debug, Clone)]
pub struct PauliEvolutionParts {
    pub hermitian: SparseObservable,
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
    debug_assert_eq!(matrix.nrows(), matrix.ncols());

    let dim = matrix.nrows();
    let matrix = DMatrix::from_row_iterator(dim, dim, matrix.iter().copied());

    let solver = matrix.symmetric_eigen();
    let eigenvectors = solver.eigenvectors;
    let eigenvalues = solver.eigenvalues;

    let mut diagonal = DMatrix::zeros(dim, dim);
    for i in 0..dim {
        let phase = -time * eigenvalues[i];
        diagonal[(i, i)] = Complex64::new(phase.cos(), phase.sin());
    }

    let adjoint = eigenvectors.adjoint();
    let evolved = eigenvectors * diagonal * adjoint;

    Array2::from_shape_fn((dim, dim), |(i, j)| evolved[(i, j)])
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::ArrayView2;
    use num_complex::c64;
    use qiskit_circuit::{
        operations::OperationRef, parameter::parameter_expression::ParameterExpression,
    };
    use qiskit_quantum_info::sparse_observable::BitTerm;

    use super::*;

    #[test]
    fn test_inverse_float() {
        let obs = xy();

        let gate = PauliEvolution::new(obs, Param::Float(3.0)).unwrap();
        let (packed, _) = gate.inverse(&[]).unwrap();

        let OperationRef::CustomOperation(custom) = packed.view() else {
            panic!("inverse is not custom operation");
        };

        let res: &PauliEvolution = custom.downcast_ref().unwrap();
        assert!(matches!(
            res.time(),
            Param::Float(time) if *time == -3.0
        ));
    }

    #[test]
    fn test_inverse_param() {
        let obs = xy();

        let expr = Arc::new(ParameterExpression::from_f64(3.0));
        let gate = PauliEvolution::new(obs, Param::ParameterExpression(expr)).unwrap();
        let (packed, _) = gate.inverse(&[]).unwrap();

        let OperationRef::CustomOperation(custom) = packed.view() else {
            panic!("inverse is not custom operation");
        };

        let exp = ParameterExpression::from_f64(-3.0);
        let res: &PauliEvolution = custom.downcast_ref().unwrap();
        assert!(matches!(res.time(), Param::ParameterExpression(expr) if expr.as_ref() == &exp));
    }

    #[test]
    fn test_to_matrix() {
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

    fn xy() -> SparseObservable {
        SparseObservable::new(
            2,
            vec![1.0.into()],
            vec![BitTerm::Y, BitTerm::X],
            vec![0, 1],
            vec![0, 2],
        )
        .expect("is valid")
    }
}

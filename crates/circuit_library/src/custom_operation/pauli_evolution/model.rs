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

use std::{result, sync::Arc};

use nalgebra::DMatrix;
use ndarray::Array2;
use num_complex::Complex64;
use qiskit_circuit::{operations::Param, parameter::parameter_expression::ParameterExpression};
use qiskit_quantum_info::sparse_observable::{MatrixError, SparseObservable, SparseTermView};
use thiserror::Error;

pub type Result<T> = result::Result<T, PauliEvolutionError>;

#[derive(Debug, Error)]
pub enum PauliEvolutionError {
    #[error("operator has complex coefficients")]
    HasComplex,
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
    label: String,
}

impl PauliEvolution {
    /// Construct a new [`PauliEvolution`] with the default label.
    ///
    /// # Errors
    ///
    /// Returns an error if `operator` has complex coefficients or `time` isn't a float or
    /// expression.
    pub fn new(operator: SparseObservable, time: Param) -> Result<Self> {
        let label = format_default_label(&operator);
        Self::with_label(operator, time, label)
    }

    /// Construct a new [`PauliEvolution`] with the given `label`.
    ///
    /// # Errors
    ///
    /// Returns an error if `operator` has complex coefficients or `time` isn't a float or
    /// expression.
    pub fn with_label(
        operator: SparseObservable,
        time: Param,
        label: impl Into<String>,
    ) -> Result<Self> {
        validate_operator(&operator)?;
        validate_time(&time)?;

        Ok(Self {
            operator,
            time: ComparableParam(time),
            label: label.into(),
        })
    }

    /// Returns a reference to the `operator`.
    pub fn operator(&self) -> &SparseObservable {
        &self.operator
    }

    /// Set the `operator`.
    ///
    /// # Errors
    ///
    /// Returns an error if `operator` has complex coefficients.
    pub fn set_operator(&mut self, operator: SparseObservable) -> Result<()> {
        validate_operator(&operator)?;
        self.operator = operator;
        Ok(())
    }

    /// Returns a reference to the `time` parameter.
    pub fn time(&self) -> &Param {
        &self.time.0
    }

    /// Set the `time` parameter.
    ///
    /// # Errors
    ///
    /// Returns an error if `time` isn't a float or expression.
    pub fn set_time(&mut self, time: Param) -> Result<()> {
        validate_time(&time)?;
        self.time = ComparableParam(time);
        Ok(())
    }

    /// Returns a reference to the `label`.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Set the `label`.
    pub fn set_label(&mut self, label: impl Into<String>) {
        self.label = label.into();
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

    /// Raise the evolution to the `exponent` power.
    pub fn power(&mut self, exponent: f64) {
        match &mut self.time.0 {
            Param::ParameterExpression(time) => {
                let exponent = ParameterExpression::from_f64(exponent);
                let raised = time.mul(&exponent).expect("exponent has empty name map");
                *time = Arc::new(raised);
            }
            Param::Float(time) => {
                *time *= exponent;
            }
            _ => {
                unreachable!()
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
    pub fn to_matrix(&self) -> Result<Array2<Complex64>> {
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
            label: self.label,
        }
    }
}

fn validate_operator(operator: &SparseObservable) -> Result<()> {
    if operator.coeffs().iter().all(|c| c.im == 0.0) {
        Ok(())
    } else {
        Err(PauliEvolutionError::HasComplex)
    }
}

fn validate_time(time: &Param) -> Result<()> {
    if matches!(time, Param::Float(_) | Param::ParameterExpression(_)) {
        Ok(())
    } else {
        Err(PauliEvolutionError::TimeInvalid)
    }
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

fn format_default_label(operator: &SparseObservable) -> String {
    let mut label = "exp(-it (".to_string();

    for (i, term) in operator.iter().enumerate() {
        if i != 0 {
            label.push_str(" + ");
        }

        push_term(&mut label, &term);
    }

    label.push_str("))");
    label
}

fn push_term(label: &mut String, term: &SparseTermView<'_>) {
    let mut prev = term.num_qubits;

    for (bit_term, i) in term.bit_terms.iter().zip(term.indices).rev() {
        let num_identity = prev - *i - 1;
        push_identity(label, num_identity);

        label.push_str(bit_term.py_label());
        prev = *i;
    }

    push_identity(label, prev);
}

fn push_identity(label: &mut String, count: u32) {
    for _ in 0..count {
        label.push('I');
    }
}

#[derive(Debug, Clone)]
pub struct PauliEvolutionParts {
    pub operator: SparseObservable,
    pub time: Param,
    pub label: String,
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
    fn test_label_xy() {
        let obs = xy();
        let evolution = PauliEvolution::new(obs, Param::Float(3.0)).unwrap();
        assert_eq!(evolution.label(), "exp(-it (XY))");
    }

    #[test]
    fn test_label_xy_zz() {
        let obs = xy_zz();
        let evolution = PauliEvolution::new(obs, Param::Float(3.0)).unwrap();
        assert_eq!(evolution.label(), "exp(-it (XY + ZZ))");
    }

    #[test]
    fn test_label_xyi_izz_yix() {
        let obs = xyi_izz_yix();
        let evolution = PauliEvolution::new(obs, Param::Float(3.0)).unwrap();
        assert_eq!(evolution.label(), "exp(-it (XYI + IZZ + YIX))")
    }

    #[test]
    fn test_label_custom() {
        let obs = xy();
        let evolution = PauliEvolution::with_label(obs, Param::Float(3.0), "custom").unwrap();
        assert_eq!(evolution.label(), "custom");
    }

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
            Param::ParameterExpression(expr) if expr.as_ref() == &expected
        ));
    }

    #[test]
    fn test_power_float() {
        let obs = xy();
        let mut evolution = PauliEvolution::new(obs, Param::Float(3.0)).unwrap();
        evolution.power(3.0);

        assert!(matches!(
            evolution.time(),
            Param::Float(time) if *time == 9.0,
        ));
    }

    #[test]
    fn test_power_param() {
        let obs = xy();
        let time = Arc::new(ParameterExpression::from_f64(3.0));
        let mut evolution = PauliEvolution::new(obs, Param::ParameterExpression(time)).unwrap();
        evolution.power(3.0);

        let expected = ParameterExpression::from_f64(9.0);
        assert!(matches!(
            evolution.time(),
            Param::ParameterExpression(expr) if expr.as_ref() == &expected
        ));
    }

    #[test]
    fn test_to_matrix_xy() {
        let obs = xy();
        let evolution = PauliEvolution::new(obs, Param::Float(3.0)).unwrap();
        let matrix = evolution.to_matrix().unwrap();

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

        let expected = ArrayView2::from_shape((4, 4), data).expect("shape fits data");
        assert_abs_diff_eq!(matrix, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_to_matrix_ry() {
        let obs = SparseObservable::new(1, vec![1.0.into()], vec![BitTerm::Y], vec![0], vec![0, 1])
            .expect("should be coherent");

        for time in [1.0, 3.0, 5.0] {
            let evolution = PauliEvolution::new(obs.clone(), Param::Float(time * 0.5)).unwrap();
            let matrix = evolution.to_matrix().unwrap();

            let ry = StandardGate::RY;
            let expected = ry.matrix(&[Param::Float(time)]).unwrap();
            assert_abs_diff_eq!(matrix, expected, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_to_matrix_rx() {
        let obs = SparseObservable::new(1, vec![1.0.into()], vec![BitTerm::X], vec![0], vec![0, 1])
            .expect("should be coherent");

        for time in [1.0, 3.0, 5.0] {
            let evolution = PauliEvolution::new(obs.clone(), Param::Float(time * 0.5)).unwrap();
            let matrix = evolution.to_matrix().unwrap();

            let ry = StandardGate::RX;
            let expected = ry.matrix(&[Param::Float(time)]).unwrap();
            assert_abs_diff_eq!(matrix, expected, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_to_matrix_rz() {
        let obs = SparseObservable::new(1, vec![1.0.into()], vec![BitTerm::Z], vec![0], vec![0, 1])
            .expect("should be coherent");

        for time in [1.0, 3.0, 5.0] {
            let evolution = PauliEvolution::new(obs.clone(), Param::Float(time * 0.5)).unwrap();
            let matrix = evolution.to_matrix().unwrap();

            let rz = StandardGate::RZ;
            let expected = rz.matrix(&[Param::Float(time)]).unwrap();
            assert_abs_diff_eq!(matrix, expected, epsilon = 1e-8);
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
        .expect("should be coherent")
    }

    fn xy_zz() -> SparseObservable {
        SparseObservable::new(
            2,
            vec![1.0.into(), 1.0.into()],
            vec![BitTerm::Y, BitTerm::X, BitTerm::Z, BitTerm::Z],
            vec![0, 1, 0, 1],
            vec![0, 2, 4],
        )
        .expect("should be coherent")
    }

    fn xyi_izz_yix() -> SparseObservable {
        SparseObservable::new(
            3,
            vec![1.0.into(), 1.0.into(), 1.0.into()],
            vec![
                BitTerm::Y,
                BitTerm::X,
                BitTerm::Z,
                BitTerm::Z,
                BitTerm::X,
                BitTerm::Y,
            ],
            vec![1, 2, 0, 1, 0, 2],
            vec![0, 2, 4, 6],
        )
        .expect("should be coherent")
    }
}

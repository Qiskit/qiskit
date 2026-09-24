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

use ndarray::Array2;
use num_complex::Complex64;
use qiskit_circuit::{
    operations::{CustomOperation, Operation, Param},
    packed_instruction::PackedOperation,
};
use qiskit_quantum_info::sparse_observable::{BitTerm, MatrixError, SparseObservable};
use smallvec::SmallVec;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PauliEvolutionError {
    #[error("time is python object")]
    TimePython,
    #[error("time is int")]
    TimeInt,
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
    /// Returns an error if `time` is [`Param::Int`], `time` is [`Param::Obj`], or `hermitian` has
    /// 0 qubits.
    pub fn new(hermitian: SparseObservable, time: Param) -> Result<Self, PauliEvolutionError> {
        if matches!(time, Param::Obj(_)) {
            Err(PauliEvolutionError::TimePython)
        } else if matches!(time, Param::Int(_)) {
            Err(PauliEvolutionError::TimeInt)
        } else {
            Ok(Self {
                hermitian,
                time: ComparableParam(time),
            })
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

    /// Decomposes `PauliEvolution` into its raw components.
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
        if let Param::Float(time) = self.time() {
            let matrix = evolve_matrix(self.hermitian(), *time)?;
            Ok(Some(matrix))
        } else {
            Ok(None)
        }
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

fn evolve_matrix(
    hermitian: &SparseObservable,
    time: f64,
) -> Result<Array2<Complex64>, PauliEvolutionError> {
    let evolve_matrix = |matrix| {
        if is_pauli(hermitian) {
            evolve_matrix_pauli(matrix, time)
        } else {
            evolve_matrix_pade(matrix, time)
        }
    };

    hermitian
        .to_matrix()
        .map(evolve_matrix)
        .map_err(PauliEvolutionError::from)
}

fn is_pauli(hermitian: &SparseObservable) -> bool {
    hermitian
        .bit_terms()
        .iter()
        .all(|bit_term| matches!(bit_term, BitTerm::X | BitTerm::Y | BitTerm::Z))
}

fn evolve_matrix_pauli(matrix: Array2<Complex64>, time: f64) -> Array2<Complex64> {
    todo!()
}

fn evolve_matrix_pade(matrix: Array2<Complex64>, time: f64) -> Array2<Complex64> {
    todo!()
}

#[cfg(test)]
mod tests {
    use qiskit_circuit::{
        operations::OperationRef, parameter::parameter_expression::ParameterExpression,
    };
    use qiskit_quantum_info::sparse_observable::BitTerm;

    use super::*;

    #[test]
    fn test_inverse_float() {
        let obs = create_observable();

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
        let obs = create_observable();

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

    fn create_observable() -> SparseObservable {
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

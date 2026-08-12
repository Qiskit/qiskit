// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use std::sync::Arc;

use ndarray::Array2;
use num_complex::Complex64;
use qiskit_circuit::{
    operations::{CustomOperation, Operation, Param},
    packed_instruction::PackedOperation,
};
use qiskit_quantum_info::sparse_observable::SparseObservable;
use smallvec::SmallVec;
use thiserror::Error;

#[derive(Debug, Error)]
#[error("time is python object")]
pub struct PauliEvolutionError;

#[derive(Debug, Clone, PartialEq)]
pub struct PauliEvolution {
    operator: SparseObservable,
    // TODO: We should have team discussion to decide whether `time` should be
    // owned by `PauliEvolution` or passed as a `param`.
    time: ComparableParam,
}

impl PauliEvolution {
    pub fn new(operator: SparseObservable, time: Param) -> Result<Self, PauliEvolutionError> {
        if matches!(time, Param::Obj(_)) {
            return Err(PauliEvolutionError);
        }

        Ok(Self {
            operator,
            time: ComparableParam(time),
        })
    }

    pub fn operator(&self) -> &SparseObservable {
        &self.operator
    }

    pub fn time(&self) -> &Param {
        &self.time.0
    }

    pub fn into_parts(self) -> PauliEvolutionParts {
        PauliEvolutionParts {
            operator: self.operator,
            time: self.time.0,
        }
    }
}

impl Operation for PauliEvolution {
    fn name(&self) -> &'static str {
        "PauliEvolution"
    }

    fn num_qubits(&self) -> u32 {
        self.operator.num_qubits()
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
    // TODO: We need to have a discussion about whether this trait member
    // should be removed or replaced with a dynamic label function
    // returning `Option<Box<String>>`.
    fn label(&self) -> Option<&str> {
        None
    }

    // TODO: We'd like `SparseObservable::to_matrix`. This requires some
    // discussion and should be completed seperately from the introduction of
    // this gate.
    fn matrix(&self, _param: &[Param]) -> Option<Array2<Complex64>> {
        None
    }

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

#[cfg(test)]
mod tests {
    use qiskit_circuit::operations::OperationRef;
    use qiskit_quantum_info::sparse_observable::BitTerm;

    use super::*;

    #[test]
    fn test_inverse_float() {
        let gate = PauliEvolution::new(mock_xy(), Param::Float(3.0)).unwrap();
        let (inverse, _) = gate.inverse(&[]).unwrap();

        let OperationRef::CustomOperation(inverse) = inverse.view() else {
            panic!("inverse is not custom operation");
        };

        let inverse: &PauliEvolution = inverse.downcast_ref().unwrap();

        assert!(matches!(
            inverse.time(),
            Param::Float(time) if *time == -3.0
        ));
    }

    fn mock_xy() -> SparseObservable {
        SparseObservable::new(
            2,
            vec![1.0.into()],
            vec![BitTerm::X, BitTerm::Y],
            vec![0, 1],
            vec![0, 2],
        )
        .expect("is valid")
    }
}

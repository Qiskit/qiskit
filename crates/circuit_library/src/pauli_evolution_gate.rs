use std::{fmt, sync::Arc};

use ndarray::Array2;
use num_complex::Complex64;
use qiskit_circuit::{
    operations::{CustomOperation, Operation, Param},
    packed_instruction::PackedOperation,
    parameter::symbol_expr::Value,
};
use qiskit_quantum_info::sparse_observable::{BitTerm, SparseObservable};
use smallvec::SmallVec;
use thiserror::Error;

#[derive(Debug, Error)]
#[error("time is python object")]
pub struct PauliEvolutionError;

#[derive(Debug, Clone, PartialEq)]
pub struct PauliEvolution {
    operator: SparseObservable,
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

impl fmt::Display for PauliEvolution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "exp(-it (")?;

        for (i, term) in self.operator.iter().enumerate() {
            if i > 0 {
                write!(f, " + ")?;
            }

            for bit in term.bit_terms {
                write!(f, "{}", bit.py_label())?;
            }
        }

        write!(f, "))")
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
    use std::fmt::Write;

    use qiskit_quantum_info::sparse_observable::BitTerm;

    use super::*;

    #[test]
    fn test_inverse_float() {
        let gate = PauliEvolution::new(mock_xy(), Param::Float(3.0)).unwrap();
        let (_, params) = gate.inverse(&[Param::Float(3.0)]).unwrap();

        assert!(matches!(
            params.first(),
            Some(Param::Float(time)) if *time == -3.0
        ));
    }

    #[test]
    fn test_display_label_xy() {
        let gate = PauliEvolution::new(mock_xy(), Param::Float(1.0)).unwrap();

        let mut label = String::new();
        let _ = write!(label, "{gate}");

        assert_eq!(label, "exp(-it (XY))");
    }

    #[test]
    fn test_display_label_xy_zz() {
        let gate = PauliEvolution::new(mock_xy_zz(), Param::Float(1.0)).unwrap();

        let mut label = String::new();
        let _ = write!(label, "{gate}");

        assert_eq!(label, "exp(-it (XY + ZZ))");
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

    fn mock_xy_zz() -> SparseObservable {
        SparseObservable::new(
            2,
            vec![1.0.into(), (-1.0).into()],
            vec![BitTerm::X, BitTerm::Y, BitTerm::Z, BitTerm::Z],
            vec![0, 1, 0, 1],
            vec![0, 2, 4],
        )
        .expect("is valid")
    }
}

use std::{fmt, sync::Arc};

use ndarray::Array2;
use num_complex::Complex64;
use qiskit_circuit::{
    operations::{CustomOperation, Operation, Param},
    packed_instruction::PackedOperation,
    parameter::symbol_expr::Value,
};
use qiskit_quantum_info::sparse_observable::SparseObservable;
use smallvec::SmallVec;
use thiserror::Error;

#[derive(Debug, Error)]
#[error("time is python object")]
pub struct PauliEvolutionError;

#[derive(Debug, Clone, PartialEq)]
pub struct PauliEvolution {
    hermitian: SparseObservable,
    time: ComparableParam,
    label: Option<String>,
}

impl PauliEvolution {
    pub fn new(hermitian: SparseObservable, time: Param) -> Result<Self, PauliEvolutionError> {
        if matches!(time, Param::Obj(_)) {
            return Err(PauliEvolutionError);
        }

        Ok(Self {
            hermitian,
            time: ComparableParam(time),
            label: None,
        })
    }

    pub fn with_label(
        hermitian: SparseObservable,
        time: Param,
        label: impl Into<String>,
    ) -> Result<Self, PauliEvolutionError> {
        let mut gate = Self::new(hermitian, time)?;
        gate.label = Some(label.into());
        Ok(gate)
    }

    pub fn hermitian(&self) -> &SparseObservable {
        &self.hermitian
    }

    pub fn time(&self) -> &Param {
        &self.time.0
    }

    pub fn into_parts(self) -> PauliEvolutionParts {
        PauliEvolutionParts {
            obs: self.hermitian,
            time: self.time.0,
            label: self.label,
        }
    }
}

impl fmt::Display for PauliEvolution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(label) = &self.label {
            write!(f, "{label}")
        } else {
            format_label(f, &self.hermitian)
        }
    }
}

fn format_label(f: &mut fmt::Formatter<'_>, hermitian: &SparseObservable) -> fmt::Result {
    write!(f, "exp(-it (")?;

    for (i, term) in hermitian.iter().enumerate() {
        if i > 0 {
            write!(f, " + ")?;
        }

        for bit in term.bit_terms {
            write!(f, "{}", bit.py_label())?;
        }
    }

    write!(f, "))")
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
        1
    }

    fn directive(&self) -> bool {
        false
    }
}

impl CustomOperation for PauliEvolution {
    fn is_unitary(&self) -> bool {
        true
    }

    fn inverse(&self, params: &[Param]) -> Option<(PackedOperation, SmallVec<[Param; 3]>)> {
        let param = params.first()?;

        let mut inverse = self.clone();
        inverse_time(&mut inverse.time.0);

        let inverse = PackedOperation::from_custom_operation(Box::new(inverse));
        let mut param = param.clone();
        inverse_time(&mut param);

        let mut params: SmallVec<_> = SmallVec::new();
        params.push(param);

        Some((inverse, params))
    }

    fn matrix(&self, params: &[Param]) -> Option<Array2<Complex64>> {
        let time = params.first().and_then(extract_time)?;

        let size = 2usize.pow(self.num_qubits());
        let mut matrix = Array2::zeros((size, size));

        Some(matrix)
    }
}

fn inverse_time(param: &mut Param) {
    match param {
        Param::ParameterExpression(time) => {
            *time = Arc::new(time.neg());
        }
        Param::Float(time) => {
            *time *= -1.0;
        }
        _ => (),
    }
}

fn extract_time(param: &Param) -> Option<f64> {
    match param {
        Param::ParameterExpression(time) => {
            if let Value::Real(time) = time.try_to_value(true).ok()? {
                Some(time)
            } else {
                None
            }
        }
        Param::Float(time) => Some(*time),
        Param::Obj(_) => None,
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

#[derive(Debug, Clone)]
pub struct PauliEvolutionParts {
    pub obs: SparseObservable,
    pub time: Param,
    pub label: Option<String>,
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

        assert!(matches!(gate.time(), Param::Float(time) if *time == -3.0));
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

    #[test]
    fn test_display_label_custom() {
        const HELLO: &str = "Hello, world!";
        let gate = PauliEvolution::with_label(mock_xy(), Param::Float(1.0), HELLO).unwrap();

        let mut label = String::new();
        let _ = write!(label, "{gate}");

        assert_eq!(label, HELLO);
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

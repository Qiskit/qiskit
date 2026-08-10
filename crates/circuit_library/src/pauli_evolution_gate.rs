use std::sync::Arc;

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
pub enum PauliEvolutionError {
    #[error("operator not set")]
    MissingOperator,
    #[error("time is python object")]
    PythonTime,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PauliEvolution {
    operator: SparseObservable,
    time: ComparableParam,
    label: String,
}

impl PauliEvolution {
    pub fn new(operator: SparseObservable, time: f64) -> PauliEvolution {
        Self::builder()
            .operator(operator)
            .time(Param::Float(time))
            .build()
            .expect("is valid")
    }

    pub fn builder() -> PauliEvolutionBuilder {
        PauliEvolutionBuilder::new()
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
            label: self.label,
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
        1
    }

    fn directive(&self) -> bool {
        false
    }
}

impl CustomOperation for PauliEvolution {
    fn label(&self) -> Option<&str> {
        Some(&self.label)
    }

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

#[derive(Debug, Default, Clone)]
pub struct PauliEvolutionBuilder {
    operator: Option<SparseObservable>,
    time: Option<Param>,
    label: Option<String>,
}

impl PauliEvolutionBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn operator(mut self, operator: SparseObservable) -> Self {
        self.operator = Some(operator);
        self
    }

    pub fn time(mut self, time: Param) -> Self {
        self.time = Some(time);
        self
    }

    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn build(self) -> Result<PauliEvolution, PauliEvolutionError> {
        const DEFAULT_TIME: f64 = 1.0;

        let operator = self.operator.ok_or(PauliEvolutionError::MissingOperator)?;

        if matches!(self.time, Some(Param::Obj(_))) {
            return Err(PauliEvolutionError::PythonTime);
        }

        let time = self
            .time
            .map(ComparableParam)
            .unwrap_or(ComparableParam(Param::Float(DEFAULT_TIME)));

        let label = self.label.unwrap_or_else(|| format_label(&operator));

        Ok(PauliEvolution {
            operator,
            time,
            label,
        })
    }
}

fn format_label(operator: &SparseObservable) -> String {
    let mut label = String::from("exp(-it (");

    for (i, term) in operator.iter().enumerate() {
        if i > 0 {
            label.push_str(" + ");
        }

        for bit in term.bit_terms {
            label.push_str(bit.py_label());
        }
    }

    label.push_str("))");
    label
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
    pub operator: SparseObservable,
    pub time: Param,
    pub label: String,
}

#[cfg(test)]
mod tests {
    use qiskit_quantum_info::sparse_observable::BitTerm;

    use super::*;

    #[test]
    fn test_inverse_float() {
        let evolution = PauliEvolution::new(mock_xy(), 3.0);
        let (_, params) = evolution.inverse(&[Param::Float(3.0)]).expect("has time");

        assert!(matches!(
            params.first(),
            Some(Param::Float(time)) if *time == -3.0
        ));
    }

    #[test]
    fn test_label_default() {
        let evolution = PauliEvolution::new(mock_xy(), 1.0);
        assert_eq!(evolution.label(), Some("exp(-it (XY))"));

        let evolution = PauliEvolution::new(mock_xy_zz(), 1.0);
        assert_eq!(evolution.label(), Some("exp(-it (XY + ZZ))"))
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

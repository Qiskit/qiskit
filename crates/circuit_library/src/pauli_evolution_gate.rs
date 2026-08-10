use std::sync::Arc;

use qiskit_circuit::{
    operations::{CustomOperation, Operation, Param},
    packed_instruction::PackedOperation,
    parameter::parameter_expression::ParameterExpression,
};
use qiskit_quantum_info::sparse_observable::SparseObservable;
use smallvec::SmallVec;

#[derive(Debug, Clone, PartialEq)]
pub struct PauliEvolution {
    operator: SparseObservable,
    time: Time,
    label: String,
}

impl PauliEvolution {
    pub fn builder(operator: SparseObservable) -> PauliEvolutionBuilder {
        PauliEvolutionBuilder::new(operator)
    }

    pub fn operator(&self) -> &SparseObservable {
        &self.operator
    }

    pub fn time(&self) -> &Time {
        &self.time
    }

    pub fn into_parts(self) -> PauliEvolutionParts {
        PauliEvolutionParts {
            operator: self.operator,
            time: self.time,
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
        debug_assert!(params.len() == 1);

        params.first().map(|param| {
            let mut inverse = self.clone();
            inverse_time(&mut inverse.time);

            let inverse = PackedOperation::from_custom_operation(Box::new(inverse));
            let mut param = param.clone();
            inverse_param(&mut param);

            let mut params: SmallVec<_> = SmallVec::new();
            params.push(param);

            (inverse, params)
        })
    }
}

fn inverse_time(time: &mut Time) {
    match time {
        Time::Float(time) => {
            *time *= -1.0;
        }
        Time::Expression(time) => {
            *time = Box::new(time.neg());
        }
    }
}

fn inverse_param(param: &mut Param) {
    match param {
        Param::Float(time) => {
            *time *= -1.0;
        }
        Param::ParameterExpression(time) => {
            *time = Arc::new(time.neg());
        }
        _ => (),
    }
}

#[derive(Debug, Clone)]
pub struct PauliEvolutionBuilder {
    operator: SparseObservable,
    time: Option<Time>,
    label: Option<String>,
}

impl PauliEvolutionBuilder {
    pub fn new(operator: SparseObservable) -> Self {
        Self {
            operator,
            time: None,
            label: None,
        }
    }

    pub fn time(mut self, time: Time) -> Self {
        self.time = Some(time);
        self
    }

    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn build(self) -> PauliEvolution {
        const DEFAULT_TIME: f64 = 1.0;

        let label = self.label.unwrap_or_else(|| format_label(&self.operator));

        PauliEvolution {
            operator: self.operator,
            time: self.time.unwrap_or(Time::Float(DEFAULT_TIME)),
            label,
        }
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
pub struct PauliEvolutionParts {
    pub operator: SparseObservable,
    pub time: Time,
    pub label: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Time {
    Float(f64),
    Expression(Box<ParameterExpression>),
}

#[cfg(test)]
mod tests {
    use qiskit_quantum_info::sparse_observable::BitTerm;

    use super::*;

    #[test]
    #[should_panic]
    fn test_inverse_empty() {
        let evolution = PauliEvolution::builder(mock_xy()).build();
        evolution.inverse(&[]);
    }

    #[test]
    #[should_panic]
    fn test_inverse_too_many() {
        let evolution = PauliEvolution::builder(mock_xy()).build();
        evolution.inverse(&[Param::Float(3.0), Param::Float(1.0)]);
    }

    #[test]
    fn test_inverse_float() {
        let evolution = PauliEvolution::builder(mock_xy())
            .time(Time::Float(3.0))
            .build();

        let (_, params) = evolution.inverse(&[Param::Float(3.0)]).expect("has time");
        assert!(matches!(
            params.first(),
            Some(Param::Float(time)) if *time == -3.0
        ));
    }

    #[test]
    fn test_label_default() {
        let evolution = PauliEvolution::builder(mock_xy()).build();
        assert_eq!(evolution.label(), Some("exp(-it (XY))"));

        let evolution = PauliEvolution::builder(mock_xy_zz()).build();
        assert_eq!(evolution.label(), Some("exp(-it (XY + ZZ))"))
    }

    #[test]
    fn test_label_custom() {
        let evolution = PauliEvolution::builder(mock_xy())
            .label("Hello, World!")
            .build();
        assert_eq!(evolution.label(), Some("Hello, World!"));
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

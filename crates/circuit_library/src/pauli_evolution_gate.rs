use std::{
    fmt::{Display, Write},
    sync::Arc,
};

use qiskit_circuit::{
    operations::{CustomOperation, Operation, Param},
    parameter::parameter_expression::ParameterExpression,
};
use qiskit_quantum_info::sparse_observable::SparseObservable;
use thiserror::Error;

#[derive(Error, Debug)]
#[error("invalid time")]
pub struct PauliEvolutionError;

#[derive(Debug, Clone, PartialEq)]
pub struct PauliEvolution {
    operator: SparseObservable,
    time: ComparableParam,
    label: String,
}

impl PauliEvolution {
    pub fn new(operator: SparseObservable, time: f64) -> Self {
        Self::builder(operator)
            .time(Param::Float(time))
            .build()
            .expect("time is float")
    }

    pub fn builder(operator: SparseObservable) -> PauliEvolutionBuilder {
        PauliEvolutionBuilder::new(operator)
    }
}

#[derive(Debug, Clone, PartialEq)]
enum ComparableParam {
    Float(f64),
    Expression(Arc<ParameterExpression>),
}

impl Display for ComparableParam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComparableParam::Float(float) => float.fmt(f),
            ComparableParam::Expression(expr) => expr.fmt(f),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PauliEvolutionBuilder {
    operator: SparseObservable,
    time: Option<Param>,
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

    pub fn time(mut self, time: Param) -> Self {
        self.time = Some(time);
        self
    }

    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn build(self) -> Result<PauliEvolution, PauliEvolutionError> {
        let time = self
            .time
            .map(|param| build_comparable_param(param).ok_or(PauliEvolutionError))
            .transpose()?
            .unwrap_or(ComparableParam::Float(1.0));

        let label = self.label.unwrap_or_else(|| format_label(&self.operator));

        Ok(PauliEvolution {
            operator: self.operator,
            time,
            label,
        })
    }
}

fn build_comparable_param(param: Param) -> Option<ComparableParam> {
    match param {
        Param::Float(param) => Some(ComparableParam::Float(param)),
        Param::ParameterExpression(param) => Some(ComparableParam::Expression(param)),
        Param::Obj(_) => None,
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

    label.push(')');
    label
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
        if matches!(self.time, ComparableParam::Expression(_)) {
            1
        } else {
            0
        }
    }

    fn directive(&self) -> bool {
        false
    }
}

impl CustomOperation for PauliEvolution {
    fn is_unitary(&self) -> bool {
        true
    }
}

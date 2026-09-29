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

//! An op that evaluates parameter expressions.

use hashbrown::{HashMap, HashSet};
use ndarray::{ArrayD, IxDyn};
use num_complex::Complex64;
use qiskit_circuit::parameter::parameter_expression::{ParameterError, ParameterExpression};
use qiskit_circuit::parameter::symbol_expr::{Symbol, Value};
use thiserror::Error;

use super::inference::{float, leading_axes};
use super::{ProgramOp, QISKIT};
use crate::tensor::{DType, Dim, Tensor, TensorType};

/// Evaluate parameter expressions over a batch of parameter values.
///
/// The operand has one value per declared parameter in its trailing axis, and the result has one
/// value per expression. Leading axes on the operand are a batch prefix. The result dtype is
/// `F64` or `C128`. When an expression evaluates to a complex number, an `F64` result keeps just
/// the real part.
#[derive(Clone, Debug)]
pub struct BindParameters {
    expressions: Vec<ParameterExpression>,
    parameters: Vec<Symbol>,
    dtype: DType,
}

impl BindParameters {
    /// Construct an op evaluating `expressions` over values for `parameters`, producing `dtype`.
    ///
    /// `dtype` must be either `F64` or `C128`.
    pub fn new(
        expressions: Vec<ParameterExpression>,
        parameters: Vec<Symbol>,
        dtype: DType,
    ) -> Result<Self, BindParametersError> {
        if !matches!(dtype, DType::F64 | DType::C128) {
            return Err(BindParametersError::ResultDType { dtype });
        }
        if let Some((expression, parameter)) = undeclared_parameter(&expressions, &parameters) {
            return Err(BindParametersError::UndeclaredParameter {
                expression,
                parameter,
            });
        }
        Ok(Self {
            expressions,
            parameters,
            dtype,
        })
    }

    /// Return the expressions this op evaluates.
    pub fn expressions(&self) -> &[ParameterExpression] {
        &self.expressions
    }

    /// Return the parameters the operand has values for, in order.
    pub fn parameters(&self) -> &[Symbol] {
        &self.parameters
    }

    /// Return the dtype of the result.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Evaluate every expression over each of `rows` consecutive sets of values from `values`,
    /// returning one tensor of `shape` in this op's dtype.
    fn bind_all(
        &self,
        values: impl Iterator<Item = f64>,
        rows: usize,
        shape: &[usize],
    ) -> Result<Tensor, BindParametersError> {
        match self.dtype {
            DType::C128 => Ok(Tensor::from(reshape(
                self.evaluate(values, rows, complex)?,
                shape,
            ))),
            DType::F64 => Ok(Tensor::from(reshape(
                self.evaluate(values, rows, real)?,
                shape,
            ))),
            other => panic!("{} cannot produce {}", self.full_name(), other),
        }
    }

    /// Evaluate every expression over each of `rows` consecutive sets of values from `values`,
    /// mapping each result through `convert`.
    fn evaluate<T>(
        &self,
        mut values: impl Iterator<Item = f64>,
        rows: usize,
        convert: fn(Value) -> T,
    ) -> Result<Vec<T>, BindParametersError> {
        let mut results = Vec::with_capacity(rows * self.expressions.len());
        for _ in 0..rows {
            let bindings: HashMap<&Symbol, Value> = self
                .parameters
                .iter()
                .zip(values.by_ref())
                .map(|(parameter, value)| (parameter, Value::Real(value)))
                .collect();
            for (index, expression) in self.expressions.iter().enumerate() {
                let bound = match expression.bind(&bindings, true) {
                    Ok(bound) => bound,
                    Err(ParameterError::BindingInf | ParameterError::BindingNaN) => {
                        return Err(BindParametersError::NonFinite { expression: index });
                    }
                    Err(other) => panic!("{} cannot bind a value: {other}", self.full_name()),
                };
                let value = match bound.try_to_value(true) {
                    Ok(value) => value,
                    Err(ParameterError::UnboundParameters(_)) => {
                        return Err(BindParametersError::UnboundParameter { expression: index });
                    }
                    Err(other) => panic!("{} cannot read a value: {other}", self.full_name()),
                };
                results.push(convert(value));
            }
        }
        Ok(results)
    }
}

impl ProgramOp for BindParameters {
    type Error = BindParametersError;

    fn name(&self) -> &str {
        "bind_parameters"
    }
    fn namespace(&self) -> &str {
        QISKIT
    }
    fn arity(&self) -> usize {
        1
    }
    fn has_builtin_eval(&self) -> bool {
        true
    }
    fn infer_output_types(&self, inputs: &[TensorType]) -> Result<Vec<TensorType>, Self::Error> {
        crate::unpack_operands!(self, inputs, [values]);
        let parameters = self.parameters.len();
        let refuse = || BindParametersError::ValueType {
            parameters,
            actual: values.clone(),
        };
        if !float(values.dtype) {
            return Err(refuse());
        }
        let batch = leading_axes(&values.shape, parameters).ok_or_else(refuse)?;
        let mut shape = batch.to_vec();
        shape.push(Dim::Fixed(self.expressions.len()));
        Ok(vec![TensorType {
            dtype: self.dtype,
            shape,
        }])
    }
    fn eval(&self, args: &[Tensor]) -> Result<Vec<Tensor>, Self::Error> {
        crate::unpack_operands!(self, args, [values]);
        let parameters = self.parameters.len();
        let (&supplied, batch) = values
            .shape()
            .split_last()
            .expect("the operand has a trailing axis of parameter values");
        assert_eq!(
            supplied,
            parameters,
            "{} expects one value per parameter",
            self.full_name()
        );
        let rows = batch.iter().product();
        let mut shape = batch.to_vec();
        shape.push(self.expressions.len());
        let results = match values {
            Tensor::F32(x) => self.bind_all(x.iter().map(|&value| f64::from(value)), rows, &shape),
            Tensor::F64(x) => self.bind_all(x.iter().copied(), rows, &shape),
            other => panic!(
                "{} expects floating-point parameter values, got {}",
                self.full_name(),
                other.dtype()
            ),
        }?;
        Ok(vec![results])
    }
}

/// Return the real part of `value`.
fn real(value: Value) -> f64 {
    match value {
        Value::Real(value) => value,
        Value::Int(value) => value as f64,
        Value::Complex(value) => value.re,
    }
}

/// Return `value` as a complex number.
fn complex(value: Value) -> Complex64 {
    match value {
        Value::Real(value) => Complex64::from(value),
        Value::Int(value) => Complex64::from(value as f64),
        Value::Complex(value) => value,
    }
}

/// Return `results` as an array of `shape`.
fn reshape<T>(results: Vec<T>, shape: &[usize]) -> ArrayD<T> {
    ArrayD::from_shape_vec(IxDyn(shape), results).expect("one value per expression per batch entry")
}

/// Return a parameter some expression references that `parameters` does not contain, with the index
/// of that expression.
fn undeclared_parameter(
    expressions: &[ParameterExpression],
    parameters: &[Symbol],
) -> Option<(usize, Symbol)> {
    let declared: HashSet<&Symbol> = parameters.iter().collect();
    expressions
        .iter()
        .enumerate()
        .find_map(|(index, expression)| {
            expression
                .iter_symbols()
                .find(|symbol| !declared.contains(*symbol))
                .map(|symbol| (index, symbol.clone()))
        })
}

/// Errors returned by [`BindParameters`].
#[derive(Debug, Error)]
pub enum BindParametersError {
    /// An expression references an undeclared parameter.
    #[error("expression {expression} references undeclared parameter {}", .parameter.fullname())]
    UndeclaredParameter {
        expression: usize,
        parameter: Symbol,
    },
    /// An operand has the wrong type or shape.
    #[error("expected a floating-point tensor of shape [..., {parameters}], got {actual}")]
    ValueType {
        parameters: usize,
        actual: TensorType,
    },
    /// The result dtype is neither `F64` nor `C128`.
    #[error("expected a result dtype of F64 or C128, got {dtype}")]
    ResultDType { dtype: DType },
    /// An expression did not evaluate to a finite number.
    #[error("expression {expression} did not evaluate to a finite number")]
    NonFinite { expression: usize },
    /// An expression has an unbound parameter.
    #[error("expression {expression} has an unbound parameter")]
    UnboundParameter { expression: usize },
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::data_tree::DataTree;
    use crate::program::{ProgramFunction, QuantumProgram};

    /// Return a standalone symbol named `name`.
    fn symbol(name: &str) -> Symbol {
        Symbol::standalone(name.to_string(), None)
    }

    /// Return the expression `symbol`, as an expression rather than a symbol.
    fn expr(symbol: &Symbol) -> ParameterExpression {
        ParameterExpression::from_symbol(symbol.clone())
    }

    /// Return a `TensorType` of `dtype` over fixed axes `shape`.
    fn ty(dtype: DType, shape: &[usize]) -> TensorType {
        TensorType {
            dtype,
            shape: shape.iter().copied().map(Dim::Fixed).collect(),
        }
    }

    /// Return `data` as an `F64` tensor of `shape` in row-major order.
    fn tensor(shape: &[usize], data: &[f64]) -> Tensor {
        Tensor::from(ArrayD::from_shape_vec(IxDyn(shape), data.to_vec()).unwrap())
    }

    /// Return `data` as a `C128` tensor of `shape` in row-major order.
    fn complex_tensor(shape: &[usize], data: &[Complex64]) -> Tensor {
        Tensor::from(ArrayD::from_shape_vec(IxDyn(shape), data.to_vec()).unwrap())
    }

    /// Return an op evaluating `expressions` over `parameters`, producing `F64`.
    fn real_op(expressions: Vec<ParameterExpression>, parameters: Vec<Symbol>) -> BindParameters {
        BindParameters::new(expressions, parameters, DType::F64).unwrap()
    }

    #[test]
    fn test_bind_parameters_full_name_and_arity() {
        let x = symbol("x");
        let op = real_op(vec![expr(&x)], vec![x.clone()]);
        assert_eq!(op.full_name(), "qiskit.bind_parameters");
        assert_eq!(op.arity(), 1);
        assert!(op.has_builtin_eval(), "Qiskit evaluates expressions");
        assert_eq!(op.expressions(), [expr(&x)]);
        assert_eq!(op.parameters(), [x]);
        assert_eq!(op.dtype(), DType::F64);
    }

    #[test]
    fn test_infer_output_types_gives_one_value_per_expression() {
        let (x, y) = (symbol("x"), symbol("y"));
        let op = real_op(
            vec![expr(&x).add(&expr(&y)).unwrap(), expr(&x), expr(&y)],
            vec![x, y],
        );
        assert_eq!(
            op.infer_output_types(&[ty(DType::F64, &[2])]).unwrap(),
            vec![ty(DType::F64, &[3])],
            "two parameters in, three expressions out"
        );
    }

    #[test]
    fn test_infer_output_types_carries_the_batch_prefix() {
        // The prefix is opaque, so any rank of it passes through.
        let x = symbol("x");
        let op = real_op(vec![expr(&x), expr(&x)], vec![x]);
        assert_eq!(
            op.infer_output_types(&[ty(DType::F32, &[5, 3, 1])])
                .unwrap(),
            vec![ty(DType::F64, &[5, 3, 2])]
        );
        let bounded = Dim::Bounded { max: 4 };
        assert_eq!(
            op.infer_output_types(&[TensorType {
                dtype: DType::F64,
                shape: vec![bounded, Dim::Fixed(1)],
            }])
            .unwrap(),
            vec![TensorType {
                dtype: DType::F64,
                shape: vec![bounded, Dim::Fixed(2)],
            }]
        );
    }

    #[test]
    fn test_infer_output_types_rejects_values_that_are_not_the_declared_parameters() {
        let x = symbol("x");
        let op = real_op(vec![expr(&x)], vec![x, symbol("y")]);

        // A dtype that is not floating point is refused the same way as a parameter axis of the
        // wrong length.
        for operand in [
            ty(DType::F64, &[3]),
            ty(DType::I64, &[2]),
            ty(DType::Bit, &[2]),
        ] {
            let err = op
                .infer_output_types(std::slice::from_ref(&operand))
                .unwrap_err();
            assert!(
                matches!(
                    &err,
                    BindParametersError::ValueType { parameters: 2, actual } if *actual == operand
                ),
                "{operand} is refused: {err}"
            );
        }
    }

    #[test]
    fn test_an_expression_may_not_reference_an_undeclared_parameter() {
        let (x, y) = (symbol("x"), symbol("y"));
        let Err(err) = BindParameters::new(vec![expr(&x), expr(&y)], vec![x], DType::F64) else {
            panic!("y has no value to be evaluated at")
        };
        assert!(
            matches!(
                &err,
                BindParametersError::UndeclaredParameter { expression: 1, parameter } if *parameter == y
            ),
            "the expression and the parameter that has no value are both named: {err}"
        );
    }

    #[test]
    fn test_surplus_declared_parameters_are_ignored() {
        // A caller can declare a whole circuit's parameters without pruning them to what each
        // expression uses.
        let (x, y, z) = (symbol("x"), symbol("y"), symbol("z"));
        let op = real_op(vec![expr(&y)], vec![x, y, z]);
        assert_eq!(
            op.infer_output_types(&[ty(DType::F64, &[3])]).unwrap(),
            vec![ty(DType::F64, &[1])]
        );
        assert_eq!(
            op.eval(&[tensor(&[3], &[0.5, 1.5, 2.5])]).unwrap(),
            vec![tensor(&[1], &[1.5])]
        );
    }

    #[test]
    fn test_eval_evaluates_every_expression_over_every_set_of_values() {
        let (x, y) = (symbol("x"), symbol("y"));
        let op = real_op(
            vec![
                expr(&x).add(&expr(&y)).unwrap(),
                expr(&x).mul(&expr(&x)).unwrap(),
                expr(&y).sin(),
            ],
            vec![x, y],
        );
        assert_eq!(
            op.eval(&[tensor(&[2, 2], &[0.5, 1.5, 2.0, -0.25])])
                .unwrap(),
            vec![tensor(
                &[2, 3],
                &[2.0, 0.25, 1.5_f64.sin(), 1.75, 4.0, (-0.25_f64).sin(),]
            )],
            "the values match evaluating the expressions directly"
        );
    }

    #[test]
    fn test_eval_widens_a_single_precision_operand_over_a_batch_of_any_rank() {
        let x = symbol("x");
        let op = real_op(
            vec![expr(&x).mul(&ParameterExpression::from_f64(2.0)).unwrap()],
            vec![x],
        );
        let values = Tensor::from(
            ArrayD::from_shape_vec(IxDyn(&[3, 1, 1]), vec![0.5_f32, 1.5, -2.0]).unwrap(),
        );
        assert_eq!(
            op.eval(std::slice::from_ref(&values)).unwrap(),
            vec![tensor(&[3, 1, 1], &[1.0, 3.0, -4.0])],
            "expressions evaluate in double precision, whatever the operand's precision"
        );
        assert_eq!(
            op.infer_output_types(&[values.tensor_type()]).unwrap(),
            vec![ty(DType::F64, &[3, 1, 1])]
        );
    }

    #[test]
    fn test_eval_over_an_empty_batch_and_over_no_parameters() {
        let x = symbol("x");
        let op = real_op(vec![expr(&x)], vec![x]);
        assert_eq!(
            op.eval(&[tensor(&[0, 1], &[])]).unwrap(),
            vec![tensor(&[0, 1], &[])]
        );

        let constant = real_op(vec![ParameterExpression::from_f64(0.25)], vec![]);
        assert_eq!(
            constant
                .infer_output_types(&[ty(DType::F64, &[2, 0])])
                .unwrap(),
            vec![ty(DType::F64, &[2, 1])]
        );
        assert_eq!(
            constant.eval(&[tensor(&[2, 0], &[])]).unwrap(),
            vec![tensor(&[2, 1], &[0.25, 0.25])]
        );
    }

    #[test]
    #[should_panic(expected = "qiskit.bind_parameters expects one value per parameter")]
    fn test_eval_of_the_wrong_number_of_values_per_set_panics() {
        // Unreachable through a program function, which type-checks the operand while adding the
        // op, but reading the values as sets of the wrong size would be silently wrong.
        let x = symbol("x");
        let op = real_op(vec![expr(&x)], vec![x]);
        let _ = op.eval(&[tensor(&[2, 3], &[0.0; 6])]);
    }

    #[test]
    fn test_a_non_finite_result_is_reported() {
        let x = symbol("x");
        let op = real_op(
            vec![
                expr(&x),
                ParameterExpression::from_f64(1.0).div(&expr(&x)).unwrap(),
            ],
            vec![x],
        );
        assert_eq!(
            op.eval(&[tensor(&[1], &[2.0])]).unwrap(),
            vec![tensor(&[2], &[2.0, 0.5])]
        );
        let err = op.eval(&[tensor(&[1], &[0.0])]).unwrap_err();
        assert!(
            matches!(err, BindParametersError::NonFinite { expression: 1 }),
            "the expression that has no value is named: {err}"
        );
    }

    #[test]
    fn test_an_f64_result_keeps_the_real_part_of_a_complex_value() {
        // `log(-1)` is `i * pi`, so this expression is real or complex depending on the value.
        let x = symbol("x");
        let op = real_op(vec![expr(&x).log()], vec![x]);
        assert_eq!(
            op.eval(&[tensor(&[1], &[-1.0])]).unwrap(),
            vec![tensor(&[1], &[0.0])],
            "the real part of i * pi is zero"
        );
        assert_eq!(
            op.eval(&[tensor(&[1], &[1.0])]).unwrap(),
            vec![tensor(&[1], &[0.0])]
        );
    }

    #[test]
    fn test_every_expression_kind_evaluates_to_a_value() {
        // Binding every declared parameter leaves nothing symbolic, whatever the expression is
        // built from, which is why an unbound parameter cannot reach the result.
        let x = symbol("x");
        let expressions = vec![
            expr(&x).sin(),
            expr(&x).cos(),
            expr(&x).tan(),
            expr(&x).atan(),
            expr(&x).exp(),
            expr(&x).abs(),
            expr(&x).sign(),
            expr(&x).conjugate(),
            expr(&x).pow(&ParameterExpression::from_f64(3.0)).unwrap(),
            expr(&x).sub(&expr(&x)).unwrap(),
        ];
        let count = expressions.len();
        let op = real_op(expressions, vec![x]);
        let results = op.eval(&[tensor(&[1], &[0.5])]).unwrap();
        assert_eq!(results[0].shape(), [count]);
    }

    #[test]
    fn test_a_c128_result_admits_a_complex_value() {
        let x = symbol("x");
        let op = BindParameters::new(vec![expr(&x).log()], vec![x], DType::C128).unwrap();
        assert_eq!(op.dtype(), DType::C128);
        assert_eq!(
            op.infer_output_types(&[ty(DType::F64, &[1])]).unwrap(),
            vec![ty(DType::C128, &[1])]
        );
        assert_eq!(
            op.eval(&[tensor(&[1], &[-1.0])]).unwrap(),
            vec![complex_tensor(
                &[1],
                &[Complex64::new(0.0, std::f64::consts::PI)]
            )],
            "the value an F64 result refuses is the value a C128 result returns"
        );
        assert_eq!(
            op.eval(&[tensor(&[1], &[1.0])]).unwrap(),
            vec![complex_tensor(&[1], &[Complex64::new(0.0, 0.0)])],
            "a real value widens to a complex one"
        );
    }

    #[test]
    fn test_a_result_dtype_other_than_f64_or_c128_is_refused() {
        // Expressions evaluate in double precision, so a narrower result comes from a `Cast`.
        let x = symbol("x");
        for dtype in [DType::F32, DType::C64, DType::I64, DType::Bit] {
            let Err(err) = BindParameters::new(vec![expr(&x)], vec![x.clone()], dtype) else {
                panic!("{dtype} is neither F64 nor C128")
            };
            assert!(
                matches!(err, BindParametersError::ResultDType { dtype: got } if got == dtype),
                "{dtype} is refused: {err}"
            );
        }
    }

    #[test]
    fn test_a_program_computes_parameters_from_its_inputs() {
        let theta = symbol("theta");
        let op = real_op(
            vec![
                expr(&theta)
                    .mul(&ParameterExpression::from_f64(2.0))
                    .unwrap(),
                ParameterExpression::from_f64(0.25),
            ],
            vec![theta],
        );

        let mut function = ProgramFunction::new();
        let sweep = function.add_parameter(ty(DType::F64, &[2, 1]));
        let angles = function.add_op(op, &[sweep]).unwrap()[0];
        function.add_result(angles).unwrap();
        assert_eq!(function.type_of(angles), Some(&ty(DType::F64, &[2, 2])));

        let program = QuantumProgram::new(
            vec![function],
            DataTree::mapping([("theta", DataTree::Leaf(()))]).unwrap(),
            DataTree::mapping([("angles", DataTree::Leaf(()))]).unwrap(),
        )
        .unwrap();
        assert!(program.entry_function().has_builtin_eval());

        let inputs =
            DataTree::mapping([("theta", DataTree::Leaf(tensor(&[2, 1], &[0.5, 1.5])))]).unwrap();
        assert_eq!(
            program.eval(inputs).unwrap(),
            DataTree::mapping([(
                "angles",
                DataTree::Leaf(tensor(&[2, 2], &[1.0, 0.25, 3.0, 0.25]))
            )])
            .unwrap()
        );
    }
}

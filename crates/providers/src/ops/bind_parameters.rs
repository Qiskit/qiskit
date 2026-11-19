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

use hashbrown::HashMap;
use ndarray::{ArrayD, IxDyn};
use num_complex::Complex64;
use qiskit_circuit::parameter::parameter_expression::ParameterExpression;
use qiskit_circuit::parameter::symbol_expr::{Symbol, Value};
use thiserror::Error;

use super::inference::{float, leading_axes};
use super::{ProgramOp, QISKIT};
use crate::tensor::{DType, Dim, Tensor, TensorType};

/// A strategy for evaluating one expression of a [`BindParameters`], allowing us to short-cut the cases
/// where an expression is just a parameter or a constant.
#[derive(Clone, Debug)]
enum Evaluation {
    /// The expression is the parameter at this index in `parameters`.
    Parameter(usize),
    /// The expression has no parameters. The value is `None` if the evaluation fails (e.g. `1 / 0`).
    Constant(Option<Value>),
    /// The expression is evaluated with [`ParameterExpression::bind`]. Its parameters are at the
    /// indices `indices[start..end]` in `parameters`.
    General { start: usize, end: usize },
}

/// Evaluate parameter expressions over a batch of parameter values.
///
/// The operand has one value per declared parameter in its trailing axis, and the result has one
/// value per expression. Leading axes on the operand are a batch prefix. The result dtype is
/// `F64` or `C128`. When an expression evaluates to a complex number, an `F64` result keeps just
/// the real part.
///
/// An expression that does not evaluate to a finite number, for example by dividing by zero,
/// is set to NaN. In a `C128` result, both the real and imaginary parts are set to NaN.
#[derive(Clone, Debug)]
pub struct BindParameters {
    expressions: Vec<ParameterExpression>,
    parameters: Vec<Symbol>,
    dtype: DType,
    /// The evaluation strategy for each expression, in the order of `expressions`.
    evaluations: Vec<Evaluation>,
    /// The indices in `parameters` of the parameters of each general expression, concatenated.
    indices: Vec<usize>,
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
        let mut evaluations = Vec::with_capacity(expressions.len());
        let mut indices = Vec::new();
        {
            let parameter_indices: HashMap<&Symbol, usize> = parameters
                .iter()
                .enumerate()
                .map(|(index, parameter)| (parameter, index))
                .collect();
            for (expression_index, expression) in expressions.iter().enumerate() {
                let index = |symbol: &Symbol| {
                    parameter_indices.get(symbol).copied().ok_or_else(|| {
                        BindParametersError::UndeclaredParameter {
                            expression: expression_index,
                            parameter: symbol.clone(),
                        }
                    })
                };
                let evaluation = if let Ok(symbol) = expression.try_to_symbol_ref() {
                    Evaluation::Parameter(index(symbol)?)
                } else if expression.num_symbols() == 0 {
                    Evaluation::Constant(evaluate_expression(expression, &HashMap::new()))
                } else {
                    let start = indices.len();
                    for symbol in expression.iter_symbols() {
                        indices.push(index(symbol)?);
                    }
                    Evaluation::General {
                        start,
                        end: indices.len(),
                    }
                };
                evaluations.push(evaluation);
            }
        }
        Ok(Self {
            expressions,
            parameters,
            dtype,
            evaluations,
            indices,
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
    fn evaluate_tensor(
        &self,
        values: impl Iterator<Item = f64>,
        rows: usize,
        shape: &[usize],
    ) -> Tensor {
        match self.dtype {
            DType::C128 => Tensor::from(reshape(
                self.evaluate_flat(values, rows, complex, Complex64::new(f64::NAN, f64::NAN)),
                shape,
            )),
            DType::F64 => Tensor::from(reshape(
                self.evaluate_flat(values, rows, real, f64::NAN),
                shape,
            )),
            other => panic!("{} cannot produce {}", self.full_name(), other),
        }
    }

    /// Evaluate every expression over each of `rows` consecutive sets of values from `values`,
    /// mapping each result through `convert`.
    ///
    /// The results are returned in one flat vector, in row-major order.
    fn evaluate_flat<T: Copy>(
        &self,
        mut values: impl Iterator<Item = f64>,
        rows: usize,
        convert: fn(Value) -> T,
        nan: T,
    ) -> Vec<T> {
        let mut bindings: HashMap<&Symbol, Value> = HashMap::new();
        let mut row = Vec::with_capacity(self.parameters.len());
        let mut results = Vec::with_capacity(rows * self.expressions.len());
        for _ in 0..rows {
            row.clear();
            row.extend(values.by_ref().take(self.parameters.len()));
            for (expression, evaluation) in self.expressions.iter().zip(&self.evaluations) {
                let value = match *evaluation {
                    Evaluation::Parameter(index) => {
                        let value = row[index];
                        value.is_finite().then_some(Value::Real(value))
                    }
                    Evaluation::Constant(value) => value,
                    Evaluation::General { start, end } => {
                        // The map has only this expression's parameters. The cost of
                        // `ParameterExpression::bind` is linear in the size of the map.
                        bindings.clear();
                        bindings.extend(
                            self.indices[start..end]
                                .iter()
                                .map(|&index| (&self.parameters[index], Value::Real(row[index]))),
                        );
                        evaluate_expression(expression, &bindings)
                    }
                };
                results.push(value.map_or(nan, convert));
            }
        }
        results
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
            Tensor::F32(x) => {
                self.evaluate_tensor(x.iter().map(|&value| f64::from(value)), rows, &shape)
            }
            Tensor::F64(x) => self.evaluate_tensor(x.iter().copied(), rows, &shape),
            other => panic!(
                "{} expects floating-point parameter values, got {}",
                self.full_name(),
                other.dtype()
            ),
        };
        Ok(vec![results])
    }
}

/// Return the value of `expression` with `bindings`, or `None` if the evaluation fails.
fn evaluate_expression(
    expression: &ParameterExpression,
    bindings: &HashMap<&Symbol, Value>,
) -> Option<Value> {
    expression
        .bind(bindings, true)
        .and_then(|bound| bound.try_to_value(true))
        .ok()
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
}

#[cfg(test)]
mod test {
    use super::*;

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
    fn test_infer_output_types_keeps_the_batch_prefix() {
        let x = symbol("x");
        let op = real_op(vec![expr(&x), expr(&x)], vec![x]);
        assert_eq!(
            op.infer_output_types(&[ty(DType::F32, &[5, 3, 1])])
                .unwrap(),
            vec![ty(DType::F64, &[5, 3, 2])]
        );
    }

    #[test]
    fn test_infer_output_types_rejects_values_that_are_not_the_declared_parameters() {
        let x = symbol("x");
        let op = real_op(vec![expr(&x)], vec![x, symbol("y")]);

        // A dtype that is not floating point is rejected with the same error as a parameter axis
        // of the wrong length.
        for operand in [ty(DType::F64, &[3]), ty(DType::I64, &[2])] {
            let err = op
                .infer_output_types(std::slice::from_ref(&operand))
                .unwrap_err();
            assert!(
                matches!(
                    &err,
                    BindParametersError::ValueType { parameters: 2, actual } if *actual == operand
                ),
                "{operand} is rejected: {err}"
            );
        }
    }

    #[test]
    fn test_an_expression_may_not_reference_an_undeclared_parameter() {
        // `y` is a bare parameter, and `x + y` is a general expression.
        let (x, y) = (symbol("x"), symbol("y"));
        for undeclared in [expr(&y), expr(&x).add(&expr(&y)).unwrap()] {
            let Err(err) =
                BindParameters::new(vec![expr(&x), undeclared], vec![x.clone()], DType::F64)
            else {
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
    }

    #[test]
    fn test_surplus_declared_parameters_are_ignored() {
        // A caller can declare a whole circuit's parameters without pruning them to what each
        // expression uses. `y` is a bare parameter, and `z - y` is a general expression.
        let (x, y, z) = (symbol("x"), symbol("y"), symbol("z"));
        let op = real_op(
            vec![expr(&y), expr(&z).sub(&expr(&y)).unwrap()],
            vec![x, y, z],
        );
        assert_eq!(
            op.infer_output_types(&[ty(DType::F64, &[3])]).unwrap(),
            vec![ty(DType::F64, &[2])]
        );
        assert_eq!(
            op.eval(&[tensor(&[3], &[0.5, 1.5, 2.5])]).unwrap(),
            vec![tensor(&[2], &[1.5, 1.0])]
        );
    }

    #[test]
    fn test_eval_evaluates_every_expression_over_every_set_of_values() {
        // The six results are distinct, so any transposition of expressions against sets of
        // values is caught. Each is exact in binary floating point, which keeps the comparison
        // exact: Miri's shims for the transcendental functions are deliberately perturbed by up
        // to an ULP, and independently per call.
        let (x, y) = (symbol("x"), symbol("y"));
        let op = real_op(
            vec![
                expr(&x).add(&expr(&y)).unwrap(),
                expr(&x).mul(&expr(&x)).unwrap(),
                expr(&y).sub(&expr(&x)).unwrap(),
            ],
            vec![x, y],
        );
        assert_eq!(
            op.eval(&[tensor(&[2, 2], &[0.5, 1.5, 2.0, -0.25])])
                .unwrap(),
            vec![tensor(&[2, 3], &[2.0, 0.25, 1.0, 1.75, 4.0, -2.25])],
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
            op.eval(&[values]).unwrap(),
            vec![tensor(&[3, 1, 1], &[1.0, 3.0, -4.0])],
            "expressions evaluate in double precision, whatever the operand's precision"
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
    #[should_panic]
    fn test_eval_of_the_wrong_number_of_values_per_set_panics() {
        // Unreachable through a program function, which type-checks the operand while adding the
        // op, but reading the values as sets of the wrong size would be silently wrong.
        let x = symbol("x");
        let op = real_op(vec![expr(&x)], vec![x]);
        let _ = op.eval(&[tensor(&[2, 3], &[0.0; 6])]);
    }

    #[test]
    fn test_a_non_finite_result_is_nan() {
        let x = symbol("x");
        let expressions = vec![
            expr(&x),
            ParameterExpression::from_f64(1.0).div(&expr(&x)).unwrap(),
        ];
        let values = tensor(&[2, 1], &[2.0, 0.0]);

        let op = real_op(expressions.clone(), vec![x.clone()]);
        let results = op.eval(std::slice::from_ref(&values)).unwrap();
        let Tensor::F64(results) = &results[0] else {
            panic!("the result is F64")
        };
        assert_eq!(results.shape(), [2, 2]);
        let results: Vec<Option<f64>> = results
            .iter()
            .map(|&value| (!value.is_nan()).then_some(value))
            .collect();
        assert_eq!(results, [Some(2.0), Some(0.5), Some(0.0), None]);

        let op = BindParameters::new(expressions, vec![x], DType::C128).unwrap();
        let results = op.eval(std::slice::from_ref(&values)).unwrap();
        let Tensor::C128(results) = &results[0] else {
            panic!("the result is C128")
        };
        let nan = results[[1, 1]];
        assert!(
            nan.re.is_nan() && nan.im.is_nan(),
            "{nan} is NaN in both parts"
        );
    }

    #[test]
    fn test_bare_parameters_and_constants_match_bind() {
        // Bare parameters and constants are evaluated without `ParameterExpression::bind`. The
        // product is bound in the usual way, to check the order of mixed results.
        let (x, y) = (symbol("x"), symbol("y"));
        let expressions = vec![
            expr(&y),
            expr(&x),
            ParameterExpression::from_f64(0.25),
            ParameterExpression::from_f64(f64::INFINITY),
            ParameterExpression::from_f64(f64::NAN),
            ParameterExpression::from_f64(0.0).log(),
            ParameterExpression::from_f64(-1.0).log(),
            expr(&x).mul(&expr(&y)).unwrap(),
        ];
        let values = [
            1.5,
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            f64::MIN_POSITIVE / 2.0,
        ];
        let sets: Vec<[f64; 2]> = values
            .iter()
            .flat_map(|&a| values.iter().map(move |&b| [a, b]))
            .collect();
        let expected: Vec<Option<Value>> = sets
            .iter()
            .flat_map(|&[a, b]| {
                let bindings = HashMap::from([(&x, Value::Real(a)), (&y, Value::Real(b))]);
                expressions
                    .iter()
                    .map(|expression| evaluate_expression(expression, &bindings))
                    .collect::<Vec<_>>()
            })
            .collect();
        let values = tensor(&[sets.len(), 2], sets.as_flattened());
        let parameters = vec![x.clone(), y.clone()];

        let op = real_op(expressions.clone(), parameters.clone());
        let Tensor::F64(results) = &op.eval(std::slice::from_ref(&values)).unwrap()[0] else {
            panic!("the result is F64")
        };
        let results: Vec<Option<f64>> = results
            .iter()
            .map(|&value| (!value.is_nan()).then_some(value))
            .collect();
        assert_eq!(
            results,
            expected
                .iter()
                .map(|value| value.map(real))
                .collect::<Vec<_>>()
        );

        let op = BindParameters::new(expressions, parameters, DType::C128).unwrap();
        let Tensor::C128(results) = &op.eval(std::slice::from_ref(&values)).unwrap()[0] else {
            panic!("the result is C128")
        };
        let results: Vec<Option<Complex64>> = results
            .iter()
            .map(|&value| (!value.is_nan()).then_some(value))
            .collect();
        assert_eq!(
            results,
            expected
                .iter()
                .map(|value| value.map(complex))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_an_f64_result_keeps_the_real_part_of_a_complex_value() {
        let x = symbol("x");
        let op = real_op(vec![expr(&x).log()], vec![x]);
        assert_eq!(
            op.eval(&[tensor(&[1], &[-1.0])]).unwrap(),
            vec![tensor(&[1], &[0.0])],
            "the real part of log(-1) = i * pi is zero"
        );
    }

    #[test]
    fn test_a_c128_result_keeps_a_complex_value() {
        let x = symbol("x");
        let op = BindParameters::new(vec![expr(&x).log()], vec![x], DType::C128).unwrap();
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
            "log(-1) is i * pi"
        );
        assert_eq!(
            op.eval(&[tensor(&[1], &[1.0])]).unwrap(),
            vec![complex_tensor(&[1], &[Complex64::new(0.0, 0.0)])],
            "a real value widens to a complex one"
        );
    }

    #[test]
    fn test_a_result_dtype_other_than_f64_or_c128_is_rejected() {
        // Expressions evaluate in double precision, so a narrower result comes from a `Cast`.
        let x = symbol("x");
        for dtype in [DType::F32, DType::C64, DType::I64, DType::Bit] {
            let Err(err) = BindParameters::new(vec![expr(&x)], vec![x.clone()], dtype) else {
                panic!("{dtype} is neither F64 nor C128")
            };
            assert!(
                matches!(err, BindParametersError::ResultDType { dtype: got } if got == dtype),
                "{dtype} is rejected: {err}"
            );
        }
    }
}

// This code is part of Qiskit.
//
// (C) Copyright IBM 2023, 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

/// symbol_expr_py.rs
/// Python interface of symbolic expression
use crate::symbol_expr::{Symbol, SymbolExpr, Value};
use crate::symbol_parser::parse_expression;

use hashbrown::{HashMap, HashSet};
use num_complex::Complex64;

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use pyo3::prelude::*;

/// Python interface to SymbolExpr
#[pyclass(sequence, module = "qiskit._accelerate.circuit")]
#[derive(Clone, Debug)]
pub struct PySymbolExpr {
    pub expr: SymbolExpr,
}

/// enum for parameter value types, used to accept multiple types from Python
#[derive(FromPyObject, Clone, Debug)]
pub enum ParameterValue {
    #[pyo3(transparent, annotation = "int")]
    Int(i64),
    #[pyo3(transparent, annotation = "float")]
    Real(f64),
    #[pyo3(transparent, annotation = "complex")]
    Complex(Complex64),
    #[pyo3(transparent, annotation = "str")]
    Str(String),
    Expr(PySymbolExpr),
}

/// enum for bind value types, used to accept multiple types from Python
#[derive(FromPyObject, Clone, Debug)]
pub enum BindValue {
    #[pyo3(transparent, annotation = "int")]
    Int(i64),
    #[pyo3(transparent, annotation = "float")]
    Real(f64),
    #[pyo3(transparent, annotation = "complex")]
    Complex(Complex64),
}

/// value types to return to Python
#[derive(IntoPyObject)]
pub enum ReturnValueTypes {
    Int(i64),
    Real(f64),
    Complex(Complex64),
}

#[pymethods]
impl PySymbolExpr {
    /// parse expression from string
    #[new]
    #[pyo3(signature = (in_expr=None))]
    pub fn new(in_expr: Option<String>) -> PyResult<Self> {
        match in_expr {
            Some(e) => Ok(PySymbolExpr {
                expr: parse_expression(&e),
            }),
            None => Ok(PySymbolExpr {
                expr: SymbolExpr::Value(Value::Real(0.0)),
            }),
        }
    }

    /// create new expression as a symbol
    #[allow(non_snake_case)]
    #[staticmethod]
    pub fn Symbol(name: String) -> Self {
        PySymbolExpr {
            expr: SymbolExpr::Symbol(Symbol::new(&name)),
        }
    }

    /// create new expression as a value
    #[allow(non_snake_case)]
    #[staticmethod]
    pub fn Value(value: ParameterValue) -> Self {
        match value {
            ParameterValue::Real(r) => PySymbolExpr {
                expr: SymbolExpr::Value(Value::from(r)),
            },
            ParameterValue::Complex(c) => PySymbolExpr {
                expr: SymbolExpr::Value(Value::from(c)),
            },
            ParameterValue::Int(r) => PySymbolExpr {
                expr: SymbolExpr::Value(Value::from(r)),
            },
            ParameterValue::Str(s) => PySymbolExpr {
                expr: parse_expression(&s),
            },
            ParameterValue::Expr(e) => PySymbolExpr { expr: e.expr },
        }
    }

    /// this is called for np.complex128 because np.complex is recognized as Real in Value function
    #[allow(non_snake_case)]
    #[staticmethod]
    pub fn Complex(value: Complex64) -> Self {
        PySymbolExpr {
            expr: SymbolExpr::Value(Value::from(value)),
        }
    }

    /// create new expression from string
    #[allow(non_snake_case)]
    #[staticmethod]
    pub fn Expression(name: String) -> Self {
        PySymbolExpr {
            expr: parse_expression(&name),
        }
    }

    /// unary functions
    pub fn sin(&self) -> Self {
        Self {
            expr: self.expr.sin(),
        }
    }
    pub fn cos(&self) -> Self {
        Self {
            expr: self.expr.cos(),
        }
    }
    pub fn tan(&self) -> Self {
        Self {
            expr: self.expr.tan(),
        }
    }
    pub fn asin(&self) -> Self {
        Self {
            expr: self.expr.asin(),
        }
    }
    pub fn acos(&self) -> Self {
        Self {
            expr: self.expr.acos(),
        }
    }
    pub fn atan(&self) -> Self {
        Self {
            expr: self.expr.atan(),
        }
    }
    pub fn exp(&self) -> Self {
        Self {
            expr: self.expr.exp(),
        }
    }
    pub fn log(&self) -> Self {
        Self {
            expr: self.expr.log(),
        }
    }
    pub fn abs(&self) -> Self {
        Self {
            expr: self.expr.abs(),
        }
    }
    pub fn sign(&self) -> Self {
        Self {
            expr: self.expr.sign(),
        }
    }
    pub fn pow(&self, rhs: ParameterValue) -> Self {
        match rhs {
            ParameterValue::Real(r) => Self {
                expr: self.expr.pow(&SymbolExpr::Value(Value::from(r))),
            },
            ParameterValue::Complex(c) => Self {
                expr: self.expr.pow(&SymbolExpr::Value(Value::from(c))),
            },
            ParameterValue::Int(r) => Self {
                expr: self.expr.pow(&SymbolExpr::Value(Value::from(r))),
            },
            ParameterValue::Str(s) => Self {
                expr: self.expr.pow(&parse_expression(&s)),
            },
            ParameterValue::Expr(e) => Self {
                expr: self.expr.pow(&e.expr),
            },
        }
    }

    /// return complex number if expression does not have symbols
    pub fn complex(&self) -> PyResult<Complex64> {
        match self.expr.eval(true) {
            Some(v) => match v {
                Value::Real(r) => Ok(Complex64::from(r)),
                Value::Int(r) => Ok(Complex64::from(r as f64)),
                Value::Complex(c) => Ok(c),
            },
            None => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Expression has some undefined symbols.",
            )),
        }
    }
    /// return floating number if expression does not have symbols
    pub fn float(&self) -> PyResult<f64> {
        match self.expr.eval(true) {
            Some(v) => match v {
                Value::Real(r) => Ok(r),
                Value::Int(r) => Ok(r as f64),
                Value::Complex(_) => Err(pyo3::exceptions::PyTypeError::new_err(
                    "complex can not be converted to float",
                )),
            },
            None => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Expression has some undefined symbols.",
            )),
        }
    }

    // return real number if expression does not have symbols
    pub fn real(&self) -> PyResult<f64> {
        match self.expr.real() {
            Some(r) => Ok(r),
            None => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Expression has some undefined symbols.",
            )),
        }
    }

    // return imaginary number if expression does not have symbols
    pub fn imag(&self) -> PyResult<f64> {
        match self.expr.imag() {
            Some(r) => Ok(r),
            None => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Expression has some undefined symbols.",
            )),
        }
    }

    /// return integer number if expression does not have symbols
    pub fn int(&self) -> PyResult<i64> {
        match self.expr.eval(true) {
            Some(v) => match v {
                Value::Real(r) => Ok(r as i64),
                Value::Int(r) => Ok(r),
                Value::Complex(_) => Err(pyo3::exceptions::PyTypeError::new_err(
                    "complex can not be converted to int",
                )),
            },
            None => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Expression has some undefined symbols.",
            )),
        }
    }

    /// clone expression
    pub fn copy(&self) -> Self {
        Self {
            expr: self.expr.clone(),
        }
    }
    /// return conjugate of expression
    pub fn conjugate(&self) -> Self {
        Self {
            expr: self.expr.conjugate(),
        }
    }
    /// return derivative of this expression for param
    pub fn derivative(&self, param: &Self) -> Self {
        Self {
            expr: self.expr.derivative(&param.expr),
        }
    }

    /// expand expression
    pub fn expand(&self) -> Self {
        Self {
            expr: self.expr.expand(),
        }
    }

    /// helper function to calculate reciprocal of the equation
    pub fn rcp(&self) -> Self {
        Self {
            expr: self.expr.rcp(),
        }
    }

    /// helper function to calculate square root of the equation
    pub fn sqrt(&self) -> Self {
        Self {
            expr: self.expr.sqrt(),
        }
    }

    /// check if this expression is real number of not
    #[getter]
    pub fn is_real(&self) -> Option<bool> {
        self.expr.is_real()
    }
    /// check if this expression is complex number of not
    #[getter]
    pub fn is_complex(&self) -> Option<bool> {
        self.expr.is_complex()
    }
    /// check if this expression is integer of not
    #[getter]
    pub fn is_int(&self) -> Option<bool> {
        self.expr.is_int()
    }

    /// get hashset of all the symbols used in this expression
    #[getter]
    pub fn symbols(&self) -> HashSet<String> {
        self.expr.symbols()
    }

    /// return all values in this equation
    pub fn values(&self) -> PyResult<Vec<ReturnValueTypes>> {
        let ret: Vec<ReturnValueTypes> = self
            .expr
            .values()
            .iter()
            .map(|val| match val {
                Value::Real(r) => ReturnValueTypes::Real(*r),
                Value::Int(i) => ReturnValueTypes::Int(*i),
                Value::Complex(c) => ReturnValueTypes::Complex(*c),
            })
            .collect();
        Ok(ret)
    }

    /// return expression as a string
    #[getter]
    pub fn name(&self) -> String {
        self.expr.to_string()
    }

    /// bind values to symbols given by input hashmap
    pub fn bind(&self, in_maps: HashMap<String, BindValue>) -> PyResult<Self> {
        let maps: HashMap<String, Value> = in_maps
            .into_iter()
            .map(|(key, val)| {
                (
                    key,
                    match val {
                        BindValue::Complex(c) => Value::from(c),
                        BindValue::Real(r) => Value::from(r),
                        BindValue::Int(r) => Value::from(r),
                    },
                )
            })
            .collect();
        let bound = self.expr.bind(&maps);
        match bound {
            SymbolExpr::Value(ref v) => match v {
                Value::Real(r) => {
                    if *r == f64::INFINITY {
                        Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                            "zero division occurs while binding parameter",
                        ))
                    } else {
                        Ok(Self { expr: bound })
                    }
                }
                Value::Int(_) => Ok(Self { expr: bound }),
                Value::Complex(c) => {
                    if c.re == f64::INFINITY || c.im == f64::INFINITY {
                        Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                            "zero division occurs while binding parameter",
                        ))
                    } else if c.im < f64::EPSILON && c.im > -f64::EPSILON {
                        Ok(Self {
                            expr: SymbolExpr::Value(Value::Real(c.re)),
                        })
                    } else {
                        Ok(Self { expr: bound })
                    }
                }
            },
            _ => Ok(Self { expr: bound }),
        }
    }
    // this function is used for numpy.complex128
    pub fn bind_complex(&self, in_maps: HashMap<String, Complex64>) -> PyResult<Self> {
        let maps: HashMap<String, Value> = in_maps
            .iter()
            .map(|(key, val)| (key.clone(), Value::from(*val)))
            .collect();
        let bound = self.expr.bind(&maps);
        match bound {
            SymbolExpr::Value(ref v) => match v {
                Value::Real(r) => {
                    if *r == f64::INFINITY {
                        Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                            "zero division occurs while binding parameter",
                        ))
                    } else {
                        Ok(Self { expr: bound })
                    }
                }
                Value::Int(_) => Ok(Self { expr: bound }),
                Value::Complex(c) => {
                    if c.re == f64::INFINITY || c.im == f64::INFINITY {
                        Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                            "zero division occurs while binding parameter",
                        ))
                    } else if c.im < f64::EPSILON && c.im > -f64::EPSILON {
                        Ok(Self {
                            expr: SymbolExpr::Value(Value::Real(c.re)),
                        })
                    } else {
                        Ok(Self { expr: bound })
                    }
                }
            },
            _ => Ok(Self { expr: bound }),
        }
    }

    /// substitute symbols to expressions (or values) given by hash map
    pub fn subs(&self, in_maps: HashMap<String, Self>) -> Self {
        let maps: HashMap<String, SymbolExpr> = in_maps
            .iter()
            .map(|(key, val)| (key.clone(), val.expr.clone()))
            .collect();
        Self {
            expr: self.expr.subs(&maps),
        }
    }

    // ====================================
    // operator overrides
    // ====================================
    pub fn __eq__(&self, rhs: ParameterValue) -> bool {
        match rhs {
            ParameterValue::Real(r) => self.expr == SymbolExpr::Value(Value::from(r)),
            ParameterValue::Complex(c) => self.expr == SymbolExpr::Value(Value::from(c)),
            ParameterValue::Int(r) => self.expr == SymbolExpr::Value(Value::from(r)),
            ParameterValue::Str(s) => self.expr == parse_expression(&s),
            ParameterValue::Expr(e) => self.expr == e.expr,
        }
    }
    pub fn __ne__(&self, rhs: ParameterValue) -> bool {
        match rhs {
            ParameterValue::Real(r) => self.expr != SymbolExpr::Value(Value::from(r)),
            ParameterValue::Complex(c) => self.expr != SymbolExpr::Value(Value::from(c)),
            ParameterValue::Int(r) => self.expr != SymbolExpr::Value(Value::from(r)),
            ParameterValue::Str(s) => self.expr != parse_expression(&s),
            ParameterValue::Expr(e) => self.expr != e.expr,
        }
    }
    pub fn __neg__(&self) -> Self {
        Self { expr: -&self.expr }
    }
    pub fn __add__(&self, rhs: ParameterValue) -> Self {
        match rhs {
            ParameterValue::Real(r) => Self {
                expr: &self.expr + &SymbolExpr::Value(Value::from(r)),
            },
            ParameterValue::Complex(c) => Self {
                expr: &self.expr + &SymbolExpr::Value(Value::from(c)),
            },
            ParameterValue::Int(r) => Self {
                expr: &self.expr + &SymbolExpr::Value(Value::from(r)),
            },
            ParameterValue::Str(s) => Self {
                expr: &self.expr + &parse_expression(&s),
            },
            ParameterValue::Expr(e) => Self {
                expr: &self.expr + &e.expr,
            },
        }
    }
    pub fn __radd__(&self, rhs: ParameterValue) -> Self {
        match rhs {
            ParameterValue::Real(r) => Self {
                expr: &SymbolExpr::Value(Value::from(r)) + &self.expr,
            },
            ParameterValue::Complex(c) => Self {
                expr: &SymbolExpr::Value(Value::from(c)) + &self.expr,
            },
            ParameterValue::Int(r) => Self {
                expr: &SymbolExpr::Value(Value::from(r)) + &self.expr,
            },
            ParameterValue::Str(s) => Self {
                expr: &parse_expression(&s) + &self.expr,
            },
            ParameterValue::Expr(e) => Self {
                expr: &e.expr + &self.expr,
            },
        }
    }
    pub fn __sub__(&self, rhs: ParameterValue) -> Self {
        match rhs {
            ParameterValue::Real(r) => Self {
                expr: &self.expr - &SymbolExpr::Value(Value::from(r)),
            },
            ParameterValue::Complex(c) => Self {
                expr: &self.expr - &SymbolExpr::Value(Value::from(c)),
            },
            ParameterValue::Int(r) => Self {
                expr: &self.expr - &SymbolExpr::Value(Value::from(r)),
            },
            ParameterValue::Str(s) => Self {
                expr: &self.expr - &parse_expression(&s),
            },
            ParameterValue::Expr(e) => Self {
                expr: &self.expr - &e.expr,
            },
        }
    }
    pub fn __rsub__(&self, rhs: ParameterValue) -> Self {
        match rhs {
            ParameterValue::Real(r) => Self {
                expr: &SymbolExpr::Value(Value::from(r)) - &self.expr,
            },
            ParameterValue::Complex(c) => Self {
                expr: &SymbolExpr::Value(Value::from(c)) - &self.expr,
            },
            ParameterValue::Int(r) => Self {
                expr: &SymbolExpr::Value(Value::from(r)) - &self.expr,
            },
            ParameterValue::Str(s) => Self {
                expr: &parse_expression(&s) - &self.expr,
            },
            ParameterValue::Expr(e) => Self {
                expr: &e.expr - &self.expr,
            },
        }
    }
    pub fn __mul__(&self, rhs: ParameterValue) -> Self {
        match rhs {
            ParameterValue::Real(r) => Self {
                expr: &self.expr * &SymbolExpr::Value(Value::from(r)),
            },
            ParameterValue::Complex(c) => Self {
                expr: &self.expr * &SymbolExpr::Value(Value::from(c)),
            },
            ParameterValue::Int(r) => Self {
                expr: &self.expr * &SymbolExpr::Value(Value::from(r)),
            },
            ParameterValue::Str(s) => Self {
                expr: &self.expr * &parse_expression(&s),
            },
            ParameterValue::Expr(e) => Self {
                expr: &self.expr * &e.expr,
            },
        }
    }
    pub fn __rmul__(&self, rhs: ParameterValue) -> Self {
        match rhs {
            ParameterValue::Real(r) => Self {
                expr: &SymbolExpr::Value(Value::from(r)) * &self.expr,
            },
            ParameterValue::Complex(c) => Self {
                expr: &SymbolExpr::Value(Value::from(c)) * &self.expr,
            },
            ParameterValue::Int(r) => Self {
                expr: &SymbolExpr::Value(Value::from(r)) * &self.expr,
            },
            ParameterValue::Str(s) => Self {
                expr: &parse_expression(&s) * &self.expr,
            },
            ParameterValue::Expr(e) => Self {
                expr: &e.expr * &self.expr,
            },
        }
    }
    pub fn __truediv__(&self, rhs: ParameterValue) -> PyResult<Self> {
        match rhs {
            ParameterValue::Real(r) => {
                if r < f64::EPSILON && r > -f64::EPSILON {
                    Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                        "Division by zero",
                    ))
                } else {
                    Ok(Self {
                        expr: &self.expr / &SymbolExpr::Value(Value::from(r)),
                    })
                }
            }
            ParameterValue::Complex(c) => {
                let t = (c.re * c.re + c.im * c.im).sqrt();
                if t < f64::EPSILON && t > -f64::EPSILON {
                    Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                        "Division by zero",
                    ))
                } else {
                    Ok(Self {
                        expr: &self.expr / &SymbolExpr::Value(Value::from(c)),
                    })
                }
            }
            ParameterValue::Int(r) => {
                if r == 0 {
                    Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                        "Division by zero",
                    ))
                } else {
                    Ok(Self {
                        expr: &self.expr / &SymbolExpr::Value(Value::from(r)),
                    })
                }
            }
            ParameterValue::Str(s) => {
                let r = parse_expression(&s);
                if r == 0.0 {
                    Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                        "Division by zero",
                    ))
                } else {
                    Ok(Self {
                        expr: &self.expr / &r,
                    })
                }
            }
            ParameterValue::Expr(e) => {
                if e.expr == 0.0 {
                    Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                        "Division by zero",
                    ))
                } else {
                    Ok(Self {
                        expr: &self.expr / &e.expr,
                    })
                }
            }
        }
    }
    pub fn __rtruediv__(&self, rhs: ParameterValue) -> PyResult<Self> {
        if self.expr == 0.0 {
            return Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                "Division by zero",
            ));
        }
        match rhs {
            ParameterValue::Real(r) => Ok(Self {
                expr: &SymbolExpr::Value(Value::from(r)) / &self.expr,
            }),
            ParameterValue::Complex(c) => Ok(Self {
                expr: &SymbolExpr::Value(Value::from(c)) / &self.expr,
            }),
            ParameterValue::Int(r) => Ok(Self {
                expr: &SymbolExpr::Value(Value::from(r)) / &self.expr,
            }),
            ParameterValue::Str(s) => Ok(Self {
                expr: &parse_expression(&s) / &self.expr,
            }),
            ParameterValue::Expr(e) => Ok(Self {
                expr: &e.expr / &self.expr,
            }),
        }
    }
    pub fn __pow__(&self, rhs: ParameterValue, _modulo: Option<i32>) -> Self {
        self.pow(rhs)
    }

    pub fn __rpow__(&self, rhs: ParameterValue, _modulo: Option<i32>) -> Self {
        match rhs {
            ParameterValue::Real(r) => Self {
                expr: SymbolExpr::Value(Value::from(r)).pow(&self.expr),
            },
            ParameterValue::Complex(c) => Self {
                expr: SymbolExpr::Value(Value::from(c)).pow(&self.expr),
            },
            ParameterValue::Int(r) => Self {
                expr: SymbolExpr::Value(Value::from(r)).pow(&self.expr),
            },
            ParameterValue::Str(s) => Self {
                expr: parse_expression(&s).pow(&self.expr),
            },
            ParameterValue::Expr(e) => Self {
                expr: e.expr.pow(&self.expr),
            },
        }
    }
    pub fn __str__(&self) -> String {
        self.expr.to_string()
    }

    pub fn __float__(&self) -> PyResult<f64> {
        self.float()
    }
    pub fn __complex__(&self) -> PyResult<Complex64> {
        self.complex()
    }
    pub fn __int__(&self) -> PyResult<i64> {
        self.int()
    }

    pub fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.expr.to_string().hash(&mut hasher);
        hasher.finish()
    }

    // for pickle, we can reproduce equation from expression string
    fn __getstate__(&self) -> String {
        self.expr.to_string()
    }
    fn __setstate__(&mut self, state: String) {
        self.expr = parse_expression(&state);
    }
}

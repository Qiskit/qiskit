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

// symbol_expr_py.rs
// Python interface of symbolic expression
use crate::symbol_expr::{SymbolExpr, Value, SYMEXPR_EPSILON};
use crate::symbol_parser::parse_expression;

use hashbrown::{HashMap, HashSet};
use num_complex::Complex64;

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use pyo3::prelude::*;
use pyo3::IntoPyObjectExt;

// Python interface to SymbolExpr
#[pyclass(sequence, module = "qiskit._accelerate.circuit")]
#[derive(Clone, Debug)]
pub struct PySymbolExpr {
    pub expr: SymbolExpr,
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
                expr: SymbolExpr::Value(Value::Int(0)),
            }),
        }
    }

    /// create new expression as a symbol
    #[allow(non_snake_case)]
    #[staticmethod]
    pub fn Symbol(name: String) -> Self {
        // check if expr contains replacements for sympy
        let name = name
            .replace("__begin_sympy_replace__", "$\\")
            .replace("__end_sympy_replace__", "$");

        PySymbolExpr {
            expr: SymbolExpr::Symbol(Box::new(name)),
        }
    }

    /// create new expression as a value
    #[allow(non_snake_case)]
    #[staticmethod]
    pub fn Value(py: Python, value: PyObject) -> PyResult<Self> {
        match Self::_extract_value(py, value) {
            Some(v) => Ok(v),
            None => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Unsupported data type to initialize SymbolExpr as a value",
            )),
        }
    }

    #[inline]
    #[staticmethod]
    fn _extract_value(py: Python, value: PyObject) -> Option<Self> {
        if let Ok(i) = value.extract::<i64>(py) {
            Some(PySymbolExpr {
                expr: SymbolExpr::Value(Value::from(i)),
            })
        } else if let Ok(r) = value.extract::<f64>(py) {
            Some(PySymbolExpr {
                expr: SymbolExpr::Value(Value::from(r)),
            })
        } else if let Ok(c) = value.extract::<Complex64>(py) {
            Some(PySymbolExpr {
                expr: SymbolExpr::Value(Value::from(c)),
            })
        } else if let Ok(s) = value.extract::<String>(py) {
            Some(PySymbolExpr {
                expr: parse_expression(&s),
            })
        } else if let Ok(e) = value.extract::<PySymbolExpr>(py) {
            Some(PySymbolExpr {
                expr: e.expr.clone(),
            })
        } else {
            None
        }
    }

    /// create new expression from string
    #[allow(non_snake_case)]
    #[staticmethod]
    pub fn Expression(expr: String) -> Self {
        // check if expr contains replacements for sympy
        let expr = expr
            .replace("__begin_sympy_replace__", "$\\")
            .replace("__end_sympy_replace__", "$");
        PySymbolExpr {
            expr: parse_expression(&expr),
        }
    }

    // return string to pass to sympify
    pub fn expr_for_sympy(&self) -> String {
        let ret = self.expr.optimize().to_string();
        ret.replace("$\\", "__begin_sympy_replace__")
            .replace('$', "__end_sympy_replace__")
    }

    // unary functions
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

    /// return value if expression does not contain any symbols
    pub fn value(&self, py: Python) -> PyResult<PyObject> {
        match self.expr.eval(true) {
            Some(v) => match v {
                Value::Real(r) => r.into_py_any(py),
                Value::Int(i) => i.into_py_any(py),
                Value::Complex(c) => {
                    if (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&c.im) {
                        c.re.into_py_any(py)
                    } else {
                        c.into_py_any(py)
                    }
                }
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

    /// get hashset of all the symbols used in this expression
    pub fn symbols(&self) -> HashSet<String> {
        self.expr.symbols()
    }

    /// return all values in this equation
    pub fn values(&self, py: Python) -> PyResult<Vec<PyObject>> {
        let ret: Vec<PyObject> = self
            .expr
            .values()
            .iter()
            .map(|val| match val {
                Value::Real(r) => r.into_py_any(py).unwrap(),
                Value::Int(i) => i.into_py_any(py).unwrap(),
                Value::Complex(c) => c.into_py_any(py).unwrap(),
            })
            .collect();
        Ok(ret)
    }

    /// return expression as a string
    #[getter]
    pub fn name(&self) -> String {
        self.expr.optimize().to_string()
    }

    /// bind values to symbols given by input hashmap
    pub fn bind(&self, py: Python, map: HashMap<String, PyObject>) -> PyResult<Self> {
        let map: HashMap<String, Value> = map
            .into_iter()
            .map(|(key, val)| {
                if let Ok(i) = val.extract::<i64>(py) {
                    (key, Value::from(i))
                } else if let Ok(r) = val.extract::<f64>(py) {
                    (key, Value::from(r))
                } else if let Ok(c) = val.extract::<Complex64>(py) {
                    (key, Value::from(c))
                } else {
                    // if unsupported data type, insert empty
                    ("".to_string(), Value::Int(0))
                }
            })
            .collect();
        let bound = self.expr.bind(&map);
        match bound.eval(true) {
            Some(v) => match &v {
                Value::Real(r) => {
                    if r.is_infinite() {
                        Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                            "zero division occurs while binding parameter",
                        ))
                    } else if r.is_nan() {
                        Err(pyo3::exceptions::PyRuntimeError::new_err(
                            "NAN detected while binding parameter",
                        ))
                    } else {
                        Ok(Self {
                            expr: SymbolExpr::Value(v),
                        })
                    }
                }
                Value::Int(_) => Ok(Self {
                    expr: SymbolExpr::Value(v),
                }),
                Value::Complex(c) => {
                    if c.re.is_infinite() || c.im.is_infinite() {
                        Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                            "zero division occurs while binding parameter",
                        ))
                    } else if c.re.is_nan() || c.im.is_nan() {
                        Err(pyo3::exceptions::PyRuntimeError::new_err(
                            "NAN detected while binding parameter",
                        ))
                    } else if (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&c.im) {
                        Ok(Self {
                            expr: SymbolExpr::Value(Value::Real(c.re)),
                        })
                    } else {
                        Ok(Self {
                            expr: SymbolExpr::Value(v),
                        })
                    }
                }
            },
            None => Ok(Self { expr: bound }),
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
    pub fn __eq__(&self, py: Python, rhs: PyObject) -> bool {
        match Self::_extract_value(py, rhs) {
            Some(rhs) => match rhs.expr {
                SymbolExpr::Value(v) => match self.expr.eval(true) {
                    Some(e) => e == v,
                    None => false,
                },
                _ => self.expr == rhs.expr,
            },
            None => false,
        }
    }
    pub fn __ne__(&self, py: Python, rhs: PyObject) -> bool {
        match Self::_extract_value(py, rhs) {
            Some(rhs) => match rhs.expr {
                SymbolExpr::Value(v) => match self.expr.eval(true) {
                    Some(e) => e != v,
                    None => true,
                },
                _ => self.expr != rhs.expr,
            },
            None => true,
        }
    }
    pub fn __neg__(&self) -> Self {
        Self { expr: -&self.expr }
    }
    pub fn __add__(&self, py: Python, rhs: PyObject) -> PyResult<Self> {
        match Self::_extract_value(py, rhs) {
            Some(rhs) => Ok(Self {
                expr: &self.expr + &rhs.expr,
            }),
            None => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Unsupported data type for __add__",
            )),
        }
    }
    pub fn __radd__(&self, py: Python, lhs: PyObject) -> PyResult<Self> {
        match Self::_extract_value(py, lhs) {
            Some(lhs) => Ok(Self {
                expr: &lhs.expr + &self.expr,
            }),
            None => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Unsupported data type for __radd__",
            )),
        }
    }
    pub fn __sub__(&self, py: Python, rhs: PyObject) -> PyResult<Self> {
        match Self::_extract_value(py, rhs) {
            Some(rhs) => Ok(Self {
                expr: &self.expr - &rhs.expr,
            }),
            None => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Unsupported data type for __sub__",
            )),
        }
    }
    pub fn __rsub__(&self, py: Python, lhs: PyObject) -> PyResult<Self> {
        match Self::_extract_value(py, lhs) {
            Some(lhs) => Ok(Self {
                expr: &lhs.expr - &self.expr,
            }),
            None => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Unsupported data type for __rsub__",
            )),
        }
    }
    pub fn __mul__(&self, py: Python, rhs: PyObject) -> PyResult<Self> {
        match Self::_extract_value(py, rhs) {
            Some(rhs) => Ok(Self {
                expr: &self.expr * &rhs.expr,
            }),
            None => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Unsupported data type for __mul__",
            )),
        }
    }
    pub fn __rmul__(&self, py: Python, lhs: PyObject) -> PyResult<Self> {
        match Self::_extract_value(py, lhs) {
            Some(lhs) => Ok(Self {
                expr: &lhs.expr * &self.expr,
            }),
            None => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Unsupported data type for __rmul__",
            )),
        }
    }

    pub fn __truediv__(&self, py: Python, rhs: PyObject) -> PyResult<Self> {
        match Self::_extract_value(py, rhs) {
            Some(rhs) => Ok(Self {
                expr: &self.expr / &rhs.expr,
            }),
            None => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Unsupported data type for __truediv__",
            )),
        }
    }
    pub fn __rtruediv__(&self, py: Python, lhs: PyObject) -> PyResult<Self> {
        match Self::_extract_value(py, lhs) {
            Some(lhs) => Ok(Self {
                expr: &lhs.expr / &self.expr,
            }),
            None => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Unsupported data type for __truediv__",
            )),
        }
    }
    pub fn __pow__(&self, py: Python, rhs: PyObject, _modulo: Option<i32>) -> PyResult<Self> {
        match Self::_extract_value(py, rhs) {
            Some(rhs) => Ok(Self {
                expr: self.expr.pow(&rhs.expr),
            }),
            None => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Unsupported data type for __pow__",
            )),
        }
    }
    pub fn __rpow__(&self, py: Python, lhs: PyObject, _modulo: Option<i32>) -> PyResult<Self> {
        match Self::_extract_value(py, lhs) {
            Some(lhs) => Ok(Self {
                expr: lhs.expr.pow(&self.expr),
            }),
            None => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Unsupported data type for __rpow__",
            )),
        }
    }

    pub fn __str__(&self) -> String {
        if let SymbolExpr::Symbol(s) = &self.expr {
            return s.as_ref().clone();
        }
        match self.expr.eval(true) {
            Some(e) => e.to_string(),
            None => self.expr.optimize().to_string(),
        }
    }

    pub fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.expr.optimize().to_string().hash(&mut hasher);
        hasher.finish()
    }

    // for pickle, we can reproduce equation from expression string
    fn __getstate__(&self) -> String {
        self.expr.optimize().to_string()
    }
    fn __setstate__(&mut self, state: String) {
        self.expr = parse_expression(&state);
    }
}

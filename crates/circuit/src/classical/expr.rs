// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::bit::{ClassicalRegister, ShareableClbit};
use crate::classical::Type;
use crate::duration::Duration;
use pyo3::prelude::*;
use pyo3::IntoPyObjectExt;

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Unary(Box<Unary>),
    Binary(Box<Binary>),
    Cast(Box<Cast>),
    Value(Value),
    Var(Var),
    Stretch(Stretch),
    Index(Box<Index>),
}

impl From<Unary> for Expr {
    fn from(value: Unary) -> Self {
        Expr::Unary(Box::new(value))
    }
}

impl From<Binary> for Expr {
    fn from(value: Binary) -> Self {
        Expr::Binary(Box::new(value))
    }
}

impl From<Cast> for Expr {
    fn from(value: Cast) -> Self {
        Expr::Cast(Box::new(value))
    }
}

impl From<Value> for Expr {
    fn from(value: Value) -> Self {
        Expr::Value(value)
    }
}

impl From<Var> for Expr {
    fn from(value: Var) -> Self {
        Expr::Var(value)
    }
}

impl From<Stretch> for Expr {
    fn from(value: Stretch) -> Self {
        Expr::Stretch(value)
    }
}

impl From<Index> for Expr {
    fn from(value: Index) -> Self {
        Expr::Index(Box::new(value))
    }
}

/// The base class of all Python expressions.
///
/// We store a [ExprKind] enum inside to quickly determine which Python expression
/// we're looking at during extraction.
#[pyclass(
    eq,
    hash,
    subclass,
    frozen,
    name = "Expr",
    module = "qiskit._accelerate.circuit"
)]
#[derive(PartialEq, Clone, Copy, Debug, Hash)]
struct PyExpr(ExprKind);

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum ExprKind {
    Unary,
    Binary,
    Value,
    Var,
    Cast,
    Stretch,
    Index,
}

impl<'py> IntoPyObject<'py> for Expr {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            Expr::Unary(u) => {
                Ok(Bound::new(py, (PyUnary(*u), PyExpr(ExprKind::Unary)))?.into_any())
            }
            Expr::Binary(b) => {
                Ok(Bound::new(py, (PyBinary(*b), PyExpr(ExprKind::Binary)))?.into_any())
            }
            Expr::Cast(c) => Ok(Bound::new(py, (PyCast(*c), PyExpr(ExprKind::Cast)))?.into_any()),
            Expr::Value(v) => Ok(Bound::new(py, (PyValue(v), PyExpr(ExprKind::Value)))?.into_any()),
            Expr::Var(v) => Ok(Bound::new(py, (PyVar(v), PyExpr(ExprKind::Var)))?.into_any()),
            Expr::Stretch(s) => {
                Ok(Bound::new(py, (PyStretch(s), PyExpr(ExprKind::Stretch)))?.into_any())
            }
            Expr::Index(i) => {
                Ok(Bound::new(py, (PyIndex(*i), PyExpr(ExprKind::Index)))?.into_any())
            }
        }
    }
}

impl<'py> FromPyObject<'py> for Expr {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let PyExpr(kind) = ob.extract()?;
        match kind {
            ExprKind::Unary => {
                let PyUnary(u) = ob.extract()?;
                Ok(u.into())
            }
            ExprKind::Binary => {
                let PyBinary(b) = ob.extract()?;
                Ok(b.into())
            }
            ExprKind::Value => {
                let PyValue(v) = ob.extract()?;
                Ok(v.into())
            }
            ExprKind::Var => {
                let PyVar(v) = ob.extract()?;
                Ok(v.into())
            }
            ExprKind::Cast => {
                let PyCast(c) = ob.extract()?;
                Ok(c.into())
            }
            ExprKind::Stretch => {
                let PyStretch(s) = ob.extract()?;
                Ok(s.into())
            }
            ExprKind::Index => {
                let PyIndex(i) = ob.extract()?;
                Ok(i.into())
            }
        }
    }
}

impl Expr {
    fn is_const(&self) -> bool {
        match self {
            Expr::Unary(u) => u.constant,
            Expr::Binary(b) => b.constant,
            Expr::Cast(c) => c.constant,
            Expr::Value(_) => true,
            Expr::Var(_) => false,
            Expr::Stretch(_) => true,
            Expr::Index(i) => i.constant,
        }
    }

    fn ty(&self) -> Type {
        match self {
            Expr::Unary(u) => u.ty,
            Expr::Binary(b) => b.ty,
            Expr::Cast(c) => c.ty,
            Expr::Value(v) => match v {
                Value::Duration(_) => Type::Duration,
                Value::Float { ty, .. } => *ty,
                Value::Uint { ty, .. } => *ty,
            },
            Expr::Var(v) => match v {
                Var::Standalone { ty, .. } => *ty,
                Var::Bit { .. } => Type::Bool,
                Var::Register { ty, .. } => *ty,
            },
            Expr::Stretch(_) => Type::Duration,
            Expr::Index(i) => i.ty,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Unary {
    op: UnaryOp,
    operand: Expr,
    ty: Type,
    constant: bool,
}

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum UnaryOp {
    BitNot = 1,
    LogicNot = 2,
}

#[pyclass(eq, extends = PyExpr, name = "Unary", module = "qiskit._accelerate.circuit")]
#[derive(PartialEq, Clone, Debug)]
struct PyUnary(Unary);

#[pymethods]
impl PyUnary {
    #[getter]
    fn get_type(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.0.ty.clone().into_py_any(py)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Binary {
    op: BinaryOp,
    left: Expr,
    right: Expr,
    ty: Type,
    constant: bool,
}

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    BitAnd = 1,
    BitOr = 2,
    BitXor = 3,
    LogicAnd = 4,
    LogicOr = 5,
    Equal = 6,
    NotEqual = 7,
    Less = 8,
    LessEqual = 9,
    Greater = 10,
    GreaterEqual = 11,
    ShiftLeft = 12,
    ShiftRight = 13,
    Add = 14,
    Sub = 15,
    Mul = 16,
    Div = 17,
}

#[pyclass(eq, extends = PyExpr, name = "Binary", module = "qiskit._accelerate.circuit")]
#[derive(PartialEq, Clone, Debug)]
struct PyBinary(Binary);

#[pymethods]
impl PyBinary {
    #[getter]
    fn get_type(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.0.ty.clone().into_py_any(py)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Cast {
    operand: Expr,
    ty: Type,
    constant: bool,
    implicit: bool,
}

#[pyclass(eq, extends = PyExpr, name = "Cast", module = "qiskit._accelerate.circuit")]
#[derive(PartialEq, Clone, Debug)]
struct PyCast(Cast);

#[pymethods]
impl PyCast {
    #[getter]
    fn get_type(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.0.ty.clone().into_py_any(py)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Duration(Duration),
    Float { raw: f64, ty: Type },
    Uint { raw: u64, ty: Type },
}

#[pyclass(eq, extends = PyExpr, name = "Value", module = "qiskit._accelerate.circuit")]
#[derive(PartialEq, Clone, Debug)]
struct PyValue(Value);

#[pymethods]
impl PyValue {
    #[getter]
    fn get_type(&self, py: Python) -> PyResult<Py<PyAny>> {
        match self.0 {
            Value::Duration(_) => Type::Duration.into_py_any(py),
            Value::Float { ty, .. } | Value::Uint { ty, .. } => ty.into_py_any(py),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Hash)]
pub enum Var {
    Standalone {
        uuid: u128,
        name: String,
        ty: Type,
    },
    Bit {
        bit: ShareableClbit,
    },
    Register {
        register: ClassicalRegister,
        ty: Type,
    },
}

#[pymethods]
impl PyVar {
    #[getter]
    fn get_type(&self, py: Python) -> PyResult<Py<PyAny>> {
        match self.0 {
            Var::Bit { .. } => Type::Bool.into_py_any(py),
            Var::Standalone { ty, .. } | Var::Register { ty, .. } => ty.into_py_any(py),
        }
    }
}

#[pyclass(eq, hash, frozen, extends = PyExpr, name = "Var", module = "qiskit._accelerate.circuit")]
#[derive(PartialEq, Clone, Debug, Hash)]
struct PyVar(Var);

#[derive(Clone, Debug, PartialEq, Hash)]
pub struct Stretch {
    uuid: u128,
    name: String,
}

#[pyclass(eq, hash, frozen, extends = PyExpr, name = "Stretch", module = "qiskit._accelerate.circuit")]
#[derive(PartialEq, Clone, Debug, Hash)]
struct PyStretch(Stretch);

#[pymethods]
impl PyStretch {
    #[getter]
    fn get_type(&self, py: Python) -> PyResult<Py<PyAny>> {
        Type::Duration.into_py_any(py)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Index {
    target: Expr,
    index: Expr,
    ty: Type,
    constant: bool,
}

#[pyclass(eq, extends = PyExpr, name = "Index", module = "qiskit._accelerate.circuit")]
#[derive(PartialEq, Clone, Debug)]
struct PyIndex(Index);

#[pymethods]
impl PyIndex {
    #[getter]
    fn get_type(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.0.ty.clone().into_py_any(py)
    }
}

pub(crate) fn register_python(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyExpr>()?;
    m.add_class::<PyUnary>()?;
    m.add_class::<PyBinary>()?;
    m.add_class::<PyCast>()?;
    m.add_class::<PyValue>()?;
    m.add_class::<PyVar>()?;
    m.add_class::<PyStretch>()?;
    m.add_class::<PyIndex>()?;
    Ok(())
}

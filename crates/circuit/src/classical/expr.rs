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
use crate::classical::types::Type;
use crate::duration::Duration;
use crate::imports;
use crate::imports::UUID;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyTuple};
use pyo3::{intern, BoundObject, IntoPyObjectExt, PyTypeInfo};
use uuid::Uuid;

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

    pub fn vars(&self) -> impl Iterator<Item = &Var> {
        VarIterator(ExprIterator { stack: vec![self] })
    }
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

/// Root base class of all nodes in the expression tree.  The base case should never be
/// instantiated directly.
///
/// This must not be subclassed by users; subclasses form the internal data of the representation of
/// expressions, and it does not make sense to add more outside of Qiskit library code.
///
/// All subclasses are responsible for setting their ``type`` attribute in their ``__init__``, and
/// should not call the parent initializer."""
#[pyclass(
    eq,
    hash,
    subclass,
    frozen,
    name = "Expr",
    module = "qiskit._accelerate.circuit"
)]
#[derive(PartialEq, Clone, Copy, Debug, Hash)]
struct PyExpr(ExprKind); // ExprKind is used for fast extraction from Python

#[pymethods]
impl PyExpr {
    /// Call the relevant ``visit_*`` method on the given :class:`ExprVisitor`.  The usual entry
    /// point for a simple visitor is to construct it, and then call :meth:`accept` on the root
    /// object to be visited.  For example::
    ///
    ///     expr = ...
    ///     visitor = MyVisitor()
    ///     visitor.accept(expr)
    ///
    /// Subclasses of :class:`Expr` should override this to call the correct virtual method on the
    /// visitor.  This implements double dispatch with the visitor."""
    /// return visitor.visit_generic(self)
    fn accept<'py>(
        slf: PyRef<'py, Self>,
        visitor: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        visitor.call_method1(intern!(visitor.py(), "visit_generic"), (slf,))
    }
}

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

impl<'py> IntoPyObject<'py> for Unary {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(Bound::new(py, (PyUnary(self), PyExpr(ExprKind::Unary)))?.into_any())
    }
}

impl<'py> FromPyObject<'py> for Unary {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let PyUnary(u) = ob.extract()?;
        Ok(u.into())
    }
}

impl<'py> IntoPyObject<'py> for Binary {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(Bound::new(py, (PyBinary(self), PyExpr(ExprKind::Binary)))?.into_any())
    }
}

impl<'py> FromPyObject<'py> for Binary {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let PyBinary(b) = ob.extract()?;
        Ok(b.into())
    }
}

impl<'py> IntoPyObject<'py> for Cast {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(Bound::new(py, (PyCast(self), PyExpr(ExprKind::Cast)))?.into_any())
    }
}

impl<'py> FromPyObject<'py> for Cast {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let PyCast(c) = ob.extract()?;
        Ok(c.into())
    }
}

impl<'py> IntoPyObject<'py> for Value {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(Bound::new(py, (PyValue(self), PyExpr(ExprKind::Value)))?.into_any())
    }
}

impl<'py> FromPyObject<'py> for Value {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let PyValue(v) = ob.extract()?;
        Ok(v.into())
    }
}

impl<'py> IntoPyObject<'py> for Var {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(Bound::new(py, (PyVar(self), PyExpr(ExprKind::Var)))?.into_any())
    }
}

impl<'py> FromPyObject<'py> for Var {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let PyVar(v) = ob.extract()?;
        Ok(v.into())
    }
}

impl<'py> IntoPyObject<'py> for Stretch {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(Bound::new(py, (PyStretch(self), PyExpr(ExprKind::Stretch)))?.into_any())
    }
}

impl<'py> FromPyObject<'py> for Stretch {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let PyStretch(s) = ob.extract()?;
        Ok(s.into())
    }
}

impl<'py> IntoPyObject<'py> for Index {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(Bound::new(py, (PyIndex(self), PyExpr(ExprKind::Index)))?.into_any())
    }
}

impl<'py> FromPyObject<'py> for Index {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let PyIndex(i) = ob.extract()?;
        Ok(i.into())
    }
}

impl<'py> IntoPyObject<'py> for Expr {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            Expr::Unary(u) => u.into_bound_py_any(py),
            Expr::Binary(b) => b.into_bound_py_any(py),
            Expr::Cast(c) => c.into_bound_py_any(py),
            Expr::Value(v) => v.into_bound_py_any(py),
            Expr::Var(v) => v.into_bound_py_any(py),
            Expr::Stretch(s) => s.into_bound_py_any(py),
            Expr::Index(i) => i.into_bound_py_any(py),
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

#[derive(Clone, Debug, PartialEq)]
pub struct Unary {
    pub op: UnaryOp,
    pub operand: Expr,
    pub ty: Type,
    pub constant: bool,
}

// WARNING: these must EXACTLY match _UnaryOp from expr.py!
#[repr(u8)]
#[derive(Copy, Hash, Clone, Debug, PartialEq)]
pub enum UnaryOp {
    BitNot = 1,
    LogicNot = 2,
}

unsafe impl ::bytemuck::CheckedBitPattern for UnaryOp {
    type Bits = u8;

    fn is_valid_bit_pattern(bits: &Self::Bits) -> bool {
        *bits > 0 && *bits < 3
    }
}

impl<'py> IntoPyObject<'py> for UnaryOp {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        imports::UNARY_OP.get_bound(py).call1((self as usize,))
    }
}

impl<'py> FromPyObject<'py> for UnaryOp {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let value = ob.getattr(intern!(ob.py(), "value"))?;
        Ok(bytemuck::checked::cast(value.extract::<u8>()?))
    }
}

/// A Python descriptor to prevent PyO3 from attempting to import the Python-side
/// enum before we're initialized.
#[pyclass(module = "qiskit._accelerate.circuit.classical.expr")]
struct PyUnaryOp;

#[pymethods]
impl PyUnaryOp {
    fn __get__(&self, obj: &Bound<PyAny>, _obj_type: &Bound<PyAny>) -> Py<PyAny> {
        imports::UNARY_OP.get_bound(obj.py()).clone().unbind()
    }
}

#[pyclass(eq, extends = PyExpr, name = "Unary", module = "qiskit._accelerate.circuit.classical.expr")]
#[derive(PartialEq, Clone, Debug)]
struct PyUnary(Unary);

#[pymethods]
impl PyUnary {
    #[classattr]
    fn Op(py: Python) -> PyResult<Py<PyAny>> {
        PyUnaryOp.into_py_any(py)
    }

    #[new]
    #[pyo3(text_signature = "(op, operand, type)")]
    fn new(py: Python, op: UnaryOp, operand: Expr, ty: Type) -> PyResult<Py<Self>> {
        let constant = operand.is_const();
        Py::new(
            py,
            (
                PyUnary(Unary {
                    op,
                    operand,
                    ty,
                    constant,
                }),
                PyExpr(ExprKind::Unary),
            ),
        )
    }

    #[getter]
    fn get_op(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.0.op.into_py_any(py)
    }

    #[getter]
    fn get_operand(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.0.operand.clone().into_py_any(py)
    }

    #[getter]
    fn get_const(&self, py: Python) -> bool {
        self.0.constant
    }

    #[getter]
    fn get_type(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.0.ty.clone().into_py_any(py)
    }

    fn accept<'py>(
        slf: PyRef<'py, Self>,
        visitor: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        visitor.call_method1(intern!(visitor.py(), "visit_unary"), (slf,))
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        (
            py.get_type::<Self>(),
            (self.get_op(py)?, self.get_operand(py)?, self.get_type(py)?),
        )
            .into_pyobject(py)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Binary {
    pub op: BinaryOp,
    pub left: Expr,
    pub right: Expr,
    pub ty: Type,
    pub constant: bool,
}

// WARNING: these must EXACTLY match _BinaryOp from expr.py!
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

unsafe impl ::bytemuck::CheckedBitPattern for BinaryOp {
    type Bits = u8;

    fn is_valid_bit_pattern(bits: &Self::Bits) -> bool {
        *bits > 0 && *bits < 18
    }
}

impl<'py> IntoPyObject<'py> for BinaryOp {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        imports::BINARY_OP.get_bound(py).call1((self as usize,))
    }
}

impl<'py> FromPyObject<'py> for BinaryOp {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let value = ob.getattr(intern!(ob.py(), "value"))?;
        Ok(bytemuck::checked::cast(value.extract::<u8>()?))
    }
}

/// A Python descriptor to prevent PyO3 from attempting to import the Python-side
/// enum before we're initialized.
#[pyclass(module = "qiskit._accelerate.circuit.classical.expr")]
struct PyBinaryOp;

#[pymethods]
impl PyBinaryOp {
    fn __get__(&self, obj: &Bound<PyAny>, _obj_type: &Bound<PyAny>) -> Py<PyAny> {
        imports::BINARY_OP.get_bound(obj.py()).clone().unbind()
    }
}

#[pyclass(eq, extends = PyExpr, name = "Binary", module = "qiskit._accelerate.circuit.classical.expr")]
#[derive(PartialEq, Clone, Debug)]
struct PyBinary(Binary);

#[pymethods]
impl PyBinary {
    #[classattr]
    fn Op(py: Python) -> PyResult<Py<PyAny>> {
        PyBinaryOp.into_py_any(py)
    }

    #[new]
    #[pyo3(text_signature = "(op, left, right, type)")]
    fn new(py: Python, op: BinaryOp, left: Expr, right: Expr, ty: Type) -> PyResult<Py<Self>> {
        let constant = left.is_const() && right.is_const();
        Py::new(
            py,
            (
                PyBinary(Binary {
                    op,
                    left,
                    right,
                    ty,
                    constant,
                }),
                PyExpr(ExprKind::Binary),
            ),
        )
    }

    #[getter]
    fn get_op(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.0.op.into_py_any(py)
    }

    #[getter]
    fn get_left(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.0.left.clone().into_py_any(py)
    }

    #[getter]
    fn get_right(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.0.right.clone().into_py_any(py)
    }

    #[getter]
    fn get_const(&self, py: Python) -> bool {
        self.0.constant
    }

    #[getter]
    fn get_type(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.0.ty.clone().into_py_any(py)
    }

    fn accept<'py>(
        slf: PyRef<'py, Self>,
        visitor: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        visitor.call_method1(intern!(visitor.py(), "visit_binary"), (slf,))
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        (
            py.get_type::<Self>(),
            (
                self.get_op(py)?,
                self.get_left(py)?,
                self.get_right(py)?,
                self.get_type(py)?,
            ),
        )
            .into_pyobject(py)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Cast {
    pub operand: Expr,
    pub ty: Type,
    pub constant: bool,
    pub implicit: bool,
}

/// A cast from one type to another, implied by the use of an expression in a different
/// context.
#[pyclass(eq, extends = PyExpr, name = "Cast", module = "qiskit._accelerate.circuit.classical.expr")]
#[derive(PartialEq, Clone, Debug)]
struct PyCast(Cast);

#[pymethods]
impl PyCast {
    #[new]
    #[pyo3(signature=(operand, ty, implicit=false), text_signature="(operand, type, implicit=False)")]
    fn new(py: Python, operand: Expr, ty: Type, implicit: bool) -> PyResult<Py<Self>> {
        let constant = operand.is_const();
        Py::new(
            py,
            (
                PyCast(Cast {
                    operand,
                    ty,
                    constant,
                    implicit,
                }),
                PyExpr(ExprKind::Cast),
            ),
        )
    }

    #[getter]
    fn get_operand(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.0.operand.clone().into_py_any(py)
    }

    #[getter]
    fn get_implicit(&self, py: Python) -> bool {
        self.0.implicit
    }

    #[getter]
    fn get_const(&self, py: Python) -> bool {
        self.0.constant
    }

    #[getter]
    fn get_type(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.0.ty.clone().into_py_any(py)
    }

    fn accept<'py>(
        slf: PyRef<'py, Self>,
        visitor: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        visitor.call_method1(intern!(visitor.py(), "visit_cast"), (slf,))
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        (
            py.get_type::<Self>(),
            (
                self.get_operand(py)?,
                self.get_type(py)?,
                self.get_implicit(py),
            ),
        )
            .into_pyobject(py)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Duration(Duration),
    Float { raw: f64, ty: Type },
    Uint { raw: u64, ty: Type },
}

#[pyclass(eq, extends = PyExpr, name = "Value", module = "qiskit._accelerate.circuit.classical.expr")]
#[derive(PartialEq, Clone, Debug)]
struct PyValue(Value);

#[pymethods]
impl PyValue {
    #[new]
    #[pyo3(text_signature = "(value, type)")]
    fn new(py: Python, value: Bound<PyAny>, ty: Type) -> PyResult<Py<Self>> {
        let value = if let Ok(raw) = value.extract::<u64>() {
            Value::Uint { raw, ty }
        } else if let Ok(raw) = value.extract::<f64>() {
            Value::Float { raw, ty }
        } else {
            Value::Duration(value.extract()?)
        };
        Py::new(py, (PyValue(value), PyExpr(ExprKind::Value)))
    }

    #[getter]
    fn get_value(&self, py: Python) -> PyResult<Py<PyAny>> {
        match self.0 {
            Value::Duration(d) => d.into_py_any(py),
            Value::Float { raw, .. } => raw.into_py_any(py),
            Value::Uint { raw, .. } => raw.into_py_any(py),
        }
    }

    #[getter]
    fn get_const(&self, py: Python) -> bool {
        true
    }

    #[getter]
    fn get_type(&self, py: Python) -> PyResult<Py<PyAny>> {
        match self.0 {
            Value::Duration(_) => Type::Duration.into_py_any(py),
            Value::Float { ty, .. } | Value::Uint { ty, .. } => ty.into_py_any(py),
        }
    }

    fn accept<'py>(
        slf: PyRef<'py, Self>,
        visitor: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        visitor.call_method1(intern!(visitor.py(), "visit_value"), (slf,))
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        (
            py.get_type::<Self>(),
            (self.get_value(py)?, self.get_type(py)?),
        )
            .into_pyobject(py)
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

struct ExprIterator<'a> {
    stack: Vec<&'a Expr>,
}

impl<'a> Iterator for ExprIterator<'a> {
    type Item = &'a Expr;

    fn next(&mut self) -> Option<Self::Item> {
        let Some(expr) = self.stack.pop() else {
            return None;
        };

        match expr {
            Expr::Unary(u) => {
                self.stack.push(&u.operand);
            }
            Expr::Binary(b) => {
                self.stack.push(&b.left);
                self.stack.push(&b.right);
            }
            Expr::Cast(c) => self.stack.push(&c.operand),
            Expr::Value(_) => {}
            Expr::Var(_) => {}
            Expr::Stretch(_) => {}
            Expr::Index(i) => {
                self.stack.push(&i.index);
                self.stack.push(&i.target);
            }
        }
        Some(expr)
    }
}

struct VarIterator<'a>(ExprIterator<'a>);

impl<'a> Iterator for VarIterator<'a> {
    type Item = &'a Var;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(expr) = self.0.next() {
            if let Expr::Var(v) = expr {
                return Some(v);
            }
        }
        None
    }
}

/// A classical variable.
///
/// These variables take two forms: a new-style variable that owns its storage location and has an
/// associated name; and an old-style variable that wraps a :class:`.Clbit` or
/// :class:`.ClassicalRegister` instance that is owned by some containing circuit.  In general,
/// construction of variables for use in programs should use :meth:`Var.new` or
/// :meth:`.QuantumCircuit.add_var`.
///
/// Variables are immutable after construction, so they can be used as dictionary keys."""
#[pyclass(eq, hash, frozen, extends = PyExpr, name = "Var", module = "qiskit._accelerate.circuit.classical.expr")]
#[derive(PartialEq, Clone, Debug, Hash)]
struct PyVar(Var);

#[pymethods]
impl PyVar {
    #[new]
    #[pyo3(signature = (var, ty, *, name = None), text_signature="(var, type, *, name=None)")]
    fn new0(var: &Bound<PyAny>, ty: Type, name: Option<String>) -> PyResult<Py<Self>> {
        let v = if let Some(name) = name {
            Var::Standalone {
                uuid: var.getattr(intern!(var.py(), "int"))?.extract()?,
                name,
                ty,
            }
        } else if let Ok(register) = var.extract::<ClassicalRegister>() {
            Var::Register { register, ty }
        } else {
            Var::Bit {
                bit: var.extract()?,
            }
        };
        Py::new(var.py(), (PyVar(v), PyExpr(ExprKind::Var)))
    }

    /// Generate a new named variable that owns its own backing storage.
    #[classmethod]
    fn new(cls: &Bound<'_, pyo3::types::PyType>, name: String, ty: Type) -> PyResult<Py<Self>> {
        Py::new(
            cls.py(),
            (
                PyVar(Var::Standalone {
                    uuid: Uuid::new_v4().as_u128(),
                    name,
                    ty,
                }),
                PyExpr(ExprKind::Var),
            ),
        )
    }

    #[getter]
    fn get_const(&self) -> bool {
        false
    }

    #[getter]
    fn get_type(&self, py: Python) -> PyResult<Py<PyAny>> {
        match self.0 {
            Var::Bit { .. } => Type::Bool.into_py_any(py),
            Var::Standalone { ty, .. } | Var::Register { ty, .. } => ty.into_py_any(py),
        }
    }

    /// A reference to the backing data storage of the :class:`Var` instance.  When lifting
    /// old-style :class:`.Clbit` or :class:`.ClassicalRegister` instances into a :class:`Var`,
    /// this is exactly the :class:`.Clbit` or :class:`.ClassicalRegister`.  If the variable is a
    /// new-style classical variable (one that owns its own storage separate to the old
    /// :class:`.Clbit`/:class:`.ClassicalRegister` model), this field will be a :class:`~uuid.UUID`
    /// to uniquely identify it."""
    #[getter]
    fn get_var(&self, py: Python) -> PyResult<Py<PyAny>> {
        match &self.0 {
            Var::Standalone { uuid, .. } => {
                let kwargs = [("int", uuid)].into_py_dict(py)?;
                Ok(UUID.get_bound(py).call((), Some(&kwargs))?.unbind())
            }
            Var::Bit { bit } => bit.into_py_any(py),
            Var::Register { register, .. } => register.into_py_any(py),
        }
    }

    /// The name of the variable.  This is required to exist if the backing :attr:`var` attribute
    /// is a :class:`~uuid.UUID`, i.e. if it is a new-style variable, and must be ``None`` if it is
    /// an old-style variable."""
    #[getter]
    fn get_name(&self, py: Python) -> PyResult<Py<PyAny>> {
        match &self.0 {
            Var::Standalone { name, .. } => Some(name),
            Var::Bit { .. } | Var::Register { .. } => None,
        }
        .into_py_any(py)
    }

    /// Whether this :class:`Var` is a standalone variable that owns its storage
    /// location, if applicable. If false, this is a wrapper :class:`Var` around a
    /// pre-existing circuit object."""
    #[getter]
    fn get_standalone(&self) -> bool {
        match self.0 {
            Var::Standalone { .. } => true,
            Var::Bit { .. } | Var::Register { .. } => false,
        }
    }

    fn accept<'py>(
        slf: PyRef<'py, Self>,
        visitor: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        visitor.call_method1(intern!(visitor.py(), "visit_var"), (slf,))
    }

    fn __copy__(slf: PyRef<Self>) -> PyRef<Self> {
        // I am immutable...
        slf
    }

    fn __deepcopy__<'py>(slf: PyRef<'py, Self>, _memo: Bound<'py, PyAny>) -> PyRef<'py, Self> {
        // ... as are all my constituent parts.
        slf
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        if matches!(self.0, Var::Bit { .. } | Var::Register { .. }) {
            return Ok(format!(
                "Var({}, {:?})",
                self.get_var(py)?,
                self.get_type(py)?
            ));
        };
        Ok(format!(
            "Var({}, {:?}, name='{}')",
            self.get_var(py)?,
            self.get_type(py)?,
            self.get_name(py)?
        ))
    }

    // This is needed to handle the 'name' kwarg for __reduce__.
    #[classmethod]
    fn _from_pickle(
        _cls: &Bound<'_, pyo3::types::PyType>,
        var: &Bound<PyAny>,
        ty: Type,
        name: &Bound<PyAny>,
    ) -> PyResult<Py<Self>> {
        PyVar::new0(var, ty, name.extract()?)
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        (
            py.get_type::<Self>().getattr(intern!(py, "_from_pickle"))?,
            (self.get_var(py)?, self.get_type(py)?, self.get_name(py)?),
        )
            .into_pyobject(py)
    }
}

#[derive(Clone, Debug, PartialEq, Hash)]
pub struct Stretch {
    pub uuid: u128,
    pub name: String,
}

#[pyclass(eq, hash, frozen, extends = PyExpr, name = "Stretch", module = "qiskit._accelerate.circuit.classical.expr")]
#[derive(PartialEq, Clone, Debug, Hash)]
struct PyStretch(Stretch);

#[pymethods]
impl PyStretch {
    #[new]
    fn new0(var: &Bound<PyAny>, name: String) -> PyResult<Py<Self>> {
        Py::new(
            var.py(),
            (
                PyStretch(Stretch {
                    uuid: var.getattr(intern!(var.py(), "int"))?.extract()?,
                    name,
                }),
                PyExpr(ExprKind::Stretch),
            ),
        )
    }

    /// Generate a new named stretch variable.
    #[classmethod]
    fn new(cls: &Bound<'_, pyo3::types::PyType>, name: String) -> PyResult<Py<Self>> {
        Py::new(
            cls.py(),
            (
                PyStretch(Stretch {
                    uuid: Uuid::new_v4().as_u128(),
                    name,
                }),
                PyExpr(ExprKind::Stretch),
            ),
        )
    }

    #[getter]
    fn get_var(&self, py: Python) -> PyResult<Py<PyAny>> {
        let kwargs = [("int", self.0.uuid)].into_py_dict(py)?;
        Ok(UUID.get_bound(py).call((), Some(&kwargs))?.unbind())
    }

    #[getter]
    fn get_name(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.0.name.clone().into_py_any(py)
    }

    #[getter]
    fn get_const(&self) -> bool {
        true
    }

    #[getter]
    fn get_type(&self, py: Python) -> PyResult<Py<PyAny>> {
        Type::Duration.into_py_any(py)
    }

    fn accept<'py>(
        slf: PyRef<'py, Self>,
        visitor: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        visitor.call_method1(intern!(visitor.py(), "visit_stretch"), (slf,))
    }

    fn __copy__(slf: PyRef<Self>) -> PyRef<Self> {
        // I am immutable...
        slf
    }

    fn __deepcopy__<'py>(slf: PyRef<'py, Self>, _memo: Bound<'py, PyAny>) -> PyRef<'py, Self> {
        // ... as are all my constituent parts.
        slf
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        (
            py.get_type::<Self>(),
            (self.get_var(py)?, self.get_name(py)?),
        )
            .into_pyobject(py)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Index {
    pub target: Expr,
    pub index: Expr,
    pub ty: Type,
    pub constant: bool,
}

#[pyclass(eq, extends = PyExpr, name = "Index", module = "qiskit._accelerate.circuit.classical.expr")]
#[derive(PartialEq, Clone, Debug)]
struct PyIndex(Index);

#[pymethods]
impl PyIndex {
    #[new]
    #[pyo3(text_signature = "(target, index, type)")]
    fn new(py: Python, target: Expr, index: Expr, ty: Type) -> PyResult<Py<Self>> {
        let constant = target.is_const() && index.is_const();
        Py::new(
            py,
            (
                PyIndex(Index {
                    target,
                    index,
                    ty,
                    constant,
                }),
                PyExpr(ExprKind::Index),
            ),
        )
    }

    #[getter]
    fn get_target(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.0.target.clone().into_py_any(py)
    }

    #[getter]
    fn get_index(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.0.index.clone().into_py_any(py)
    }

    #[getter]
    fn get_const(&self, py: Python) -> bool {
        self.0.constant
    }

    #[getter]
    fn get_type(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.0.ty.clone().into_py_any(py)
    }

    fn accept<'py>(
        slf: PyRef<'py, Self>,
        visitor: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        visitor.call_method1(intern!(visitor.py(), "visit_index"), (slf,))
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        (
            py.get_type::<Self>(),
            (
                self.get_target(py)?,
                self.get_index(py)?,
                self.get_type(py)?,
            ),
        )
            .into_pyobject(py)
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

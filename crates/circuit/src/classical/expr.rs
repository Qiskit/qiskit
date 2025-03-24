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
use crate::imports::UUID;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use pyo3::{intern, BoundObject, IntoPyObjectExt};
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

#[derive(Clone, Debug, PartialEq)]
pub struct Unary {
    op: UnaryOp,
    operand: Expr,
    ty: Type,
    constant: bool,
}

#[repr(u8)]
#[derive(Copy, Hash, Clone, Debug, PartialEq)]
#[pyclass(
    eq,
    hash,
    frozen,
    name = "Unary",
    module = "qiskit._accelerate.circuit"
)]
pub enum UnaryOp {
    BitNot = 1,
    LogicNot = 2,
}

#[pymethods]
impl UnaryOp {
    fn __str__(&self) -> String {
        format!("Unary.Op.{:?}", self)
    }

    fn __repr__(&self) -> String {
        format!("Unary.Op.{:?}", self)
    }
}

#[pyclass(eq, extends = PyExpr, name = "Unary", module = "qiskit._accelerate.circuit.classical")]
#[derive(PartialEq, Clone, Debug)]
struct PyUnary(Unary);

#[pymethods]
impl PyUnary {
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
#[pyclass(
    eq,
    hash,
    frozen,
    name = "Unary",
    module = "qiskit._accelerate.circuit.classical.Binary"
)]
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

#[pymethods]
impl BinaryOp {
    fn __str__(&self) -> String {
        format!("Binary.Op.{:?}", self)
    }

    fn __repr__(&self) -> String {
        format!("Binary.Op.{:?}", self)
    }
}

#[pyclass(eq, extends = PyExpr, name = "Binary", module = "qiskit._accelerate.circuit.classical")]
#[derive(PartialEq, Clone, Debug)]
struct PyBinary(Binary);

#[pymethods]
impl PyBinary {
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
}

#[derive(Clone, Debug, PartialEq)]
pub struct Cast {
    operand: Expr,
    ty: Type,
    constant: bool,
    implicit: bool,
}

/// A cast from one type to another, implied by the use of an expression in a different
/// context.
#[pyclass(eq, extends = PyExpr, name = "Cast", module = "qiskit._accelerate.circuit.classical")]
#[derive(PartialEq, Clone, Debug)]
struct PyCast(Cast);

#[pymethods]
impl PyCast {
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
}

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Duration(Duration),
    Float { raw: f64, ty: Type },
    Uint { raw: u64, ty: Type },
}

#[pyclass(eq, extends = PyExpr, name = "Value", module = "qiskit._accelerate.circuit.classical")]
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

    fn accept<'py>(
        slf: PyRef<'py, Self>,
        visitor: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        visitor.call_method1(intern!(visitor.py(), "visit_value"), (slf,))
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

/// A classical variable.
///
/// These variables take two forms: a new-style variable that owns its storage location and has an
/// associated name; and an old-style variable that wraps a :class:`.Clbit` or
/// :class:`.ClassicalRegister` instance that is owned by some containing circuit.  In general,
/// construction of variables for use in programs should use :meth:`Var.new` or
/// :meth:`.QuantumCircuit.add_var`.
///
/// Variables are immutable after construction, so they can be used as dictionary keys."""
#[pyclass(eq, hash, frozen, extends = PyExpr, name = "Var", module = "qiskit._accelerate.circuit.classical")]
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
}

#[derive(Clone, Debug, PartialEq, Hash)]
pub struct Stretch {
    uuid: u128,
    name: String,
}

#[pyclass(eq, hash, frozen, extends = PyExpr, name = "Stretch", module = "qiskit._accelerate.circuit.classical")]
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
}

#[derive(Clone, Debug, PartialEq)]
pub struct Index {
    target: Expr,
    index: Expr,
    ty: Type,
    constant: bool,
}

#[pyclass(eq, extends = PyExpr, name = "Index", module = "qiskit._accelerate.circuit.classical")]
#[derive(PartialEq, Clone, Debug)]
struct PyIndex(Index);

#[pymethods]
impl PyIndex {
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

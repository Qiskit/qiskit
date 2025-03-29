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

use crate::classical::expr::{Binary, Cast, Index, Stretch, Unary, Value, Var};
use crate::classical::types::Type;
use pyo3::prelude::*;
use pyo3::{intern, IntoPyObjectExt};

/// A classical expression.
///
/// Variants that themselves contain [Expr]s are boxed. This is done instead
/// of boxing the contained [Expr]s within the specific type to reduce the
/// number of boxes we need (e.g. Binary would otherwise contain two boxed
/// expressions).
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
    /// The const-ness of the expression.
    pub fn is_const(&self) -> bool {
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

    /// The expression's [Type].
    pub fn ty(&self) -> Type {
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

    /// Returns an iterator over the [Var] nodes in this expression in some
    /// deterministic order.
    pub fn vars(&self) -> impl Iterator<Item = &Var> {
        VarIterator(ExprIterator { stack: vec![self] })
    }
}

/// A private iterator over the [Expr] nodes of an expression
/// by reference.
///
/// The first node reference returned is the [Expr] itself.
struct ExprIterator<'a> {
    stack: Vec<&'a Expr>,
}

impl<'a> Iterator for ExprIterator<'a> {
    type Item = &'a Expr;

    fn next(&mut self) -> Option<Self::Item> {
        let expr = self.stack.pop()?;
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

/// A private iterator over the [Var] nodes contained within an [Expr].
struct VarIterator<'a>(ExprIterator<'a>);

impl<'a> Iterator for VarIterator<'a> {
    type Item = &'a Var;

    fn next(&mut self) -> Option<Self::Item> {
        for expr in self.0.by_ref() {
            if let Expr::Var(v) = expr {
                return Some(v);
            }
        }
        None
    }
}

impl From<Unary> for Expr {
    fn from(value: Unary) -> Self {
        Expr::Unary(Box::new(value))
    }
}

impl From<Box<Unary>> for Expr {
    fn from(value: Box<Unary>) -> Self {
        Expr::Unary(value)
    }
}

impl From<Binary> for Expr {
    fn from(value: Binary) -> Self {
        Expr::Binary(Box::new(value))
    }
}

impl From<Box<Binary>> for Expr {
    fn from(value: Box<Binary>) -> Self {
        Expr::Binary(value)
    }
}

impl From<Cast> for Expr {
    fn from(value: Cast) -> Self {
        Expr::Cast(Box::new(value))
    }
}

impl From<Box<Cast>> for Expr {
    fn from(value: Box<Cast>) -> Self {
        Expr::Cast(value)
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

impl From<Box<Index>> for Expr {
    fn from(value: Box<Index>) -> Self {
        Expr::Index(value)
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
    module = "qiskit._accelerate.circuit.classical.expr"
)]
#[derive(PartialEq, Clone, Copy, Debug, Hash)]
pub struct PyExpr(pub ExprKind); // ExprKind is used for fast extraction from Python

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

/// The expression's kind, used internally during Python instance extraction to avoid
/// `isinstance` checks.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ExprKind {
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
        let expr: PyRef<'_, PyExpr> = ob.downcast()?.borrow();
        match expr.0 {
            ExprKind::Unary => Ok(Expr::Unary(Box::new(ob.extract()?))),
            ExprKind::Binary => Ok(Expr::Binary(Box::new(ob.extract()?))),
            ExprKind::Value => Ok(Expr::Value(ob.extract()?)),
            ExprKind::Var => Ok(Expr::Var(ob.extract()?)),
            ExprKind::Cast => Ok(Expr::Cast(Box::new(ob.extract()?))),
            ExprKind::Stretch => Ok(Expr::Stretch(ob.extract()?)),
            ExprKind::Index => Ok(Expr::Index(Box::new(ob.extract()?))),
        }
    }
}

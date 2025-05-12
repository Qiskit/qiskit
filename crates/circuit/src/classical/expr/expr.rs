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

use std::ops::{Deref, DerefMut};
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

#[derive(Clone, Debug, PartialEq)]
pub enum ExprRefMut<'a> {
    Unary(&'a mut Unary),
    Binary(&'a mut Binary),
    Cast(&'a mut Cast),
    Value(Value),
    Var(Var),
    Stretch(Stretch),
    Index(&'a mut Index),
}

// impl Deref for Expr {
//     type Target = ();
//
//     fn deref(&self) -> &Self::Target {
//         todo!()
//     }
// }
//
// impl DerefMut for Expr {
//     fn deref_mut(&mut self) -> &mut Self::Target {
//         todo!()
//     }
// }

impl Expr {
    pub fn as_deref_mut(&mut self) -> ExprRefMut<'_> {
        match self {
            Expr::Unary(u) => ExprRefMut::Unary(u.as_mut()),
            Expr::Binary(b) => ExprRefMut::Binary(b.as_mut()),
            Expr::Cast(c) => ExprRefMut::Cast(c.as_mut()),
            Expr::Value(v) => ExprRefMut::Value(v.clone()),
            Expr::Var(v) => ExprRefMut::Var(v.clone()),
            Expr::Stretch(s) => ExprRefMut::Stretch(s.clone()),
            Expr::Index(i) => ExprRefMut::Index(i.as_mut()),
        }
    }

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
    pub fn identifiers(&self) -> impl Iterator<Item = IdentifierRef<'_>> {
        IdentIterator(ExprIterator { stack: vec![self] })
    }

    /// Returns an iterator over the [Var] nodes in this expression in some
    /// deterministic order.
    pub fn identifiers_mut(&mut self) -> impl Iterator<Item = IdentifierRefMut<'_>> {
        IdentIteratorMut(ExprIteratorMut { stack: vec![self] })
    }

    /// Returns an iterator over the [Var] nodes in this expression in some
    /// deterministic order.
    pub fn vars(&self) -> impl Iterator<Item = &Var> {
        VarIterator(ExprIterator { stack: vec![self] })
    }

    /// Returns an iterator over the [Var] nodes in this expression in some
    /// deterministic order.
    pub fn vars_mut(&mut self) -> impl Iterator<Item = &mut Var> {
        VarIteratorMut(ExprIteratorMut { stack: vec![self] })
    }

    /// Returns an iterator over all nodes in this expression in some deterministic
    /// order.
    pub fn iter(&self) -> impl Iterator<Item = &Expr> {
        ExprIterator { stack: vec![self] }
    }

    /// Returns an iterator over all nodes in this expression in some deterministic
    /// order.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Expr> {
        ExprIteratorMut { stack: vec![self] }
    }
}


pub enum IdentifierRef<'a> {
    Var(&'a Var),
    Stretch(&'a Stretch),
}

pub enum IdentifierRefMut<'a> {
    Var(&'a mut Var),
    Stretch(&'a mut Stretch),
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

/// A private iterator over the [Var] and [Stretch] nodes contained within an [Expr].
struct IdentIterator<'a>(ExprIterator<'a>);

impl<'a> Iterator for IdentIterator<'a> {
    type Item = IdentifierRef<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        for expr in self.0.by_ref() {
            if let Expr::Var(v) = expr {
                return Some(IdentifierRef::Var(v));
            }
            if let Expr::Stretch(s) = expr {
                return Some(IdentifierRef::Stretch(s));
            }
        }
        None
    }
}

/// A private iterator over the [Expr] nodes of an expression
/// by reference.
///
/// The first node reference returned is the [Expr] itself.
struct ExprIteratorMut<'a> {
    stack: Vec<&'a mut Expr>,
}

impl<'a> Iterator for ExprIteratorMut<'a> {
    type Item = &'a mut Expr;

    fn next(&mut self) -> Option<Self::Item> {
        let mut old_stack = std::mem::take(&mut self.stack);
        let expr = old_stack.pop()?;
        match expr {
            Expr::Unary(u) => {
                old_stack.push(&mut u.operand);
            }
            Expr::Binary(b) => {
                old_stack.push(&mut b.left);
                old_stack.push(&mut b.right);
            }
            Expr::Cast(c) => old_stack.push(&mut c.operand),
            Expr::Value(_) => {}
            Expr::Var(_) => {}
            Expr::Stretch(_) => {}
            Expr::Index(i) => {
                old_stack.push(&mut i.index);
                old_stack.push(&mut i.target);
            }
        }
        Some(expr)
    }
}

/// A private iterator over the [Var] nodes contained within an [Expr].
struct VarIteratorMut<'a>(ExprIteratorMut<'a>);

impl<'a> Iterator for VarIteratorMut<'a> {
    type Item = &'a mut Var;

    fn next(&mut self) -> Option<Self::Item> {
        for expr in self.0.by_ref() {
            if let Expr::Var(v) = expr {
                return Some(v);
            }
        }
        None
    }
}

/// A private iterator over the [Var] and [Stretch] nodes contained within an [Expr].
struct IdentIteratorMut<'a>(ExprIteratorMut<'a>);

impl<'a> Iterator for IdentIteratorMut<'a> {
    type Item = IdentifierRefMut<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        for expr in &mut self.0 {
            if let Expr::Var(v) = expr {
                return Some(IdentifierRefMut::Var(v));
            }
            if let Expr::Stretch(s) = expr {
                return Some(IdentifierRefMut::Stretch(s));
            }
        }
        None
    }
}

// pub trait ExprVisitor {
//     type Output;
//
//     fn visit_expr(&self, expr: &Expr) -> Self::Output {
//         match expr {
//             Expr::Unary(u) => self.visit_unary(u),
//             Expr::Binary(b) => self.visit_binary(b),
//             Expr::Cast(c) => self.visit_cast(c),
//             Expr::Value(v) => self.visit_value(v),
//             Expr::Var(v) => self.visit_var(v),
//             Expr::Stretch(s) => self.visit_stretch(s),
//             Expr::Index(i) => self.visit_index(i),
//         }
//     }
//
//     fn visit_unary(&self, u: &Unary) -> Self::Output;
//     fn visit_binary(&self, b: &Binary) -> Self::Output;
//     fn visit_cast(&self, c: &Cast) -> Self::Output;
//     fn visit_value(&self, v: &Value) -> Self::Output;
//     fn visit_var(&self, v: &Var) -> Self::Output;
//     fn visit_stretch(&self, s: &Stretch) -> Self::Output;
//     fn visit_index(&self, i: &Index) -> Self::Output;
// }
//
// pub struct ExprRewriterDefault;
//
// impl ExprVisitor for ExprRewriterDefault {
//     type Output = Expr;
//
//     fn visit_unary(&self, u: &Unary) -> Expr {
//         Expr::Unary(Box::new(Unary {
//             op: u.op,
//             ty: u.ty,
//             constant: u.constant,
//             operand: self.visit_expr(&u.operand),
//         }))
//     }
//
//     fn visit_binary(&self, b: &Binary) -> Expr {
//         Expr::Binary(Box::new(Binary {
//             op: b.op,
//             ty: b.ty,
//             constant: b.constant,
//             left: self.visit_expr(&b.left),
//             right: self.visit_expr(&b.right),
//         }))
//     }
//
//     fn visit_cast(&self, c: &Cast) -> Expr {
//         Expr::Cast(Box::new(Cast {
//             ty: c.ty,
//             constant: c.constant,
//             operand: self.visit_expr(&c.operand),
//             implicit: c.implicit,
//         }))
//     }
//
//     fn visit_value(&self, v: &Value) -> Expr {
//         Expr::Value(v.clone())
//     }
//
//     fn visit_var(&self, v: &Var) -> Expr {
//         Expr::Var(v.clone())
//     }
//
//     fn visit_stretch(&self, s: &Stretch) -> Expr {
//         Expr::Stretch(s.clone())
//     }
//
//     fn visit_index(&self, i: &Index) -> Expr {
//         Expr::Index(Box::new(Index {
//             ty: i.ty,
//             constant: i.constant,
//             target: self.visit_expr(&i.target),
//             index: self.visit_expr(&i.index),
//         }))
//     }
// }

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

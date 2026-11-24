// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

mod binary;
mod cast;
#[allow(clippy::module_inception)]
mod expr;
mod index;
mod stretch;
mod unary;
mod value;
mod var;

pub use binary::{Binary, BinaryOp};
pub use cast::Cast;
pub use expr::{Expr, ExprRef, ExprRefMut, IdentifierRef};
pub use index::Index;
pub use stretch::Stretch;
pub use unary::{Unary, UnaryOp};
pub use value::Value;
pub use var::Var;

// These are only `pub(super)`, for registering the Python module in `classical`, since we
// probably shouldn't have a reason to use the Python class types from Rust if we're doing
// things the right way.
#[cfg(feature = "py")]
pub(super) use crate::classical::expr::binary::PyBinary;
#[cfg(feature = "py")]
pub(super) use crate::classical::expr::cast::PyCast;
#[cfg(feature = "py")]
use crate::classical::expr::expr::ExprKind;
#[cfg(feature = "py")]
pub(super) use crate::classical::expr::expr::PyExpr;
#[cfg(feature = "py")]
pub(super) use crate::classical::expr::index::PyIndex;
#[cfg(feature = "py")]
pub(super) use crate::classical::expr::stretch::PyStretch;
#[cfg(feature = "py")]
pub(super) use crate::classical::expr::unary::PyUnary;
#[cfg(feature = "py")]
pub(super) use crate::classical::expr::value::PyValue;
#[cfg(feature = "py")]
pub(super) use crate::classical::expr::var::PyVar;

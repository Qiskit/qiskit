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

//! The node contract and the node types Qiskit defines.
//!
//! The node types here share the rules in `inference` rather than each spelling out its own, so two
//! nodes in one family cannot drift apart. A node may write its own inference, and one whose
//! result types come from something other than its operand types has to: a shot loop takes its
//! result types from its circuits.

mod binary;
mod bitwise;
mod constant;
mod error;
mod inference;
mod op_node_type;
mod reduction;

pub use binary::{Add, Divide, Multiply, Power, Remainder, Subtract};
pub use bitwise::{BitwiseAnd, BitwiseNot, BitwiseOr, BitwiseXor, Parity};
pub use constant::Constant;
pub use error::MathNodeError;
pub use op_node_type::{OpNodeType, QISKIT};
pub use reduction::{Mean, Std, Variance};

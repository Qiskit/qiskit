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

//! The quantum program: the dataflow IR that Qiskit's `BackendV3` interface consumes.
//!
//! A program describes a hybrid quantum-classical computation as typed tensor values produced and
//! consumed by nodes. It is inert data: it describes a computation without performing one.
//!
//! - [`tensor`] is the value domain. A [`Tensor`](tensor::Tensor) is a dense array over one of a
//!   fixed set of dtypes, and a [`TensorType`](tensor::TensorType) is its data-less counterpart.
//! - [`nodes`] holds the [`OpNodeType`] contract and the node types Qiskit defines. The set is open: a
//!   node type may be defined outside this crate, in its own namespace.
//! - [`program`] holds [`ProgramFunction`], one dataflow graph. Nodes are the only entity it holds:
//!   a [`Value`] is an output slot of the node producing it, and a function's parameters and results
//!   are nodes too, so every value has a producer.
//! - [`data_tree`] holds [`DataTree`], the container for structured values.

pub mod data_tree;
pub mod nodes;
pub mod program;
pub mod tensor;

pub use data_tree::{ArityMismatch, DataTree, InvalidName, Name, PathEntry, TreeMatchError};
pub use nodes::{BoxedOpNodeError, BoxedOpNodeType, Constant, ErasedOpNodeType, OpNodeType};
pub use program::{
    FunctionError, FunctionEvalError, NodeId, NodeRef, NodeRole, ProgramFunction, Signature, Value,
};

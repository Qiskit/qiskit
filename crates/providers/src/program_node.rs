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

use crate::data_tree::DataTree;
use crate::tensor::{Tensor, TensorType};

/// A node in a quantum program graph that transforms tensors.
pub trait ProgramNode {
    /// The name of this program node.
    fn name(&self) -> &'static str;

    /// The namespace this program node belongs to.
    fn namespace(&self) -> &'static str;

    /// The namespace and name as one string.
    fn full_name(&self) -> String {
        format_args!("{}.{}", self.namespace(), self.name()).to_string()
    }

    /// The inputs expected at `call` time.
    fn input_types(&self) -> &DataTree<TensorType>;

    /// The outputs promised on `call` return.
    fn output_types(&self) -> &DataTree<TensorType>;

    /// Whether this program node implements the call method.
    fn implements_call(&self) -> bool;

    /// The action of this program node.
    fn call(&self, args: &DataTree<Tensor>) -> anyhow::Result<DataTree<Tensor>>;
}

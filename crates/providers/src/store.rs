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
use crate::program_node::ProgramNode;
use crate::tensor::{Tensor, TensorType};
use std::sync::LazyLock;

// An empty data tree is the input for all store nodes.
static EMPTY_DATA_TREE: LazyLock<DataTree<TensorType>> = LazyLock::new(DataTree::new);

/// A program node that owns constant data and outputs it unconditionally.
///
/// `Store` takes no inputs; its `call()` always returns the data it was constructed with.
/// In a data-flow graph, `Store` nodes play the role of constants — they are wired to
/// the input ports of computation nodes to supply fixed values.
pub struct Store {
    data: DataTree<Tensor>,
    output_types: DataTree<TensorType>,
}

impl Store {
    /// Construct a new `Store` holding the given data.
    pub fn new(data: DataTree<Tensor>) -> Self {
        let output_types = derive_output_types(&data);
        Self { data, output_types }
    }

    /// Return a reference to the stored data.
    pub fn data(&self) -> &DataTree<Tensor> {
        &self.data
    }
}

/// Recursively derive output types from concrete tensor data.
fn derive_output_types(data: &DataTree<Tensor>) -> DataTree<TensorType> {
    match data {
        DataTree::Leaf(tensor) => DataTree::new_leaf(tensor.tensor_type()),
        DataTree::Branch(_) => {
            let mut result = DataTree::with_capacity(data.len());
            for (key, child) in data.iter_children() {
                let child_type = derive_output_types(child);
                if let Some(k) = key {
                    result.insert_branch(k, child_type);
                } else {
                    result.push_branch(child_type);
                }
            }
            result
        }
    }
}

impl ProgramNode for Store {
    type CallError = std::convert::Infallible;

    fn name(&self) -> &str {
        "store"
    }

    fn namespace(&self) -> &str {
        "qiskit"
    }

    fn input_types(&self) -> &DataTree<TensorType> {
        &EMPTY_DATA_TREE
    }

    fn output_types(&self) -> &DataTree<TensorType> {
        &self.output_types
    }

    fn implements_call(&self) -> bool {
        true
    }

    fn call(&self, _args: &DataTree<Tensor>) -> Result<DataTree<Tensor>, Self::CallError> {
        Ok(self.data.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{DType, DTypeLike, Dim, Tensor};

    #[test]
    fn test_store_leaf_call() {
        let data = DataTree::new_leaf(Tensor::from([1.0_f64, 2.0, 3.0]));
        let store = Store::new(data);
        let result = store.call(&DataTree::new()).unwrap();
        let DataTree::Leaf(Tensor::F64(arr)) = result else {
            panic!("expected f64 leaf");
        };
        assert_eq!(arr.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_store_output_types_leaf() {
        let data = DataTree::new_leaf(Tensor::from([1.0_f64, 2.0, 3.0]));
        let store = Store::new(data);
        let DataTree::Leaf(tt) = store.output_types() else {
            panic!("expected leaf output type");
        };
        assert!(matches!(tt.dtype, DTypeLike::Concrete(DType::F64)));
        assert_eq!(tt.shape, vec![Dim::Fixed(3)]);
        assert!(!tt.broadcastable);
    }

    #[test]
    fn test_store_output_types_2d() {
        use ndarray::arr2;
        let data = DataTree::new_leaf(Tensor::F64(arr2(&[[1.0_f64, 2.0], [3.0, 4.0]]).into_dyn()));
        let store = Store::new(data);
        let DataTree::Leaf(tt) = store.output_types() else {
            panic!("expected leaf output type");
        };
        assert_eq!(tt.shape, vec![Dim::Fixed(2), Dim::Fixed(2)]);
    }

    #[test]
    fn test_store_branched() {
        let mut data = DataTree::new();
        data.insert_leaf("a", Tensor::from([1.0_f64, 2.0]));
        data.insert_leaf("b", Tensor::from([10_i32, 20, 30]));
        let store = Store::new(data);

        assert!(store.input_types().is_empty());
        assert_eq!(store.name(), "store");
        assert_eq!(store.namespace(), "qiskit");
        assert_eq!(store.full_name(), "qiskit.store");

        let out_types = store.output_types();
        let DataTree::Leaf(tt_a) = out_types.get_by_str_key("a").unwrap() else {
            panic!("expected leaf at a");
        };
        assert!(matches!(tt_a.dtype, DTypeLike::Concrete(DType::F64)));
        assert_eq!(tt_a.shape, vec![Dim::Fixed(2)]);

        let DataTree::Leaf(tt_b) = out_types.get_by_str_key("b").unwrap() else {
            panic!("expected leaf at b");
        };
        assert!(matches!(tt_b.dtype, DTypeLike::Concrete(DType::I32)));
        assert_eq!(tt_b.shape, vec![Dim::Fixed(3)]);
    }

    #[test]
    fn test_store_no_inputs() {
        let store = Store::new(DataTree::new_leaf(Tensor::from([42.0_f64])));
        assert!(store.input_types().is_empty());
        assert!(store.implements_call());
    }
}

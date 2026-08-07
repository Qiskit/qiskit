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

//! The static counterpart of a tensor: a dtype paired with a shape of per-axis sizes.

use std::fmt;

use super::DTypeLike;
use super::rules;

/// A tensor axis dimension.
///
/// Every axis is either concrete or bounded above.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dim {
    /// A dimension whose size is known.
    Fixed(usize),
    /// A dimension whose size is not known until run time, but is provably at most `max`.
    ///
    /// An operation that needs the true size at build time demands it through
    /// [`rules::require_static`].
    Bounded { max: usize },
}

impl fmt::Display for Dim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dim::Fixed(n) => write!(f, "{n}"),
            Dim::Bounded { max } => write!(f, "<={max}"),
        }
    }
}

/// Render a shape as `[4000, <=2]`.
pub(super) fn fmt_shape(shape: &[Dim]) -> String {
    let dims: Vec<String> = shape.iter().map(Dim::to_string).collect();
    format!("[{}]", dims.join(", "))
}

/// A specification of a tensor without any data.
#[derive(Debug, Clone)]
pub struct TensorType {
    /// The type of the tensor.
    pub dtype: DTypeLike,
    /// The dimension of each tensor axis.
    pub shape: Vec<Dim>,
    /// Whether the tensor supports leading-axis (i.e. NumPy-style) broadcasting semantics.
    pub broadcastable: bool,
}

impl TensorType {
    /// Return the dimension of every axis, or `None` if any is only bounded above.
    pub fn concrete_shape(&self) -> Option<Vec<usize>> {
        rules::require_static(&self.shape).ok()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::tensor::DType;

    /// A `TensorType` over `shape`; the dtype is irrelevant to every test that uses this.
    fn bit_type(shape: Vec<Dim>) -> TensorType {
        TensorType {
            dtype: DTypeLike::Concrete(DType::Bit),
            shape,
            broadcastable: false,
        }
    }

    #[test]
    fn test_tensor_type_concrete_shape() {
        assert_eq!(
            bit_type(vec![Dim::Fixed(3), Dim::Fixed(8)]).concrete_shape(),
            Some(vec![3, 8])
        );
        assert_eq!(bit_type(vec![]).concrete_shape(), Some(vec![]));

        // A bounded axis has no concrete size, so the whole shape has none.
        assert_eq!(
            bit_type(vec![Dim::Fixed(3), Dim::Bounded { max: 8 }]).concrete_shape(),
            None
        );
    }

    #[test]
    fn test_dim_display() {
        assert_eq!(Dim::Fixed(4000).to_string(), "4000");
        assert_eq!(Dim::Bounded { max: 2 }.to_string(), "<=2");
    }
}

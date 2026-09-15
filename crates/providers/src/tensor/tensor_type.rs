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

use super::DType;
use super::rules;

/// A tensor axis dimension.
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
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorType {
    /// The element type of the tensor.
    pub dtype: DType,
    /// The dimension of each tensor axis.
    pub shape: Vec<Dim>,
}

impl TensorType {
    /// Return the dimension of every axis, or `None` if any is only bounded above.
    pub fn concrete_shape(&self) -> Option<Vec<usize>> {
        rules::require_static(&self.shape).ok()
    }
}

/// Render as `F64[4000, <=2]`, so that a type can be named in an error a caller reads.
impl fmt::Display for TensorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.dtype, fmt_shape(&self.shape))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    /// A `TensorType` over `shape`; the dtype is irrelevant to every test that uses this.
    fn bit_type(shape: Vec<Dim>) -> TensorType {
        TensorType {
            dtype: DType::Bit,
            shape,
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

    #[test]
    fn test_tensor_type_display() {
        assert_eq!(
            bit_type(vec![Dim::Fixed(4000), Dim::Bounded { max: 2 }]).to_string(),
            "Bit[4000, <=2]"
        );
        assert_eq!(bit_type(vec![]).to_string(), "Bit[]");
    }
}

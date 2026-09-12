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

impl Dim {
    /// Whether every size `offered` allows is a size this dimension allows.
    ///
    /// A fixed dimension allows only its own size. Broadcasting, where a size of `1` stands for any
    /// size, is [`rules::broadcast_dims`].
    pub fn admits(self, offered: Dim) -> bool {
        match (self, offered) {
            (Dim::Fixed(n), Dim::Fixed(m)) => n == m,
            (Dim::Fixed(_), Dim::Bounded { .. }) => false,
            (Dim::Bounded { max }, Dim::Fixed(m)) => m <= max,
            (Dim::Bounded { max }, Dim::Bounded { max: bound }) => bound <= max,
        }
    }
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

    /// Whether every tensor satisfying `other` also satisfies this type.
    ///
    /// A value of type `other` already fits this one, rather than fitting after broadcasting. It is
    /// [`Tensor::matches`](super::Tensor::matches) with a type in place of the tensor.
    pub fn admits(&self, other: &TensorType) -> bool {
        self.dtype == other.dtype
            && self.shape.len() == other.shape.len()
            && self
                .shape
                .iter()
                .zip(&other.shape)
                .all(|(&dim, &offered)| dim.admits(offered))
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
    fn test_dim_admits() {
        let bounded = Dim::Bounded { max: 4 };

        assert!(Dim::Fixed(3).admits(Dim::Fixed(3)));
        assert!(bounded.admits(bounded));

        // A bound admits a true size within it, up to and including the bound itself.
        assert!(bounded.admits(Dim::Fixed(3)));
        assert!(bounded.admits(Dim::Fixed(4)));
        assert!(!bounded.admits(Dim::Fixed(5)));

        // A tighter bound is admitted by a looser one, and not the other way round.
        assert!(bounded.admits(Dim::Bounded { max: 2 }));
        assert!(!Dim::Bounded { max: 2 }.admits(bounded));

        // A true size is required where a true size is declared.
        assert!(!Dim::Fixed(3).admits(bounded));

        // A size of 1 stands for any size when broadcasting, which this is not.
        assert!(!Dim::Fixed(3).admits(Dim::Fixed(1)));
        assert!(!Dim::Fixed(1).admits(Dim::Fixed(3)));
    }

    #[test]
    fn test_tensor_type_admits() {
        let fixed = bit_type(vec![Dim::Fixed(3)]);

        assert!(fixed.admits(&fixed));
        assert!(
            bit_type(vec![Dim::Bounded { max: 4 }]).admits(&fixed),
            "per axis"
        );

        // The dtype and the number of axes must agree.
        assert!(
            !TensorType {
                dtype: DType::F64,
                shape: vec![Dim::Fixed(3)],
            }
            .admits(&fixed)
        );
        assert!(!fixed.admits(&bit_type(vec![Dim::Fixed(1), Dim::Fixed(3)])));
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

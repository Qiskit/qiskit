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

use super::broadcast::align_axes;
use super::{DType, TensorError};

/// A tensor axis dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dim {
    /// A dimension whose size is known.
    Fixed(usize),
    /// A dimension whose size is not known until run time, but is provably at most `max`.
    ///
    /// An operation that needs the true size at build time demands it through
    /// [`require_static`].
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

/// Require every axis of `shape` to be [`Dim::Fixed`], returning their sizes.
///
/// Operations should use this helper whenever they require the true size rather than a bound.
pub fn require_static(shape: &[Dim]) -> Result<Vec<usize>, TensorError> {
    shape
        .iter()
        .map(|dim| match dim {
            Dim::Fixed(n) => Ok(*n),
            Dim::Bounded { .. } => Err(TensorError::DynamicDim {
                shape: shape.to_vec(),
            }),
        })
        .collect()
}

/// Compute the type-level NumPy-style broadcast shape for two operand shapes.
///
/// This is the [`Dim`]-level counterpart of [`broadcast_shape`](super::broadcast_shape), predicting
/// a result shape from operand shapes with no tensor data in hand. Over fixed axes the rules are
/// exactly `broadcast_shape`'s:
///
/// - `Fixed(1)` broadcasts against anything.
/// - `Fixed(m)` against `Fixed(n)` with `m != n`, neither of them `1`, is
///   [`TensorError::DimShapeMismatch`].
///
/// A [`Dim::Bounded`] axis passes through where it meets a size of `1`, including the implicit `1`s
/// that pad the shorter shape. Anywhere else it would have to be compared against the size it meets,
/// which needs its true size, so it is [`TensorError::DynamicDim`].
pub fn broadcast_dims(a: &[Dim], b: &[Dim]) -> Result<Vec<Dim>, TensorError> {
    align_axes(a, b, Dim::Fixed(1))
        .map(|pair| match pair {
            (Dim::Fixed(1), y) => Ok(y),
            (x, Dim::Fixed(1)) => Ok(x),
            (Dim::Fixed(m), Dim::Fixed(n)) if m == n => Ok(Dim::Fixed(m)),
            (Dim::Bounded { .. }, _) => Err(TensorError::DynamicDim { shape: a.to_vec() }),
            (_, Dim::Bounded { .. }) => Err(TensorError::DynamicDim { shape: b.to_vec() }),
            _ => Err(TensorError::DimShapeMismatch {
                lhs: a.to_vec(),
                rhs: b.to_vec(),
            }),
        })
        .collect()
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
        require_static(&self.shape).ok()
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
    fn test_require_static() {
        assert_eq!(
            require_static(&[Dim::Fixed(3), Dim::Fixed(8)]).unwrap(),
            vec![3, 8]
        );
        assert!(require_static(&[]).unwrap().is_empty());
    }

    #[test]
    fn test_require_static_rejects_bounded_and_reports_the_shape() {
        let shape = [Dim::Fixed(3), Dim::Bounded { max: 16 }];
        assert!(matches!(
            require_static(&shape).unwrap_err(),
            TensorError::DynamicDim { shape: reported } if reported == shape
        ));
    }

    #[test]
    fn test_broadcast_dims_compatible() {
        // [2, 3] against [3] -> [2, 3], mirroring broadcast_shape.
        assert_eq!(
            broadcast_dims(&[Dim::Fixed(2), Dim::Fixed(3)], &[Dim::Fixed(3)]).unwrap(),
            vec![Dim::Fixed(2), Dim::Fixed(3)]
        );

        // Scalar broadcast: [4] against [1] -> [4].
        assert_eq!(
            broadcast_dims(&[Dim::Fixed(4)], &[Dim::Fixed(1)]).unwrap(),
            vec![Dim::Fixed(4)]
        );

        // Differing ranks: the missing leading axes act as Fixed(1).
        let a = vec![Dim::Fixed(2), Dim::Fixed(1), Dim::Fixed(3)];
        assert_eq!(broadcast_dims(&a, &[Dim::Fixed(3)]).unwrap(), a);
    }

    #[test]
    fn test_broadcast_dims_incompatible_reports_both_operands() {
        let a = [Dim::Fixed(3)];
        let b = [Dim::Fixed(4)];
        assert!(matches!(
            broadcast_dims(&a, &b).unwrap_err(),
            TensorError::DimShapeMismatch { lhs, rhs } if lhs == a && rhs == b
        ));
    }

    #[test]
    fn test_broadcast_dims_forwards_a_bounded_axis() {
        let bounded = Dim::Bounded { max: 8 };

        // A size of 1 stretches against a bounded axis, which passes its bound through.
        assert_eq!(
            broadcast_dims(&[bounded], &[Dim::Fixed(1)]).unwrap(),
            vec![bounded]
        );
        assert_eq!(
            broadcast_dims(&[Dim::Fixed(1)], &[bounded]).unwrap(),
            vec![bounded]
        );

        // So do the implicit 1s that pad the shorter shape.
        assert_eq!(
            broadcast_dims(&[bounded, Dim::Fixed(3)], &[Dim::Fixed(3)]).unwrap(),
            vec![bounded, Dim::Fixed(3)]
        );
    }

    #[test]
    fn test_broadcast_dims_rejects_a_compared_bounded_axis() {
        // A bounded axis meeting a size it would have to be compared against needs its true size:
        // another bounded axis, or a fixed size other than 1.
        let fixed = vec![Dim::Fixed(5)];
        let bounded = vec![Dim::Bounded { max: 8 }];
        for (a, b, at_fault) in [
            (&bounded, &fixed, &bounded),
            (&fixed, &bounded, &bounded),
            (&bounded, &bounded, &bounded),
        ] {
            assert!(
                matches!(
                    broadcast_dims(a, b).unwrap_err(),
                    TensorError::DynamicDim { shape } if shape == *at_fault
                ),
                "for {a:?} against {b:?}"
            );
        }
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

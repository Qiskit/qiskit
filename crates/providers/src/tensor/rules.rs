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

//! Type-level rules for [`DType`] and [`Dim`].
//!
//! Functions in this module answer questions about what promotion and broadcasting
//! *would* do for a given pair of operand types, so a caller can understand expected
//! behaviour without actual tensors instantiated.

use super::broadcast::align_axes;
use super::{DType, Dim, TensorError};

/// Promote a pair of dtypes to the smallest type compatible with both.
///
/// This function implements the same promotion rules as NumPy, modulo that we don't
/// need to contend with the arbitrary precision types for each type kind, and that
/// we omit F16 entirely because it's unstable in Rust:
/// <https://numpy.org/doc/stable/reference/arrays.promotion.html#numerical-promotion>
/// In short, if you view the linked diagram as a DAG, this function hard-codes the
/// least-common-descendant algorithm.
pub fn promotion(lhs: DType, rhs: DType) -> DType {
    use DType::*;

    match lhs {
        C128 => C128,

        C64 => match rhs {
            U32 | U64 | I32 | I64 | F64 | C128 => C128,
            _ => C64,
        },

        F64 => match rhs {
            C64 | C128 => C128,
            _ => F64,
        },

        F32 => match rhs {
            C128 => C128,
            C64 => C64,
            U32 | U64 | I32 | I64 | F64 => F64,
            _ => F32,
        },

        I64 => match rhs {
            C64 | C128 => C128,
            U64 | F32 | F64 => F64,
            _ => I64,
        },

        I32 => match rhs {
            C64 | C128 => C128,
            U64 | F32 | F64 => F64,
            U32 | I64 => I64,
            _ => I32,
        },

        I16 => match rhs {
            U64 => F64,
            U32 => I64,
            U16 => I32,
            Bit | U8 | I8 => I16,
            _ => rhs,
        },

        I8 => match rhs {
            U64 => F64,
            U32 => I64,
            U16 => I32,
            U8 => I16,
            Bit => I8,
            _ => rhs,
        },

        U64 => match rhs {
            C128 | C64 => C128,
            F32 | F64 | I8 | I16 | I32 | I64 => F64,
            _ => U64,
        },

        U32 => match rhs {
            C64 | C128 => C128,
            F32 | F64 => F64,
            I8 | I16 | I32 | I64 => I64,
            U64 => U64,
            _ => U32,
        },

        U16 => match rhs {
            I8 | I16 => I32,
            Bit | U8 => U16,
            _ => rhs,
        },

        U8 => match rhs {
            I8 => I16,
            Bit => U8,
            _ => rhs,
        },

        Bit => rhs,
    }
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

#[cfg(test)]
mod test {
    use super::*;

    const ALL_DTYPES: [DType; 13] = [
        DType::Bit,
        DType::U8,
        DType::U16,
        DType::U32,
        DType::U64,
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::F32,
        DType::F64,
        DType::C64,
        DType::C128,
    ];

    #[test]
    fn test_promotion_against_promotion_dag() {
        use DType::*;
        use hashbrown::{HashMap, HashSet};
        use rustworkx_core::dag_algo::lexicographical_topological_sort;
        use rustworkx_core::petgraph::graph::{DiGraph, NodeIndex};
        use rustworkx_core::traversal::descendants;

        // define a DAG that implements all promotion rules; two DTypes
        // should be promoted to their least common descendant in the DAG
        let mut g: DiGraph<DType, ()> = DiGraph::new();
        let mut idx: HashMap<DType, NodeIndex> = HashMap::new();

        for &dtype in &ALL_DTYPES {
            idx.insert(dtype, g.add_node(dtype));
        }

        // within-kind promotions
        g.add_edge(idx[&U8], idx[&U16], ());
        g.add_edge(idx[&U16], idx[&U32], ());
        g.add_edge(idx[&U32], idx[&U64], ());

        g.add_edge(idx[&I8], idx[&I16], ());
        g.add_edge(idx[&I16], idx[&I32], ());
        g.add_edge(idx[&I32], idx[&I64], ());

        g.add_edge(idx[&F32], idx[&F64], ());

        g.add_edge(idx[&C64], idx[&C128], ());

        // bit promotions
        g.add_edge(idx[&Bit], idx[&U8], ());
        g.add_edge(idx[&Bit], idx[&I8], ());

        // uint promotions
        g.add_edge(idx[&U8], idx[&I16], ());
        g.add_edge(idx[&U16], idx[&I32], ());
        g.add_edge(idx[&U16], idx[&F32], ());
        g.add_edge(idx[&U32], idx[&I64], ());
        g.add_edge(idx[&U64], idx[&F64], ());

        // int promotions
        g.add_edge(idx[&I16], idx[&F32], ());
        g.add_edge(idx[&I32], idx[&F64], ());
        g.add_edge(idx[&I64], idx[&F64], ());

        // float promotions
        g.add_edge(idx[&F32], idx[&C64], ());
        g.add_edge(idx[&F64], idx[&C128], ());

        let order = lexicographical_topological_sort(
            &g,
            |n: NodeIndex| Ok::<usize, std::convert::Infallible>(n.index()),
            false,
            None,
        )
        .unwrap();

        let least_common_descendant = move |a: &DType, b: &DType| -> DType {
            let da: HashSet<_> = descendants(&g, idx[a]).collect();
            let db: HashSet<_> = descendants(&g, idx[b]).collect();
            let common: HashSet<NodeIndex> = da.intersection(&db).copied().collect();
            let least_idx = order.iter().find(|n| common.contains(*n)).unwrap();
            ALL_DTYPES[least_idx.index()]
        };

        for &a in &ALL_DTYPES {
            for &b in &ALL_DTYPES {
                assert_eq!(
                    promotion(a, b),
                    least_common_descendant(&a, &b),
                    "For promotion ({a}, {b})"
                )
            }
        }
    }

    #[test]
    fn test_promotion_idempotence() {
        for &a in &ALL_DTYPES {
            assert_eq!(promotion(a, a), a, "For promotion ({a}, {a})")
        }
    }

    #[test]
    fn test_promotion_commutativity() {
        for &a in &ALL_DTYPES {
            for &b in &ALL_DTYPES {
                assert_eq!(promotion(a, b), promotion(b, a), "For promotion ({a}, {b})")
            }
        }
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
        let err = require_static(&shape).unwrap_err();
        assert_eq!(
            err,
            TensorError::DynamicDim {
                shape: shape.to_vec()
            }
        );
        assert_eq!(
            err.to_string(),
            "shape [3, <=16] has an axis whose size is only bounded above, \
             where a true size is required"
        );
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
        let err = broadcast_dims(&a, &b).unwrap_err();
        assert_eq!(
            err,
            TensorError::DimShapeMismatch {
                lhs: a.to_vec(),
                rhs: b.to_vec()
            }
        );
        assert_eq!(
            err.to_string(),
            "shapes [3] and [4] are not broadcast-compatible"
        );
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
            let err = broadcast_dims(a, b).unwrap_err();
            assert_eq!(
                err,
                TensorError::DynamicDim {
                    shape: at_fault.clone()
                },
                "for {a:?} against {b:?}"
            );
        }
    }
}

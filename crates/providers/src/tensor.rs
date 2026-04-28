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

use ndarray::ArrayD;
use num_complex::Complex;
use std::fmt;

/// The possible data types for a Tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    C128, // complex
    C64,
    F64, // float
    F32,
    I64, // signed ints
    I32,
    I16,
    I8,
    U64, // unsigned ints
    U32,
    U16,
    U8,
    Bit, // bool
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let string_repr = match self {
            DType::C128 => "C128",
            DType::C64 => "C64",
            DType::F64 => "F64",
            DType::F32 => "F32",
            DType::I64 => "I64",
            DType::I32 => "I32",
            DType::I16 => "I16",
            DType::I8 => "I8",
            DType::U64 => "U64",
            DType::U32 => "U32",
            DType::U16 => "U16",
            DType::U8 => "U8",
            DType::Bit => "Bit",
        };
        write!(f, "{string_repr}")
    }
}

// A tensor data type whose value is yet unknown, but named.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DTypeVar {
    pub name: String,
}

impl<T: Into<String>> From<T> for DTypeVar {
    fn from(value: T) -> Self {
        Self { name: value.into() }
    }
}

/// A tensor data type whose value is yet unknown, but will be the promotion of others.
#[derive(Debug, Clone)]
pub struct DTypePromotion {
    pub args: Vec<DTypeLike>,
}

impl<T: Into<Vec<DTypeLike>>> From<T> for DTypePromotion {
    fn from(args: T) -> Self {
        Self { args: args.into() }
    }
}

/// A tensor data type, known or unknown.
#[derive(Debug, Clone)]
pub enum DTypeLike {
    Concrete(DType),
    Var(DTypeVar),
    Promotion(DTypePromotion),
}

/// Promote a pair of DTypes to the smallest type compatible with both.
///
/// QuantumProgram operations often, but not necessarily, use this promotion rule
/// to determine their output type.
///
/// This function implements the same promotion rules as NumPy, modulo that we don't
/// need to contend with the arbitrary precision types for each type kind, and that
/// we omit F16 entirely because it's ustable in rust:
/// https://numpy.org/doc/stable/reference/arrays.promotion.html#numerical-promotion
/// In short, if you view the linked diagram as a DAG, this function hard-codes the
/// least-common-descendent algorithm.
pub fn promotion(lhs: DType, rhs: DType) -> DType {
    use DType::*;

    // painfully write a lookup table as a nested match statement. to check if it's right,
    // compare agaist the linked image, or more easily, study the test
    // test_promotion_against_promotion_dag that tests every input combination.
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

/// A tensor axis dimension.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Dim {
    /// A known size.
    Fixed(usize),
    /// An unresolved, named size.
    Named(String),
}

/// A specification of a tensor without any data.
#[derive(Debug, Clone)]
pub struct TensorType {
    /// The type of the tensor.
    pub dtype: DTypeLike,
    /// The shape of the tensor, possibly with axes of unknown size.
    pub shape: Vec<Dim>,
    /// Whether the tensor supports leading-axis (i.e. NumPy-style) broadcasting semantics.
    pub broadcastable: bool,
}

impl TensorType {
    // Return a dimension vector if all are sizes are fixed, None otherwise.
    pub fn concrete_shape(&self) -> Option<Vec<usize>> {
        let mut out = Vec::with_capacity(self.shape.len());
        for d in &self.shape {
            match d {
                Dim::Fixed(n) => out.push(*n),
                Dim::Named(_) => return None,
            }
        }
        Some(out)
    }
}

/// A tensor of one of the supported dtypes.
#[derive(Debug, Clone)]
pub enum Tensor {
    C64(ArrayD<Complex<f32>>),
    C128(ArrayD<Complex<f64>>),
    F32(ArrayD<f32>),
    F64(ArrayD<f64>),
    I8(ArrayD<i8>),
    I16(ArrayD<i16>),
    I32(ArrayD<i32>),
    I64(ArrayD<i64>),
    U8(ArrayD<u8>),
    U16(ArrayD<u16>),
    U32(ArrayD<u32>),
    U64(ArrayD<u64>),
    Bit(ArrayD<u8>),
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_promotion_against_promotion_dag() {
        use DType::*;
        use rustworkx_core::dag_algo::lexicographical_topological_sort;
        use rustworkx_core::petgraph::graph::{DiGraph, NodeIndex};
        use rustworkx_core::traversal::descendants;
        use std::collections::{HashMap, HashSet};

        // define a DAG that implements all promotion rules; two DTypes
        // should be promoted to their least common descendent in the DAG
        let mut g: DiGraph<DType, ()> = DiGraph::new();
        let mut idx: HashMap<DType, NodeIndex> = HashMap::new();

        let nodes = [
            Bit, U8, U16, U32, U64, I8, I16, I32, I64, F32, F64, C64, C128,
        ];

        for &dtype in &nodes {
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
        .ok()
        .unwrap();

        let least_common_decendent = move |a: &DType, b: &DType| -> DType {
            let da: HashSet<_> = descendants(&g, idx[&a]).collect();
            let db: HashSet<_> = descendants(&g, idx[&b]).collect();
            let common: HashSet<NodeIndex> = da.intersection(&db).copied().collect();
            let least_idx = order.iter().find(|n| common.contains(n)).unwrap();
            nodes[least_idx.index()]
        };

        for &a in &nodes {
            for &b in &nodes {
                assert_eq!(
                    promotion(a, b),
                    least_common_decendent(&a, &b),
                    "For promotion ({a}, {b})"
                )
            }
        }
    }

    #[test]
    fn test_promotion_idempotence() {
        use DType::*;
        let nodes = [
            Bit, U8, U16, U32, U64, I8, I16, I32, I64, F32, F64, C64, C128,
        ];

        for &a in &nodes {
            assert_eq!(promotion(a, a), a, "For promotion ({a}, {a})")
        }
    }

    #[test]
    fn test_promotion_commutativity() {
        use DType::*;
        let nodes = [
            Bit, U8, U16, U32, U64, I8, I16, I32, I64, F32, F64, C64, C128,
        ];

        for &a in &nodes {
            for &b in &nodes {
                assert_eq!(promotion(a, b), promotion(b, a), "For promotion ({a}, {b})")
            }
        }
    }

    #[test]
    fn test_tensor_type_concrete_shape() {
        assert_eq!(
            TensorType {
                dtype: DTypeLike::Concrete(DType::Bit),
                shape: vec![Dim::Fixed(3)],
                broadcastable: false,
            }
            .concrete_shape(),
            Some(vec![3])
        );

        assert_eq!(
            TensorType {
                dtype: DTypeLike::Concrete(DType::Bit),
                shape: vec![Dim::Fixed(3), Dim::Fixed(8)],
                broadcastable: true,
            }
            .concrete_shape(),
            Some(vec![3, 8])
        );

        assert_eq!(
            TensorType {
                dtype: DTypeLike::Concrete(DType::Bit),
                shape: vec![Dim::Fixed(3), Dim::Named("foo".into())],
                broadcastable: false,
            }
            .concrete_shape(),
            None
        );
    }
}

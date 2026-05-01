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

use ndarray::{ArrayD, IxDyn, Zip};
use num_complex::Complex;
use std::fmt;

/// The possible data types for a Tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    C128, // complex
    C64,
    F64, // real
    F32,
    I64, // signed integer
    I32,
    I16,
    I8,
    U64, // unsigned integer
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

/// A tensor dtype that is unknown but identified by name.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DTypeVar {
    /// The variable name.
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
    /// The dtype arguments to promote over.
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
    /// A fully resolved dtype.
    Concrete(DType),
    /// A dtype identified by a variable name, to be resolved later.
    Var(DTypeVar),
    /// A dtype that is the promotion of one or more other dtypes.
    Promotion(DTypePromotion),
}

/// Promote a pair of DTypes to the smallest type compatible with both.
///
/// QuantumProgram nodes often, but not necessarily, use this promotion rule
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
    /// Return a dimension vector if all sizes are fixed, or `None` if any are named.
    pub fn concrete_shape(&self) -> Option<Vec<usize>> {
        self.shape
            .iter()
            .map(|d| match d {
                Dim::Fixed(n) => Some(*n),
                Dim::Named(_) => None,
            })
            .collect()
    }
}

/// A tensor of one of the supported dtypes.
#[derive(Debug, Clone)]
pub enum Tensor {
    C64(ArrayD<Complex<f32>>), // complex
    C128(ArrayD<Complex<f64>>),
    F32(ArrayD<f32>), // real
    F64(ArrayD<f64>),
    I8(ArrayD<i8>), // signed integer
    I16(ArrayD<i16>),
    I32(ArrayD<i32>),
    I64(ArrayD<i64>),
    U8(ArrayD<u8>), // unsigned integer
    U16(ArrayD<u16>),
    U32(ArrayD<u32>),
    U64(ArrayD<u64>),
    Bit(ArrayD<u8>), // bool
}

/// Cast an `ArrayD` of a real numeric type to any supported dtype.
macro_rules! cast_real {
    ($arr:expr, $src:ty, $target:expr) => {
        match $target {
            DType::Bit => Tensor::Bit($arr.mapv(|x: $src| x as u8)),
            DType::U8 => Tensor::U8($arr.mapv(|x: $src| x as u8)),
            DType::U16 => Tensor::U16($arr.mapv(|x: $src| x as u16)),
            DType::U32 => Tensor::U32($arr.mapv(|x: $src| x as u32)),
            DType::U64 => Tensor::U64($arr.mapv(|x: $src| x as u64)),
            DType::I8 => Tensor::I8($arr.mapv(|x: $src| x as i8)),
            DType::I16 => Tensor::I16($arr.mapv(|x: $src| x as i16)),
            DType::I32 => Tensor::I32($arr.mapv(|x: $src| x as i32)),
            DType::I64 => Tensor::I64($arr.mapv(|x: $src| x as i64)),
            DType::F32 => Tensor::F32($arr.mapv(|x: $src| x as f32)),
            DType::F64 => Tensor::F64($arr.mapv(|x: $src| x as f64)),
            DType::C64 => Tensor::C64($arr.mapv(|x: $src| Complex::new(x as f32, 0.0))),
            DType::C128 => Tensor::C128($arr.mapv(|x: $src| Complex::new(x as f64, 0.0))),
        }
    };
}

/// Cast an `ArrayD` of a complex type to a complex dtype (panics for real targets).
macro_rules! cast_complex {
    ($arr:expr, $target:expr) => {
        match $target {
            DType::C64 => Tensor::C64($arr.mapv(|x| Complex::new(x.re as f32, x.im as f32))),
            DType::C128 => Tensor::C128($arr.mapv(|x| Complex::new(x.re as f64, x.im as f64))),
            _ => panic!("cannot cast complex tensor to a real dtype"),
        }
    };
}

/// Element-wise binary operation on two arrays with NumPy-style broadcasting.
///
/// Unlike ndarray's built-in arithmetic operators which handle broadcasting automatically,
/// this helper is needed for operations without a Rust operator (e.g. `pow`).
fn broadcast_elementwise<T, F>(a: &ArrayD<T>, b: &ArrayD<T>, op: F) -> ArrayD<T>
where
    T: Clone,
    F: Fn(&T, &T) -> T,
{
    let ndim = a.ndim().max(b.ndim());
    let out_shape: Vec<usize> = (0..ndim)
        .map(|i| {
            let d_a = if i >= ndim - a.ndim() {
                a.shape()[i - (ndim - a.ndim())]
            } else {
                1
            };
            let d_b = if i >= ndim - b.ndim() {
                b.shape()[i - (ndim - b.ndim())]
            } else {
                1
            };
            match (d_a, d_b) {
                (x, y) if x == y => x,
                (1, y) => y,
                (x, 1) => x,
                _ => panic!(
                    "shapes {:?} and {:?} are not broadcast-compatible",
                    a.shape(),
                    b.shape()
                ),
            }
        })
        .collect();
    let out_ix = IxDyn(&out_shape);
    let a_bc = a.broadcast(out_ix.clone()).expect("broadcast failed");
    let b_bc = b.broadcast(out_ix).expect("broadcast failed");
    Zip::from(a_bc).and(b_bc).map_collect(op)
}

impl Tensor {
    /// Return the dtype of this tensor.
    pub fn dtype(&self) -> DType {
        match self {
            Tensor::C128(_) => DType::C128,
            Tensor::C64(_) => DType::C64,
            Tensor::F64(_) => DType::F64,
            Tensor::F32(_) => DType::F32,
            Tensor::I64(_) => DType::I64,
            Tensor::I32(_) => DType::I32,
            Tensor::I16(_) => DType::I16,
            Tensor::I8(_) => DType::I8,
            Tensor::U64(_) => DType::U64,
            Tensor::U32(_) => DType::U32,
            Tensor::U16(_) => DType::U16,
            Tensor::U8(_) => DType::U8,
            Tensor::Bit(_) => DType::Bit,
        }
    }

    /// Return the shape of this tensor as a slice of dimension sizes.
    pub fn shape(&self) -> &[usize] {
        match self {
            Tensor::C128(a) => a.shape(),
            Tensor::C64(a) => a.shape(),
            Tensor::F64(a) => a.shape(),
            Tensor::F32(a) => a.shape(),
            Tensor::I64(a) => a.shape(),
            Tensor::I32(a) => a.shape(),
            Tensor::I16(a) => a.shape(),
            Tensor::I8(a) => a.shape(),
            Tensor::U64(a) => a.shape(),
            Tensor::U32(a) => a.shape(),
            Tensor::U16(a) => a.shape(),
            Tensor::U8(a) => a.shape(),
            Tensor::Bit(a) => a.shape(),
        }
    }

    /// Return the [`TensorType`] that describes this tensor's dtype and concrete shape.
    pub fn tensor_type(&self) -> TensorType {
        TensorType {
            dtype: DTypeLike::Concrete(self.dtype()),
            shape: self.shape().iter().map(|&n| Dim::Fixed(n)).collect(),
            broadcastable: false,
        }
    }

    /// Element-wise power with NumPy-style broadcasting.
    ///
    /// For integer types the exponent is cast to `u32`; negative integer exponents
    /// are not supported.
    pub fn pow(&self, rhs: &Tensor) -> Tensor {
        match (self, rhs) {
            (Tensor::F32(a), Tensor::F32(b)) => {
                Tensor::F32(broadcast_elementwise(a, b, |&x, &y| x.powf(y)))
            }
            (Tensor::F64(a), Tensor::F64(b)) => {
                Tensor::F64(broadcast_elementwise(a, b, |&x, &y| x.powf(y)))
            }
            (Tensor::C64(a), Tensor::C64(b)) => {
                Tensor::C64(broadcast_elementwise(a, b, |&x, &y| x.powc(y)))
            }
            (Tensor::C128(a), Tensor::C128(b)) => {
                Tensor::C128(broadcast_elementwise(a, b, |&x, &y| x.powc(y)))
            }
            (Tensor::I8(a), Tensor::I8(b)) => {
                Tensor::I8(broadcast_elementwise(a, b, |&x, &y| x.pow(y as u32)))
            }
            (Tensor::I16(a), Tensor::I16(b)) => {
                Tensor::I16(broadcast_elementwise(a, b, |&x, &y| x.pow(y as u32)))
            }
            (Tensor::I32(a), Tensor::I32(b)) => {
                Tensor::I32(broadcast_elementwise(a, b, |&x, &y| x.pow(y as u32)))
            }
            (Tensor::I64(a), Tensor::I64(b)) => {
                Tensor::I64(broadcast_elementwise(a, b, |&x, &y| x.pow(y as u32)))
            }
            (Tensor::U8(a), Tensor::U8(b)) => {
                Tensor::U8(broadcast_elementwise(a, b, |&x, &y| x.pow(y as u32)))
            }
            (Tensor::U16(a), Tensor::U16(b)) => {
                Tensor::U16(broadcast_elementwise(a, b, |&x, &y| x.pow(y as u32)))
            }
            (Tensor::U32(a), Tensor::U32(b)) => {
                Tensor::U32(broadcast_elementwise(a, b, |&x, &y| x.pow(y)))
            }
            (Tensor::U64(a), Tensor::U64(b)) => {
                Tensor::U64(broadcast_elementwise(a, b, |&x, &y| x.pow(y as u32)))
            }
            _ => panic!("type mismatch in Tensor::pow"),
        }
    }

    /// Cast this tensor to `target`, consuming it. Returns `self` unchanged if already that dtype.
    pub fn cast(self, target: DType) -> Tensor {
        if self.dtype() == target {
            return self;
        }
        match &self {
            Tensor::Bit(a) | Tensor::U8(a) => cast_real!(a, u8, target),
            Tensor::U16(a) => cast_real!(a, u16, target),
            Tensor::U32(a) => cast_real!(a, u32, target),
            Tensor::U64(a) => cast_real!(a, u64, target),
            Tensor::I8(a) => cast_real!(a, i8, target),
            Tensor::I16(a) => cast_real!(a, i16, target),
            Tensor::I32(a) => cast_real!(a, i32, target),
            Tensor::I64(a) => cast_real!(a, i64, target),
            Tensor::F32(a) => cast_real!(a, f32, target),
            Tensor::F64(a) => cast_real!(a, f64, target),
            Tensor::C64(a) => cast_complex!(a, target),
            Tensor::C128(a) => cast_complex!(a, target),
        }
    }
}

/// Implement `From<&[T]>`, `From<&[T; N]>`, and `From<ArrayD<T>>` for a given `Tensor` variant.
macro_rules! impl_tensor_from {
    ($variant:ident, $t:ty) => {
        impl From<&[$t]> for Tensor {
            fn from(data: &[$t]) -> Self {
                Tensor::$variant(ndarray::arr1(data).into_dyn())
            }
        }
        impl<const N: usize> From<[$t; N]> for Tensor {
            fn from(data: [$t; N]) -> Self {
                Tensor::$variant(ndarray::arr1(&data).into_dyn())
            }
        }
        impl From<ArrayD<$t>> for Tensor {
            fn from(data: ArrayD<$t>) -> Self {
                Tensor::$variant(data)
            }
        }
    };
}

impl_tensor_from!(C128, Complex<f64>);
impl_tensor_from!(C64, Complex<f32>);
impl_tensor_from!(F64, f64);
impl_tensor_from!(F32, f32);
impl_tensor_from!(I64, i64);
impl_tensor_from!(I32, i32);
impl_tensor_from!(I16, i16);
impl_tensor_from!(I8, i8);
impl_tensor_from!(U64, u64);
impl_tensor_from!(U32, u32);
impl_tensor_from!(U16, u16);
impl_tensor_from!(U8, u8); // u8 → U8; Bit requires explicit construction

/// Implement a standard Rust binary operator trait for `Tensor` and `&Tensor`.
macro_rules! impl_tensor_binop {
    ($trait:ident, $method:ident, $op:tt) => {
        impl std::ops::$trait for &Tensor {
            type Output = Tensor;
            fn $method(self, rhs: Self) -> Tensor {
                match (self, rhs) {
                    (Tensor::C128(a), Tensor::C128(b)) => Tensor::C128(a $op b),
                    (Tensor::C64(a), Tensor::C64(b)) => Tensor::C64(a $op b),
                    (Tensor::F64(a), Tensor::F64(b)) => Tensor::F64(a $op b),
                    (Tensor::F32(a), Tensor::F32(b)) => Tensor::F32(a $op b),
                    (Tensor::I64(a), Tensor::I64(b)) => Tensor::I64(a $op b),
                    (Tensor::I32(a), Tensor::I32(b)) => Tensor::I32(a $op b),
                    (Tensor::I16(a), Tensor::I16(b)) => Tensor::I16(a $op b),
                    (Tensor::I8(a), Tensor::I8(b)) => Tensor::I8(a $op b),
                    (Tensor::U64(a), Tensor::U64(b)) => Tensor::U64(a $op b),
                    (Tensor::U32(a), Tensor::U32(b)) => Tensor::U32(a $op b),
                    (Tensor::U16(a), Tensor::U16(b)) => Tensor::U16(a $op b),
                    (Tensor::U8(a), Tensor::U8(b)) => Tensor::U8(a $op b),
                    _ => panic!("type mismatch in Tensor::{}", stringify!($method)),
                }
            }
        }
        impl std::ops::$trait for Tensor {
            type Output = Tensor;
            fn $method(self, rhs: Self) -> Tensor { &self $op &rhs }
        }
    };
}

/// Like [`impl_tensor_binop!`], but omits complex variants for ops that don't support them
/// (e.g. `Rem`, which `num_complex` does not implement).
macro_rules! impl_tensor_binop_real {
    ($trait:ident, $method:ident, $op:tt) => {
        impl std::ops::$trait for &Tensor {
            type Output = Tensor;
            fn $method(self, rhs: Self) -> Tensor {
                match (self, rhs) {
                    (Tensor::F64(a), Tensor::F64(b)) => Tensor::F64(a $op b),
                    (Tensor::F32(a), Tensor::F32(b)) => Tensor::F32(a $op b),
                    (Tensor::I64(a), Tensor::I64(b)) => Tensor::I64(a $op b),
                    (Tensor::I32(a), Tensor::I32(b)) => Tensor::I32(a $op b),
                    (Tensor::I16(a), Tensor::I16(b)) => Tensor::I16(a $op b),
                    (Tensor::I8(a), Tensor::I8(b)) => Tensor::I8(a $op b),
                    (Tensor::U64(a), Tensor::U64(b)) => Tensor::U64(a $op b),
                    (Tensor::U32(a), Tensor::U32(b)) => Tensor::U32(a $op b),
                    (Tensor::U16(a), Tensor::U16(b)) => Tensor::U16(a $op b),
                    (Tensor::U8(a), Tensor::U8(b)) => Tensor::U8(a $op b),
                    _ => panic!("type mismatch or unsupported dtype in Tensor::{}", stringify!($method)),
                }
            }
        }
        impl std::ops::$trait for Tensor {
            type Output = Tensor;
            fn $method(self, rhs: Self) -> Tensor { &self $op &rhs }
        }
    };
}

impl_tensor_binop!(Add, add, +);
impl_tensor_binop!(Sub, sub, -);
impl_tensor_binop!(Mul, mul, *);
impl_tensor_binop!(Div, div, /);
impl_tensor_binop_real!(Rem, rem, %);

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

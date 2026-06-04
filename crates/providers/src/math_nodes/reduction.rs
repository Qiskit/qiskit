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
use crate::tensor::{DType, DTypeLike, Tensor, TensorType};
use crate::unpack_tensor_args;
use ndarray::Axis;
use num_complex::Complex;
use std::sync::LazyLock;

/// Shared input type spec for reduction nodes: a single broadcastable tensor of any dtype.
static INPUT_TYPES: LazyLock<DataTree<TensorType>> = LazyLock::new(|| {
    DataTree::new_leaf(TensorType {
        dtype: DTypeLike::Var("x".into()),
        shape: vec![],
        broadcastable: true,
    })
});

/// Shared output type spec for reduction nodes: a single broadcastable tensor of any dtype.
static OUTPUT_TYPES: LazyLock<DataTree<TensorType>> = LazyLock::new(|| {
    DataTree::new_leaf(TensorType {
        dtype: DTypeLike::Var("out".into()),
        shape: vec![],
        broadcastable: true,
    })
});

/// Mean of a tensor along a specified axis, removing that axis.
///
/// Integer inputs are cast to `F64` before computing the mean. `F32` inputs
/// produce `F32` output; all other float and integer types produce `F64`.
/// Complex inputs (`C64`, `C128`) preserve their complex dtype.
pub struct Mean {
    axis: usize,
}

impl Mean {
    /// Construct a `Mean` node that reduces along `axis`.
    pub fn new(axis: usize) -> Self {
        Self { axis }
    }
}

impl ProgramNode for Mean {
    type CallError = super::MathNodeError;

    fn name(&self) -> &'static str {
        "mean"
    }
    fn namespace(&self) -> &'static str {
        "math"
    }
    fn input_types(&self) -> &DataTree<TensorType> {
        &INPUT_TYPES
    }
    fn output_types(&self) -> &DataTree<TensorType> {
        &OUTPUT_TYPES
    }
    fn implements_call(&self) -> bool {
        true
    }
    fn call_flat(&self, args: &[Tensor]) -> Result<Vec<Tensor>, Self::CallError> {
        unpack_tensor_args!(args, [x]);
        let result = match x {
            Tensor::F32(a) => Tensor::F32(a.mean_axis(Axis(self.axis)).unwrap().into_shared()),
            Tensor::F64(a) => Tensor::F64(a.mean_axis(Axis(self.axis)).unwrap().into_shared()),
            Tensor::C64(a) => {
                let n = a.shape()[self.axis] as f32;
                Tensor::C64((a.sum_axis(Axis(self.axis)) / Complex::new(n, 0.0)).into_shared())
            }
            Tensor::C128(a) => {
                let n = a.shape()[self.axis] as f64;
                Tensor::C128((a.sum_axis(Axis(self.axis)) / Complex::new(n, 0.0)).into_shared())
            }
            other => {
                let Tensor::F64(a) = other.cast_ref(DType::F64).into_owned() else {
                    unreachable!()
                };
                Tensor::F64(a.mean_axis(Axis(self.axis)).unwrap().into_shared())
            }
        };
        Ok(vec![result])
    }
}

/// Variance of a tensor along a specified axis, removing that axis.
///
/// The `ddof` (delta degrees of freedom) parameter adjusts the divisor: the result
/// is divided by `n - ddof` where `n` is the number of elements along the axis.
/// Use `ddof=0` for population variance and `ddof=1` for sample variance.
///
/// Integer inputs are cast to `F64`. `F32` produces `F32`; all other real types
/// produce `F64`. Complex inputs (`C64`, `C128`) produce real output (`F32`, `F64`
/// respectively), computed as the mean squared modulus of the deviations.
pub struct Variance {
    axis: usize,
    ddof: f64,
}

impl Variance {
    /// Construct a `Variance` node that reduces along `axis` with degrees-of-freedom
    /// correction `ddof`.
    pub fn new(axis: usize, ddof: f64) -> Self {
        Self { axis, ddof }
    }
}

impl ProgramNode for Variance {
    type CallError = super::MathNodeError;

    fn name(&self) -> &'static str {
        "variance"
    }
    fn namespace(&self) -> &'static str {
        "math"
    }
    fn input_types(&self) -> &DataTree<TensorType> {
        &INPUT_TYPES
    }
    fn output_types(&self) -> &DataTree<TensorType> {
        &OUTPUT_TYPES
    }
    fn implements_call(&self) -> bool {
        true
    }
    fn call_flat(&self, args: &[Tensor]) -> Result<Vec<Tensor>, Self::CallError> {
        unpack_tensor_args!(args, [x]);
        let result = match x {
            Tensor::F32(a) => {
                Tensor::F32(a.var_axis(Axis(self.axis), self.ddof as f32).into_shared())
            }
            Tensor::F64(a) => Tensor::F64(a.var_axis(Axis(self.axis), self.ddof).into_shared()),
            Tensor::C64(a) => {
                let n = a.shape()[self.axis] as f32;
                let mean = (a.sum_axis(Axis(self.axis)) / Complex::new(n, 0.0))
                    .insert_axis(Axis(self.axis));
                let sq_mod = (a - &mean).mapv(|c| c.re * c.re + c.im * c.im);
                Tensor::F32(
                    (sq_mod.sum_axis(Axis(self.axis)) / (n - self.ddof as f32)).into_shared(),
                )
            }
            Tensor::C128(a) => {
                let n = a.shape()[self.axis] as f64;
                let mean = (a.sum_axis(Axis(self.axis)) / Complex::new(n, 0.0))
                    .insert_axis(Axis(self.axis));
                let sq_mod = (a - &mean).mapv(|c| c.re * c.re + c.im * c.im);
                Tensor::F64((sq_mod.sum_axis(Axis(self.axis)) / (n - self.ddof)).into_shared())
            }
            other => {
                let Tensor::F64(a) = other.cast_ref(DType::F64).into_owned() else {
                    unreachable!()
                };
                Tensor::F64(a.var_axis(Axis(self.axis), self.ddof).into_shared())
            }
        };
        Ok(vec![result])
    }
}

/// Standard deviation of a tensor along a specified axis, removing that axis.
///
/// This is the square root of [`Variance`]. See that type for details on `ddof`,
/// output dtypes, and complex handling.
pub struct Std {
    axis: usize,
    ddof: f64,
}

impl Std {
    /// Construct a `Std` node that reduces along `axis` with degrees-of-freedom
    /// correction `ddof`.
    pub fn new(axis: usize, ddof: f64) -> Self {
        Self { axis, ddof }
    }
}

impl ProgramNode for Std {
    type CallError = super::MathNodeError;

    fn name(&self) -> &'static str {
        "std"
    }
    fn namespace(&self) -> &'static str {
        "math"
    }
    fn input_types(&self) -> &DataTree<TensorType> {
        &INPUT_TYPES
    }
    fn output_types(&self) -> &DataTree<TensorType> {
        &OUTPUT_TYPES
    }
    fn implements_call(&self) -> bool {
        true
    }
    fn call_flat(&self, args: &[Tensor]) -> Result<Vec<Tensor>, Self::CallError> {
        unpack_tensor_args!(args, [x]);
        let result = match x {
            Tensor::F32(a) => {
                Tensor::F32(a.std_axis(Axis(self.axis), self.ddof as f32).into_shared())
            }
            Tensor::F64(a) => Tensor::F64(a.std_axis(Axis(self.axis), self.ddof).into_shared()),
            Tensor::C64(a) => {
                let n = a.shape()[self.axis] as f32;
                let mean = (a.sum_axis(Axis(self.axis)) / Complex::new(n, 0.0))
                    .insert_axis(Axis(self.axis));
                let sq_mod = (a - &mean).mapv(|c| c.re * c.re + c.im * c.im);
                Tensor::F32(
                    (sq_mod.sum_axis(Axis(self.axis)) / (n - self.ddof as f32))
                        .mapv(f32::sqrt)
                        .into_shared(),
                )
            }
            Tensor::C128(a) => {
                let n = a.shape()[self.axis] as f64;
                let mean = (a.sum_axis(Axis(self.axis)) / Complex::new(n, 0.0))
                    .insert_axis(Axis(self.axis));
                let sq_mod = (a - &mean).mapv(|c| c.re * c.re + c.im * c.im);
                Tensor::F64(
                    (sq_mod.sum_axis(Axis(self.axis)) / (n - self.ddof))
                        .mapv(f64::sqrt)
                        .into_shared(),
                )
            }
            other => {
                let Tensor::F64(a) = other.cast_ref(DType::F64).into_owned() else {
                    unreachable!()
                };
                Tensor::F64(a.std_axis(Axis(self.axis), self.ddof).into_shared())
            }
        };
        Ok(vec![result])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math_nodes::MathNodeError;
    use crate::program_node::{CallError, CallInputError, ProgramNodeExt};
    use crate::tensor::{DType, Tensor};
    use ndarray::arr2;

    fn approx_eq_slice(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len(), "slice lengths differ");
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-10, "{x} != {y}");
        }
    }

    // --- Mean tests ---

    #[test]
    fn test_mean_f64_axis0() {
        // [[1,2,3],[4,5,6]] along axis 0 → [2.5, 3.5, 4.5]
        let x = Tensor::F64(
            arr2(&[[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]])
                .into_dyn()
                .into_shared(),
        );
        let result = Mean::new(0).call_flat(&[x]).unwrap();
        let Tensor::F64(arr) = &result[0] else {
            panic!("expected F64 leaf");
        };
        approx_eq_slice(arr.as_slice().unwrap(), &[2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_mean_i32_casts_to_f64() {
        let x = Tensor::from([1_i32, 2, 3, 4]);
        let result = Mean::new(0).call_flat(&[x]).unwrap();
        assert_eq!(
            result[0].dtype(),
            DType::F64,
            "integer input should produce F64 mean"
        );
        let Tensor::F64(arr) = &result[0] else {
            panic!()
        };
        approx_eq_slice(arr.as_slice().unwrap(), &[2.5]);
    }

    #[test]
    fn test_mean_c128() {
        use num_complex::Complex;
        let data: Vec<Complex<f64>> = vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 6.0),
        ];
        let x = Tensor::C128(ndarray::Array1::from(data).into_dyn().into_shared());
        let result = Mean::new(0).call_flat(&[x]).unwrap();
        let Tensor::C128(arr) = &result[0] else {
            panic!("expected C128 leaf");
        };
        let v = arr.as_slice().unwrap()[0];
        assert!((v.re - 3.0).abs() < 1e-10);
        assert!((v.im - 4.0).abs() < 1e-10);
    }

    // --- Variance tests ---

    #[test]
    fn test_variance_f64_ddof0() {
        // [2, 4, 4, 4, 5, 5, 7, 9] — classic example, population variance = 4.0
        let x = Tensor::from([2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        let result = Variance::new(0, 0.0).call_flat(&[x]).unwrap();
        let Tensor::F64(arr) = &result[0] else {
            panic!("expected F64 leaf");
        };
        approx_eq_slice(arr.as_slice().unwrap(), &[4.0]);
    }

    #[test]
    fn test_variance_f64_ddof1() {
        // Sample variance (ddof=1) of the same sequence
        let x = Tensor::from([2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        let result = Variance::new(0, 1.0).call_flat(&[x]).unwrap();
        let Tensor::F64(arr) = &result[0] else {
            panic!("expected F64 leaf");
        };
        // sample variance = population variance * n / (n-1) = 4.0 * 8/7
        approx_eq_slice(arr.as_slice().unwrap(), &[4.0 * 8.0 / 7.0]);
    }

    #[test]
    fn test_variance_c128_returns_real() {
        use num_complex::Complex;
        // [1+1i, 3+3i] — mean = 2+2i, deviations = [−1−i, 1+i], |.|^2 = [2, 2], var = 2.0
        let data: Vec<Complex<f64>> = vec![Complex::new(1.0, 1.0), Complex::new(3.0, 3.0)];
        let x = Tensor::C128(ndarray::Array1::from(data).into_dyn().into_shared());
        let result = Variance::new(0, 0.0).call_flat(&[x]).unwrap();
        assert_eq!(
            result[0].dtype(),
            DType::F64,
            "C128 variance should return F64"
        );
        let Tensor::F64(arr) = &result[0] else {
            panic!()
        };
        approx_eq_slice(arr.as_slice().unwrap(), &[2.0]);
    }

    // --- Std tests ---

    #[test]
    fn test_std_matches_sqrt_of_variance() {
        // Verify std = sqrt(variance) numerically
        let x = Tensor::from([1.0_f64, 3.0, 5.0, 7.0, 9.0]);
        let var_result = Variance::new(0, 0.0).call_flat(&[x.clone()]).unwrap();
        let std_result = Std::new(0, 0.0).call_flat(&[x]).unwrap();

        let Tensor::F64(var_arr) = &var_result[0] else {
            panic!()
        };
        let Tensor::F64(std_arr) = &std_result[0] else {
            panic!()
        };

        let var_val = var_arr.as_slice().unwrap()[0];
        let std_val = std_arr.as_slice().unwrap()[0];
        assert!((std_val - var_val.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_std_c128_returns_real() {
        use num_complex::Complex;
        let data: Vec<Complex<f64>> = vec![Complex::new(1.0, 1.0), Complex::new(3.0, 3.0)];
        let x = Tensor::C128(ndarray::Array1::from(data).into_dyn().into_shared());
        let result = Std::new(0, 0.0).call_flat(&[x]).unwrap();
        assert_eq!(result[0].dtype(), DType::F64, "C128 std should return F64");
        let Tensor::F64(arr) = &result[0] else {
            panic!()
        };
        // std = sqrt(2.0)
        approx_eq_slice(arr.as_slice().unwrap(), &[2.0_f64.sqrt()]);
    }

    #[test]
    fn test_call_branch_where_leaf_expected_errors() {
        let mut tree = DataTree::new();
        tree.insert_leaf("x", Tensor::from([1.0_f64, 2.0]));
        let err = Mean::new(0).call(&tree).unwrap_err();
        assert!(matches!(
            err,
            CallError::<MathNodeError>::Input(CallInputError::ExpectedLeaf {
                ref key,
            }) if key.is_empty()
        ));
    }

    #[test]
    fn test_mean_wrong_arity_errors() {
        let err = Mean::new(0)
            .call_flat(&[Tensor::from([1.0_f64]), Tensor::from([2.0_f64])])
            .unwrap_err();
        assert_eq!(
            err,
            MathNodeError::Input(CallInputError::WrongArity {
                expected: 1,
                actual: 2,
            })
        );
    }
}

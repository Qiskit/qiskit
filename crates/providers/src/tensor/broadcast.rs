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

//! NumPy-style broadcasting helpers.

use ndarray::{ArcArrayD, IxDyn, Zip};

use super::TensorError;

/// Pair up the axes of two shapes, right-aligned, padding the shorter one with `pad`.
///
/// This is the axis correspondence NumPy-style broadcasting uses, shared by [`broadcast_shape`] and
/// [`broadcast_dims`](super::rules::broadcast_dims) so that the two agree on which axes meet.
pub(super) fn align_axes<'a, T: Copy>(
    a: &'a [T],
    b: &'a [T],
    pad: T,
) -> impl Iterator<Item = (T, T)> + 'a {
    let ndim = a.len().max(b.len());
    (0..ndim).map(move |i| {
        let axis = |shape: &[T]| {
            let offset = ndim - shape.len();
            if i >= offset { shape[i - offset] } else { pad }
        };
        (axis(a), axis(b))
    })
}

/// Compute the NumPy-style broadcast shape for two operand shapes, or
/// return [`TensorError::ShapeMismatch`] if they are not broadcast-compatible.
pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>, TensorError> {
    align_axes(a, b, 1)
        .map(|pair| match pair {
            (x, y) if x == y => Ok(x),
            (1, y) => Ok(y),
            (x, 1) => Ok(x),
            _ => Err(TensorError::ShapeMismatch {
                lhs: a.to_vec(),
                rhs: b.to_vec(),
            }),
        })
        .collect()
}

/// Element-wise binary operation on two arrays with NumPy-style broadcasting.
///
/// Unlike ndarray's built-in arithmetic operators which handle broadcasting automatically,
/// this helper is needed for operations without a Rust operator (e.g. `pow`). Returns
/// [`TensorError::ShapeMismatch`] if the operand shapes are not broadcast-compatible.
pub(super) fn broadcast_elementwise<T, F>(
    a: &ArcArrayD<T>,
    b: &ArcArrayD<T>,
    op: F,
) -> Result<ArcArrayD<T>, TensorError>
where
    T: Clone,
    F: Fn(&T, &T) -> T,
{
    let out_shape = broadcast_shape(a.shape(), b.shape())?;
    let out_ix = IxDyn(&out_shape);
    let a_bc = a.broadcast(out_ix.clone()).expect("broadcast failed");
    let b_bc = b.broadcast(out_ix).expect("broadcast failed");
    Ok(Zip::from(a_bc).and(b_bc).map_collect(op).into_shared())
}

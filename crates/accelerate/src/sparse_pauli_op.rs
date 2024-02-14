// This code is part of Qiskit.
//
// (C) Copyright IBM 2022
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::Python;

use hashbrown::HashMap;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use num_complex::Complex64;
use num_traits::Zero;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};

/// Find the unique elements of an array.
///
/// This function is a drop-in replacement of
/// ``np.unique(array, return_index=True, return_inverse=True, axis=0)``
/// where ``array`` is a ``numpy.ndarray`` of ``dtype=u16`` and ``ndim=2``.
///
/// Note that the order of the output of this function is not sorted while ``numpy.unique``
/// returns the sorted elements.
///
/// Args:
///     array (numpy.ndarray): An array of ``dtype=u16`` and ``ndim=2``
///
/// Returns:
///     (indexes, inverses): A tuple of the following two indices.
///
///         - the indices of the input array that give the unique values
///         - the indices of the unique array that reconstruct the input array
///
#[pyfunction]
pub fn unordered_unique(py: Python, array: PyReadonlyArray2<u16>) -> (PyObject, PyObject) {
    let array = array.as_array();
    let shape = array.shape();
    let mut table = HashMap::<ArrayView1<u16>, usize>::with_capacity(shape[0]);
    let mut indices = Vec::new();
    let mut inverses = vec![0; shape[0]];
    for (i, v) in array.axis_iter(Axis(0)).enumerate() {
        match table.get(&v) {
            Some(id) => inverses[i] = *id,
            None => {
                let new_id = table.len();
                table.insert(v, new_id);
                inverses[i] = new_id;
                indices.push(i);
            }
        }
    }
    (
        indices.into_pyarray(py).into(),
        inverses.into_pyarray(py).into(),
    )
}

#[derive(Clone, Copy)]
enum Pauli {
    I,
    X,
    Y,
    Z,
}

/// A complete ZX-convention representation of a Pauli decomposition.  This is all the components
/// necessary to construct a Qiskit-space :class:`.SparsePauliOp`, where :attr:`phases` is in the
/// ZX convention.
#[pyclass(module = "qiskit._accelerate.sparse_pauli_op")]
pub struct ZXPaulis {
    #[pyo3(get)]
    pub z: Py<PyArray2<bool>>,
    #[pyo3(get)]
    pub x: Py<PyArray2<bool>>,
    #[pyo3(get)]
    pub phases: Py<PyArray1<u8>>,
    #[pyo3(get)]
    pub coeffs: Py<PyArray1<Complex64>>,
}

/// Decompose a dense complex operator into the symplectic Pauli representation in the
/// ZX-convention.
///
/// This is an implementation of the "tensorized Pauli decomposition" presented in
/// `Hantzko, Binkowski and Gupta (2023) <https://arxiv.org/abs/2310.13421>`__.
#[pyfunction]
pub fn decompose_dense(
    py: Python,
    operator: PyReadonlyArray2<Complex64>,
    tolerance: f64,
) -> PyResult<ZXPaulis> {
    let num_qubits = operator.shape()[0].ilog2() as usize;
    let size = 1 << num_qubits;
    if operator.shape() != [size, size] {
        return Err(PyValueError::new_err(format!(
            "input with shape {:?} cannot be interpreted as a multiqubit operator",
            operator.shape()
        )));
    }
    let mut paulis = vec![];
    let mut coeffs = vec![];
    if num_qubits > 0 {
        decompose_dense_inner(
            Complex64::new(1.0, 0.0),
            num_qubits,
            &[],
            operator.as_array(),
            &mut paulis,
            &mut coeffs,
            tolerance * tolerance,
        );
    }
    if coeffs.is_empty() {
        Ok(ZXPaulis {
            z: PyArray2::zeros(py, [0, num_qubits], false).into(),
            x: PyArray2::zeros(py, [0, num_qubits], false).into(),
            phases: PyArray1::zeros(py, [0], false).into(),
            coeffs: PyArray1::zeros(py, [0], false).into(),
        })
    } else {
        // Constructing several arrays of different shapes at once is rather awkward in iterator
        // logic, so we just loop manually.
        let mut z = Array2::<bool>::uninit([paulis.len(), num_qubits]);
        let mut x = Array2::<bool>::uninit([paulis.len(), num_qubits]);
        let mut phases = Array1::<u8>::uninit(paulis.len());
        for (i, paulis) in paulis.drain(..).enumerate() {
            let mut phase = 0u8;
            for (j, pauli) in paulis.into_iter().rev().enumerate() {
                match pauli {
                    Pauli::I => {
                        z[[i, j]].write(false);
                        x[[i, j]].write(false);
                    }
                    Pauli::X => {
                        z[[i, j]].write(false);
                        x[[i, j]].write(true);
                    }
                    Pauli::Y => {
                        z[[i, j]].write(true);
                        x[[i, j]].write(true);
                        phase = phase.wrapping_add(1);
                    }
                    Pauli::Z => {
                        z[[i, j]].write(true);
                        x[[i, j]].write(false);
                    }
                }
            }
            phases[i].write(phase % 4);
        }
        // These are safe because the above loops write into every element.  It's guaranteed that
        // each of the elements of the `paulis` vec will have `num_qubits` because they're all
        // reading from the same base array.
        let z = unsafe { z.assume_init() };
        let x = unsafe { x.assume_init() };
        let phases = unsafe { phases.assume_init() };
        Ok(ZXPaulis {
            z: z.into_pyarray(py).into(),
            x: x.into_pyarray(py).into(),
            phases: phases.into_pyarray(py).into(),
            coeffs: PyArray1::from_vec(py, coeffs).into(),
        })
    }
}

/// Recurse worker routine of `decompose_dense`.  Should be called with at least one qubit.
fn decompose_dense_inner(
    factor: Complex64,
    num_qubits: usize,
    paulis: &[Pauli],
    block: ArrayView2<Complex64>,
    out_paulis: &mut Vec<Vec<Pauli>>,
    out_coeffs: &mut Vec<Complex64>,
    square_tolerance: f64,
) {
    if num_qubits == 0 {
        // It would be safe to `return` here, but if it's unreachable then LLVM is allowed to
        // optimise out this branch entirely in release mode, which is good for a ~2% speedup.
        unreachable!("should not call this with an empty operator")
    }
    // Base recursion case.
    if num_qubits == 1 {
        let mut push_if_nonzero = |extra: Pauli, value: Complex64| {
            if value.norm_sqr() <= square_tolerance {
                return;
            }
            let paulis = {
                let mut vec = Vec::with_capacity(paulis.len() + 1);
                vec.extend_from_slice(paulis);
                vec.push(extra);
                vec
            };
            out_paulis.push(paulis);
            out_coeffs.push(value);
        };
        push_if_nonzero(Pauli::I, 0.5 * factor * (block[[0, 0]] + block[[1, 1]]));
        push_if_nonzero(Pauli::X, 0.5 * factor * (block[[0, 1]] + block[[1, 0]]));
        push_if_nonzero(
            Pauli::Y,
            0.5 * Complex64::i() * factor * (block[[0, 1]] - block[[1, 0]]),
        );
        push_if_nonzero(Pauli::Z, 0.5 * factor * (block[[0, 0]] - block[[1, 1]]));
        return;
    }
    let mut recurse_if_nonzero = |extra: Pauli, factor: Complex64, values: Array2<Complex64>| {
        let mut is_zero = true;
        for value in values.iter() {
            if !value.is_zero() {
                is_zero = false;
                break;
            }
        }
        if is_zero {
            return;
        }
        let mut new_paulis = Vec::with_capacity(paulis.len() + 1);
        new_paulis.extend_from_slice(paulis);
        new_paulis.push(extra);
        decompose_dense_inner(
            factor,
            num_qubits - 1,
            &new_paulis,
            values.view(),
            out_paulis,
            out_coeffs,
            square_tolerance,
        );
    };
    let mid = 1usize << (num_qubits - 1);
    recurse_if_nonzero(
        Pauli::I,
        0.5 * factor,
        &block.slice(s![..mid, ..mid]) + &block.slice(s![mid.., mid..]),
    );
    recurse_if_nonzero(
        Pauli::X,
        0.5 * factor,
        &block.slice(s![..mid, mid..]) + &block.slice(s![mid.., ..mid]),
    );
    recurse_if_nonzero(
        Pauli::Y,
        0.5 * Complex64::i() * factor,
        &block.slice(s![..mid, mid..]) - &block.slice(s![mid.., ..mid]),
    );
    recurse_if_nonzero(
        Pauli::Z,
        0.5 * factor,
        &block.slice(s![..mid, ..mid]) - &block.slice(s![mid.., mid..]),
    );
}

#[pymodule]
pub fn sparse_pauli_op(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(unordered_unique))?;
    m.add_wrapped(wrap_pyfunction!(decompose_dense))?;
    m.add_class::<ZXPaulis>()?;
    Ok(())
}

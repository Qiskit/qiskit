// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use ndarray::ArrayView2;
use qiskit_circuit::{circuit_data::CircuitData, operations::Param};
use qiskit_circuit_library::iqp::{iqp, py_random_iqp};

/// Internal helper: build `CircuitData` from a *validated* view.
///
/// Assumes:
///   - `view` is square and symmetric.
///   - Any error from `from_standard_gates` is considered unreachable for valid
///     inputs, so we use `unwrap()` like other cext code.
fn iqp_from_view(view: ArrayView2<'_, i64>) -> CircuitData {
    let nrows = view.nrows();

    CircuitData::from_standard_gates(nrows as u32, iqp(view), Param::Float(0.0)).unwrap()
}

/// Generate an IQP circuit from an integer interaction matrix.
///
/// The `interactions` matrix is interpreted as an `n x n` row-major array of
/// 64-bit integers, where `n = num_qubits`. The diagonal entries set T-like
/// phase powers, and the upper triangle encodes two-qubit CPhase interactions.
/// The matrix must be symmetric; otherwise this function returns `NULL`.
///
/// # Parameters
///
/// - `num_qubits`: Number of logical qubits (`n`). Must match the dimension of
///   the `interactions` matrix.
/// - `interactions`: Pointer to a row-major `n x n` matrix of type `int64_t`.
///
/// # Returns
///
/// - A newly allocated `QkCircuit*` on success (caller must free with
///   `qk_circuit_free`).
/// - `NULL` if the input pointer is `NULL`, `num_qubits` is zero, or the
///   matrix is not symmetric.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_circuit_library_iqp_(
    num_qubits: u32,
    interactions: *const i64, // row major nxn
) -> *mut CircuitData {
    if interactions.is_null() {
        return std::ptr::null_mut();
    }

    let num_qubits = num_qubits as usize;
    if num_qubits == 0 {
        return std::ptr::null_mut();
    }

    // SAFETY: caller guarantees at least n*n elements are readable.
    let len = num_qubits * num_qubits;
    let buf = unsafe { std::slice::from_raw_parts(interactions, len) };

    // Wrap the flat buffer as an `ndarray` view with shape (n, n).
    let view = ndarray::ArrayView2::from_shape((num_qubits, num_qubits), buf).unwrap();

    // Symmetry on upper triangle only.
    let mut i = 0;
    while i < num_qubits {
        let mut j = i + 1;
        while j < num_qubits {
            if view[[i, j]] != view[[j, i]] {
                // Non-symmetric: return NULL to signal invalid input.
                return std::ptr::null_mut();
            }
            j += 1;
        }
        i += 1;
    }

    // Matrix is square and symmetric: build CircuitData and return an owning pointer.
    let circuit_data = iqp_from_view(view);
    Box::into_raw(Box::new(circuit_data))
}

/// Generate a random IQP circuit.
///
/// # Parameters
/// - `num_qubits`: Number of qubits.
/// - `seed`: RNG seed. If negative, entropy is drawn from the OS.
///
/// # Returns
/// - A newly allocated `QkCircuit*` (caller must free with `qk_circuit_free`).
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_circuit_library_random_iqp_(num_qubits: u32, seed: i64) -> *mut CircuitData {
    let seed = if seed < 0 { None } else { Some(seed as u64) };
    Box::into_raw(Box::new(py_random_iqp(num_qubits, seed).unwrap()))
}

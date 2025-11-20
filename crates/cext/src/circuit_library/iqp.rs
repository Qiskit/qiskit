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
use qiskit_circuit::{circuit_data::CircuitData, operations::{Param, StandardGate}, Qubit};
use smallvec::{smallvec, SmallVec};
use qiskit_circuit_library::iqp::py_random_iqp;

/// π/2 and π/4 constants reused for phase computations.
const PI2: f64 = std::f64::consts::PI / 2.0;
const PI4: f64 = std::f64::consts::PI / 4.0;

/// Build the IQP instruction stream from the integer interaction matrix `m`.
/// Layout:
///   H^{⊗n}
///   CPhase(π/2 * m[i,j]) for i<j where m[i,j] % 4 != 0
///   Phase(π/4 * m[i,i])  where m[i,i] % 8 != 0
///   H^{⊗n}

fn iqp_instructions(interactions: ArrayView2<'_, i64>,
) -> impl Iterator<Item = (StandardGate, SmallVec<[Param; 3]>, SmallVec<[Qubit; 2]>)> + '_ {

    let num_qubits = interactions.ncols();

    // The initial and final Hadamard layer.
    let h_layer = (0..num_qubits).map(|i| (StandardGate::H, smallvec![], smallvec![Qubit(i as u32)]));

    // The circuit interactions are powers of the CS gate, which is implemented by calling
    // the CPhase gate with angles of Pi/2 times the power. The gate powers are given by the
    // upper triangular part of the symmetric ``interactions`` matrix.
    let connections = (0..num_qubits).flat_map(move |i| {
        (i + 1..num_qubits)
            .map(move |j| (j, interactions[(i, j)]))
            .filter(move |(_, value)| value % 4 != 0)
            .map(move |(j, value)| {
                (
                    StandardGate::CPhase,
                    smallvec![Param::Float(PI2 * value as f64)],
                    smallvec![Qubit(i as u32), Qubit(j as u32)],
                )
            })
    });

    // The layer of T gates. Again we use the Phase gate, now with powers of Pi/4. The powers
    // are given by the diagonal of the ``interactions`` matrix.
    let shifts = (0..num_qubits)
        .map(move |i| interactions[(i, i)])
        .enumerate()
        .filter(|(_, value)| value % 8 != 0)
        .map(|(i, value)| {
            (
                StandardGate::Phase,
                smallvec![Param::Float(PI4 * value as f64)],
                smallvec![Qubit(i as u32)],
            )
        });

    h_layer
        .clone()
        .chain(connections)
        .chain(shifts)
        .chain(h_layer)
}


/// Internal helper: build `CircuitData` from a *validated* view.
///
/// Assumes:
///   - `view` is square and symmetric.
///   - Any error from `from_standard_gates` is considered unreachable for valid
///     inputs, so we use `unwrap()` like other cext code.
fn iqp_from_view(view : ArrayView2<'_, i64>) -> CircuitData {
    let nrows = view.nrows();

    CircuitData::from_standard_gates(
        nrows as u32,
        iqp_instructions(view),
        Param::Float(0.0)
    ).unwrap()
}


#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_circuit_library_iqp_(
    num_qubits: u32,
    interactions: *const i64, // row major nxn
) -> *mut CircuitData {
    if interactions.is_null(){
        return std::ptr::null_mut();
    }

    let num_qubits = num_qubits as usize;
    if num_qubits == 0 {
        return std::ptr::null_mut();
    }

    // SAFETY: caller guarantees at least n*n elements are readable.
    let len = num_qubits * num_qubits;
    let buf = unsafe{std::slice::from_raw_parts(interactions, len)};

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


#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_circuit_library_random_iqp_(
    num_qubits: u32,
    seed: i64
) -> *mut CircuitData {
    let seed = if seed < 0 { None } else { Some(seed as u64) };
    Box::into_raw(Box::new(py_random_iqp(num_qubits, seed).unwrap()))
}
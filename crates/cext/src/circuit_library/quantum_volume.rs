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

use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::library::quantum_volume::quantum_volume;

/// @ingroup QkCircuitLibrary
/// Generate a Quantum Volume circuit
///
///
///
/// # Example
///
/// ```c
///     QkCircuit *qc = qk_circuit_library_quantum_volume(10, 10, -1)
/// ```
///
/// # Safety
///
/// It's SAFE!
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_circuit_library_quantum_volume(
    num_qubits: u32,
    depth: usize,
    seed: i64,
) -> *mut CircuitData {
    let seed = if seed < 0 { None } else { Some(seed as u64) };
    Box::into_raw(Box::new(quantum_volume(num_qubits, depth, seed).unwrap()))
}

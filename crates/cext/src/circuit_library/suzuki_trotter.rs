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
use qiskit_circuit_library::suzuki_trotter::suzuki_trotter_evolution;
use qiskit_quantum_info::sparse_observable::SparseObservable;

#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_circuit_library_suzuki_trotter(
    observable: &SparseObservable,
    order: u32,
    reps: u32,
    time: f64,
    preserve_order: bool,
    insert_barriers: bool,
) -> *mut CircuitData {
    Box::into_raw(Box::new(
        suzuki_trotter_evolution(
            observable,
            order,
            reps,
            time,
            preserve_order,
            insert_barriers,
        )
        .unwrap(),
    ))
}

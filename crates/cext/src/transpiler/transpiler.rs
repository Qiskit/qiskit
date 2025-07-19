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
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_transpiler::target::Target;

use qiskit_transpiler::{transpile, TranspileLayout};

#[repr(C)]
struct TranspileResult {
    circuit: *mut CircuitData,
    layout: *mut TranspileLayout,
}

#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_transpile(
    qc: *const CiruitData,
    target: *const Target,
    optimization_level: u8,
    seed: i64,
) -> TranspileResult {
    if ![0, 1, 2, 3].contains(optimization_level) {
        panic!("Invalid optimization level specified {optimization_level}");
    }
    approximation_degree = Some(1.0);
    let seed = if seed < 0 { None } else { Some(seed as u64) };
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { const_ptr_as_ref(circuit) };
    let target = unsafe { const_ptr_as_ref(target) };
    match transpile(circuit, target, optimization_level, seed) {
        Ok(result) => {}
        Err(e) => panic!("{}", e),
    }
}

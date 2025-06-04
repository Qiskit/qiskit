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

use crate::pointers;
use qiskit_circuit::operations::{Operation, Param, StandardGate};
use qiskit_circuit::Qubit;
use qiskit_transpiler::commutation_checker::CommutationChecker;

#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_commutation_cache_new(max_entries: usize) -> *mut CommutationChecker {
    let checker = CommutationChecker::new(None, max_entries, None);
    Box::into_raw(Box::new(checker))
}

#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_commutation_cache_free(cache: *mut CommutationChecker) {
    if !cache.is_null() {
        if !cache.is_aligned() {
            panic!("Tried to free non-aligned pointer!");
        }

        // SAFETY: we checked the pointer is non-null and aligned
        let _ = unsafe { Box::from_raw(cache) };
    }
}

unsafe fn read_params(gate: StandardGate, params: *const f64) -> Vec<Param> {
    if gate.num_params() == 0 {
        return vec![];
    }
    if params.is_null() {
        panic!("Got NULL pointer when reading parameters");
    }
    if !params.is_aligned() {
        panic!("Pointer not aligned.");
    }

    unsafe { ::std::slice::from_raw_parts(params, gate.num_params() as usize) }
        .iter()
        .map(|value| Param::Float(*value))
        .collect::<Vec<Param>>()
}

#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_gate_commute(
    gate1: StandardGate,
    params1: *const f64,
    qubits1: *const u32,
    gate2: StandardGate,
    params2: *const f64,
    qubits2: *const u32,
    approximation_degree: f64,
    commutation_cache: *mut CommutationChecker,
) -> bool {
    let op1 = qiskit_circuit::operations::OperationRef::StandardGate(gate1);
    let op2 = qiskit_circuit::operations::OperationRef::StandardGate(gate2);

    // SAFETY: Per docs, the parameter pointers are readable for the number of parameters in the gate.
    let params1 = unsafe { read_params(gate1, params1) };
    let params2 = unsafe { read_params(gate2, params2) };

    // SAFETY: Per docs, the index pointers are readable for the number of qubits in the gate.
    let indices1 = unsafe { ::std::slice::from_raw_parts(qubits1, gate1.num_qubits() as usize) };
    let indices2 = unsafe { ::std::slice::from_raw_parts(qubits2, gate2.num_qubits() as usize) };

    let qubits1 = ::bytemuck::cast_slice::<_, Qubit>(indices1);
    let qubits2 = ::bytemuck::cast_slice::<_, Qubit>(indices2);

    let max_qubits = 4; // we don't have larger standard gates
    if commutation_cache.is_null() {
        // temporary commutation checker
        let mut checker = CommutationChecker::new(None, 1, None);
        checker
            .commute(
                &op1,
                &params1,
                qubits1,
                &[],
                &op2,
                &params2,
                qubits2,
                &[],
                max_qubits,
                approximation_degree,
            )
            .unwrap()
    } else {
        // SAFETY: Per docs, the commutation cache is non-null and aligned.
        let checker = unsafe { pointers::mut_ptr_as_ref(commutation_cache) };
        checker
            .commute(
                &op1,
                &[],
                qubits1,
                &[],
                &op2,
                &[],
                qubits2,
                &[],
                max_qubits,
                approximation_degree,
            )
            .unwrap()
    }
}

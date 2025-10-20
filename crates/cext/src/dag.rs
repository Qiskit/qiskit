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

use crate::pointers::{const_ptr_as_ref, mut_ptr_as_ref};

use qiskit_circuit::bit::{ClassicalRegister, QuantumRegister};
use qiskit_circuit::dag_circuit::DAGCircuit;

/// @ingroup QkDag
/// Construct a new empty DAG.
///
/// You must free the returned DAG with qk_dag_free when done with it.
///
/// @return A pointer to the created DAG.
///
/// # Example
/// ```c
/// QkDag *empty = qk_dag_new();
/// ```
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_dag_new() -> *mut DAGCircuit {
    let dag = DAGCircuit::new();
    Box::into_raw(Box::new(dag))
}

/// @ingroup QkDag
/// Add a quantum register to the DAG.
///
/// @param dag A pointer to the DAG.
/// @param reg A pointer to the quantum register.
///
/// # Example
/// ```c
/// QkDag *dag = qk_dag_new();
/// QkQuantumRegister *qr = qk_quantum_register_new(1024, "my_register");
/// qk_dag_add_quantum_register(dag, qr);
/// qk_quantum_register_free(qr);
/// qk_dag_free(dag);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag`` and
/// if ``reg`` is not a valid, non-null pointer to a ``QkQuantumRegister``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_dag_add_quantum_register(
    dag: *mut DAGCircuit,
    reg: *const QuantumRegister,
) {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let dag = unsafe { mut_ptr_as_ref(dag) };
    let qreg = unsafe { const_ptr_as_ref(reg) };

    dag.add_qreg(qreg.clone())
        .expect("Invalid register unable to be added to DAG");
}

/// @ingroup QkDag
/// Add a classical register to the DAG.
///
/// @param dag A pointer to the DAG.
/// @param reg A pointer to the classical register.
///
/// # Example
/// ```c
/// QkDag *dag = qk_dag_new();
/// QkClassicalRegister *cr = qk_classical_register_new(24, "my_register");
/// qk_dag_add_classical_register(dag, cr);
/// qk_classical_register_free(cr);
/// qk_dag_free(dag);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag`` and
/// if ``reg`` is not a valid, non-null pointer to a ``QkClassicalRegister``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_dag_add_classical_register(
    dag: *mut DAGCircuit,
    reg: *const ClassicalRegister,
) {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let dag = unsafe { mut_ptr_as_ref(dag) };
    let creg = unsafe { const_ptr_as_ref(reg) };

    dag.add_creg(creg.clone())
        .expect("Invalid register unable to be added to DAG");
}

/// @ingroup QkDag
/// Get the number of qubits the DAG contains.
///
/// @param dag A pointer to the DAG.
///
/// @return The number of qubits the DAG is defined on.
///
/// # Example
/// ```c
/// QkDag *dag = qk_dag_new();
/// QkQuantumRegister *qr = qk_quantum_register_new(24, "my_register");
/// qk_dag_add_quantum_register(dag, qr);
/// uint32_t num_qubits = qk_dag_num_qubits(dag);  // num_qubits==24
/// qk_quantum_register_free(qr);
/// qk_dag_free(dag);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_dag_num_qubits(dag: *const DAGCircuit) -> u32 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let dag = unsafe { const_ptr_as_ref(dag) };

    dag.num_qubits() as u32
}

/// @ingroup QkDag
/// Get the number of clbits the DAG contains.
///
/// @param dag A pointer to the DAG.
///
/// @return The number of clbits the DAG is defined on.
///
/// # Example
/// ```c
/// QkDag *dag = qk_dag_new();
/// QkClassicalRegister *cr = qk_classical_register_new(24, "my_register");
/// qk_dag_add_classical_register(dag, cr);
/// uint32_t num_clbits = qk_dag_num_clbits(dag);  // num_clbits==24
/// qk_classical_register_free(cr);
/// qk_dag_free(dag);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_dag_num_clbits(dag: *const DAGCircuit) -> u32 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let dag = unsafe { const_ptr_as_ref(dag) };

    dag.num_clbits() as u32
}

/// @ingroup QkDag
/// Free the DAG.
///
/// @param dag A pointer to the DAG to free.
///
/// # Example
/// ```c
/// QkDag *dag = qk_dag_new();
/// qk_dag_free(dag);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not either null or a valid pointer to a
/// ``QkDag``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_dag_free(dag: *mut DAGCircuit) {
    if !dag.is_null() {
        if !dag.is_aligned() {
            panic!("Attempted to free a non-aligned pointer.")
        }

        // SAFETY: We have verified the pointer is non-null and aligned, so it should be
        // readable by Box.
        unsafe {
            let _ = Box::from_raw(dag);
        }
    }
}

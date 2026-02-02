// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::exit_codes::ExitCode;
use crate::pointers::{const_ptr_as_ref, mut_ptr_as_ref};

use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::converters::dag_to_circuit;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_transpiler::commutation_checker::get_standard_commutation_checker;
use qiskit_transpiler::passes::cancel_commutations;
use qiskit_transpiler::target::Target;

/// @ingroup QkTranspilerPasses
/// Run the CommutativeCancellation transpiler pass on a circuit.
///
/// Refer to the ``qk_transpiler_pass_commutative_cancellation`` function for more details about the pass.
///
/// @param circuit A pointer to the circuit to run CommutativeCancellation on.
/// @param target Optional target for gate selection fallback.
/// @param approximation_degree The approximation degree used when analyzing commutations.
/// @returns The integer return code where 0 represents no error and 1 indicates an error.
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` or ``target`` is not a valid, ``QkCircuit`` and ``QkTarget``.
/// ``QkCircuit`` is not expected to be null and behavior is undefined if it is.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_transpiler_pass_standalone_commutative_cancellation(
    circuit: *mut CircuitData,
    target: *const Target,
    approximation_degree: f64,
) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { mut_ptr_as_ref(circuit) };
    let target = if target.is_null() {
        None
    } else {
        // SAFETY: Per documentation, the pointer is non-null and aligned.
        Some(unsafe { const_ptr_as_ref(target) })
    };
    if !(0.0..=1.0).contains(&approximation_degree) {
        panic!(
            "Invalid value provided for approximation degree, only NAN or values between 0.0 and 1.0 inclusive are valid"
        );
    }
    let mut dag = match DAGCircuit::from_circuit_data(circuit, false, None, None, None, None) {
        Ok(dag) => dag,
        Err(_) => panic!("Internal circuit -> DAG conversion failed"),
    };
    let mut commutation_checker = get_standard_commutation_checker();
    let basis = target.map(|t| t.operation_names().map(|n| n.to_string()).collect());
    if cancel_commutations(
        &mut dag,
        &mut commutation_checker,
        basis,
        approximation_degree,
    )
    .is_err()
    {
        return ExitCode::TranspilerError;
    }
    let out_circuit = match dag_to_circuit(&dag, false) {
        Ok(qc) => qc,
        Err(_) => panic!("Internal DAG -> circuit conversion failed"),
    };
    *circuit = out_circuit;
    ExitCode::Success
}

/// @ingroup QkTranspilerPasses
/// Run the CommutativeCancellation transpiler pass on a DAG Circuit.
///
/// This pass cancels the redundant (self-adjoint) gates through commutation relations.
///
/// @param dag A pointer to the DAG Circuit to run CommutativeCancellation on. This circuit
/// pointer to will be updated with the modified DAG Circuit if the pass is able to remove any gates.
/// @param target This pass will attempt to accumulate all Z rotations into either
/// an RZ, P or U1 gate, depending on which is already used in the circuit. If none
/// is present in the circuit, this (optional) target argument is used as fallback to
/// decide which gate to use. If none of RZ, P or U1 are in the circuit or the target,
/// single-qubit Z rotations will not be optimized.
/// @param approximation_degree The approximation degree used when
/// analyzing commutations. Must be within ``(0, 1]``.
/// @returns The integer return code where 0 represents no error and 1 is
/// used to indicate an error was encountered during the execution of the pass.
///
/// # Example
///
/// ```c
/// QkDag *dag = qk_dag_new();
/// QkQuantumRegister *qr = qk_quantum_register_new(4, "qr");
/// qk_dag_add_quantum_register(dag, qr);
///
/// uint32_t cx_qargs[2] = {0, 1};
/// qk_dag_apply_gate(dag, QkGate_CX, cx_qargs, NULL, false);
/// qk_dag_apply_gate(dag, QkGate_Z, (uint32_t[]){0}, NULL, false);
/// qk_dag_apply_gate(dag, QkGate_CX, cx_qargs, NULL, false);
/// qk_transpiler_pass_commutative_cancellation(dag, NULL, 1.0);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``dag`` or ``target`` is not a valid, ``QkDag`` and ``QkTarget``.
/// ``QkDag`` is not expected to be null and behavior is undefined if it is.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_transpiler_pass_commutative_cancellation(
    dag: *mut DAGCircuit,
    target: *const Target,
    approximation_degree: f64,
) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let mut dag_from_ptr = unsafe { mut_ptr_as_ref(dag) };
    let target = if target.is_null() {
        None
    } else {
        // SAFETY: Per documentation, the pointer is non-null and aligned.
        Some(unsafe { const_ptr_as_ref(target) })
    };
    if !(0.0..=1.0).contains(&approximation_degree) {
        panic!(
            "Invalid value provided for approximation degree, only values between 0.0 and 1.0 inclusive are valid"
        );
    }
    let mut commutation_checker = get_standard_commutation_checker();
    let basis = target.map(|t| t.operation_names().map(|n| n.to_string()).collect());
    if cancel_commutations(
        &mut dag_from_ptr,
        &mut commutation_checker,
        basis,
        approximation_degree,
    )
    .is_err()
    {
        return ExitCode::TranspilerError;
    }
    ExitCode::Success
}

#[cfg(test)]
mod tests {
    use super::*;
    use qiskit_circuit::Qubit;
    use qiskit_circuit::bit::QuantumRegister;
    use qiskit_circuit::bit::ShareableQubit;
    use qiskit_circuit::circuit_data::CircuitData;
    use qiskit_circuit::operations::StandardGate;

    #[test]
    fn test_commutative_cancellation() {
        let mut qc = CircuitData::new(
            Some((0..2).map(|_| ShareableQubit::new_anonymous()).collect()),
            None,
            (0.).into(),
        )
        .unwrap();
        qc.push_standard_gate(StandardGate::CX, &[], &[Qubit(0), Qubit(1)])
            .unwrap();
        qc.push_standard_gate(StandardGate::Z, &[], &[Qubit(0)])
            .unwrap();
        qc.push_standard_gate(StandardGate::CX, &[], &[Qubit(0), Qubit(1)])
            .unwrap();
        let result = unsafe {
            qk_transpiler_pass_standalone_commutative_cancellation(&mut qc, std::ptr::null(), 1.0)
        };
        assert_eq!(result, ExitCode::Success);
        assert_eq!(qc.__len__(), 1);
        let Some(gate) = qc.data()[0].op.try_standard_gate() else {
            panic!("Not a standard gate");
        };
        assert_eq!(StandardGate::Z, gate);
    }

    #[test]
    fn test_dag_commutative_cancellation() {
        let mut dag = DAGCircuit::new();
        dag.add_qreg(QuantumRegister::new_owning("q", 2)).unwrap();
        dag.apply_operation_back(
            StandardGate::CX.into(),
            &[Qubit(0), Qubit(1)],
            &[],
            None,
            None,
            #[cfg(feature = "cache_pygates")]
            None,
        )
        .unwrap();
        dag.apply_operation_back(
            StandardGate::Z.into(),
            &[Qubit(0)],
            &[],
            None,
            None,
            #[cfg(feature = "cache_pygates")]
            None,
        )
        .unwrap();
        dag.apply_operation_back(
            StandardGate::CX.into(),
            &[Qubit(0), Qubit(1)],
            &[],
            None,
            None,
            #[cfg(feature = "cache_pygates")]
            None,
        )
        .unwrap();
        let result = unsafe {
            qk_transpiler_pass_commutative_cancellation(
                &mut dag,
                std::ptr::null(),
                1.0,
            )
        };
        assert_eq!(result, ExitCode::Success);
        assert_eq!(dag.num_ops(), 1);
        let (_, instruction) = dag.op_nodes(false).next().unwrap();
        let Some(gate) = instruction.op.try_standard_gate() else {
            panic!("Not a standard gate");
        };
        assert_eq!(StandardGate::Z, gate);
    }
}

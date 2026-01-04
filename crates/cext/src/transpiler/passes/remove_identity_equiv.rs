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

use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::converters::dag_to_circuit;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_transpiler::passes::run_remove_identity_equiv;
use qiskit_transpiler::target::Target;

/// @ingroup QkTranspilerPasses
/// Run the RemoveIdentityEquivalent transpiler pass on a circuit.
///
/// Removes gates whose effect is close to an identity operation up to a global phase
/// and up to the specified tolerance. Parameterized gates are not considered by this pass.
///
/// For a cutoff fidelity \f$f\f$, this pass removes gates whose average
/// gate fidelity with respect to the identity is below \f$f\f$. Concretely,
/// a gate \f$G\f$ is removed if \f$\bar F < f\f$ where
///
/// \f[
/// bar{F} = \frac{1 + d F_{\text{process}}}{1 + d},\
///
/// F_{\text{process}} = \frac{|\mathrm{Tr}(G)|^2}{d^2}
/// \f]
///
/// where \f$d = 2^n\f$ is the dimension of the gate for \f$n\f$ qubits.
///
/// @param circuit A pointer to the circuit to run RemoveIdentityEquivalent on. This circuit
/// pointed to will be updated with the modified circuit if the pass is able to remove any gates.
/// @param target The target for the RemoveIdentityEquivalent pass. If ``approximation_degree`` is set to
/// ``NAN`` the tolerance for determining whether an operation is equivalent to
/// identity will be set to the reported error rate in the target. Otherwise
/// the ``target`` is not used as the tolerance is independent of the target.
/// @param approximation_degree The degree to approximate for the equivalence check. This can be a
/// floating point value between 0 and 1, or ``NAN``. If the value is 1 this does not
/// approximate above the floating point precision. For a value < 1 this is used as a
/// scaling factor for the cutoff fidelity. If the value is ``NAN`` this approximates up
/// to the fidelity for the gate specified in ``target``.
///
/// # Example
///
/// ```c
///     QkTarget *target = qk_target_new(5);
///     uint32_t current_num_qubits = qk_target_num_qubits(target);
///     QkTargetEntry *cx_entry = qk_target_entry_new(QkGate_CX);
///     for (uint32_t i = 0; i < current_num_qubits - 1; i++) {
///         uint32_t qargs[2] = {i, i + 1};
///         double inst_error = 0.0090393 * (current_num_qubits - i);
///         double inst_duration = 0.020039;
///         qk_target_entry_add_property(cx_entry, qargs, 2, inst_duration, inst_error);
///     }
///     QkExitCode result_cx = qk_target_add_instruction(target, cx_entry);
///     QkCircuit *qc = qk_circuit_new(4, 0);
///     for (uint32_t i = 0; i < qk_circuit_num_qubits(qc) - 1; i++) {
///         uint32_t qargs[2] = {i, i + 1};
///         for (uint32_t j = 0; j<i+1; j++) {
///             qk_circuit_gate(qc, QkGate_CX, qargs, NULL);
///         }
///     }
///     uint32_t rz_qargs[1] = {1,};
///     double rz_params[1] = {0.,};
///     qk_circuit_gate(qc, QkGate_RZ, rz_qargs, rz_params);
///     qk_transpiler_pass_standalone_remove_identity_equivalent(qc, target, 1.0);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` or ``target`` is not a valid, non-null pointer to a ``QkCircuit`` and ``QkTarget``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_transpiler_pass_standalone_remove_identity_equivalent(
    circuit: *mut CircuitData,
    target: *const Target,
    approximation_degree: f64,
) {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { mut_ptr_as_ref(circuit) };
    let target = unsafe { const_ptr_as_ref(target) };
    let mut dag = match DAGCircuit::from_circuit_data(circuit, false, None, None, None, None) {
        Ok(dag) => dag,
        Err(e) => panic!("{}", e),
    };
    let approximation_degree = if approximation_degree.is_nan() {
        None
    } else {
        if !(0.0..=1.0).contains(&approximation_degree) {
            panic!(
                "Invalid value provided for approximation degree, only NAN or values between 0.0 and 1.0 inclusive are valid"
            );
        }
        Some(approximation_degree)
    };

    run_remove_identity_equiv(&mut dag, approximation_degree, Some(target))
        .unwrap_or_else(|_| panic!("Remove identity equiv failed."));
    let out_circuit = match dag_to_circuit(&dag, false) {
        Ok(qc) => qc,
        Err(e) => panic!("{}", e),
    };
    *circuit = out_circuit;
}

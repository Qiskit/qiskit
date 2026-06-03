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

use crate::pointers::const_ptr_as_ref;
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit_library::qdrift::qdrift_evolution;
use qiskit_quantum_info::sparse_observable::SparseObservable;

/// @ingroup QkCircuitLibrary
///
/// Implements the QDrift Trotterization method, which selects Hamiltonian terms
/// randomly with probability proportional to their absolute coefficients.
///
/// This implementation follows the method introduced by Earl Campbell [1].
///
/// @param obs
/// Pointer to a valid QkObs. Requirements:
/// - All coefficients c_j must be real (imaginary parts numerically zero).
/// - Terms must be Pauli operators only (no projectors).
///   Projectors can be computationally inefficient: a term containing n
///   projectors may expand to 2^n Pauli terms. If your observable contains
///   projectors, consider decomposing it into Pauli terms first.
///
/// @param reps
/// Number of outer repetitions (independent segments). Must be strictly
/// positive. A value of reps == 0 is rejected with a non-success ExitCode.
///
/// @param time
/// Evolution time t in exp(-i t H). May be positive, negative, or zero.
/// The target gate count scales quadratically in |t|.
///
/// @param[out] out
/// Output parameter. On success, *out is set to a newly allocated circuit.
/// On failure, *out is set to NULL.
///
/// @return
/// ExitCode::Success on success. Otherwise a non-success ExitCode if:
/// - obs is NULL or otherwise invalid (e.g. contains projectors),
/// - reps is zero,
/// - the observable contains non-real coefficients,
/// - an internal allocation or conversion fails.
///
/// @details
/// Behavior and guarantees:
/// - Gate count scaling: for lambda = sum_j |c_j|, the target gate count is
///   N = ceil(2 * lambda^2 * t^2 * reps).
/// - Identity terms: pure-identity Hamiltonians (e.g. H = I) produce circuits
///   with no nontrivial instructions and a global phase equal to the
///   analytically expected value (e.g. -t for H = I).
/// - Stochasticity: repeated calls with identical inputs may produce different
///   gate sequences, but share the same distribution over term types and the
///   same expected gate count.
///
/// @warning
/// # Safety
/// Safety assumptions:
/// - obs must be a valid pointer managed by the Qiskit C API.
/// - out must be a valid writable pointer to a QkCircuit* location.
///  
/// Violating these conditions may cause undefined behavior.
///
/// @par Example
/// @code{.c}
/// // 2-qubit observable H = XI + ZZ
/// // QkObs *obs = qk_obs_zero(2);
///
/// // Term 1: X on qubit 1 (XI).
/// QkBitTerm bit_term_1[1] = { QkBitTerm_X };
/// QkComplex64 coeff_1 = { 1, 0 };
/// uint32_t indices_1[1] = { 1 };
/// QkObsTerm term_1 = { coeff_1, 1, bit_term_1, indices_1, 2 };
/// code = qk_obs_add_term(obs, &term_1);
///
/// // Term 2: ZZ on qubits {0,1}.
/// QkBitTerm bit_term_2[2] = { QkBitTerm_Z, QkBitTerm_Z };
/// QkComplex64 coeff_2 = { 1, 0 };
/// uint32_t indices_2[2] = { 0, 1 };
/// QkObsTerm term_2 = { coeff_2, 2, bit_term_2, indices_2, 2 };
/// code = qk_obs_add_term(obs, &term_2);
///
/// QkCircuit *circ = NULL;
/// code = qk_qdrift(obs, 1, 0.5, &circ);
/// @endcode
///
/// @par References
/// - [1] E. Campbell, "A random compiler for fast Hamiltonian simulation",
///   Phys. Rev. Lett. 123, 070503 (2019).
///   https://arxiv.org/abs/1811.08017
///
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_qdrift(
    obs: *const SparseObservable, // H in e^{-i t H}
    reps: usize,                  // n in e^{-it/n H}^n
    time: f64,                    // evolution time e in e^{-i t H}
    seed: u64,
    preserve_order: bool,
    insert_barriers: bool,
) -> *mut CircuitData {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let obs = unsafe { const_ptr_as_ref(obs) };

    let seed = if seed == 0 { None } else { Some(seed) };

    match qdrift_evolution(
        obs,
        time,
        reps as u32,
        seed,
        preserve_order,
        insert_barriers,
    ) {
        Ok(circuit) => Box::into_raw(Box::new(circuit)),
        Err(_) => std::ptr::null_mut(),
    }
}

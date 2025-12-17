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

use crate::exit_codes::ExitCode;
use crate::pointers::{const_ptr_as_ref, mut_ptr_as_ref};

use num_complex::Complex64;

use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::operations::Param;
use qiskit_circuit_library::pauli_evolution::Instruction;
use qiskit_circuit_library::pauli_evolution::sparse_term_evolution;
use qiskit_quantum_info::sparse_observable::SparseObservable;

use rand::distr::{Distribution, weighted::WeightedIndex};

use std::ptr;

/// Internal helper to extract real part of a complex number,
/// returning an error if imaginary part is non-zero.
fn approx_real(z: &Complex64) -> Result<f64, ExitCode> {
    if z.im.abs() > 1e-12 {
        return Err(ExitCode::CInputError);
    }
    Ok(z.re)
}

/// This is an internal helper function to build the QDrift circuit.
/// The same safety requirements as for `qk_circuit_library_qdrift` apply.
fn qdrift_build_circuit(
    obs: &SparseObservable,
    reps: usize,
    time: f64,
) -> Result<CircuitData, ExitCode> {
    let num_qubits = obs.num_qubits();
    if reps == 0 || num_qubits == 0 || obs.coeffs().is_empty() {
        return Err(ExitCode::CInputError);
    }

    // If the observable contains projectors, return an error. 
    // User should explicitly opt in to Pauli decomposition.
    if obs.bit_terms().iter().any(|b| b.is_projector()) {
        return Err(ExitCode::CInputError);
    }

    let bit_terms = obs.bit_terms();
    let boundaries = obs.boundaries();
    let coeffs = obs.coeffs();
    let indices = obs.indices();

    let n_terms = coeffs.len();
    let mut mags: Vec<f64> = Vec::with_capacity(n_terms);
    let mut signs: Vec<f64> = Vec::with_capacity(n_terms);
    let mut lambda = 0.0f64;
    let mut kappa = 0.0f64;

    for (i, &c) in obs.coeffs().iter().enumerate() {
        let start = boundaries[i];
        let end = boundaries[i + 1];
        let is_identity = start == end;

        let r = approx_real(&c);
        let r = match r {
            Err(e) => return Err(e),
            Ok(r) => r,
        };

        if is_identity {
            kappa += r;
            mags.push(0.0);
            signs.push(0.0);
            continue;
        } else {
            // artificially make weights positive
            let m = r.abs();
            if m == 0.0 {
                mags.push(0.0);
                signs.push(0.0);
                continue;
            }
            mags.push(m);
            signs.push(r.signum());
            lambda += m;
        }
    }

    // Zero hamiltonian
    let global_phase = Param::Float(-time * kappa);
    if lambda == 0.0 {
        return CircuitData::from_packed_operations(
            num_qubits,
            0,
            std::iter::empty(),
            global_phase,
        )
        .map_err(|_| ExitCode::ArithmeticError);
    }

    let num_gates = (2.0 * lambda.powi(2) * time.powi(2) * reps as f64).ceil() as usize;

    let probs: Vec<f64> = mags.iter().map(|m| m / lambda).collect();

    let dist = WeightedIndex::new(&probs);
    let dist = match dist {
        Err(_) => return Err(ExitCode::CInputError),
        Ok(dist) => dist,
    };
    // let mut rng = StdRng::seed_from_u64(seed as u64);
    let mut rng = rand::rng();

    let rescaled_time = 2.0 * lambda / num_gates as f64 * time;

    let evos = (0..num_gates)
        .map(|_| dist.sample(&mut rng))
        .filter_map(|i| {
            let start = boundaries[i];
            let end = boundaries[i + 1];
            if start == end {
                return None; // identity term: skip
            }

            let term_bits = &bit_terms[start..end];
            let term_indices = &indices[start..end];
            debug_assert_eq!(term_bits.len(), term_indices.len());

            let theta = signs[i] * rescaled_time;

            // length of pauli string is number of terms
            let mut pauli_string = String::with_capacity(term_bits.len());
            for bit in term_bits.iter() {
                pauli_string.push_str(bit.py_label());
            }

            Some((pauli_string, theta, term_indices.to_vec()))
        })
        .flat_map(move |(pauli_string, theta, idxs)| {
            let insts: Vec<Instruction> = sparse_term_evolution(
                pauli_string.as_str(),
                idxs,
                Param::Float(theta),
                false,
                false,
            )
            .collect();

            insts.into_iter().map(Ok::<Instruction, _>)
        });

    CircuitData::from_packed_operations(num_qubits, 0, evos, global_phase)
        .map_err(|_| ExitCode::ArithmeticError)
}

/// The QDrift Trotterization method, which selects each term in the Trotterization randomly,
/// with a probability proportional to its weight.
/// Based on the work of Earl Campbell in Ref. [1].
///
/// # Parameters
///
/// - `obs`: Pointer to a valid `QkObs` with the following conditions?
///   - Contain only *real* coefficients `c_j` (imaginary parts must be
///     numerically zero).
///   - Projectors are computationally inefficient - For example, the decomposition acting on a
///     term with :math:`n` projectors will use :math:`2^n` Pauli terms.
/// - `reps`: The number of outer repetitions `n` in the product formula,
///   i.e. QDRIFT approximates `exp(-i * time * H)` by a product of
///   `reps` independently sampled segments.  Must be strictly positive;
///   `reps == 0` is rejected with a non-success `ExitCode`.
/// - `time`: Evolution time `t` in `exp(-i * t * H)`.  This is a real
///   scalar and may be positive, negative, or zero.  The total number of
///   gates scales quadratically in `|time|`.
/// - `out`: Output parameter. On success, `*out` is set to point to a newly
///   allocated circuit object (*QkCircuitData) representing the synthesized
///   QDRIFT circuit.  On failure, `*out` is set to `NULL`.
///
/// # Returns
///
/// - `ExitCode::Success` if a circuit was successfully synthesized and
///   written to `*out`.
/// - A non-success `ExitCode` if:
///   - `obs` is null or otherwise invalid,
///   - `reps` is zero,
///   - the observable contains non-real coefficients,
///   - an internal allocation or conversion fails.
///
/// In all error cases, `*out` is set to `NULL` and no circuit is leaked.
///
/// # Behavior and guarantees
///
/// - **Gate count scaling**: For a Hamiltonian with :math:`\lambda = \sum_j |c_j|`, the
///   target gate count is :math:`N = \lceil 2 \lambda^2 \, \text{time}^2 \, \text{reps} \rceil`.  For small
///   examples this yields, for instance, `N = 1` for :math:`\lambda = 1`, `time = 0.5`,
///   `reps = 1`; `N = 2` for :math:`\lambda = 1`, `time = 1.0`, `reps = 1`; and
///   `N = 4` for the same Hamiltonian with `reps = 2`.
/// - **Identity terms**: Pure-identity Hamiltonians (e.g. `H = I`) produce
///   circuits with zero nontrivial instructions and a nonzero global phase
///   equal to the analytically expected value (e.g. `-time` for `H = I`).
/// - **Stochasticity**: Because term indices are sampled randomly,
///   different calls with the same `(obs, reps, time)` may produce
///   different gate *sequences*, but they will share the same distribution
///   over term types and the same expected gate count.
///
/// # Safety
///
/// This function assumes:
///
/// - `obs` is a valid pointer to an observable allocated and managed by the
///   Qiskit C API.
/// - `out` is a valid, writable pointer to a location that can hold a
///   `*mut CircuitData`.
///
/// Violating these conditions may cause undefined behavior.
///
/// # References
/// - [1]: E. Campbell, “A random compiler for fast Hamiltonian simulation” (2018).[https://arxiv.org/abs/1811.08017](https://arxiv.org/abs/1811.08017)
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_circuit_library_qdrift(
    obs: *const SparseObservable, // H in e^{-i t H}
    reps: usize,                  // n in e^{-it/n H}^n
    // insert_barriers: bool, // TODO wait for C implementation of barriers
    // cx_structure: CXStructure, // use chain by default
    // atomic_evolution: AtomicEvolution, // TODO add later
    // seed: usize, // TODO use seed or ThreadRng from rand::rng()
    // wrap: bool,  // TODO wait for C support
    // preserve_order: bool, // TODO add later
    // atomic_evolution_sparse_observable: bool // TODO add later
    time: f64,                  // evolution time e in e^{-i t H}
    out: *mut *mut CircuitData, // output pointer to CircuitData
) -> ExitCode {
    // Safety: obs must be a valid pointer to a QkObs object created by Rust code.
    let obs = unsafe { const_ptr_as_ref(obs) };
    let out_ref = unsafe { mut_ptr_as_ref(out) };

    *out_ref = ptr::null_mut();

    match qdrift_build_circuit(obs, reps, time) {
        Ok(circuit) => {
            let boxed = Box::new(circuit);
            *out_ref = Box::into_raw(boxed);
        }
        Err(e) => return e,
    }

    ExitCode::Success
}

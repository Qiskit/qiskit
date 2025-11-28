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

use rand::distr::{
    Distribution,
    weighted::WeightedIndex
};

use std::ptr;


fn approx_real(z: &Complex64) -> f64 {
    if z.im.abs() > 1e-12 {
        panic!("Complex coefficient with imaginary part encountered in QDRIFT.")
    }
    z.re
}

fn qdrift_build_circuit(
    obs: &SparseObservable,
    reps: usize,
    time: f64,
) -> Result<CircuitData, ExitCode> {
    let num_qubits = obs.num_qubits();
    if reps == 0 || num_qubits == 0 || obs.coeffs().is_empty() {
        return Err(ExitCode::CInputError)
    }

    // If the observable contains projectors, convert to Paulis
    let pauli_obs = obs.as_paulis();
    let obs = &pauli_obs;
    
    let bit_terms = obs.bit_terms();
    let boundaries = obs.boundaries();
    let indices = obs.indices();

    let n_terms = obs.coeffs().len();
    let mut mags: Vec<f64> = Vec::with_capacity(n_terms);
    let mut signs: Vec<f64> = Vec::with_capacity(n_terms);
    let mut lambda = 0.0f64;
    
    for &c in obs.coeffs() {
        let r = approx_real(&c);
        // artificially make weights positive
        let m = r.abs();
        if m == 0.0 {
            mags.push(0.0);
            signs.push(0.0);
            continue
        }
        mags.push(m);
        signs.push(r.signum());
        lambda += m;
    }
    
    // Zero hamiltonian
    if lambda == 0.0 {
        return Err(ExitCode::Success)
    }

    let num_gates = (2.0 * lambda.powi(2) * time.powi(2) * reps as f64).ceil() as usize;

    let probs: Vec<f64> = mags.iter().map(|m| m / lambda).collect();

    let dist = WeightedIndex::new(&probs).unwrap();
    // let mut rng = StdRng::seed_from_u64(seed as u64);
    let mut rng = rand::rng();

    let rescaled_time = 2.0 * lambda / num_gates as f64 * time;

    // sample term indices (0..n_terms)
    let sampled_indices: Vec<usize> =
        (0..num_gates).map(|_| dist.sample(&mut rng)).collect();
    let mut pauli_strings: Vec<String> = Vec::with_capacity(num_gates);
    let mut pauli_indices: Vec<Vec<u32>> = Vec::with_capacity(num_gates);
    let mut pauli_thetas: Vec<Param> = Vec::with_capacity(num_gates);
    for i in sampled_indices {
        let start = boundaries[i];
        let end = boundaries[i + 1];

        let term_bits = &bit_terms[start..end];
        let term_indices = &indices[start..end];
        debug_assert_eq!(term_bits.len(), term_indices.len());

        let theta = signs[i] * rescaled_time;

        let mut pauli_chars = vec!["I"; num_qubits as usize];
        for (bit, &idx) in term_bits.iter().zip(term_indices.iter()) {
            pauli_chars[idx as usize] = bit.py_label();
        }
        let pauli_string: String = pauli_chars.join("");
        let dense_indices: Vec<u32> = (0..num_qubits).collect();

        pauli_strings.push(pauli_string);
        pauli_indices.push(dense_indices);
        pauli_thetas.push(Param::Float(theta));
    }

    // Construct the evolutions
    let evos = pauli_strings.iter()
        .zip(pauli_indices.into_iter())
        .zip(pauli_thetas.into_iter())
        .flat_map(move |((pauli_str, idxs), time_param)| {
            let inner = sparse_term_evolution(
                pauli_str.as_str(),
                idxs,
                time_param,
                false,  // no phase gate for Paulis
                false,  // use chain CX structure
            );
            // This will never return a PyErr, so we can safely infer the error type
            // Infallible will not work directly since CircuitData::from_packed_operations expects
            // the items in the iterator to be PyResult<_>
            inner.map(|inst: Instruction| Ok::<Instruction, _>(inst))
        });
    
    let global_phase = Param::Float(0.0);
    
    // use CircuitData::from_packed_operations(num_qubits as u32, 0, evos, global_phase);
    // to assemble the circuit
    CircuitData::from_packed_operations(
        num_qubits as u32,
        0,
        evos,
        global_phase,
    ).map_err(|_| ExitCode::ArithmeticError)
}

#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_circuit_library_qdrift(
    obs: *const SparseObservable, // H in e^{-i t H}
    reps: usize, // n in e^{-it/n H}^n
    // insert_barriers: bool, // TODO wait for C implementation of barriers
    // cx_structure: CXStructure, // use chain by default
    // atomic_evolution: AtomicEvolution, // TODO add later
    // seed: usize, // TODO use seed or ThreadRng from rand::rng()
    // wrap: bool,  // TODO wait for C support
    // preserve_order: bool, // TODO add later
    // atomic_evolution_sparse_observable: bool // TODO add later
    time: f64, // evolution time e in e^{-i t H}
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
        Err(e) => return e
    }

    return ExitCode::Success
}
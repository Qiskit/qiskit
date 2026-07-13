// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

mod chunks;
mod mcts;
mod pauli_network;

use pyo3::prelude::*;
use pyo3::types::{PyList, PyString, PyTuple};

use qiskit_circuit::circuit_data::{CircuitDataError, PyCircuitData};
use qiskit_circuit::operations::Param;
use qiskit_quantum_info::clifford::PauliListError;

use crate::QiskitError;
use crate::evolution::mcts::pauli_network_mcts_inner;
use crate::evolution::pauli_network::pauli_network_synthesis_inner;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum EvolutionSynthesisError {
    // wraps PauliList error
    #[error(transparent)]
    ErrorFromPauliList(#[from] PauliListError),

    // wraps CircuitData error
    #[error(transparent)]
    ErrorFromCircuitData(#[from] CircuitDataError),
}

impl From<EvolutionSynthesisError> for PyErr {
    fn from(error: EvolutionSynthesisError) -> Self {
        match error {
            EvolutionSynthesisError::ErrorFromPauliList(err) => {
                QiskitError::new_err(err.to_string())
            }
            EvolutionSynthesisError::ErrorFromCircuitData(err) => err.into(),
        }
    }
}

/// Expands the sparse pauli string representation to the full representation.
///
/// For example: for the input `sparse_pauli = "XY", qubits = [1, 3], num_qubits = 6`,
/// the function returns `"IXIYII"`.
fn expand_pauli(sparse_pauli: String, qubits: &[u32], num_qubits: usize) -> String {
    let mut v: Vec<char> = vec!['I'; num_qubits];
    for (q, p) in qubits.iter().zip(sparse_pauli.chars()) {
        v[*q as usize] = p;
    }
    v.into_iter().collect()
}

fn extract_paulis_and_angles(
    num_qubits: usize,
    pauli_network: &Bound<PyList>,
) -> PyResult<(Vec<String>, Vec<Param>)> {
    let mut paulis: Vec<String> = Vec::with_capacity(pauli_network.len());
    let mut angles: Vec<Param> = Vec::with_capacity(pauli_network.len());

    let allowed_chars = ['I', 'X', 'Y', 'Z'];

    // go over the input pauli network and extract a list of pauli rotations and
    // the corresponding rotation angles
    for item in pauli_network {
        let tuple = item.cast::<PyTuple>()?;

        let sparse_pauli: String = tuple.get_item(0)?.cast::<PyString>()?.extract()?;
        let qubits: Vec<u32> = tuple.get_item(1)?.extract()?;
        let angle: Param = tuple.get_item(2)?.extract()?;

        if sparse_pauli.chars().any(|c| !allowed_chars.contains(&c)) {
            return Err(QiskitError::new_err(format!(
                "Pauli network contains invalid Pauli string {sparse_pauli}"
            )));
        }

        paulis.push(expand_pauli(sparse_pauli, &qubits, num_qubits));
        angles.push(angle);
    }
    Ok((paulis, angles))
}

/// Calls Rustiq's pauli network synthesis algorithm and returns the
/// Qiskit circuit data with Clifford gates and rotations.
///
/// # Arguments
///
/// * py: a GIL handle, needed to add and negate rotation parameters in Python space.
/// * num_qubits: total number of qubits.
/// * pauli_network: pauli network represented in sparse format. It's a list
///     of triples such as `[("XX", [0, 3], theta), ("ZZ", [0, 1], 0.1)]`.
/// * optimize_count: if `true`, Rustiq's synthesis algorithms aims to optimize
///     the 2-qubit gate count; and if `false`, then the 2-qubit depth.
/// * preserve_order: whether the order of paulis should be preserved, up to
///     commutativity. If the order is not preserved, the returned circuit will
///     generally not be equivalent to the given pauli network.
/// * upto_clifford: if `true`, the final Clifford operator is not synthesized
///     and the returned circuit will generally not be equivalent to the given
///     pauli network. In addition, the argument `upto_phase` would be ignored.
/// * upto_phase: if `true`, the global phase of the returned circuit may differ
///     from the global phase of the given pauli network. The argument is considered
///     to be `true` when `upto_clifford` is `true`.
/// * resynth_clifford_method: describes the strategy to synthesize the final
///     Clifford operator. If `0` a naive approach is used, which doubles the number
///     of gates but preserves the global phase of the circuit. If `1`, the Clifford is
///     resynthesized using Qiskit's greedy Clifford synthesis algorithm. If `2`, it
///     is resynthesized by Rustiq itself. If `upto_phase` is `false`, the naive
///     approach is used, as neither synthesis method preserves the global phase.
///
/// If `preserve_order` is `true` and both `upto_clifford` and `upto_phase` are `false`,
/// the returned circuit is equivalent to the given pauli network.
#[pyfunction]
#[pyo3(signature = (num_qubits, pauli_network, optimize_count=true, preserve_order=true, upto_clifford=false, upto_phase=false, resynth_clifford_method=1))]
#[allow(clippy::too_many_arguments)]
pub fn pauli_network_synthesis(
    num_qubits: usize,
    pauli_network: &Bound<PyList>,
    optimize_count: bool,
    preserve_order: bool,
    upto_clifford: bool,
    upto_phase: bool,
    resynth_clifford_method: usize,
) -> PyResult<PyCircuitData> {
    let (paulis, angles) = extract_paulis_and_angles(num_qubits, pauli_network)?;
    pauli_network_synthesis_inner(
        num_qubits,
        paulis,
        angles,
        optimize_count,
        preserve_order,
        upto_clifford,
        upto_phase,
        resynth_clifford_method,
    )
    .map(Into::into)
    .map_err(Into::into)
}

#[pyfunction]
#[pyo3(signature = (num_qubits, pauli_network, preserve_order=true, upto_clifford=false, upto_phase=false, num_simulations=0))]
#[allow(clippy::too_many_arguments)]
pub fn pauli_network_mcts(
    num_qubits: usize,
    pauli_network: &Bound<PyList>,
    preserve_order: bool,
    upto_clifford: bool,
    upto_phase: bool,
    num_simulations: usize,
) -> PyResult<PyCircuitData> {
    let (paulis, angles) = extract_paulis_and_angles(num_qubits, pauli_network)?;
    pauli_network_mcts_inner(
        num_qubits,
        paulis,
        angles,
        preserve_order,
        upto_clifford,
        upto_phase,
        num_simulations,
    )
    .map(Into::into)
    .map_err(Into::into)
}

pub fn evolution(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pauli_network_synthesis, m)?)?;
    m.add_function(wrap_pyfunction!(pauli_network_mcts, m)?)?;
    Ok(())
}

pub mod suzuki_trotter;

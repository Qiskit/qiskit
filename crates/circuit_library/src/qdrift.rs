// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use qiskit_circuit::Qubit;
use qiskit_circuit::circuit_data::{CircuitData, CircuitDataError};
use qiskit_circuit::operations::{Param, StandardInstruction, multiply_param, radd_param};
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_quantum_info::sparse_observable::{SparseObservable, SparseTermView};
use qiskit_synthesis::evolution::trotterization::{
    TrotterizationError, qdrift_evolution as qdrift, reorder_terms,
};
use qiskit_synthesis::pauli_evolution::{Instruction, sparse_term_evolution};
use smallvec::smallvec;
use thiserror::Error;

pub fn qdrift_evolution(
    observable: &SparseObservable,
    time: f64,
    reps: u32,
    seed: Option<u64>,
    preserve_order: bool,
    insert_barriers: bool,
) -> Result<CircuitData, EvolutionError> {
    let num_qubits = observable.num_qubits();

    let evo: Vec<(usize, f64)> = match qdrift(time, reps, seed, observable.coeffs().iter()) {
        Ok(evo) => evo,
        Err(err) => return Err(EvolutionError::FailedEvolutionError(err)),
    };

    let sampled_paulis = evo.iter().map(|(index, coeff)| {
        let start = observable.boundaries()[*index];
        let end = observable.boundaries()[index + 1];
        SparseTermView {
            num_qubits,
            coeff: coeff.into(),
            bit_terms: &observable.bit_terms()[start..end],
            indices: &observable.indices()[start..end],
        }
    });

    let sampled_paulis = if preserve_order || observable.bit_terms().len() <= 1 {
        sampled_paulis.collect()
    } else {
        match reorder_terms(sampled_paulis) {
            Ok(sampled_paulis) => sampled_paulis,
            Err(msg) => return Err(EvolutionError::TermsReorder(msg.to_string())),
        }
    };

    let mut global_phase = Param::Float(0.0);
    let mut modified_phase = false;

    let instructions = sampled_paulis.iter().enumerate().flat_map(|(i, view)| {
        let coeff_param: Param = view.coeff.re.into();
        if view.bit_terms.is_empty() {
            global_phase = radd_param(global_phase.clone(), coeff_param.clone());
            modified_phase = true;
        }
        let inst_iter = sparse_term_evolution(
            view.bit_terms
                .iter()
                .map(|bit| bit.py_label())
                .collect::<String>()
                .leak(),
            view.indices.into(),
            coeff_param,
            false,
            false,
        )
        .map(Ok::<Instruction, _>);

        let maybe_barrier = (insert_barriers && i != evo.len() - 1)
            .then_some(Ok((
                PackedOperation::from_standard_instruction(StandardInstruction::Barrier(
                    num_qubits,
                )),
                smallvec![],
                (0..num_qubits).map(Qubit).collect(),
                vec![],
            )))
            .into_iter();
        inst_iter.chain(maybe_barrier)
    });

    match CircuitData::from_packed_operations(
        observable.num_qubits(),
        0,
        instructions,
        Param::Float(0.0),
    ) {
        Ok(mut circuit) => {
            if modified_phase {
                let _ = circuit.set_global_phase_param(multiply_param(&global_phase, -0.5));
            }
            Ok(circuit)
        }
        Err(err) => Err(EvolutionError::CircuitBuild(err)),
    }
}

#[derive(Debug, Error)]
pub enum EvolutionError {
    /// A general error for terms reordering error
    #[error["Error ocurred when trying to reorder terms: {0}"]]
    TermsReorder(String),

    /// A general error when trying to build circuit from generated instructions
    #[error["Failed building circuit"]]
    CircuitBuild(#[from] CircuitDataError),

    #[error["qDrift evolution failed"]]
    FailedEvolutionError(#[from] TrotterizationError),
}

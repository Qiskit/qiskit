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

use crate::pauli_evolution::{Instruction, sparse_term_evolution};
use qiskit_circuit::Qubit;
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::operations::{Param, StandardInstruction, multiply_param, radd_param};
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_quantum_info::sparse_observable::SparseObservable;
use qiskit_synthesis::evolution::suzuki_trotter::{
    StrSparseTerm, bit_term_as_char, evolution, reorder_terms,
};
use smallvec::smallvec;

pub fn suzuki_trotter_evolution(
    observable: &SparseObservable,
    order: u32,
    reps: u32,
    time: f64,
    preserve_order: bool,
    insert_barriers: bool,
) -> Result<CircuitData, String> {
    if order > 1 && order % 2 == 1 {
        return Err(format!(
            "Suzuki product formulae are symmetric and therefore only defined \
            for when the order is 1 or even, not {}.",
            order
        ));
    }

    let parsed_observable: Vec<StrSparseTerm> = observable
        .coeffs()
        .iter()
        .enumerate()
        .map(|(i, coeff)| {
            let start = observable.boundaries()[i];
            let end = observable.boundaries()[i + 1];
            let mut coeff = *coeff;
            coeff.re = coeff.re * time * 2.0 / (reps as f64);
            StrSparseTerm {
                terms: observable.bit_terms()[start..end]
                    .iter()
                    .map(bit_term_as_char)
                    .collect(),
                coeff: coeff,
                indices: &observable.indices()[start..end],
            }
        })
        .collect();

    // execute evolution
    let evo: Vec<StrSparseTerm> = evolution(
        order,
        if preserve_order || !preserve_order && parsed_observable.len() <= 1 {
            parsed_observable
        } else {
            reorder_terms(&parsed_observable)
        },
    );

    let mut global_phase = Param::Float(0.0);
    let mut modified_phase = false;

    // convert terms into instructions for circuit creation
    let repetitions = evo.len() as u32 * reps;
    let iter_repeated = evo.iter().cycle().take(repetitions as usize);

    for view in iter_repeated.clone() {
        if view.terms.chars().all(|p| p == 'i') {
            global_phase = radd_param(global_phase, view.coeff.re.into());
            modified_phase = true;
        }
    }

    let repeated_evo = iter_repeated
        .enumerate()
        .map(|(i, view)| {
            let instructions = sparse_term_evolution(
                &view.terms,
                view.indices.into(),
                view.coeff.re.into(),
                false,
                false,
            )
            .map(Ok::<Instruction, _>);

            let maybe_barrier = (insert_barriers && i as u32 != repetitions - 1)
                .then_some(Ok(create_barrier(observable.num_qubits())))
                .into_iter();
            instructions.chain(maybe_barrier)
        })
        .flatten();

    if modified_phase {
        global_phase = multiply_param(&global_phase, -0.5);
    }

    match CircuitData::from_packed_operations(
        observable.num_qubits() as u32,
        0,
        repeated_evo,
        global_phase,
    ) {
        Ok(circuit) => Ok(circuit),
        Err(_) => Err("Failed building circuit".to_string()),
    }
}

fn create_barrier(num_qubits: u32) -> Instruction {
    (
        PackedOperation::from_standard_instruction(StandardInstruction::Barrier(num_qubits)),
        smallvec![],
        (0..num_qubits as u32).map(Qubit).collect(),
        vec![],
    )
}

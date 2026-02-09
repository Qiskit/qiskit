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
use qiskit_synthesis::evolution::suzuki_trotter::{evolution, reorder_terms};
use smallvec::smallvec;

pub fn suzuki_trotter_evolution(
    observable: &SparseObservable,
    order: u32,
    reps: u32,
    time: f64,
    preserve_order: bool,
    insert_barriers: bool,
) -> Result<CircuitData, String> {
    if order > 1 && order % 2 != 0 || order == 0 {
        return Err(format!(
            "Suzuki product formulae are symmetric and therefore only defined \
            for when the order is 1 or even, not {}.",
            order
        ));
    }

    let terms_iter = observable.iter().map(|mut view| {
        view.coeff.re *= time * 2.0 / (reps as f64);
        view
    });
    let terms = if preserve_order || !preserve_order && observable.bit_terms().len() <= 1 {
        terms_iter.collect()
    } else {
        reorder_terms(terms_iter)?
    };

    // execute evolution
    let evo: Vec<(usize, f64)> = evolution(order, terms.len());

    // convert terms into instructions for circuit creation
    let repetitions = evo.len() as u32 * reps;
    let iter_repeated = evo.iter().cycle().take(repetitions as usize);

    let mut global_phase = Param::Float(0.0);
    let mut modified_phase = false;
    for (index, coeff) in iter_repeated.clone() {
        let view = &terms[*index];
        if view.bit_terms.len() == 0 {
            global_phase = radd_param(global_phase, (view.coeff.re * coeff).into());
            modified_phase = true;
        }
    }

    let repeated_evo = iter_repeated
        .enumerate()
        .map(|(i, (index, coeff))| {
            let view = &terms[*index];
            let instructions = sparse_term_evolution(
                view.bit_terms
                    .iter()
                    .map(|bit| bit.py_label())
                    .collect::<String>()
                    .leak(),
                view.indices.into(),
                (view.coeff.re * coeff).into(),
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

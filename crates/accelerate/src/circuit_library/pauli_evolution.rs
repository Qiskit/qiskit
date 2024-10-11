// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use itertools::Itertools;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString, PyTuple};
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::operations::{radd_param, Param, StandardGate};
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::{Clbit, Qubit};
use smallvec::{smallvec, SmallVec};
use std::f64::consts::PI;

use crate::circuit_library::utils;

// custom math and types for a more readable code
const PI2: f64 = PI / 2.;
type StandardInstruction = (StandardGate, SmallVec<[Param; 3]>, SmallVec<[Qubit; 2]>);
type Instruction = (
    PackedOperation,
    SmallVec<[Param; 3]>,
    Vec<Qubit>,
    Vec<Clbit>,
);

/// Return instructions (using only StandardGate operations) to implement a Pauli evolution
/// of a given Pauli string over a given time (as Param).
///
/// The Pauli evolution is implemented as a basis transformation to the Pauli-Z basis,
/// followed by a CX-chain and then a single Pauli-Z rotation on the last qubit. Then the CX-chain
/// is uncomputed and the inverse basis transformation applied. E.g. for the evolution under the
/// Pauli string XIYZ we have the circuit
///                     ┌───┐┌───────┐┌───┐
/// 0: ─────────────────┤ X ├┤ Rz(2) ├┤ X ├──────────────────
///    ┌──────────┐┌───┐└─┬─┘└───────┘└─┬─┘┌───┐┌───────────┐
/// 1: ┤ Rx(pi/2) ├┤ X ├──■─────────────■──┤ X ├┤ Rx(-pi/2) ├
///    └──────────┘└─┬─┘                   └─┬─┘└───────────┘
/// 2: ──────────────┼───────────────────────┼───────────────
///     ┌───┐        │                       │  ┌───┐
/// 3: ─┤ H ├────────■───────────────────────■──┤ H ├────────
///     └───┘                                   └───┘
///
/// If ``phase_gate`` is ``true``, use the ``PhaseGate`` instead of ``RZGate`` as single-qubit
/// rotation.
pub fn pauli_evolution<'a>(
    pauli: &'a str,
    indices: Vec<u32>,
    time: Param,
    phase_gate: bool,
    // check_sparse: bool
) -> Box<dyn Iterator<Item = StandardInstruction> + 'a> {
    // ensure the Pauli has no identity terms
    let binding = pauli.to_lowercase(); // lowercase for convenience
    let active = binding
        .as_str()
        .chars()
        .rev()
        .zip(indices)
        .filter(|(pauli, _)| *pauli != 'i');
    let (paulis, indices): (Vec<char>, Vec<u32>) = active.unzip();

    match (phase_gate, indices.len()) {
        (_, 0) => Box::new(std::iter::empty()),
        (false, 1) => Box::new(single_qubit_evolution(paulis[0], indices[0], time)),
        (false, 2) => two_qubit_evolution(paulis.clone(), indices.clone(), time),
        _ => Box::new(multi_qubit_evolution(
            paulis.clone(),
            indices.clone(),
            time,
            phase_gate,
        )),
    }
}

fn single_qubit_evolution(
    pauli: char,
    index: u32,
    time: Param,
) -> impl Iterator<Item = StandardInstruction> {
    let qubit: SmallVec<[Qubit; 2]> = smallvec![Qubit(index)];
    let param: SmallVec<[Param; 3]> = smallvec![time.clone()];

    std::iter::once(match pauli {
        'x' => (StandardGate::RXGate, param, qubit),
        'y' => (StandardGate::RXGate, param, qubit),
        'z' => (StandardGate::RXGate, param, qubit),
        _ => unreachable!("Unsupported Pauli, at this point we expected one of x, y, z."),
    })
}

fn two_qubit_evolution<'a>(
    pauli: Vec<char>,
    indices: Vec<u32>,
    time: Param,
) -> Box<dyn Iterator<Item = StandardInstruction> + 'a> {
    let qubits: SmallVec<[Qubit; 2]> = smallvec![Qubit(indices[0]), Qubit(indices[1])];
    let param: SmallVec<[Param; 3]> = smallvec![time.clone()];
    let paulistring: String = pauli.iter().collect();

    match paulistring.as_str() {
        "xx" => Box::new(std::iter::once((StandardGate::RXXGate, param, qubits))),
        "zx" => Box::new(std::iter::once((StandardGate::RZXGate, param, qubits))),
        "yy" => Box::new(std::iter::once((StandardGate::RYYGate, param, qubits))),
        "zz" => Box::new(std::iter::once((StandardGate::RZZGate, param, qubits))),
        _ => Box::new(multi_qubit_evolution(pauli, indices, time, false)),
    }
}

fn multi_qubit_evolution(
    pauli: Vec<char>,
    indices: Vec<u32>,
    time: Param,
    phase_gate: bool,
) -> impl Iterator<Item = StandardInstruction> {
    // Get pairs of (pauli, qubit) that are active, i.e. that are not the identity. Note that
    // the rest of the code also works if there are only identities, in which case we will
    // effectively return an empty iterator.
    // let qubits: Vec<Qubit> = indices.iter().map(|i| Qubit(*i)).collect();
    // let binding = pauli.to_lowercase(); // lowercase for convenience
    // let active_paulis = binding
    //     .as_str()
    //     .chars()
    //     .rev() // reverse due to Qiskit's bit ordering convention
    //     .zip(qubits)
    //     .filter(|(p, _)| *p != 'i')
    //     .collect_vec();
    let active_paulis: Vec<(char, Qubit)> = pauli
        .into_iter()
        .zip(indices.into_iter().map(Qubit))
        .collect();

    // get the basis change: x -> HGate, y -> RXGate(pi/2), z -> nothing
    let basis_change = active_paulis
        .clone()
        .into_iter()
        .filter(|(p, _)| *p != 'z')
        .map(|(p, q)| match p {
            'x' => (StandardGate::HGate, smallvec![], smallvec![q]),
            'y' => (
                StandardGate::RXGate,
                smallvec![Param::Float(PI2)],
                smallvec![q],
            ),
            _ => unreachable!("Invalid Pauli string."), // "z" and "i" have been filtered out
        });

    // get the inverse basis change
    let inverse_basis_change = basis_change.clone().map(|(gate, _, qubit)| match gate {
        StandardGate::HGate => (gate, smallvec![], qubit),
        StandardGate::RXGate => (gate, smallvec![Param::Float(-PI2)], qubit),
        _ => unreachable!(),
    });

    // get the CX chain down to the target rotation qubit
    let chain_down = active_paulis
        .clone()
        .into_iter()
        .map(|(_, q)| q)
        .tuple_windows() // iterates over (q[i], q[i+1]) windows
        .map(|(ctrl, target)| (StandardGate::CXGate, smallvec![], smallvec![ctrl, target]));

    // get the CX chain up (cannot use chain_down.rev since tuple_windows is not double ended)
    let chain_up = active_paulis
        .clone()
        .into_iter()
        .rev()
        .map(|(_, q)| q)
        .tuple_windows()
        .map(|(target, ctrl)| (StandardGate::CXGate, smallvec![], smallvec![ctrl, target]));

    // get the RZ gate on the last qubit
    let last_qubit = active_paulis.last().unwrap().1;
    let z_rotation = std::iter::once((
        if phase_gate {
            StandardGate::PhaseGate
        } else {
            StandardGate::RZGate
        },
        smallvec![time],
        smallvec![last_qubit],
    ));

    // and finally chain everything together
    basis_change
        .chain(chain_down)
        .chain(z_rotation)
        .chain(chain_up)
        .chain(inverse_basis_change)
}

#[pyfunction]
#[pyo3(signature = (num_qubits, sparse_paulis, insert_barriers=false))]
pub fn py_pauli_evolution(
    py: Python,
    num_qubits: i64,
    sparse_paulis: &Bound<PyList>,
    insert_barriers: bool,
) -> PyResult<CircuitData> {
    let num_paulis = sparse_paulis.len();
    let mut paulis: Vec<String> = Vec::with_capacity(num_paulis);
    let mut indices: Vec<Vec<u32>> = Vec::with_capacity(num_paulis);
    let mut times: Vec<Param> = Vec::with_capacity(num_paulis);
    let mut global_phase = Param::Float(0.0);

    for el in sparse_paulis.iter() {
        let tuple = el.downcast::<PyTuple>()?;
        let pauli = tuple.get_item(0)?.downcast::<PyString>()?.to_string();
        let time = Param::extract_no_coerce(&tuple.get_item(2)?)?;

        if pauli.as_str().chars().all(|p| p == 'i') {
            global_phase = radd_param(global_phase, time, py); // apply factor -1 at the end
            continue;
        }

        paulis.push(pauli);
        times.push(time);
        indices.push(
            tuple
                .get_item(1)?
                .downcast::<PyList>()?
                .iter()
                .map(|index| index.extract::<u32>())
                .collect::<PyResult<_>>()?,
        );
    }

    let evos = paulis
        .iter()
        .zip(indices)
        .zip(times)
        .flat_map(|((pauli, qubits), time)| {
            let as_packed = pauli_evolution(pauli, qubits, time, false).map(
                |(gate, params, qubits)| -> PyResult<Instruction> {
                    Ok((
                        gate.into(),
                        params.clone(),
                        Vec::from_iter(qubits.into_iter()),
                        Vec::new(),
                    ))
                },
            );
            as_packed.chain(utils::maybe_barrier(py, num_qubits as u32, insert_barriers))
        });

    CircuitData::from_packed_operations(py, num_qubits as u32, 0, evos, global_phase)
}

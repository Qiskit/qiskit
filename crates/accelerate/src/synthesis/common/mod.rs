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

#![allow(dead_code)]

use qiskit_circuit::operations::{OperationRef, Param, StandardGate, StandardInstruction};
use smallvec::SmallVec;

/// A triple consisting of the circuit operation, its parameters, and the indices of the
/// qubits its defined on.
type SynthesisEntry<'a> = (OperationRef<'a>, SmallVec<[Param; 3]>, SmallVec<[usize; 2]>);

/// Represents the synthesized circuit. Ideally, this should be replaced by [CircuitData],
/// however at the moment [CircuitData] has too high of an overhead.
#[derive(Debug)]
pub struct SynthesisData<'a> {
    num_qubits: usize,
    data: Vec<SynthesisEntry<'a>>,
    global_phase: f64,
}

impl<'a> SynthesisData<'a> {
    /// Creates a new empty circuit.
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            data: Vec::new(),
            global_phase: 0.0,
        }
    }

    /// Appends a [StandardGate] to the  circuit.
    #[inline]
    pub fn push_standard_gate(
        &mut self,
        operation: StandardGate,
        params: &[Param],
        qargs: &[usize],
    ) {
        self.data.push((
            OperationRef::StandardGate(operation),
            params.into(),
            qargs.into(),
        ));
    }

    /// Appends a [StandardInstruction] to the circuit.
    #[inline]
    pub fn push_standard_instruction(
        &mut self,
        operation: StandardInstruction,
        params: &[Param],
        qargs: &[usize],
    ) {
        self.data.push((
            OperationRef::StandardInstruction(operation),
            params.into(),
            qargs.into(),
        ));
    }

    /// Composes ``other`` into ``self``, while optionally remapping the
    /// qubits over which ``other`` is defined.
    pub fn compose(&mut self, other: &Self, qubit_map: Option<&[usize]>) {
        for entry in &other.data {
            let remapped_qubits: SmallVec<[usize; 2]> = match qubit_map {
                Some(qubit_map) => (entry.2).iter().map(|q| qubit_map[*q]).collect(),
                None => entry.2.clone(),
            };

            match entry.0 {
                OperationRef::StandardGate(operation) => {
                    self.push_standard_gate(operation, &entry.1, &remapped_qubits);
                }
                OperationRef::StandardInstruction(operation) => {
                    self.push_standard_instruction(operation, &entry.1, &remapped_qubits);
                }
                _ => {
                    panic!("Other OperationRef types are not yet supported.");
                }
            }
        }

        self.global_phase += other.global_phase;
    }
}

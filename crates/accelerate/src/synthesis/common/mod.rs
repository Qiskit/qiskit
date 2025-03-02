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

use pyo3::{PyResult, Python};
use qiskit_circuit::{
    circuit_data::CircuitData,
    operations::{OperationRef, Param, StandardGate, StandardInstruction},
    Qubit,
};
use smallvec::SmallVec;

/// A triple consisting of the circuit operation, its parameters, and the indices of the
/// qubits its defined on.
// ToDo: will we ever need to reason about non-Float parameters? Should we then replace
// this by ``SmallVec<[f64; 3]>``?
type SynthesisEntry<'a> = (OperationRef<'a>, SmallVec<[Param; 3]>, SmallVec<[u32; 2]>);

/// Represents the synthesized circuit. Ideally, this should be replaced by [CircuitData],
/// however at the moment [CircuitData] has too high of an overhead.
#[derive(Debug)]
pub struct SynthesisData<'a> {
    num_qubits: u32,
    pub data: Vec<SynthesisEntry<'a>>,
    global_phase: f64,
}

impl<'a> SynthesisData<'a> {
    /// Creates a new empty circuit.
    pub fn new(num_qubits: u32) -> Self {
        Self {
            num_qubits,
            data: Vec::new(),
            global_phase: 0.0,
        }
    }

    /// Appends a [StandardGate] to the  circuit.
    #[inline]
    pub fn push_standard_gate(&mut self, operation: StandardGate, params: &[Param], qargs: &[u32]) {
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
        qargs: &[u32],
    ) {
        self.data.push((
            OperationRef::StandardInstruction(operation),
            params.into(),
            qargs.into(),
        ));
    }

    // /// Appends a [PyGate] to the circuit.
    // #[inline]
    // pub fn push_py_gate<'a>(
    //     &mut self,
    //     operation: &Bound<PyAny>,
    //     params: &[Param],
    //     qargs: &[u32],
    //     operation_name: String,
    //     num_params: u32,

    // ) {
    //     let as_py_gate = PyGate {
    //         qubits: qargs.len() as u32,
    //         clbits: 0,
    //         params: num_params,
    //         op_name: operation_name,
    //         gate: operation.clone().unbind(),
    //     };
    //     self.data.push((
    //         OperationRef::Gate(&as_py_gate),
    //         params.into(),
    //         qargs.into(),
    //     ));
    // }

    /// Composes ``other`` into ``self``, while optionally remapping the
    /// qubits over which ``other`` is defined.
    pub fn compose(&mut self, other: &Self, qubit_map: Option<&[u32]>) {
        for (op, params, qubits) in &other.data {
            let remapped_qubits: SmallVec<[u32; 2]> = match qubit_map {
                Some(qubit_map) => qubits.iter().map(|q| qubit_map[*q as usize]).collect(),
                None => qubits.clone(),
            };

            match op {
                OperationRef::StandardGate(operation) => {
                    self.push_standard_gate(*operation, params, &remapped_qubits);
                }
                OperationRef::StandardInstruction(operation) => {
                    self.push_standard_instruction(*operation, params, &remapped_qubits);
                }
                _ => {
                    panic!("Other OperationRef types are not yet supported.");
                }
            }
        }

        self.global_phase += other.global_phase;
    }

    /// Converts to [CircuitData].
    pub fn to_circuit_data(&self, py: Python) -> PyResult<CircuitData> {
        let mut circuit = CircuitData::with_capacity(
            py,
            self.num_qubits,
            0,
            self.data.len(),
            Param::Float(self.global_phase),
        )?;
        for (op, params, qubits) in &self.data {
            let circuit_qubits: Vec<Qubit> = qubits.iter().map(|q| Qubit(*q)).collect();
            match op {
                OperationRef::StandardGate(operation) => {
                    circuit.push_standard_gate(*operation, params, &circuit_qubits)?;
                }
                OperationRef::StandardInstruction(operation) => {
                    circuit.push_standard_instruction(*operation, params, &circuit_qubits, &[])?;
                }
                _ => {
                    panic!("Other OperationRef types are not yet supported.");
                }
            }
        }
        Ok(circuit)
    }

    /// Creates from [CircuitData].
    pub fn from_circuit_data(circuit_data: &'a CircuitData) -> SynthesisData<'a> {
        let mut circuit = SynthesisData::new(circuit_data.qubits().len() as u32);
        for inst in circuit_data.data() {
            circuit.data.push((
                inst.op.view(),
                inst.params_view().into(),
                circuit_data
                    .get_qargs(inst.qubits)
                    .into_iter()
                    .map(|q| q.index() as u32)
                    .collect::<Vec<u32>>()
                    .into(),
            ));
        }
        circuit
    }

    // Convenience functions

    /// Appends XGate to the circuit.
    #[inline]
    pub fn x(&mut self, q: u32) {
        self.push_standard_gate(StandardGate::XGate, &[], &[q]);
    }

    /// Appends HGate to the circuit.
    #[inline]
    pub fn h(&mut self, q: u32) {
        self.push_standard_gate(StandardGate::HGate, &[], &[q]);
    }

    /// Appends TGate to the circuit.
    #[inline]
    pub fn t(&mut self, q: u32) {
        self.push_standard_gate(StandardGate::TGate, &[], &[q]);
    }

    /// Appends TdgGate to the circuit.
    #[inline]
    pub fn tdg(&mut self, q: u32) {
        self.push_standard_gate(StandardGate::TdgGate, &[], &[q]);
    }

    /// Appends PhaseGate to the circuit.
    #[inline]
    pub fn p(&mut self, theta: f64, q: u32) {
        self.push_standard_gate(StandardGate::PhaseGate, &[Param::Float(theta)], &[q]);
    }

    /// Appends CXGate to the circuit.
    #[inline]
    pub fn cx(&mut self, q1: u32, q2: u32) {
        self.push_standard_gate(StandardGate::CXGate, &[], &[q1, q2]);
    }

    /// Appends CU1Gate to the circuit.
    #[inline]
    pub fn cu1(&mut self, theta: f64, q1: u32, q2: u32) {
        self.push_standard_gate(StandardGate::CU1Gate, &[Param::Float(theta)], &[q1, q2]);
    }
}

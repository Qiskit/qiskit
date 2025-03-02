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

// #![allow(dead_code)]

// use pyo3::{PyResult, Python};
// use qiskit_circuit::{
//     circuit_data::CircuitData,
//     operations::{Operation, OperationRef, Param, StandardGate, StandardInstruction},
//     Qubit,
// };
// use smallvec::SmallVec;

// /// A triple consisting of the circuit operation, its parameters, and the indices of the
// /// qubits its defined on.
// // ToDo: will we ever need to reason about non-Float parameters? Should we then replace
// // this by ``SmallVec<[f64; 3]>``?
// type SynthesisEntry<'a> = (OperationRef<'a>, SmallVec<[Param; 3]>, SmallVec<[u32; 2]>);

// /// Represents the synthesized circuit. Ideally, this should be replaced by [CircuitData],
// /// however at the moment [CircuitData] has too high of an overhead.
// #[derive(Debug)]
// pub struct SynthesisData<'a> {
//     num_qubits: u32,
//     pub data: Vec<SynthesisEntry<'a>>,
//     global_phase: f64,
// }

// impl<'a> SynthesisData<'a> {
//     /// Creates a new empty circuit.
//     pub fn new(num_qubits: u32) -> Self {
//         Self {
//             num_qubits,
//             data: Vec::new(),
//             global_phase: 0.0,
//         }
//     }

//     /// Appends a [StandardGate] to the  circuit.
//     #[inline]
//     pub fn push_standard_gate(&mut self, operation: StandardGate, params: &[Param], qargs: &[u32]) {
//         self.data.push((
//             OperationRef::StandardGate(operation),
//             params.into(),
//             qargs.into(),
//         ));
//     }

//     /// Appends a [StandardInstruction] to the circuit.
//     #[inline]
//     pub fn push_standard_instruction(
//         &mut self,
//         operation: StandardInstruction,
//         params: &[Param],
//         qargs: &[u32],
//     ) {
//         self.data.push((
//             OperationRef::StandardInstruction(operation),
//             params.into(),
//             qargs.into(),
//         ));
//     }

//     // /// Appends a [PyGate] to the circuit.
//     // #[inline]
//     // pub fn push_py_gate<'a>(
//     //     &mut self,
//     //     operation: &Bound<PyAny>,
//     //     params: &[Param],
//     //     qargs: &[u32],
//     //     operation_name: String,
//     //     num_params: u32,

//     // ) {
//     //     let as_py_gate = PyGate {
//     //         qubits: qargs.len() as u32,
//     //         clbits: 0,
//     //         params: num_params,
//     //         op_name: operation_name,
//     //         gate: operation.clone().unbind(),
//     //     };
//     //     self.data.push((
//     //         OperationRef::Gate(&as_py_gate),
//     //         params.into(),
//     //         qargs.into(),
//     //     ));
//     // }

// }

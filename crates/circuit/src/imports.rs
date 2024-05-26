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

// This module contains objects imported from Python that are reused. These are
// typically data model classes that are used to identify an object, or for
// python side casting

use pyo3::prelude::*;
use pyo3::sync::GILOnceCell;

// builtin list:
pub static BUILTIN_LIST: GILOnceCell<PyObject> = GILOnceCell::new();
// qiskit.circuit.operation.Operation
pub static OPERATION: GILOnceCell<PyObject> = GILOnceCell::new();
// qiskit.circuit.instruction.Instruction
pub static INSTRUCTION: GILOnceCell<PyObject> = GILOnceCell::new();
// qiskit.circuit.gate.Gate
pub static GATE: GILOnceCell<PyObject> = GILOnceCell::new();
// qiskit.circuit.quantumregister.Qubit
pub static QUBIT: GILOnceCell<PyObject> = GILOnceCell::new();
// qiskit.circuit.classicalregister.Clbit
pub static CLBIT: GILOnceCell<PyObject> = GILOnceCell::new();
// qiskit.circuit.parameterexpression.ParameterExpression
pub static PARAMETER_EXPRESSION: GILOnceCell<PyObject> = GILOnceCell::new();
// qiskit.circuit.quantumcircuit.QuantumCircuit
pub static QUANTUM_CIRCUIT: GILOnceCell<PyObject> = GILOnceCell::new();
// qiskit.circuit.singleton.SingletonGate
pub static SINGLETON_GATE: GILOnceCell<PyObject> = GILOnceCell::new();
// qiskit.circuit.singleton.SingletonControlledGate
pub static SINGLETON_CONTROLLED_GATE: GILOnceCell<PyObject> = GILOnceCell::new();

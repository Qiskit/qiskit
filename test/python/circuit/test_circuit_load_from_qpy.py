# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Test cases for the circuit qasm_file and qasm_string method."""

import io
import random

import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.classicalregister import Clbit
from qiskit.circuit.quantumregister import Qubit
from qiskit.circuit.random import random_circuit
from qiskit.circuit.gate import Gate
from qiskit.circuit.library import XGate
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.parameter import Parameter
from qiskit.extensions import UnitaryGate
from qiskit.opflow import X, Y, Z
from qiskit.test import QiskitTestCase
from qiskit.circuit.qpy_serialization import dump, load
from qiskit.quantum_info.random import random_unitary


class TestLoadFromQPY(QiskitTestCase):
    """Test circuit.from_qasm_* set of methods."""

    def test_qpy_full_path(self):
        """Test full path qpy serialization for basic circuit."""
        qr_a = QuantumRegister(4, "a")
        qr_b = QuantumRegister(4, "b")
        cr_c = ClassicalRegister(4, "c")
        cr_d = ClassicalRegister(4, "d")
        q_circuit = QuantumCircuit(
            qr_a,
            qr_b,
            cr_c,
            cr_d,
            name="MyCircuit",
            metadata={"test": 1, "a": 2},
            global_phase=3.14159,
        )
        q_circuit.h(qr_a)
        q_circuit.cx(qr_a, qr_b)
        q_circuit.barrier(qr_a)
        q_circuit.barrier(qr_b)
        q_circuit.measure(qr_a, cr_c)
        q_circuit.measure(qr_b, cr_d)
        qpy_file = io.BytesIO()
        dump(q_circuit, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(q_circuit, new_circ)
        self.assertEqual(q_circuit.global_phase, new_circ.global_phase)
        self.assertEqual(q_circuit.metadata, new_circ.metadata)
        self.assertEqual(q_circuit.name, new_circ.name)

    def test_circuit_with_conditional(self):
        """Test that instructions with conditions are correctly serialized."""
        qc = QuantumCircuit(1, 1)
        qc.x(0).c_if(qc.cregs[0], 1)
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)

    def test_int_parameter(self):
        """Test that integer parameters are correctly serialized."""
        qc = QuantumCircuit(1)
        qc.rx(3, 0)
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)

    def test_float_parameter(self):
        """Test that float parameters are correctly serialized."""
        qc = QuantumCircuit(1)
        qc.rx(3.14, 0)
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)

    def test_numpy_float_parameter(self):
        """Test that numpy float parameters are correctly serialized."""
        qc = QuantumCircuit(1)
        qc.rx(np.float32(3.14), 0)
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)

    def test_numpy_int_parameter(self):
        """Test that numpy integer parameters are correctly serialized."""
        qc = QuantumCircuit(1)
        qc.rx(np.int16(3), 0)
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)

    def test_unitary_gate(self):
        """Test that numpy array parameters are correctly serialized"""
        qc = QuantumCircuit(1)
        unitary = np.array([[0, 1], [1, 0]])
        qc.unitary(unitary, 0)
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)

    def test_opaque_gate(self):
        """Test that custom opaque gate is correctly serialized"""
        custom_gate = Gate("black_box", 1, [])
        qc = QuantumCircuit(1)
        qc.append(custom_gate, [0])
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)

    def test_opaque_instruction(self):
        """Test that custom opaque instruction is correctly serialized"""
        custom_gate = Instruction("black_box", 1, 0, [])
        qc = QuantumCircuit(1)
        qc.append(custom_gate, [0])
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)

    def test_custom_gate(self):
        """Test that custom  gate is correctly serialized"""
        custom_gate = Gate("black_box", 1, [])
        custom_definition = QuantumCircuit(1)
        custom_definition.h(0)
        custom_definition.rz(1.5, 0)
        custom_definition.sdg(0)
        custom_gate.definition = custom_definition

        qc = QuantumCircuit(1)
        qc.append(custom_gate, [0])
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)
        self.assertEqual(qc.decompose(), new_circ.decompose())

    def test_custom_instruction(self):
        """Test that custom instruction is correctly serialized"""
        custom_gate = Instruction("black_box", 1, 0, [])
        custom_definition = QuantumCircuit(1)
        custom_definition.h(0)
        custom_definition.rz(1.5, 0)
        custom_definition.sdg(0)
        custom_gate.definition = custom_definition
        qc = QuantumCircuit(1)
        qc.append(custom_gate, [0])
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)
        self.assertEqual(qc.decompose(), new_circ.decompose())

    def test_parameter(self):
        """Test that a circuit with a parameter is correctly serialized."""
        theta = Parameter("theta")
        qc = QuantumCircuit(5, 1)
        qc.h(0)
        for i in range(4):
            qc.cx(i, i + 1)

        qc.barrier()
        qc.rz(theta, range(5))
        qc.barrier()
        for i in reversed(range(4)):
            qc.cx(i, i + 1)
        qc.h(0)
        qc.measure(0, 0)

        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)
        self.assertEqual(qc.bind_parameters({theta: 3.14}), new_circ.bind_parameters({theta: 3.14}))

    def test_bound_parameter(self):
        """Test a circuit with a bound parameter is correctly serialized."""
        theta = Parameter("theta")
        qc = QuantumCircuit(5, 1)
        qc.h(0)
        for i in range(4):
            qc.cx(i, i + 1)

        qc.barrier()
        qc.rz(theta, range(5))
        qc.barrier()
        for i in reversed(range(4)):
            qc.cx(i, i + 1)
        qc.h(0)
        qc.measure(0, 0)
        qc.assign_parameters({theta: 3.14})

        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)

    def test_parameter_expression(self):
        """Test a circuit with a parameter expression."""
        theta = Parameter("theta")
        phi = Parameter("phi")
        sum_param = theta + phi
        qc = QuantumCircuit(5, 1)
        qc.h(0)
        for i in range(4):
            qc.cx(i, i + 1)

        qc.barrier()
        qc.rz(sum_param, range(3))
        qc.rz(phi, 3)
        qc.rz(theta, 4)
        qc.barrier()
        for i in reversed(range(4)):
            qc.cx(i, i + 1)
        qc.h(0)
        qc.measure(0, 0)

        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)

    def test_string_parameter(self):
        """Test a PauliGate instruction that has string parameters."""
        circ = (X ^ Y ^ Z).to_circuit_op().to_circuit()
        qpy_file = io.BytesIO()
        dump(circ, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(circ, new_circuit)

    def test_multiple_circuits(self):
        """Test multiple circuits can be serialized together."""
        circuits = []
        for i in range(10):
            circuits.append(
                random_circuit(10, 10, measure=True, conditional=True, reset=True, seed=42 + i)
            )
        qpy_file = io.BytesIO()
        dump(circuits, qpy_file)
        qpy_file.seek(0)
        new_circs = load(qpy_file)
        self.assertEqual(circuits, new_circs)

    def test_shared_bit_register(self):
        """Test a circuit with shared bit registers."""
        qubits = [Qubit() for _ in range(5)]
        qc = QuantumCircuit()
        qc.add_bits(qubits)
        qr = QuantumRegister(bits=qubits)
        qc.add_register(qr)
        qc.h(qr)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.measure_all()
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_qc = load(qpy_file)[0]
        self.assertEqual(qc, new_qc)

    def test_hybrid_standalone_register(self):
        """Test qpy serialization with registers that mix bit types"""
        qr = QuantumRegister(5, "foo")
        qr = QuantumRegister(name="bar", bits=qr[:3] + [Qubit(), Qubit()])
        cr = ClassicalRegister(5, "foo")
        cr = ClassicalRegister(name="classical_bar", bits=cr[:3] + [Clbit(), Clbit()])
        qc = QuantumCircuit(qr, cr)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.measure(qr, cr)
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)

    def test_mixed_registers(self):
        """Test circuit with mix of standalone and shared registers."""
        qubits = [Qubit() for _ in range(5)]
        clbits = [Clbit() for _ in range(5)]
        qc = QuantumCircuit()
        qc.add_bits(qubits)
        qc.add_bits(clbits)
        qr = QuantumRegister(bits=qubits)
        cr = ClassicalRegister(bits=clbits)
        qc.add_register(qr)
        qc.add_register(cr)
        qr_standalone = QuantumRegister(2, "standalone")
        qc.add_register(qr_standalone)
        cr_standalone = ClassicalRegister(2, "classical_standalone")
        qc.add_register(cr_standalone)
        qc.unitary(random_unitary(32, seed=42), qr)
        qc.unitary(random_unitary(4, seed=100), qr_standalone)
        qc.measure(qr, cr)
        qc.measure(qr_standalone, cr_standalone)
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)

    def test_standalone_and_shared_out_of_order(self):
        """Test circuit with register bits inserted out of order."""
        qr_standalone = QuantumRegister(2, "standalone")
        qubits = [Qubit() for _ in range(5)]
        clbits = [Clbit() for _ in range(5)]
        qc = QuantumCircuit()
        qc.add_bits(qubits)
        qc.add_bits(clbits)
        random.shuffle(qubits)
        random.shuffle(clbits)
        qr = QuantumRegister(bits=qubits)
        cr = ClassicalRegister(bits=clbits)
        qc.add_register(qr)
        qc.add_register(cr)
        qr_standalone = QuantumRegister(2, "standalone")
        cr_standalone = ClassicalRegister(2, "classical_standalone")
        qc.add_bits([qr_standalone[1], qr_standalone[0]])
        qc.add_bits([cr_standalone[1], cr_standalone[0]])
        qc.add_register(qr_standalone)
        qc.add_register(cr_standalone)
        qc.unitary(random_unitary(32, seed=42), qr)
        qc.unitary(random_unitary(4, seed=100), qr_standalone)
        qc.measure(qr, cr)
        qc.measure(qr_standalone, cr_standalone)
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)

    def test_unitary_gate_with_label(self):
        """Test that numpy array parameters are correctly serialized with a label"""
        qc = QuantumCircuit(1)
        unitary = np.array([[0, 1], [1, 0]])
        unitary_gate = UnitaryGate(unitary, "My Special unitary")
        qc.append(unitary_gate, [0])
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)
        self.assertEqual([x[0].label for x in qc.data], [x[0].label for x in new_circ.data])

    def test_opaque_gate_with_label(self):
        """Test that custom opaque gate is correctly serialized with a label"""
        custom_gate = Gate("black_box", 1, [])
        custom_gate.label = "My Special Black Box"
        qc = QuantumCircuit(1)
        qc.append(custom_gate, [0])
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)
        self.assertEqual([x[0].label for x in qc.data], [x[0].label for x in new_circ.data])

    def test_opaque_instruction_with_label(self):
        """Test that custom opaque instruction is correctly serialized with a label"""
        custom_gate = Instruction("black_box", 1, 0, [])
        custom_gate.label = "My Special Black Box Instruction"
        qc = QuantumCircuit(1)
        qc.append(custom_gate, [0])
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)
        self.assertEqual([x[0].label for x in qc.data], [x[0].label for x in new_circ.data])

    def test_custom_gate_with_label(self):
        """Test that custom  gate is correctly serialized with a label"""
        custom_gate = Gate("black_box", 1, [])
        custom_definition = QuantumCircuit(1)
        custom_definition.h(0)
        custom_definition.rz(1.5, 0)
        custom_definition.sdg(0)
        custom_gate.definition = custom_definition
        custom_gate.label = "My special black box with a definition"

        qc = QuantumCircuit(1)
        qc.append(custom_gate, [0])
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)
        self.assertEqual(qc.decompose(), new_circ.decompose())
        self.assertEqual([x[0].label for x in qc.data], [x[0].label for x in new_circ.data])

    def test_custom_instruction_with_label(self):
        """Test that custom instruction is correctly serialized with a label"""
        custom_gate = Instruction("black_box", 1, 0, [])
        custom_definition = QuantumCircuit(1)
        custom_definition.h(0)
        custom_definition.rz(1.5, 0)
        custom_definition.sdg(0)
        custom_gate.definition = custom_definition
        custom_gate.label = "My Special Black Box Instruction with a definition"
        qc = QuantumCircuit(1)
        qc.append(custom_gate, [0])
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)
        self.assertEqual(qc.decompose(), new_circ.decompose())
        self.assertEqual([x[0].label for x in qc.data], [x[0].label for x in new_circ.data])

    def test_standard_gate_with_label(self):
        """Test a standard gate with a label."""
        qc = QuantumCircuit(1)
        gate = XGate()
        gate.label = "My special X gate"
        qc.append(gate, [0])
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)
        self.assertEqual([x[0].label for x in qc.data], [x[0].label for x in new_circ.data])

    def test_circuit_with_conditional_with_label(self):
        """Test that instructions with conditions are correctly serialized."""
        qc = QuantumCircuit(1, 1)
        gate = XGate(label="My conditional x gate")
        gate.c_if(qc.cregs[0], 1)
        qc.append(gate, [0])
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)
        self.assertEqual([x[0].label for x in qc.data], [x[0].label for x in new_circ.data])

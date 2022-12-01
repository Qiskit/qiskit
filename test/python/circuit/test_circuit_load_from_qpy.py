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
import json
import random

import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.classicalregister import Clbit
from qiskit.circuit.quantumregister import Qubit
from qiskit.circuit.random import random_circuit
from qiskit.circuit.gate import Gate
from qiskit.circuit.library import XGate, QFT, QAOAAnsatz, PauliEvolutionGate, DCXGate, MCU1Gate
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parametervector import ParameterVector
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from qiskit.extensions import UnitaryGate
from qiskit.opflow import I, X, Y, Z
from qiskit.test import QiskitTestCase
from qiskit.circuit.qpy_serialization import dump, load
from qiskit.quantum_info.random import random_unitary
from qiskit.circuit.controlledgate import ControlledGate


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
        self.assertEqual(
            [x.operation.label for x in qc.data], [x.operation.label for x in new_circ.data]
        )

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
        self.assertEqual(
            [x.operation.label for x in qc.data], [x.operation.label for x in new_circ.data]
        )

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
        self.assertEqual(
            [x.operation.label for x in qc.data], [x.operation.label for x in new_circ.data]
        )

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
        self.assertEqual(
            [x.operation.label for x in qc.data], [x.operation.label for x in new_circ.data]
        )

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
        self.assertEqual(
            [x.operation.label for x in qc.data], [x.operation.label for x in new_circ.data]
        )

    def test_custom_gate_with_noop_definition(self):
        """Test that a custom gate whose definition contains no elements is serialized with a
        proper definition.

        Regression test of gh-7429."""
        empty = QuantumCircuit(1, name="empty").to_gate()
        opaque = Gate("opaque", 1, [])
        qc = QuantumCircuit(2)
        qc.append(empty, [0], [])
        qc.append(opaque, [1], [])

        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]

        self.assertEqual(qc, new_circ)
        self.assertEqual(qc.decompose(), new_circ.decompose())
        self.assertEqual(len(new_circ), 2)
        self.assertIsInstance(new_circ.data[0].operation.definition, QuantumCircuit)
        self.assertIs(new_circ.data[1].operation.definition, None)

    def test_custom_instruction_with_noop_definition(self):
        """Test that a custom instruction whose definition contains no elements is serialized with a
        proper definition.

        Regression test of gh-7429."""
        empty = QuantumCircuit(1, name="empty").to_instruction()
        opaque = Instruction("opaque", 1, 0, [])
        qc = QuantumCircuit(2)
        qc.append(empty, [0], [])
        qc.append(opaque, [1], [])

        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]

        self.assertEqual(qc, new_circ)
        self.assertEqual(qc.decompose(), new_circ.decompose())
        self.assertEqual(len(new_circ), 2)
        self.assertIsInstance(new_circ.data[0].operation.definition, QuantumCircuit)
        self.assertIs(new_circ.data[1].operation.definition, None)

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
        self.assertEqual(
            [x.operation.label for x in qc.data], [x.operation.label for x in new_circ.data]
        )

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
        self.assertEqual(
            [x.operation.label for x in qc.data], [x.operation.label for x in new_circ.data]
        )

    def test_initialize_qft(self):
        """Test that initialize with a complex statevector and qft work."""
        k = 5
        state = (1 / np.sqrt(8)) * np.array(
            [
                np.exp(-1j * 2 * np.pi * k * (0) / 8),
                np.exp(-1j * 2 * np.pi * k * (1) / 8),
                np.exp(-1j * 2 * np.pi * k * (2) / 8),
                np.exp(-1j * 2 * np.pi * k * 3 / 8),
                np.exp(-1j * 2 * np.pi * k * 4 / 8),
                np.exp(-1j * 2 * np.pi * k * 5 / 8),
                np.exp(-1j * 2 * np.pi * k * 6 / 8),
                np.exp(-1j * 2 * np.pi * k * 7 / 8),
            ]
        )

        qubits = 3
        qc = QuantumCircuit(qubits, qubits)
        qc.initialize(state)
        qc.append(QFT(qubits), range(qubits))
        qc.measure(range(qubits), range(qubits))
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)
        self.assertEqual(
            [x.operation.label for x in qc.data], [x.operation.label for x in new_circ.data]
        )

    def test_single_bit_teleportation(self):
        """Test a teleportation circuit with single bit conditions."""
        qr = QuantumRegister(1)
        cr = ClassicalRegister(2, name="name")
        qc = QuantumCircuit(qr, cr, name="Reset Test")
        qc.x(0)
        qc.measure(0, cr[0])
        qc.x(0).c_if(cr[0], 1)
        qc.measure(0, cr[1])
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)
        self.assertEqual(
            [x.operation.label for x in qc.data], [x.operation.label for x in new_circ.data]
        )

    def test_qaoa(self):
        """Test loading a QAOA circuit works."""
        cost_operator = Z ^ I ^ I ^ Z
        qaoa = QAOAAnsatz(cost_operator, reps=2)
        qpy_file = io.BytesIO()
        dump(qaoa, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qaoa, new_circ)
        self.assertEqual(
            [x.operation.label for x in qaoa.data], [x.operation.label for x in new_circ.data]
        )

    def test_evolutiongate(self):
        """Test loading a circuit with evolution gate works."""
        synthesis = LieTrotter(reps=2)
        evo = PauliEvolutionGate((Z ^ I) + (I ^ Z), time=2, synthesis=synthesis)
        qc = QuantumCircuit(2)
        qc.append(evo, range(2))
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]

        self.assertEqual(qc, new_circ)
        self.assertEqual(
            [x.operation.label for x in qc.data], [x.operation.label for x in new_circ.data]
        )

        new_evo = new_circ.data[0].operation
        self.assertIsInstance(new_evo, PauliEvolutionGate)

    def test_evolutiongate_param_time(self):
        """Test loading a circuit with an evolution gate that has a parameter for time."""
        synthesis = LieTrotter(reps=2)
        time = Parameter("t")
        evo = PauliEvolutionGate((Z ^ I) + (I ^ Z), time=time, synthesis=synthesis)
        qc = QuantumCircuit(2)
        qc.append(evo, range(2))
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]

        self.assertEqual(qc, new_circ)
        self.assertEqual(
            [x.operation.label for x in qc.data], [x.operation.label for x in new_circ.data]
        )

        new_evo = new_circ.data[0].operation
        self.assertIsInstance(new_evo, PauliEvolutionGate)

    def test_evolutiongate_param_expr_time(self):
        """Test loading a circuit with an evolution gate that has a parameter for time."""
        synthesis = LieTrotter(reps=2)
        time = Parameter("t")
        evo = PauliEvolutionGate((Z ^ I) + (I ^ Z), time=time * time, synthesis=synthesis)
        qc = QuantumCircuit(2)
        qc.append(evo, range(2))
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]

        self.assertEqual(qc, new_circ)
        self.assertEqual(
            [x.operation.label for x in qc.data], [x.operation.label for x in new_circ.data]
        )

        new_evo = new_circ.data[0].operation
        self.assertIsInstance(new_evo, PauliEvolutionGate)

    def test_evolutiongate_param_vec_time(self):
        """Test loading a an evolution gate that has a param vector element for time."""
        synthesis = LieTrotter(reps=2)
        time = ParameterVector("TimeVec", 1)
        evo = PauliEvolutionGate((Z ^ I) + (I ^ Z), time=time[0], synthesis=synthesis)
        qc = QuantumCircuit(2)
        qc.append(evo, range(2))
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]

        self.assertEqual(qc, new_circ)
        self.assertEqual(
            [x.operation.label for x in qc.data], [x.operation.label for x in new_circ.data]
        )

        new_evo = new_circ.data[0].operation
        self.assertIsInstance(new_evo, PauliEvolutionGate)

    def test_op_list_evolutiongate(self):
        """Test loading a circuit with evolution gate works."""
        evo = PauliEvolutionGate([(Z ^ I) + (I ^ Z)] * 5, time=0.2, synthesis=None)
        qc = QuantumCircuit(2)
        qc.append(evo, range(2))
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]

        self.assertEqual(qc, new_circ)
        self.assertEqual(
            [x.operation.label for x in qc.data], [x.operation.label for x in new_circ.data]
        )

        new_evo = new_circ.data[0].operation
        self.assertIsInstance(new_evo, PauliEvolutionGate)

    def test_op_evolution_gate_suzuki_trotter(self):
        """Test qpy path with a suzuki trotter synthesis method on an evolution gate."""
        synthesis = SuzukiTrotter()
        evo = PauliEvolutionGate((Z ^ I) + (I ^ Z), time=0.2, synthesis=synthesis)
        qc = QuantumCircuit(2)
        qc.append(evo, range(2))
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]

        self.assertEqual(qc, new_circ)
        self.assertEqual(
            [x.operation.label for x in qc.data], [x.operation.label for x in new_circ.data]
        )

        new_evo = new_circ.data[0].operation
        self.assertIsInstance(new_evo, PauliEvolutionGate)

    def test_parameter_expression_global_phase(self):
        """Test a circuit with a parameter expression global_phase."""
        theta = Parameter("theta")
        phi = Parameter("phi")
        sum_param = theta + phi
        qc = QuantumCircuit(5, 1, global_phase=sum_param)
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

    def test_parameter_global_phase(self):
        """Test a circuit with a parameter expression global_phase."""
        theta = Parameter("theta")
        qc = QuantumCircuit(2, global_phase=theta)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)

    def test_parameter_vector(self):
        """Test a circuit with a parameter vector for gate parameters."""
        qc = QuantumCircuit(11)
        input_params = ParameterVector("x_par", 11)
        user_params = ParameterVector("θ_par", 11)
        for i, param in enumerate(user_params):
            qc.ry(param, i)
        for i, param in enumerate(input_params):
            qc.rz(param, i)
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        expected_params = [x.name for x in qc.parameters]
        self.assertEqual([x.name for x in new_circuit.parameters], expected_params)

    def test_parameter_vector_element_in_expression(self):
        """Test a circuit with a parameter vector used in a parameter expression."""
        qc = QuantumCircuit(7)
        entanglement = [[i, i + 1] for i in range(7 - 1)]
        input_params = ParameterVector("x_par", 14)
        user_params = ParameterVector("\u03B8_par", 1)

        for i in range(qc.num_qubits):
            qc.ry(user_params[0], qc.qubits[i])

        for source, target in entanglement:
            qc.cz(qc.qubits[source], qc.qubits[target])

        for i in range(qc.num_qubits):
            qc.rz(-2 * input_params[2 * i + 1], qc.qubits[i])
            qc.rx(-2 * input_params[2 * i], qc.qubits[i])

        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        expected_params = [x.name for x in qc.parameters]
        self.assertEqual([x.name for x in new_circuit.parameters], expected_params)

    def test_parameter_vector_incomplete_warns(self):
        """Test that qpy's deserialization warns if a ParameterVector isn't fully identical."""
        vec = ParameterVector("test", 3)
        qc = QuantumCircuit(1, name="fun")
        qc.rx(vec[1], 0)
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        with self.assertWarnsRegex(UserWarning, r"^The ParameterVector.*Elements 0, 2.*fun$"):
            new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)

    def test_parameter_vector_global_phase(self):
        """Test that a circuit with a standalone ParameterVectorElement phase works."""
        vec = ParameterVector("phase", 1)
        qc = QuantumCircuit(1, global_phase=vec[0])
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)

    def test_custom_metadata_serializer_full_path(self):
        """Test that running with custom metadata serialization works."""

        class CustomObject:
            """Custom string container object."""

            def __init__(self, string):
                self.string = string

            def __eq__(self, other):
                return self.string == other.string

        class CustomSerializer(json.JSONEncoder):
            """Custom json encoder to handle CustomObject."""

            def default(self, o):  # pylint: disable=invalid-name
                if isinstance(o, CustomObject):
                    return {"__type__": "Custom", "value": o.string}
                return json.JSONEncoder.default(self, o)

        class CustomDeserializer(json.JSONDecoder):
            """Custom json decoder to handle CustomObject."""

            def object_hook(self, o):  # pylint: disable=invalid-name,method-hidden
                """Hook to override default decoder.

                Normally specified as a kwarg on load() that overloads the
                default decoder. Done here to avoid reimplementing the
                decode method.
                """
                if "__type__" in o:
                    obj_type = o["__type__"]
                    if obj_type == "Custom":
                        return CustomObject(o["value"])
                return o

        theta = Parameter("theta")
        qc = QuantumCircuit(2, global_phase=theta)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        circuits = [qc, qc.copy()]
        circuits[0].metadata = {"key": CustomObject("Circuit 1")}
        circuits[1].metadata = {"key": CustomObject("Circuit 2")}
        qpy_file = io.BytesIO()
        dump(circuits, qpy_file, metadata_serializer=CustomSerializer)
        qpy_file.seek(0)
        new_circuits = load(qpy_file, metadata_deserializer=CustomDeserializer)
        self.assertEqual(qc, new_circuits[0])
        self.assertEqual(circuits[0].metadata["key"], CustomObject("Circuit 1"))
        self.assertEqual(qc, new_circuits[1])
        self.assertEqual(circuits[1].metadata["key"], CustomObject("Circuit 2"))

    def test_qpy_with_ifelseop(self):
        """Test qpy serialization with an if block."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.measure(0, 0)
        with qc.if_test((qc.clbits[0], True)):
            qc.x(1)
        qc.measure(1, 1)
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)

    def test_qpy_with_ifelseop_with_else(self):
        """Test qpy serialization with an else block."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.measure(0, 0)
        with qc.if_test((qc.clbits[0], True)) as else_:
            qc.x(1)
        with else_:
            qc.y(1)
        qc.measure(1, 1)
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)

    def test_qpy_with_while_loop(self):
        """Test qpy serialization with a for loop."""
        qc = QuantumCircuit(2, 1)

        with qc.while_loop((qc.clbits[0], 0)):
            qc.h(0)
            qc.cx(0, 1)
            qc.measure(0, 0)
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)

    def test_qpy_with_for_loop(self):
        """Test qpy serialization with a for loop."""
        qc = QuantumCircuit(2, 1)

        with qc.for_loop(range(5)):
            qc.h(0)
            qc.cx(0, 1)
            qc.measure(0, 0)
            qc.break_loop().c_if(0, True)
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)

    def test_qpy_with_for_loop_iterator(self):
        """Test qpy serialization with a for loop."""
        qc = QuantumCircuit(2, 1)

        with qc.for_loop(iter(range(5))):
            qc.h(0)
            qc.cx(0, 1)
            qc.measure(0, 0)
            qc.break_loop().c_if(0, True)
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)

    def test_standalone_register_partial_bit_in_circuit(self):
        """Test qpy with only some bits from standalone register."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit([qr[0]])
        qc.x(0)
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)

    def test_nested_tuple_param(self):
        """Test qpy with an instruction that contains nested tuples."""
        inst = Instruction("tuple_test", 1, 0, [((((0, 1), (0, 1)), 2, 3), ("A", "B", "C"))])
        qc = QuantumCircuit(1)
        qc.append(inst, [0])
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)

    def test_empty_tuple_param(self):
        """Test qpy with an instruction that contains an empty tuple."""
        inst = Instruction("empty_tuple_test", 1, 0, [tuple()])
        qc = QuantumCircuit(1)
        qc.append(inst, [0])
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)

    def test_ucr_gates(self):
        """Test qpy with UCRX, UCRY, and UCRZ gates."""
        qc = QuantumCircuit(3)
        qc.ucrz([0, 0, 0, -np.pi], [0, 1], 2)
        qc.ucry([0, 0, 0, -np.pi], [0, 2], 1)
        qc.ucrx([0, 0, 0, -np.pi], [2, 1], 0)
        qc.measure_all()
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc.decompose().decompose(), new_circuit.decompose().decompose())

    def test_controlled_gate(self):
        """Test a custom controlled gate."""
        qc = QuantumCircuit(3)
        controlled_gate = DCXGate().control(1)
        qc.append(controlled_gate, [0, 1, 2])
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)

    def test_controlled_gate_open_controls(self):
        """Test a controlled gate with open controls round-trips exactly."""
        qc = QuantumCircuit(3)
        controlled_gate = DCXGate().control(1, ctrl_state=0)
        qc.append(controlled_gate, [0, 1, 2])
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)

    def test_nested_controlled_gate(self):
        """Test a custom nested controlled gate."""
        custom_gate = Gate("black_box", 1, [])
        custom_definition = QuantumCircuit(1)
        custom_definition.h(0)
        custom_definition.rz(1.5, 0)
        custom_definition.sdg(0)
        custom_gate.definition = custom_definition

        qc = QuantumCircuit(3)
        qc.append(custom_gate, [0])
        controlled_gate = custom_gate.control(2)
        qc.append(controlled_gate, [0, 1, 2])
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)
        self.assertEqual(qc.decompose(), new_circ.decompose())

    def test_open_controlled_gate(self):
        """Test an open control is preserved across serialization."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1, ctrl_state=0)
        with io.BytesIO() as fd:
            dump(qc, fd)
            fd.seek(0)
            new_circ = load(fd)[0]
        self.assertEqual(qc, new_circ)
        self.assertEqual(qc.data[0][0].ctrl_state, new_circ.data[0][0].ctrl_state)

    def test_standard_control_gates(self):
        """Test standard library controlled gates."""
        qc = QuantumCircuit(3)
        mcu1_gate = MCU1Gate(np.pi, 2)
        qc.append(mcu1_gate, [0, 2, 1])
        qc.mcp(np.pi, [0, 2], 1)
        qc.mct([0, 2], 1)
        qc.mcx([0, 2], 1)
        qc.measure_all()
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)

    def test_controlled_gate_subclass_custom_definition(self):
        """Test controlled gate with overloaded definition.

        Reproduce from: https://github.com/Qiskit/qiskit-terra/issues/8794
        """

        class CustomCXGate(ControlledGate):
            """Custom CX with overloaded _define."""

            def __init__(self, label=None, ctrl_state=None):
                super().__init__(
                    "cx", 2, [], label, num_ctrl_qubits=1, ctrl_state=ctrl_state, base_gate=XGate()
                )

            def _define(self) -> None:
                qc = QuantumCircuit(2, name=self.name)
                qc.cx(0, 1)
                self.definition = qc

        qc = QuantumCircuit(2)
        qc.append(CustomCXGate(), [0, 1])
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)
        self.assertEqual(qc.decompose(), new_circ.decompose())

    def test_load_with_loose_bits(self):
        """Test that loading from a circuit with loose bits works."""
        qc = QuantumCircuit([Qubit(), Qubit(), Clbit()])
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(tuple(new_circuit.qregs), ())
        self.assertEqual(tuple(new_circuit.cregs), ())
        self.assertEqual(qc, new_circuit)

    def test_load_with_loose_bits_and_registers(self):
        """Test that loading from a circuit with loose bits and registers works."""
        qc = QuantumCircuit(QuantumRegister(3), ClassicalRegister(1), [Clbit()])
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)

    def test_registers_after_loose_bits(self):
        """Test that a circuit whose registers appear after some loose bits roundtrips. Regression
        test of gh-9094."""
        qc = QuantumCircuit()
        qc.add_bits([Qubit(), Clbit()])
        qc.add_register(QuantumRegister(2, name="q1"))
        qc.add_register(ClassicalRegister(2, name="c1"))
        with io.BytesIO() as fptr:
            dump(qc, fptr)
            fptr.seek(0)
            new_circuit = load(fptr)[0]
        self.assertEqual(qc, new_circuit)

    def test_roundtrip_empty_register(self):
        """Test that empty registers round-trip correctly."""
        qc = QuantumCircuit(QuantumRegister(0), ClassicalRegister(0))
        with io.BytesIO() as fptr:
            dump(qc, fptr)
            fptr.seek(0)
            new_circuit = load(fptr)[0]
        self.assertEqual(qc, new_circuit)
        self.assertEqual(qc.qregs, new_circuit.qregs)
        self.assertEqual(qc.cregs, new_circuit.cregs)

    def test_roundtrip_several_empty_registers(self):
        """Test that several empty registers round-trip correctly."""
        qc = QuantumCircuit(
            QuantumRegister(0, "a"),
            QuantumRegister(0, "b"),
            ClassicalRegister(0, "c"),
            ClassicalRegister(0, "d"),
        )
        with io.BytesIO() as fptr:
            dump(qc, fptr)
            fptr.seek(0)
            new_circuit = load(fptr)[0]
        self.assertEqual(qc, new_circuit)
        self.assertEqual(qc.qregs, new_circuit.qregs)
        self.assertEqual(qc.cregs, new_circuit.cregs)

    def test_roundtrip_empty_registers_with_loose_bits(self):
        """Test that empty registers still round-trip correctly in the presence of loose bits."""
        loose = [Qubit(), Clbit()]

        qc = QuantumCircuit(loose, QuantumRegister(0), ClassicalRegister(0))
        with io.BytesIO() as fptr:
            dump(qc, fptr)
            fptr.seek(0)
            new_circuit = load(fptr)[0]
        self.assertEqual(qc, new_circuit)
        self.assertEqual(qc.qregs, new_circuit.qregs)
        self.assertEqual(qc.cregs, new_circuit.cregs)

        qc = QuantumCircuit(QuantumRegister(0), ClassicalRegister(0), loose)
        with io.BytesIO() as fptr:
            dump(qc, fptr)
            fptr.seek(0)
            new_circuit = load(fptr)[0]
        self.assertEqual(qc, new_circuit)
        self.assertEqual(qc.qregs, new_circuit.qregs)
        self.assertEqual(qc.cregs, new_circuit.cregs)

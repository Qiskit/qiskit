# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the converters."""

from qiskit.converters import circuit_to_instruction, instruction_to_gate
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit import Gate, Instruction
from qiskit.test import QiskitTestCase
from qiskit.exceptions import QiskitError


class TestInstructionToGate(QiskitTestCase):
    """Test Instruction to Gate"""

    def test_simple_instruction(self):
        """test simple instruction"""
        qr1 = QuantumRegister(4, 'qr1')
        qr2 = QuantumRegister(3, 'qr2')
        qr3 = QuantumRegister(3, 'qr3')
        circ = QuantumCircuit(qr1, qr2, qr3)
        circ.cx(qr1[1], qr2[2])

        inst = circuit_to_instruction(circ)
        gate = instruction_to_gate(inst)
        q = QuantumRegister(10, 'q')

        self.assertIsInstance(gate, Gate)
        self.assertEqual(gate.definition[0][1], [q[1], q[6]])

    def test_opaque_instruction(self):
        """test opaque instruction"""
        inst = Instruction('opaque', 3, 0, [])
        gate = instruction_to_gate(inst)
        self.assertIsInstance(gate, Gate)
        inst_attr = inst.__dict__
        gate_attr = gate.__dict__
        for att, value in inst_attr.items():
            with self.subTest(name=att):
                self.assertEqual(value, getattr(gate, att))
        self.assertEqual(set(gate_attr.keys()) - set(inst_attr.keys()),
                         {'_label'})

    def test_raises(self):
        """test instruction which can't be converted raises"""
        circ1 = QuantumCircuit(3)
        circ1.x(0)
        circ1.cx(0, 1)
        circ1.barrier()
        inst1 = circuit_to_instruction(circ1)

        circ2 = QuantumCircuit(1, 1)
        circ2.measure(0, 0)
        inst2 = circuit_to_instruction(circ2)

        with self.assertRaises(QiskitError):
            instruction_to_gate(inst1)
            instruction_to_gate(inst2)

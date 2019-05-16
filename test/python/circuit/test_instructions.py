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

"""Test Qiskit's Instruction class."""

import unittest

from qiskit.circuit import Gate
from qiskit.circuit import Parameter
from qiskit.circuit import Instruction
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister, ClassicalRegister
from qiskit.extensions.standard.h import HGate
from qiskit.extensions.standard.cx import CnotGate
from qiskit.test import QiskitTestCase
from qiskit.exceptions import QiskitError


class TestInstructions(QiskitTestCase):
    """Instructions tests."""

    def test_instructions_equal(self):
        """Test equality of two instructions."""
        hop1 = Instruction('h', 1, 0, [])
        hop2 = Instruction('s', 1, 0, [])
        hop3 = Instruction('h', 1, 0, [])

        uop1 = Instruction('u', 1, 0, [0.4, 0.5, 0.5])
        uop2 = Instruction('u', 1, 0, [0.4, 0.6, 0.5])
        uop3 = Instruction('v', 1, 0, [0.4, 0.5, 0.5])
        uop4 = Instruction('u', 1, 0, [0.4, 0.5, 0.5])
        self.assertFalse(hop1 == hop2)
        self.assertTrue(hop1 == hop3)
        self.assertFalse(uop1 == uop2)
        self.assertTrue(uop1 == uop4)
        self.assertFalse(uop1 == uop3)
        self.assertTrue(HGate() == HGate())
        self.assertFalse(HGate() == CnotGate())
        self.assertFalse(hop1 == HGate())

    def test_instructions_equal_with_parameters(self):
        """Test equality of instructions for cases with Parameters."""
        theta = Parameter('theta')
        phi = Parameter('phi')

        # Verify we can check params including parameters
        self.assertEqual(Instruction('u', 1, 0, [theta, phi, 0.4]),
                         Instruction('u', 1, 0, [theta, phi, 0.4]))

        # Verify we can test for correct parameter order
        self.assertNotEqual(Instruction('u', 1, 0, [theta, phi, 0]),
                            Instruction('u', 1, 0, [phi, theta, 0]))

        # Verify we can still find a wrong fixed param if we use parameters
        self.assertNotEqual(Instruction('u', 1, 0, [theta, phi, 0.4]),
                            Instruction('u', 1, 0, [theta, phi, 0.5]))

        # Verify we can find cases when param != float
        self.assertNotEqual(Instruction('u', 1, 0, [0.3, phi, 0.4]),
                            Instruction('u', 1, 0, [theta, phi, 0.5]))

    def circuit_instruction_circuit_roundtrip(self):
        """test converting between circuit and instruction and back
        preserves the circuit"""
        q = QuantumRegister(4)
        c = ClassicalRegister(4)
        circ1 = QuantumCircuit(q, c, name='circ1')
        circ1.h(q[0])
        circ1.crz(0.1, q[0], q[1])
        circ1.iden(q[1])
        circ1.u3(0.1, 0.2, -0.2, q[0])
        circ1.barrier()
        circ1.measure(q, c)
        circ1.rz(0.8, q[0]).c_if(c, 6)
        inst = circ1.to_instruction()

        circ2 = QuantumCircuit(q, c, name='circ2')
        circ2.append(inst, q[:])

        self.assertEqual(circ1, circ2)

    def test_append_opaque_wrong_dimension(self):
        """test appending opaque gate to wrong dimension wires.
        """
        qr = QuantumRegister(2)
        circ = QuantumCircuit(qr)
        opaque_gate = Gate(name='crz_2', num_qubits=2, params=[0.5])
        self.assertRaises(QiskitError, circ.append, opaque_gate, [qr[0]])

    def test_opaque_gate(self):
        """test opaque gate functionality"""
        q = QuantumRegister(4)
        c = ClassicalRegister(4)
        circ = QuantumCircuit(q, c, name='circ')
        opaque_gate = Gate(name='crz_2', num_qubits=2, params=[0.5])
        circ.append(opaque_gate, [q[2], q[0]])
        self.assertEqual(circ.data[0][0].name, 'crz_2')
        self.assertEqual(circ.decompose(), circ)

    def test_opaque_instruction(self):
        """test opaque instruction does not decompose"""
        q = QuantumRegister(4)
        c = ClassicalRegister(2)
        circ = QuantumCircuit(q, c)
        opaque_inst = Instruction(name='my_inst', num_qubits=3,
                                  num_clbits=1, params=[0.5])
        circ.append(opaque_inst, [q[3], q[1], q[0]], [c[1]])
        self.assertEqual(circ.data[0][0].name, 'my_inst')
        self.assertEqual(circ.decompose(), circ)

    def test_mirror_gate(self):
        """test mirroring a composite gate"""
        q = QuantumRegister(4)
        c = ClassicalRegister(4)
        circ = QuantumCircuit(q, c, name='circ')
        circ.h(q[0])
        circ.crz(0.1, q[0], q[1])
        circ.iden(q[1])
        circ.u3(0.1, 0.2, -0.2, q[0])
        gate = circ.to_instruction()

        circ = QuantumCircuit(q, c, name='circ')
        circ.u3(0.1, 0.2, -0.2, q[0])
        circ.iden(q[1])
        circ.crz(0.1, q[0], q[1])
        circ.h(q[0])
        gate_mirror = circ.to_instruction()
        self.assertEqual(gate.mirror().definition, gate_mirror.definition)

    def test_mirror_instruction(self):
        """test mirroring an instruction with conditionals"""
        q = QuantumRegister(4)
        c = ClassicalRegister(4)
        circ = QuantumCircuit(q, c, name='circ')
        circ.t(q[1])
        circ.u3(0.1, 0.2, -0.2, q[0])
        circ.barrier()
        circ.measure(q[0], c[0])
        circ.rz(0.8, q[0]).c_if(c, 6)
        inst = circ.to_instruction()

        circ = QuantumCircuit(q, c, name='circ')
        circ.rz(0.8, q[0]).c_if(c, 6)
        circ.measure(q[0], c[0])
        circ.barrier()
        circ.u3(0.1, 0.2, -0.2, q[0])
        circ.t(q[1])
        inst_mirror = circ.to_instruction()
        self.assertEqual(inst.mirror().definition, inst_mirror.definition)

    def test_mirror_opaque(self):
        """test opaque gates mirror to themselves"""
        opaque_gate = Gate(name='crz_2', num_qubits=2, params=[0.5])
        self.assertEqual(opaque_gate.mirror(), opaque_gate)
        hgate = HGate()
        self.assertEqual(hgate.mirror(), hgate)

    def test_inverse_gate(self):
        """test inverse of composite gate"""
        q = QuantumRegister(4)
        circ = QuantumCircuit(q, name='circ')
        circ.h(q[0])
        circ.crz(0.1, q[0], q[1])
        circ.iden(q[1])
        circ.u3(0.1, 0.2, -0.2, q[0])
        gate = circ.to_instruction()
        circ = QuantumCircuit(q, name='circ')
        circ.u3(-0.1, 0.2, -0.2, q[0])
        circ.iden(q[1])
        circ.crz(-0.1, q[0], q[1])
        circ.h(q[0])
        gate_inverse = circ.to_instruction()
        self.assertEqual(gate.inverse().definition, gate_inverse.definition)

    def test_inverse_recursive(self):
        """test that a hierarchical gate recursively inverts"""
        qr0 = QuantumRegister(2)
        circ0 = QuantumCircuit(qr0, name='circ0')
        circ0.t(qr0[0])
        circ0.rx(0.4, qr0[1])
        circ0.cx(qr0[1], qr0[0])
        little_gate = circ0.to_instruction()

        qr1 = QuantumRegister(4)
        circ1 = QuantumCircuit(qr1, name='circ1')
        circ1.cu1(-0.1, qr1[0], qr1[2])
        circ1.iden(qr1[1])
        circ1.append(little_gate, [qr1[2], qr1[3]])

        circ_inv = QuantumCircuit(qr1, name='circ1_dg')
        circ_inv.append(little_gate.inverse(), [qr1[2], qr1[3]])
        circ_inv.iden(qr1[1])
        circ_inv.cu1(0.1, qr1[0], qr1[2])

        self.assertEqual(circ1.inverse(), circ_inv)

    def test_inverse_instruction_with_measure(self):
        """test inverting instruction with measure fails"""
        q = QuantumRegister(4)
        c = ClassicalRegister(4)
        circ = QuantumCircuit(q, c, name='circ')
        circ.t(q[1])
        circ.u3(0.1, 0.2, -0.2, q[0])
        circ.barrier()
        circ.measure(q[0], c[0])
        inst = circ.to_instruction()
        self.assertRaises(QiskitError, inst.inverse)

    def test_inverse_instruction_with_conditional(self):
        """test inverting instruction with conditionals fails"""
        q = QuantumRegister(4)
        c = ClassicalRegister(4)
        circ = QuantumCircuit(q, c, name='circ')
        circ.t(q[1])
        circ.u3(0.1, 0.2, -0.2, q[0])
        circ.barrier()
        circ.measure(q[0], c[0])
        circ.rz(0.8, q[0]).c_if(c, 6)
        inst = circ.to_instruction()
        self.assertRaises(QiskitError, inst.inverse)

    def test_inverse_opaque(self):
        """test inverting opaque gate fails"""
        opaque_gate = Gate(name='crz_2', num_qubits=2, params=[0.5])
        self.assertRaises(QiskitError, opaque_gate.inverse)


if __name__ == '__main__':
    unittest.main()

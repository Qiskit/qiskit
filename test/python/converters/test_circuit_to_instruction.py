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

import unittest

from qiskit.converters import circuit_to_instruction
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.test import QiskitTestCase
from qiskit.exceptions import QiskitError


class TestCircuitToInstruction(QiskitTestCase):
    """Test Circuit to Instruction."""

    def test_flatten_circuit_registers(self):
        """Check correct flattening"""
        qr1 = QuantumRegister(4, 'qr1')
        qr2 = QuantumRegister(3, 'qr2')
        qr3 = QuantumRegister(3, 'qr3')
        cr1 = ClassicalRegister(4, 'cr1')
        cr2 = ClassicalRegister(1, 'cr2')
        circ = QuantumCircuit(qr1, qr2, qr3, cr1, cr2)
        circ.cx(qr1[1], qr2[2])
        circ.measure(qr3[0], cr2[0])

        inst = circuit_to_instruction(circ)
        q = QuantumRegister(10, 'q')
        c = ClassicalRegister(5, 'c')

        self.assertEqual(inst.definition[0][1], [q[1], q[6]])
        self.assertEqual(inst.definition[1][1], [q[7]])
        self.assertEqual(inst.definition[1][2], [c[4]])

    def test_flatten_parameters(self):
        """Verify parameters from circuit are moved to instruction.params"""
        qr = QuantumRegister(3, 'qr')
        qc = QuantumCircuit(qr)

        theta = Parameter('theta')
        phi = Parameter('phi')

        qc.rz(theta, qr[0])
        qc.rz(phi, qr[1])
        qc.u2(theta, phi, qr[2])

        inst = circuit_to_instruction(qc)

        self.assertEqual(inst.params, [phi, theta])
        self.assertEqual(inst.definition[0][0].params, [theta])
        self.assertEqual(inst.definition[1][0].params, [phi])
        self.assertEqual(inst.definition[2][0].params, [theta, phi])

    def test_underspecified_parameter_map_raises(self):
        """Verify we raise if not all circuit parameters are present in parameter_map."""
        qr = QuantumRegister(3, 'qr')
        qc = QuantumCircuit(qr)

        theta = Parameter('theta')
        phi = Parameter('phi')

        gamma = Parameter('gamma')

        qc.rz(theta, qr[0])
        qc.rz(phi, qr[1])
        qc.u2(theta, phi, qr[2])

        self.assertRaises(QiskitError, circuit_to_instruction, qc, {theta: gamma})

        # Raise if provided more parameters than present in the circuit
        delta = Parameter('delta')
        self.assertRaises(QiskitError, circuit_to_instruction, qc,
                          {theta: gamma, phi: phi, delta: delta})

    def test_parameter_map(self):
        """Verify alternate parameter specification"""
        qr = QuantumRegister(3, 'qr')
        qc = QuantumCircuit(qr)

        theta = Parameter('theta')
        phi = Parameter('phi')

        gamma = Parameter('gamma')

        qc.rz(theta, qr[0])
        qc.rz(phi, qr[1])
        qc.u2(theta, phi, qr[2])

        inst = circuit_to_instruction(qc, {theta: gamma, phi: phi})

        self.assertEqual(inst.params, [gamma, phi])
        self.assertEqual(inst.definition[0][0].params, [gamma])
        self.assertEqual(inst.definition[1][0].params, [phi])
        self.assertEqual(inst.definition[2][0].params, [gamma, phi])


if __name__ == '__main__':
    unittest.main(verbosity=2)

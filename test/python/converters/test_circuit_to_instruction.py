# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Tests for the converters."""

import unittest

from qiskit.converters import circuit_to_instruction
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.test import QiskitTestCase


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


if __name__ == '__main__':
    unittest.main(verbosity=2)

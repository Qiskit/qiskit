# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""Test Qiskit's QuantumCircuit class for wires."""

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.test import QiskitTestCase


class TestCircuitWires(QiskitTestCase):
    """Test QuantumCircuit with wires."""

    def test_circuit_multi_qregs(self):
        """Test circuit multi qregs and wires with Hs.
        """
        qreg0 = QuantumRegister(2)
        qreg1 = QuantumRegister(2)

        circ = QuantumCircuit(qreg0, qreg1)
        circ.h(0)
        circ.h(2)

        expected = QuantumCircuit(qreg0, qreg1)
        expected.h(qreg0[0])
        expected.h(qreg1[0])

        self.assertEqual(circ, expected)

    def test_circuit_multi_qreg_cregs(self):
        """Test circuit multi qregs/cregs and wires.
        """
        qreg0 = QuantumRegister(2)
        creg0 = ClassicalRegister(2)
        qreg1 = QuantumRegister(2)
        creg1 = ClassicalRegister(2)

        circ = QuantumCircuit(qreg0, qreg1, creg0, creg1)
        circ.measure(0, 2)
        circ.measure(2, 1)

        expected = QuantumCircuit(qreg0, qreg1, creg0, creg1)
        expected.measure(qreg0[0], creg1[0])
        expected.measure(qreg1[0], creg0[1])

        self.assertEqual(circ, expected)

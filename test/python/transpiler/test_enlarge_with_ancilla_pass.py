# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Test the EnlargeWithAncilla pass"""

import unittest

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.transpiler import Layout
from qiskit.transpiler.passes import EnlargeWithAncilla
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase


class TestEnlargeWithAncilla(QiskitTestCase):
    """Tests the EnlargeWithAncilla pass."""

    def setUp(self):
        self.qr3 = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(self.qr3)
        circuit.h(self.qr3)
        self.dag = circuit_to_dag(circuit)

    def test_no_extension(self):
        """There are no virtual qubits to extend."""
        layout = Layout({self.qr3[0]: 0, self.qr3[1]: 1, self.qr3[2]: 2})

        pass_ = EnlargeWithAncilla(layout)
        after = pass_.run(self.dag)

        qregs = list(after.qregs.values())
        self.assertEqual(1, len(qregs))
        self.assertEqual(self.qr3, qregs[0])

    def test_with_extension(self):
        """There are 2 virtual qubit to extend."""
        ancilla = QuantumRegister(2, 'ancilla')

        layout = Layout({0: self.qr3[0], 1: ancilla[0],
                         2: self.qr3[1], 3: ancilla[1],
                         4: self.qr3[2]})

        pass_ = EnlargeWithAncilla(layout)
        after = pass_.run(self.dag)

        qregs = list(after.qregs.values())
        self.assertEqual(2, len(qregs))
        self.assertEqual(self.qr3, qregs[0])
        self.assertEqual(ancilla, qregs[1])


if __name__ == '__main__':
    unittest.main()

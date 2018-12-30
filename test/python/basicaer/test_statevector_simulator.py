# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-docstring,broad-except

import unittest
from qiskit import QuantumCircuit, QuantumRegister
from qiskit import execute
from qiskit import BasicAer
from qiskit.test import QiskitTestCase


class StatevectorSimulatorTest(QiskitTestCase):
    """Test BasicAer statevector simulator."""

    def setUp(self):
        qr = QuantumRegister(2)
        self.q_circuit = QuantumCircuit(qr)
        self.q_circuit.h(qr[0])
        self.q_circuit.cx(qr[0], qr[1])

    def test_statevector_simulator(self):
        """Test final state vector for single circuit run."""
        result = execute(self.q_circuit,
                         backend=BasicAer.get_backend('statevector_simulator')).result()
        self.assertEqual(result.success, True)
        actual = result.get_statevector(self.q_circuit)

        # state is 1/sqrt(2)|00> + 1/sqrt(2)|11>, up to a global phase
        self.assertAlmostEqual((abs(actual[0]))**2, 1/2)
        self.assertEqual(actual[1], 0)
        self.assertEqual(actual[2], 0)
        self.assertAlmostEqual((abs(actual[3]))**2, 1/2)


if __name__ == '__main__':
    unittest.main()

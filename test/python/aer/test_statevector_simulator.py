# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-docstring,broad-except

import unittest
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import execute
from qiskit import Aer
from ..common import QiskitTestCase, requires_cpp_simulator


@requires_cpp_simulator
class StatevectorSimulatorTest(QiskitTestCase):
    """Test Aer's C++ statevector simulator."""

    def setUp(self):
        q = QuantumRegister(2)
        c = ClassicalRegister(2)
        self.q_circuit = QuantumCircuit(q, c)
        self.q_circuit.h(q[0])
        self.q_circuit.cx(q[0], q[1])

    def test_statevector_simulator(self):
        """Test final state vector for single circuit run."""
        result = execute(self.q_circuit, backend=Aer.get_backend('statevector_simulator')).result()
        self.assertEqual(result.get_status(), 'COMPLETED')
        actual = result.get_statevector(self.q_circuit)

        # state is 1/sqrt(2)|00> + 1/sqrt(2)|11>, up to a global phase
        self.assertAlmostEqual((abs(actual[0]))**2, 1/2)
        self.assertEqual(actual[1], 0)
        self.assertEqual(actual[2], 0)
        self.assertAlmostEqual((abs(actual[3]))**2, 1/2)


if __name__ == '__main__':
    unittest.main()

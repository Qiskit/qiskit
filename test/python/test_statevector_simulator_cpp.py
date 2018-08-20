# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-docstring,broad-except

import unittest
from qiskit import execute, load_qasm_file
from .common import QiskitTestCase, requires_cpp_simulator


@requires_cpp_simulator
class StatevectorSimulatorCppTest(QiskitTestCase):
    """Test C++ statevector simulator."""

    def setUp(self):
        self.qasm_filename = self._get_resource_path('qasm/simple.qasm')
        self.q_circuit = load_qasm_file(self.qasm_filename, name='example')

    def test_statevector_simulator_cpp(self):
        """Test final state vector for single circuit run."""
        result = execute(self.q_circuit, backend='local_statevector_simulator_cpp').result()
        self.assertEqual(result.get_status(), 'COMPLETED')
        actual = result.get_statevector(self.q_circuit)

        # state is 1/sqrt(2)|00> + 1/sqrt(2)|11>, up to a global phase
        self.assertAlmostEqual((abs(actual[0]))**2, 1/2)
        self.assertEqual(actual[1], 0)
        self.assertEqual(actual[2], 0)
        self.assertAlmostEqual((abs(actual[3]))**2, 1/2)


if __name__ == '__main__':
    unittest.main()

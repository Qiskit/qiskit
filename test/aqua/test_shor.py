# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Shor """

import unittest
import math
from test.aqua.common import QiskitAquaTestCase
from parameterized import parameterized
from qiskit import BasicAer
from qiskit.aqua import QuantumInstance, AquaError
from qiskit.aqua.algorithms import Shor


class TestShor(QiskitAquaTestCase):
    """test Shor's algorithm"""

    @parameterized.expand([
        [15, 'qasm_simulator', [3, 5]],
    ])
    def test_shor_factoring(self, n_v, backend, factors):
        """ shor factoring test """
        shor = Shor(n_v)
        result_dict = shor.run(QuantumInstance(BasicAer.get_backend(backend), shots=1000))
        self.assertListEqual(result_dict['factors'][0], factors)

    @parameterized.expand([
        [5],
        [7],
    ])
    def test_shor_no_factors(self, n_v):
        """ shor no factors test """
        shor = Shor(n_v)
        backend = BasicAer.get_backend('qasm_simulator')
        quantum_instance = QuantumInstance(backend, shots=1000)
        ret = shor.run(quantum_instance)
        self.assertTrue(ret['factors'] == [])

    @parameterized.expand([
        [3, 5],
        [5, 3],
    ])
    def test_shor_power(self, base, power):
        """ shor power test """
        n_v = int(math.pow(base, power))
        shor = Shor(n_v)
        backend = BasicAer.get_backend('qasm_simulator')
        quantum_instance = QuantumInstance(backend, shots=1000)
        ret = shor.run(quantum_instance)
        self.assertTrue(ret['factors'] == [base])

    @parameterized.expand([
        [-1],
        [0],
        [1],
        [2],
        [4],
        [16],
    ])
    def test_shor_bad_input(self, n_v):
        """ shor bad input test """
        with self.assertRaises(AquaError):
            Shor(n_v)


if __name__ == '__main__':
    unittest.main()

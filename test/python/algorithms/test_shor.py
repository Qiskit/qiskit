# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
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
from test.python.algorithms import QiskitAlgorithmsTestCase
from ddt import ddt, data, idata, unpack

from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.algorithms import Shor


@unittest.skipUnless(Aer, "qiskit-aer is required for these tests")
@ddt
class TestShor(QiskitAlgorithmsTestCase):
    """test Shor's algorithm"""

    @idata([
        [15, 'qasm_simulator', [3, 5]],
    ])
    @unpack
    def test_shor_factoring(self, n_v, backend, factors):
        """ shor factoring test """
        shor = Shor(quantum_instance=QuantumInstance(Aer.get_backend(backend), shots=1000))
        result = shor.factor(N=n_v)
        self.assertListEqual(result.factors[0], factors)
        self.assertTrue(result.total_counts >= result.successful_counts)

    @data(5, 7)
    def test_shor_no_factors(self, n_v):
        """ shor no factors test """
        backend = Aer.get_backend('qasm_simulator')
        shor = Shor(quantum_instance=QuantumInstance(backend, shots=1000))
        result = shor.factor(N=n_v)
        self.assertTrue(result.factors == [])
        self.assertTrue(result.successful_counts == 0)

    @idata([
        [3, 5],
        [5, 3],
    ])
    @unpack
    def test_shor_power(self, base, power):
        """ shor power test """
        n_v = int(math.pow(base, power))
        backend = Aer.get_backend('qasm_simulator')
        shor = Shor(quantum_instance=QuantumInstance(backend, shots=1000))
        result = shor.factor(N=n_v)
        self.assertTrue(result.factors == [base])
        self.assertTrue(result.total_counts >= result.successful_counts)

    @data(-1, 0, 1, 2, 4, 16)
    def test_shor_bad_input(self, n_v):
        """ shor bad input test """
        with self.assertRaises(ValueError):
            _ = Shor().factor(N=n_v)

    @idata([[2, 15, 8], [4, 15, 4]])
    @unpack
    def test_shor_modinv(self, a_v, m_v, expected):
        """ shor modular inverse test """
        modinv = Shor.modinv(a_v, m_v)
        self.assertTrue(modinv == expected)


if __name__ == '__main__':
    unittest.main()

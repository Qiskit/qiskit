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
from test.aqua import QiskitAquaTestCase
from ddt import ddt, data, idata, unpack
from qiskit import BasicAer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import Shor


@ddt
class TestShor(QiskitAquaTestCase):
    """test Shor's algorithm"""

    @idata([
        [15, 'qasm_simulator', [3, 5]],
    ])
    @unpack
    def test_shor_factoring(self, n_v, backend, factors):
        """ shor factoring test """
        shor = Shor(n_v)
        result_dict = shor.run(QuantumInstance(BasicAer.get_backend(backend), shots=1000))
        self.assertListEqual(result_dict['factors'][0], factors)
        self.assertTrue(result_dict["total_counts"] >= result_dict["successful_counts"])

    @data(5, 7)
    def test_shor_no_factors(self, n_v):
        """ shor no factors test """
        shor = Shor(n_v)
        backend = BasicAer.get_backend('qasm_simulator')
        quantum_instance = QuantumInstance(backend, shots=1000)
        ret = shor.run(quantum_instance)
        self.assertTrue(ret['factors'] == [])
        self.assertTrue(ret["successful_counts"] == 0)

    @idata([
        [3, 5],
        [5, 3],
    ])
    @unpack
    def test_shor_power(self, base, power):
        """ shor power test """
        n_v = int(math.pow(base, power))
        shor = Shor(n_v)
        backend = BasicAer.get_backend('qasm_simulator')
        quantum_instance = QuantumInstance(backend, shots=1000)
        ret = shor.run(quantum_instance)
        self.assertTrue(ret['factors'] == [base])
        self.assertTrue(ret["total_counts"] >= ret["successful_counts"])

    @data(-1, 0, 1, 2, 4, 16)
    def test_shor_bad_input(self, n_v):
        """ shor bad input test """
        with self.assertRaises(ValueError):
            Shor(n_v)

    @idata([[2, 15, 8], [4, 15, 4]])
    @unpack
    def test_shor_modinv(self, a_v, m_v, expected):
        """ shor modular inverse test """
        modinv = Shor.modinv(a_v, m_v)
        self.assertTrue(modinv == expected)


if __name__ == '__main__':
    unittest.main()

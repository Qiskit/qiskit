# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2022.
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
import warnings
import math
from test.python.algorithms import QiskitAlgorithmsTestCase
from ddt import ddt, data, idata, unpack

from qiskit import ClassicalRegister
from qiskit.utils import QuantumInstance, optionals
from qiskit.algorithms import Shor
from qiskit.test import slow_test


@ddt
class TestShor(QiskitAlgorithmsTestCase):
    """test Shor's algorithm"""

    @unittest.skipUnless(optionals.HAS_AER, "qiskit-aer is required to run this test")
    def setUp(self):
        super().setUp()
        from qiskit_aer import Aer

        backend = Aer.get_backend("aer_simulator")
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.filterwarnings(
                "always",
                category=DeprecationWarning,
            )
            self.instance = Shor(quantum_instance=QuantumInstance(backend, shots=1000))
            self.assertTrue("Shor class is deprecated" in str(caught_warnings[0].message))

    @slow_test
    @idata(
        [
            [15, "aer_simulator", [3, 5]],
        ]
    )
    @unpack
    def test_shor_factoring(self, n_v, backend, factors):
        """shor factoring test for n = log(N) = 4"""
        self._test_shor_factoring(backend, factors, n_v)

    @slow_test
    @idata(
        [
            [21, "aer_simulator", [3, 7]],
        ]
    )
    @unpack
    def test_shor_factoring_5_bit_number(self, n_v, backend, factors):
        """shor factoring test for n = log(N) = 5"""
        self._test_shor_factoring(backend, factors, n_v)

    def _test_shor_factoring(self, backend, factors, n_v):
        """shor factoring test"""
        from qiskit_aer import Aer

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            shor = Shor(quantum_instance=QuantumInstance(Aer.get_backend(backend), shots=1000))
            self.assertTrue("Shor class is deprecated" in str(caught_warnings[0].message))
        result = shor.factor(N=n_v)
        self.assertListEqual(result.factors[0], factors)
        self.assertTrue(result.total_counts >= result.successful_counts)

    @slow_test
    @data(5, 7)
    def test_shor_no_factors(self, n_v):
        """shor no factors test"""
        shor = self.instance
        result = shor.factor(N=n_v)
        self.assertTrue(result.factors == [])
        self.assertTrue(result.successful_counts == 0)

    @idata(
        [
            [3, 5],
            [5, 3],
        ]
    )
    @unpack
    def test_shor_input_being_power(self, base, power):
        """shor input being power test"""
        n_v = int(math.pow(base, power))
        shor = self.instance
        result = shor.factor(N=n_v)
        self.assertTrue(result.factors == [base])
        self.assertTrue(result.total_counts >= result.successful_counts)

    @idata(
        [[N, 2] for N in [-1, 0, 1, 2, 4, 16]] + [[15, a] for a in [-1, 0, 1, 3, 5, 15, 16]],
    )
    @unpack
    def test_shor_bad_input(self, n_v, a_v):
        """shor factor bad input test"""
        shor = self.instance
        with self.assertRaises(ValueError):
            _ = shor.factor(N=n_v, a=a_v)

    @slow_test
    @idata(
        [
            [15, 4, 2],
            [15, 7, 4],
        ]
    )
    @unpack
    def test_shor_quantum_result(self, n_v, a_v, order):
        """shor quantum result test (for order being power of 2)"""
        self._test_quantum_result(a_v, n_v, order)

    @slow_test
    @idata(
        [
            [17, 8, 8],
            [21, 13, 2],
        ]
    )
    @unpack
    def test_shor_quantum_result_for_5_bit_number(self, n_v, a_v, order):
        """shor quantum result test (for order being power of 2 and n = log(N) = 5)"""
        self._test_quantum_result(a_v, n_v, order)

    def _test_quantum_result(self, a_v, n_v, order):
        shor = self.instance
        circuit = shor.construct_circuit(N=n_v, a=a_v, measurement=True)

        result = shor.quantum_instance.execute(circuit)
        measurements = [int(key, base=2) for key in result.get_counts(circuit).keys()]

        # calculate values that could be measured
        values = [i << (2 * n_v.bit_length() - order.bit_length() + 1) for i in range(order)]

        for measurement in measurements:
            self.assertTrue(measurement in values)

    @slow_test
    @idata(
        [
            [15, 4, [1, 4]],
            [15, 7, [1, 4, 7, 13]],
        ]
    )
    @unpack
    def test_shor_exponentiation_result(self, n_v, a_v, values):
        """shor exponentiation result test (for n = log(N) = 4)"""
        self._test_exponentiation_result(a_v, n_v, values)

    @slow_test
    @idata(
        [
            [5, 21, [1, 4, 5, 16, 17, 20]],
            [4, 25, [1, 4, 6, 9, 11, 14, 16, 19, 21, 24]],
        ]
    )
    @unpack
    def test_shor_exponentiation_result_for_5_bit_number(self, a_v, n_v, values):
        """shor exponentiation result test (for n = log(N) = 5)"""
        self._test_exponentiation_result(a_v, n_v, values)

    def _test_exponentiation_result(self, a_v, n_v, values):
        shor = self.instance

        circuit = shor.construct_circuit(N=n_v, a=a_v, measurement=False)
        # modify circuit to measure output (down) register
        down_qreg = circuit.qregs[1]
        down_creg = ClassicalRegister(len(down_qreg), name="m")
        circuit.add_register(down_creg)
        circuit.measure(down_qreg, down_creg)

        result = shor.quantum_instance.execute(circuit)
        measurements = [int(key, base=2) for key in result.get_counts(circuit).keys()]

        for measurement in measurements:
            self.assertTrue(measurement in values)

    @idata([[2, 15, 8], [4, 15, 4]])
    @unpack
    def test_shor_modinv(self, a_v, m_v, expected):
        """shor modular inverse test"""
        modinv = Shor.modinv(a_v, m_v)
        self.assertTrue(modinv == expected)


if __name__ == "__main__":
    unittest.main()

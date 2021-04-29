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

from qiskit import Aer, ClassicalRegister
from qiskit.utils import QuantumInstance
from qiskit.algorithms import Shor


@unittest.skipUnless(Aer, "qiskit-aer is required for these tests")
@ddt
class TestShor(QiskitAlgorithmsTestCase):
    """test Shor's algorithm"""

    @idata([
        [15, 'qasm_simulator', [3, 5]],
        [21, 'qasm_simulator', [3, 7]],
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
    def test_shor_input_being_power(self, base, power):
        """ shor input being power test """
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

    @idata([
        [15, 4, 2],
        [15, 7, 4],
        [21, 13, 2],
        [35, 8, 4],
    ])
    @unpack
    def test_shor_quantum_result(self, n_v, a_v, order):
        """ shor quantum result test (for order being power of 2) """
        backend = Aer.get_backend('qasm_simulator')
        shor = Shor(quantum_instance=QuantumInstance(backend, shots=1000))

        circuit = shor.construct_circuit(N=n_v, a=a_v, measurement=True)

        result = shor.quantum_instance.execute(circuit)
        measurements = [int(key, base=2) for key in result.get_counts(circuit).keys()]

        # calculate values that could be measured
        values = [i << (2 * n_v.bit_length() - order.bit_length() + 1) for i in range(order)]

        for measurement in measurements:
            self.assertTrue(measurement in values)

    @idata([
        [15, 4, [1, 4]],
        [15, 7, [1, 4, 7, 13]],
        [21, 5, [1, 4, 5, 16, 17, 20]],
        [35, 2, [1, 2, 4, 8, 9, 11, 16, 18, 22, 23, 29, 32]],
    ])
    @unpack
    def test_shor_exponentiation_result(self, n_v, a_v, values):
        """ shor exponentiation result test """
        shor = Shor(quantum_instance=QuantumInstance(Aer.get_backend('qasm_simulator'), shots=1000))

        circuit = shor.construct_circuit(N=n_v, a=a_v, measurement=False)
        # modify circuit to measure output (down) register
        down_qreg = circuit.qregs[1]
        down_creg = ClassicalRegister(len(down_qreg), name='m')
        circuit.add_register(down_creg)
        circuit.measure(down_qreg, down_creg)

        result = shor.quantum_instance.execute(circuit)
        measurements = [int(key, base=2) for key in result.get_counts(circuit).keys()]

        for measurement in measurements:
            self.assertTrue(measurement in values)

    @idata([[2, 15, 8], [4, 15, 4]])
    @unpack
    def test_shor_modinv(self, a_v, m_v, expected):
        """ shor modular inverse test """
        modinv = Shor.modinv(a_v, m_v)
        self.assertTrue(modinv == expected)


if __name__ == '__main__':
    unittest.main()

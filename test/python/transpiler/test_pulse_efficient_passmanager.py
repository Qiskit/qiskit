# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests pulse-efficient pass manager API"""

import unittest

from qiskit.converters import circuit_to_dag

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info.synthesis.two_qubit_decompose import TwoQubitWeylDecomposition
from qiskit.compiler import transpile
from qiskit.test import QiskitTestCase
from qiskit.test.mock import (
    FakeParis,
    FakeAthens,
)

from math import pi
import numpy as np

import qiskit.quantum_info as qi



class TestPulseEfficientTranspilerPass(QiskitTestCase):
    """Test the pulse-efficient pass manager"""

    def setUp(self):
        super().setUp()
        self.passes = []
        self.backends = (FakeParis(), FakeAthens())

    def test_empty_circuit(self):
        """Test empty circuit"""

        circuit = QuantumCircuit()

        unitary_circuit = qi.Operator(circuit).data

        result = transpile(circuit, self.backends[0], optimization_level="pulse_efficient")

        unitary_result = qi.Operator(result).data

        self.assertTrue(np.allclose(unitary_circuit, unitary_result))

    def test_2q_circuit_transpilation(self):
        """Test random two-qubit circuit"""

        theta = 0.2
        epsilon = pi / 3
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])
        circuit.rzx(theta, qr[1], qr[0])
        circuit.sx(qr[1])
        circuit.rzz(epsilon, qr[1], qr[0])
        circuit.swap(qr[1], qr[0])
        circuit.h(qr[1])

        unitary_circuit = qi.Operator(circuit).data

        result = transpile(circuit, self.backends[0], optimization_level="pulse_efficient")

        unitary_result = qi.Operator(result).data

        self.assertTrue(np.allclose(unitary_circuit, unitary_result))

    def test_2q_circuit_rzx_number(self):
        """Test correct number of rzx gates"""

        theta = -2
        epsilon = pi / 9
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.rxx(theta, qr[1], qr[0])
        circuit.x(qr[1])
        circuit.cx(qr[1], qr[0])
        circuit.rzz(epsilon, qr[1], qr[0])
        circuit.cx(qr[1], qr[0])
        circuit.h(qr[1])

        unitary_circuit = qi.Operator(circuit).data

        result = transpile(circuit, self.backends[1], optimization_level="pulse_efficient")

        alpha = TwoQubitWeylDecomposition(unitary_circuit).a
        beta = TwoQubitWeylDecomposition(unitary_circuit).b
        gamma = TwoQubitWeylDecomposition(unitary_circuit).c

        # check whether after circuit has correct number of rzx gates
        expected_rzx_number = 0
        if not alpha == 0:
            expected_rzx_number += 2
        if not beta == 0:
            expected_rzx_number += 2
        if not gamma == 0:
            expected_rzx_number += 2

        circuit_rzx_number = QuantumCircuit.count_ops(result)["rzx"]

        self.assertEqual(expected_rzx_number, circuit_rzx_number)

    def test_alpha_beta_gamma(self):
        """Check if the Weyl parameters match the expected ones for rzz"""

        theta = 1
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.rzz(theta, qr[1], qr[0])

        result = transpile(circuit, self.backends[1], optimization_level="pulse_efficient")

        unitary_result = qi.Operator(result).data

        alpha = TwoQubitWeylDecomposition(unitary_result).a
        beta = TwoQubitWeylDecomposition(unitary_result).b
        gamma = TwoQubitWeylDecomposition(unitary_result).c

        self.assertEqual((alpha, beta, gamma), (0.5, 0, 0))

    def test_5q_circuit_weyl_decomposition(self):
        """Test random five-qubit circuit"""

        delta = 6 * pi / 5
        epsilon = pi / 3
        zeta = -2.1
        theta = 0.2
        qr = QuantumRegister(5, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])
        circuit.rzx(theta, qr[1], qr[0])
        circuit.rzz(epsilon, qr[1], qr[0])
        circuit.swap(qr[1], qr[0])
        circuit.rzz(delta, qr[0], qr[1])
        circuit.swap(qr[1], qr[0])
        circuit.cx(qr[1], qr[0])
        circuit.swap(qr[1], qr[0])
        circuit.h(qr[1])
        circuit.rxx(zeta, qr[1], qr[0])
        circuit.rzz(theta, qr[0], qr[1])
        circuit.swap(qr[3], qr[2])
        circuit.cx(qr[1], qr[2])
        circuit.swap(qr[1], qr[4])
        circuit.h(qr[3])

        unitary_circuit = qi.Operator(circuit).data

        result = transpile(circuit, self.backends[0], optimization_level="pulse_efficient")

        unitary_result = qi.Operator(result).data

        self.assertTrue(np.allclose(unitary_circuit, unitary_result))

    def test_rzx_calibrations(self):
        """Test whether there exist calibrations for rzx"""

        delta = 0.001
        theta = 3.8
        qr = QuantumRegister(5, "qr")
        circuit = QuantumCircuit(qr)
        circuit.rzz(delta, qr[0], qr[1])
        circuit.swap(qr[1], qr[0])
        circuit.cx(qr[1], qr[0])
        circuit.swap(qr[1], qr[0])
        circuit.s(qr[1])
        circuit.rzz(theta, qr[0], qr[1])
        circuit.swap(qr[3], qr[2])
        circuit.cx(qr[1], qr[2])
        circuit.sx(qr[4])
        circuit.swap(qr[1], qr[4])
        circuit.h(qr[3])

        result = transpile(circuit, self.backends[0], optimization_level="pulse_efficient")

        self.assertIn("rzx", result.calibrations)

    def test_params_values(self):
        """Test whether absolute value of rzx angles is smaller than or equals pi"""

        delta = 8 * pi / 5
        epsilon = pi / 2
        zeta = -5.1
        theta = 0.02
        qr = QuantumRegister(5, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])
        circuit.rzx(theta, qr[1], qr[0])
        circuit.rzz(epsilon, qr[1], qr[0])
        circuit.swap(qr[1], qr[0])
        circuit.rzz(delta, qr[0], qr[1])
        circuit.swap(qr[1], qr[0])
        circuit.cx(qr[1], qr[0])
        circuit.swap(qr[1], qr[0])
        circuit.h(qr[1])
        circuit.rxx(zeta, qr[1], qr[0])
        circuit.rzz(theta, qr[0], qr[1])
        circuit.swap(qr[3], qr[2])
        circuit.cx(qr[1], qr[2])
        circuit.swap(qr[1], qr[4])
        circuit.h(qr[3])

        result = transpile(circuit, self.backends[0], optimization_level="pulse_efficient")

        after_dag = circuit_to_dag(result)

        for node in after_dag.nodes():
            if node.type == "op" and node.op.name == "rzx":
                params = node.op.params
                self.assertTrue(abs(params[0]) <= np.pi)


if __name__ == "__main__":
    unittest.main()

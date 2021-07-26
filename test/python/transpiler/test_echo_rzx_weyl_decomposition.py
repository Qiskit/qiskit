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

"""Test the EchoRZXWeylDecomposition pass and the TwoQubitWeylEchoRZX class"""
import unittest
from math import pi
import numpy as np

from qiskit import QuantumRegister, QuantumCircuit

from qiskit.transpiler.passes.optimization.echo_rzx_weyl_decomposition import (
    EchoRZXWeylDecomposition,
)
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeParis

import qiskit.quantum_info as qi

from qiskit.quantum_info.synthesis.two_qubit_decompose import (
    TwoQubitWeylDecomposition,
    TwoQubitWeylEchoRZX,
)


class TestEchoRZXWeylDecomposition(QiskitTestCase):
    """Tests the EchoRZXWeylDecomposition pass and the TwoQubitWeylEchoRZX class."""

    def setUp(self):
        super().setUp()
        self.backend = FakeParis()
        self.inst_map = self.backend.defaults().instruction_schedule_map

    def test_native_weyl_decomposition(self):
        """The CX is in the hardware-native direction"""
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])

        unitary_circuit = qi.Operator(circuit).data

        after = EchoRZXWeylDecomposition(self.inst_map)(circuit)

        unitary_after = qi.Operator(after).data

        self.assertTrue(np.allclose(unitary_circuit, unitary_after))

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

        circuit_rzx_number = after.count_ops()["rzx"]

        self.assertEqual(expected_rzx_number, circuit_rzx_number)

    def test_non_native_weyl_decomposition(self):
        """The RZZ is not in the hardware-native direction"""
        theta = pi / 9
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.rzz(theta, qr[1], qr[0])

        unitary_circuit = qi.Operator(circuit).data

        dag = circuit_to_dag(circuit)
        pass_ = EchoRZXWeylDecomposition(self.inst_map)
        after = dag_to_circuit(pass_.run(dag))

        unitary_after = qi.Operator(after).data

        self.assertTrue(np.allclose(unitary_circuit, unitary_after))

        self.assertRZXgates(unitary_circuit, after)

    def assertRZXgates(self, unitary_circuit, after):
        """Check the number of rzx gates"""
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

        circuit_rzx_number = QuantumCircuit.count_ops(after)["rzx"]

        self.assertEqual(expected_rzx_number, circuit_rzx_number)

    def test_weyl_unitaries_random_circuit(self):
        """Weyl decomposition for random two-qubit circuit."""
        theta = pi / 9
        epsilon = 5
        delta = -1
        eta = 0.2
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)

        # random two-qubit circuit
        circuit.rzx(theta, 0, 1)
        circuit.rzz(epsilon, 0, 1)
        circuit.rz(eta, 0)
        circuit.swap(1, 0)
        circuit.h(0)
        circuit.rzz(delta, 1, 0)
        circuit.swap(0, 1)
        circuit.cx(1, 0)
        circuit.swap(0, 1)
        circuit.h(1)
        circuit.rxx(theta, 0, 1)
        circuit.ryy(theta, 1, 0)
        circuit.ecr(0, 1)

        unitary_circuit = qi.Operator(circuit).data

        dag = circuit_to_dag(circuit)
        pass_ = EchoRZXWeylDecomposition(self.inst_map)
        after = dag_to_circuit(pass_.run(dag))

        unitary_after = qi.Operator(after).data

        self.assertTrue(np.allclose(unitary_circuit, unitary_after))

    def test_weyl_parameters(self):
        """Computation of the correct RZX Weyl parameters"""
        theta = pi / 3
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        qubit_pair = (qr[0], qr[1])

        circuit.rzz(theta, qubit_pair[0], qubit_pair[1])
        circuit.swap(qubit_pair[0], qubit_pair[1])
        unitary_circuit = qi.Operator(circuit).data

        # Weyl parameters (alpha, beta, gamma)
        alpha = TwoQubitWeylDecomposition(unitary_circuit).a
        beta = TwoQubitWeylDecomposition(unitary_circuit).b
        gamma = TwoQubitWeylDecomposition(unitary_circuit).c

        # RZX Weyl parameters (rzx_alpha, rzx_beta, rzx_gamma)
        rzx_alpha = TwoQubitWeylEchoRZX(unitary_circuit, is_native=True).a
        rzx_beta = TwoQubitWeylEchoRZX(unitary_circuit, is_native=True).b
        rzx_gamma = TwoQubitWeylEchoRZX(unitary_circuit, is_native=True).c

        self.assertEqual((alpha, beta, gamma), (rzx_alpha, rzx_beta, rzx_gamma))

    def test_non_native_weyl_parameters(self):
        """Weyl parameters for a non-hardware-native CX direction"""
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        qubit_pair = (qr[1], qr[0])
        circuit.cx(qubit_pair[1], qubit_pair[0])
        unitary_circuit = qi.Operator(circuit).data

        # Weyl parameters (alpha, beta, gamma)
        alpha = TwoQubitWeylDecomposition(unitary_circuit).a
        beta = TwoQubitWeylDecomposition(unitary_circuit).b
        gamma = TwoQubitWeylDecomposition(unitary_circuit).c

        # RZX Weyl parameters (rzx_alpha, rzx_beta, rzx_gamma)
        rzx_alpha = TwoQubitWeylEchoRZX(unitary_circuit, is_native=False).a
        rzx_beta = TwoQubitWeylEchoRZX(unitary_circuit, is_native=False).b
        rzx_gamma = TwoQubitWeylEchoRZX(unitary_circuit, is_native=False).c

        self.assertAlmostEqual(alpha, rzx_alpha)
        self.assertAlmostEqual(beta, rzx_beta)
        self.assertAlmostEqual(gamma, rzx_gamma)


if __name__ == "__main__":
    unittest.main()

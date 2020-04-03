# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test library of quantum circuits."""

from ddt import ddt, data, unpack
import numpy as np

from qiskit.test import QiskitTestCase

from qiskit import execute, BasicAer, transpile
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.library import Permutation, XOR, InnerProduct, QFT


class TestBooleanLogicLibrary(QiskitTestCase):
    """Test library of boolean logic quantum circuits."""

    def test_permutation(self):
        """Test permutation circuit."""
        circuit = Permutation(num_qubits=4, pattern=[1, 0, 3, 2])
        expected = QuantumCircuit(4)
        expected.swap(0, 1)
        expected.swap(2, 3)
        self.assertEqual(circuit, expected)

    def test_permutation_bad(self):
        """Test that [0,..,n-1] permutation is required (no -1 for last element)"""
        self.assertRaises(CircuitError, Permutation, 4, [1, 0, -1, 2])

    def test_xor(self):
        """Test xor circuit."""
        circuit = XOR(num_qubits=3, amount=4)
        expected = QuantumCircuit(3)
        expected.x(2)
        self.assertEqual(circuit, expected)

    def test_inner_product(self):
        """Test inner product circuit."""
        circuit = InnerProduct(num_qubits=3)
        expected = QuantumCircuit(*circuit.qregs)
        expected.cz(0, 3)
        expected.cz(1, 4)
        expected.cz(2, 5)
        self.assertEqual(circuit, expected)


@ddt
class TestBasisChanges(QiskitTestCase):
    """Test the basis changes."""

    def to_matrix(self, circuit):
        """Get the unitary matrix from simulating the circuit with the unitary simulator."""
        backend = BasicAer.get_backend('unitary_simulator')
        return execute(circuit, backend).result().get_unitary()

    def assertQFTIsCorrect(self, qft, num_qubits=None, inverse=False, prepend_swaps=False):
        """Assert that the QFT circuit produces the correct matrix.

        Can be provided with an explicit number of qubits, if None is provided the number
        of qubits is set to ``qft.num_qubits``.
        """
        if prepend_swaps:
            circuit = QuantumCircuit(*qft.qregs)
            for i in range(circuit.num_qubits // 2):
                circuit.swap(i, circuit.num_qubits - i - 1)

            qft = circuit + qft

        simulated = self.to_matrix(qft)

        num_qubits = num_qubits or qft.num_qubits
        expected = np.empty((2 ** num_qubits, 2 ** num_qubits), dtype=complex)
        for i in range(2 ** num_qubits):
            for j in range(i, 2 ** num_qubits):
                entry = np.exp(2 * np.pi * 1j * i * j / 2 ** num_qubits) / 2 ** (num_qubits / 2)
                expected[i, j] = entry
                if i != j:
                    expected[j, i] = entry

        if inverse:
            expected = np.conj(expected)

        np.testing.assert_array_almost_equal(simulated, expected)

    @data(True, False)
    def test_qft_matrix(self, inverse):
        """Test the matrix representation of the QFT."""
        num_qubits = 4
        qft = QFT(num_qubits)
        if inverse:
            qft = qft.inverse()
        self.assertQFTIsCorrect(qft, inverse=inverse)

    def test_qft_mutability(self):
        """Test the mutability of the QFT circuit."""
        qft = QFT()

        with self.subTest(msg='empty initialization'):
            self.assertEqual(qft.num_qubits, 0)
            self.assertEqual(qft.data, [])

        with self.subTest(msg='changing number of qubits'):
            qft.num_qubits = 3
            self.assertQFTIsCorrect(qft, num_qubits=3)

        with self.subTest(msg='test diminishing the number of qubits'):
            qft.num_qubits = 1
            self.assertQFTIsCorrect(qft, num_qubits=1)

        with self.subTest(msg='test with swaps'):
            qft.num_qubits = 4
            qft.do_swaps = False
            self.assertQFTIsCorrect(qft, prepend_swaps=True)

        with self.subTest(msg='set approximation'):
            qft.approximation_degree = 2
            qft.do_swaps = True
            with self.assertRaises(AssertionError):
                self.assertQFTIsCorrect(qft)

    @data(
        (4, 0),
        (4, 2),
        (4, 5),
    )
    @unpack
    def test_qft_num_gates(self, num_qubits, approximation_degree):
        """Test the number of gates in the QFT and the approximated QFT."""
        basis_gates = ['h', 'swap', 'cu1']

        qft = QFT(num_qubits, approximation_degree=approximation_degree)
        ops = transpile(qft, basis_gates=basis_gates).count_ops()
        print(ops)
        print(qft.draw())

        with self.subTest(msg='assert H count'):
            self.assertEqual(ops['h'], num_qubits)

        with self.subTest(msg='assert swap count'):
            self.assertEqual(ops['swap'], num_qubits // 2)

        with self.subTest(msg='assert CU1 count'):
            expected = sum(max(0, min(num_qubits - 1 - k, num_qubits - 1 - approximation_degree))
                           for k in range(num_qubits))
            self.assertEqual(ops.get('cu1', 0), expected)

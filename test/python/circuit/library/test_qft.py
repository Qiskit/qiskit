# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test library of QFT circuits."""

import unittest
import numpy as np
from ddt import ddt, data, unpack

from qiskit.test.base import QiskitTestCase
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit.quantum_info import Operator


@ddt
class TestQFT(QiskitTestCase):
    """Test the QFT."""

    def assertQFTIsCorrect(self, qft, num_qubits=None, inverse=False, add_swaps_at_end=False):
        """Assert that the QFT circuit produces the correct matrix.

        Can be provided with an explicit number of qubits, if None is provided the number
        of qubits is set to ``qft.num_qubits``.
        """
        if add_swaps_at_end:
            circuit = QuantumCircuit(*qft.qregs)
            for i in range(circuit.num_qubits // 2):
                circuit.swap(i, circuit.num_qubits - i - 1)

            qft.compose(circuit, inplace=True)

        simulated = Operator(qft)

        num_qubits = num_qubits or qft.num_qubits
        expected = np.empty((2 ** num_qubits, 2 ** num_qubits), dtype=complex)
        for i in range(2 ** num_qubits):
            i_index = int(bin(i)[2:].zfill(num_qubits), 2)
            for j in range(i, 2 ** num_qubits):
                entry = np.exp(2 * np.pi * 1j * i * j / 2 ** num_qubits) / 2 ** (num_qubits / 2)
                j_index = int(bin(j)[2:].zfill(num_qubits), 2)
                expected[i_index, j_index] = entry
                if i != j:
                    expected[j_index, i_index] = entry

        if inverse:
            expected = np.conj(expected)

        expected = Operator(expected)

        self.assertTrue(expected.equiv(simulated))

    @data(True, False)
    def test_qft_matrix(self, inverse):
        """Test the matrix representation of the QFT."""
        num_qubits = 5
        qft = QFT(num_qubits)
        if inverse:
            qft = qft.inverse()
        self.assertQFTIsCorrect(qft, inverse=inverse)

    def test_qft_is_inverse(self):
        """Test the is_inverse() method."""
        qft = QFT(2)

        with self.subTest(msg="initial object is not inverse"):
            self.assertFalse(qft.is_inverse())

        qft = qft.inverse()
        with self.subTest(msg="inverted"):
            self.assertTrue(qft.is_inverse())

        qft = qft.inverse()
        with self.subTest(msg="re-inverted"):
            self.assertFalse(qft.is_inverse())

    def test_qft_mutability(self):
        """Test the mutability of the QFT circuit."""
        qft = QFT()

        with self.subTest(msg="empty initialization"):
            self.assertEqual(qft.num_qubits, 0)
            self.assertEqual(qft.data, [])

        with self.subTest(msg="changing number of qubits"):
            qft.num_qubits = 3
            self.assertQFTIsCorrect(qft, num_qubits=3)

        with self.subTest(msg="test diminishing the number of qubits"):
            qft.num_qubits = 1
            self.assertQFTIsCorrect(qft, num_qubits=1)

        with self.subTest(msg="test with swaps"):
            qft.num_qubits = 4
            qft.do_swaps = False
            self.assertQFTIsCorrect(qft, add_swaps_at_end=True)

        with self.subTest(msg="inverse"):
            qft = qft.inverse()
            qft.do_swaps = True
            self.assertQFTIsCorrect(qft, inverse=True)

        with self.subTest(msg="double inverse"):
            qft = qft.inverse()
            self.assertQFTIsCorrect(qft)

        with self.subTest(msg="set approximation"):
            qft.approximation_degree = 2
            qft.do_swaps = True
            with self.assertRaises(AssertionError):
                self.assertQFTIsCorrect(qft)

    @data(
        (4, 0, False),
        (3, 0, True),
        (6, 2, False),
        (4, 5, True),
    )
    @unpack
    def test_qft_num_gates(self, num_qubits, approximation_degree, insert_barriers):
        """Test the number of gates in the QFT and the approximated QFT."""
        basis_gates = ["h", "swap", "cu1"]

        qft = QFT(
            num_qubits, approximation_degree=approximation_degree, insert_barriers=insert_barriers
        )
        ops = transpile(qft, basis_gates=basis_gates).count_ops()

        with self.subTest(msg="assert H count"):
            self.assertEqual(ops["h"], num_qubits)

        with self.subTest(msg="assert swap count"):
            self.assertEqual(ops["swap"], num_qubits // 2)

        with self.subTest(msg="assert CU1 count"):
            expected = sum(
                max(0, min(num_qubits - 1 - k, num_qubits - 1 - approximation_degree))
                for k in range(num_qubits)
            )
            self.assertEqual(ops.get("cu1", 0), expected)

        with self.subTest(msg="assert barrier count"):
            expected = qft.num_qubits if insert_barriers else 0
            self.assertEqual(ops.get("barrier", 0), expected)


if __name__ == "__main__":
    unittest.main()

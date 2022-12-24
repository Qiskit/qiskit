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

# pylint: disable=invalid-name

"""Tests for Pauli operator class."""

import itertools as it
import unittest
from functools import lru_cache
from test.python.opflow import QiskitOpflowTestCase

import numpy as np
from ddt import data, ddt, unpack
from scipy.sparse import csr_matrix

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.opflow import DictStateFn, EvolvedOp, I, PauliOp, SummedOp, X, Y, Z, Zero
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.quantum_info.operators.symplectic.pauli import _phase_from_label, _split_pauli_label


@lru_cache(maxsize=8)
def pauli_group_labels(nq, full_group=True):
    """Generate list of the N-qubit pauli group string labels"""
    labels = ["".join(i) for i in it.product(("I", "X", "Y", "Z"), repeat=nq)]
    if full_group:
        labels = ["".join(i) for i in it.product(("", "-i", "-", "i"), labels)]
    return labels


def operator_from_label(label):
    """Construct operator from full Pauli group label"""
    pauli, coeff = _split_pauli_label(label)
    coeff = (-1j) ** _phase_from_label(coeff)
    return coeff * Operator.from_label(pauli)


@ddt
class TestPauliOp(QiskitOpflowTestCase):
    """PauliOp tests."""

    def test_construct(self):
        """constructor test"""
        pauli = Pauli("XYZX")
        coeff = 3.0
        pauli_op = PauliOp(pauli, coeff)
        self.assertIsInstance(pauli_op, PauliOp)
        self.assertEqual(pauli_op.primitive, pauli)
        self.assertEqual(pauli_op.coeff, coeff)
        self.assertEqual(pauli_op.num_qubits, 4)

    def test_add(self):
        """add test"""
        pauli_sum = X + Y
        summed_op = SummedOp([X, Y])
        self.assertEqual(pauli_sum, summed_op)

        a = Parameter("a")
        b = Parameter("b")
        actual = PauliOp(Pauli("X"), a) + PauliOp(Pauli("Y"), b)
        expected = SummedOp([PauliOp(Pauli("X"), a), PauliOp(Pauli("Y"), b)])
        self.assertEqual(actual, expected)

    def test_adjoint(self):
        """adjoint test"""
        pauli_op = PauliOp(Pauli("XYZX"), coeff=3)
        expected = PauliOp(Pauli("XYZX"), coeff=3)

        self.assertEqual(~pauli_op, expected)

        pauli_op = PauliOp(Pauli("XXY"), coeff=2j)
        expected = PauliOp(Pauli("XXY"), coeff=-2j)
        self.assertEqual(~pauli_op, expected)

        pauli_op = PauliOp(Pauli("XYZX"), coeff=2 + 3j)
        expected = PauliOp(Pauli("XYZX"), coeff=2 - 3j)
        self.assertEqual(~pauli_op, expected)

    @data(*it.product(pauli_group_labels(2, full_group=True), repeat=2))
    @unpack
    def test_compose(self, label1, label2):
        """compose test"""
        p1 = PauliOp(Pauli(label1))
        p2 = PauliOp(Pauli(label2))
        value = Operator(p1 @ p2)
        op1 = operator_from_label(label1)
        op2 = operator_from_label(label2)
        target = op1 @ op2
        self.assertEqual(value, target)

    def test_equals(self):
        """equality test"""

        self.assertEqual(I @ X, X)
        self.assertEqual(X, I @ X)

        theta = Parameter("theta")
        pauli_op = theta * X ^ Z
        expected = PauliOp(
            Pauli("XZ"),
            coeff=1.0 * theta,
        )
        self.assertEqual(pauli_op, expected)

    def test_eval(self):
        """eval test"""
        target0 = (X ^ Y ^ Z).eval("000")
        target1 = (X ^ Y ^ Z).eval(Zero ^ 3)
        expected = DictStateFn({"110": 1j})
        self.assertEqual(target0, expected)
        self.assertEqual(target1, expected)

    def test_exp_i(self):
        """exp_i test"""
        target = (2 * X ^ Z).exp_i()
        expected = EvolvedOp(PauliOp(Pauli("XZ"), coeff=2.0), coeff=1.0)
        self.assertEqual(target, expected)

    @data(([1, 2, 4], "XIYZI"), ([2, 1, 0], "ZYX"))
    @unpack
    def test_permute(self, permutation, expected_pauli):
        """Test the permute method."""
        pauli_op = PauliOp(Pauli("XYZ"), coeff=1.0)
        expected = PauliOp(Pauli(expected_pauli), coeff=1.0)
        permuted = pauli_op.permute(permutation)

        with self.subTest(msg="test permutated object"):
            self.assertEqual(permuted, expected)

        with self.subTest(msg="test original object is unchanged"):
            original = PauliOp(Pauli("XYZ"))
            self.assertEqual(pauli_op, original)

    def test_primitive_strings(self):
        """primitive strings test"""
        target = (2 * X ^ Z).primitive_strings()
        expected = {"Pauli"}
        self.assertEqual(target, expected)

    def test_tensor(self):
        """tensor test"""
        pauli_op = X ^ Y ^ Z
        tensored_op = PauliOp(Pauli("XYZ"))
        self.assertEqual(pauli_op, tensored_op)

    def test_to_instruction(self):
        """to_instruction test"""
        target = (X ^ Z).to_instruction()
        qc = QuantumCircuit(2)
        qc.u(0, 0, np.pi, 0)
        qc.u(np.pi, 0, np.pi, 1)
        qc_out = QuantumCircuit(2)
        qc_out.append(target, qc_out.qubits)
        qc_out = transpile(qc_out, basis_gates=["u"])
        self.assertEqual(qc, qc_out)

    def test_to_matrix(self):
        """to_matrix test"""
        target = (X ^ Y).to_matrix()
        expected = np.kron(np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([[0.0, -1j], [1j, 0.0]]))
        np.testing.assert_array_equal(target, expected)

    def test_to_spmatrix(self):
        """to_spmatrix test"""
        target = X ^ Y
        expected = csr_matrix(
            np.kron(np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([[0.0, -1j], [1j, 0.0]]))
        )
        self.assertEqual((target.to_spmatrix() - expected).nnz, 0)


if __name__ == "__main__":
    unittest.main()

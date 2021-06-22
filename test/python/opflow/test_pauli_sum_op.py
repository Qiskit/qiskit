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

""" Test PauliSumOp """

import unittest
from itertools import product
from test.python.opflow import QiskitOpflowTestCase

import numpy as np
from scipy.sparse import csr_matrix

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter, ParameterVector
from qiskit.opflow import (
    CX,
    CircuitStateFn,
    DictStateFn,
    H,
    I,
    One,
    OperatorStateFn,
    PauliSumOp,
    SummedOp,
    X,
    Y,
    Z,
    Zero,
)
from qiskit.quantum_info import Pauli, PauliTable, SparsePauliOp


class TestPauliSumOp(QiskitOpflowTestCase):
    """PauliSumOp tests."""

    def test_construct(self):
        """constructor test"""
        sparse_pauli = SparsePauliOp(Pauli("XYZX"), coeffs=[2.0])
        coeff = 3.0
        pauli_sum = PauliSumOp(sparse_pauli, coeff=coeff)
        self.assertIsInstance(pauli_sum, PauliSumOp)
        self.assertEqual(pauli_sum.primitive, sparse_pauli)
        self.assertEqual(pauli_sum.coeff, coeff)
        self.assertEqual(pauli_sum.num_qubits, 4)

    def test_add(self):
        """add test"""
        pauli_sum = 3 * X + Y
        self.assertIsInstance(pauli_sum, PauliSumOp)
        expected = PauliSumOp(3.0 * SparsePauliOp(Pauli("X")) + SparsePauliOp(Pauli("Y")))
        self.assertEqual(pauli_sum, expected)

        pauli_sum = X + Y
        summed_op = SummedOp([X, Y])
        self.assertEqual(pauli_sum, summed_op)

        a = Parameter("a")
        b = Parameter("b")
        actual = a * PauliSumOp.from_list([("X", 2)]) + b * PauliSumOp.from_list([("Y", 1)])
        expected = SummedOp(
            [PauliSumOp.from_list([("X", 2)], a), PauliSumOp.from_list([("Y", 1)], b)]
        )
        self.assertEqual(actual, expected)

    def test_mul(self):
        """multiplication test"""
        target = 2 * (X + Z)
        self.assertEqual(target.coeff, 1)
        self.assertListEqual(target.primitive.to_list(), [("X", (2 + 0j)), ("Z", (2 + 0j))])

        target = 0 * (X + Z)
        self.assertEqual(target.coeff, 0)
        self.assertListEqual(target.primitive.to_list(), [("X", (1 + 0j)), ("Z", (1 + 0j))])

        beta = Parameter("β")
        target = beta * (X + Z)
        self.assertEqual(target.coeff, 1.0 * beta)
        self.assertListEqual(target.primitive.to_list(), [("X", (1 + 0j)), ("Z", (1 + 0j))])

    def test_adjoint(self):
        """adjoint test"""
        pauli_sum = PauliSumOp(SparsePauliOp(Pauli("XYZX"), coeffs=[2]), coeff=3)
        expected = PauliSumOp(SparsePauliOp(Pauli("XYZX")), coeff=6)

        self.assertEqual(pauli_sum.adjoint(), expected)

        pauli_sum = PauliSumOp(SparsePauliOp(Pauli("XYZY"), coeffs=[2]), coeff=3j)
        expected = PauliSumOp(SparsePauliOp(Pauli("XYZY")), coeff=-6j)
        self.assertEqual(pauli_sum.adjoint(), expected)

        pauli_sum = PauliSumOp(SparsePauliOp(Pauli("X"), coeffs=[1]))
        self.assertEqual(pauli_sum.adjoint(), pauli_sum)

        pauli_sum = PauliSumOp(SparsePauliOp(Pauli("Y"), coeffs=[1]))
        self.assertEqual(pauli_sum.adjoint(), pauli_sum)

        pauli_sum = PauliSumOp(SparsePauliOp(Pauli("Z"), coeffs=[1]))
        self.assertEqual(pauli_sum.adjoint(), pauli_sum)

        pauli_sum = (Z ^ Z) + (Y ^ I)
        self.assertEqual(pauli_sum.adjoint(), pauli_sum)

    def test_equals(self):
        """equality test"""

        self.assertNotEqual((X ^ X) + (Y ^ Y), X + Y)
        self.assertEqual((X ^ X) + (Y ^ Y), (Y ^ Y) + (X ^ X))

        theta = ParameterVector("theta", 2)
        pauli_sum0 = theta[0] * (X + Z)
        pauli_sum1 = theta[1] * (X + Z)
        expected = PauliSumOp(
            SparsePauliOp(Pauli("X")) + SparsePauliOp(Pauli("Z")),
            coeff=1.0 * theta[0],
        )
        self.assertEqual(pauli_sum0, expected)
        self.assertNotEqual(pauli_sum1, expected)

    def test_tensor(self):
        """Test for tensor operation"""
        with self.subTest("Test 1"):
            pauli_sum = ((I - Z) ^ (I - Z)) + ((X - Y) ^ (X + Y))
            expected = (I ^ I) - (I ^ Z) - (Z ^ I) + (Z ^ Z) + (X ^ X) + (X ^ Y) - (Y ^ X) - (Y ^ Y)
            self.assertEqual(pauli_sum, expected)

        with self.subTest("Test 2"):
            pauli_sum = (Z + I) ^ Z
            expected = (Z ^ Z) + (I ^ Z)
            self.assertEqual(pauli_sum, expected)

        with self.subTest("Test 3"):
            pauli_sum = Z ^ (Z + I)
            expected = (Z ^ Z) + (Z ^ I)
            self.assertEqual(pauli_sum, expected)

    def test_permute(self):
        """permute test"""
        pauli_sum = PauliSumOp(SparsePauliOp((X ^ Y ^ Z).primitive))
        expected = PauliSumOp(SparsePauliOp((X ^ I ^ Y ^ Z ^ I).primitive))

        self.assertEqual(pauli_sum.permute([1, 2, 4]), expected)

    def test_compose(self):
        """compose test"""
        target = (X + Z) @ (Y + Z)
        expected = 1j * Z - 1j * Y - 1j * X + I
        self.assertEqual(target, expected)

        observable = (X ^ X) + (Y ^ Y) + (Z ^ Z)
        state = CircuitStateFn((CX @ (X ^ H @ X)).to_circuit())
        self.assertAlmostEqual((~OperatorStateFn(observable) @ state).eval(), -3)

    def test_to_matrix(self):
        """test for to_matrix method"""
        target = (Z + Y).to_matrix()
        expected = np.array([[1.0, -1j], [1j, -1]])
        np.testing.assert_array_equal(target, expected)

    def test_str(self):
        """str test"""
        target = 3.0 * (X + 2.0 * Y - 4.0 * Z)
        expected = "3.0 * X\n+ 6.0 * Y\n- 12.0 * Z"
        self.assertEqual(str(target), expected)

        alpha = Parameter("α")
        target = alpha * (X + 2.0 * Y - 4.0 * Z)
        expected = "1.0*α * (\n  1.0 * X\n  + 2.0 * Y\n  - 4.0 * Z\n)"
        self.assertEqual(str(target), expected)

    def test_eval(self):
        """eval test"""
        target0 = (2 * (X ^ Y ^ Z) + 3 * (X ^ X ^ Z)).eval("000")
        target1 = (2 * (X ^ Y ^ Z) + 3 * (X ^ X ^ Z)).eval(Zero ^ 3)
        expected = DictStateFn({"110": (3 + 2j)})
        self.assertEqual(target0, expected)
        self.assertEqual(target1, expected)

        phi = 0.5 * ((One + Zero) ^ 2)
        zero_op = (Z + I) / 2
        one_op = (I - Z) / 2
        h1 = one_op ^ I
        h2 = one_op ^ (one_op + zero_op)
        h2a = one_op ^ one_op
        h2b = one_op ^ zero_op
        self.assertEqual((~OperatorStateFn(h1) @ phi).eval(), 0.5)
        self.assertEqual((~OperatorStateFn(h2) @ phi).eval(), 0.5)
        self.assertEqual((~OperatorStateFn(h2a) @ phi).eval(), 0.25)
        self.assertEqual((~OperatorStateFn(h2b) @ phi).eval(), 0.25)

        pauli_op = (Z ^ I ^ X) + (I ^ I ^ Y)
        mat_op = pauli_op.to_matrix_op()
        full_basis = ["".join(b) for b in product("01", repeat=pauli_op.num_qubits)]
        for bstr1, bstr2 in product(full_basis, full_basis):
            self.assertEqual(pauli_op.eval(bstr1).eval(bstr2), mat_op.eval(bstr1).eval(bstr2))

    def test_exp_i(self):
        """exp_i test"""
        # TODO: add tests when special methods are added
        pass

    def test_to_instruction(self):
        """test for to_instruction"""
        target = ((X + Z) / np.sqrt(2)).to_instruction()
        qc = QuantumCircuit(1)
        qc.u(np.pi / 2, 0, np.pi, 0)
        qc_out = transpile(target.definition, basis_gates=["u"])
        self.assertEqual(qc_out, qc)

    def test_to_pauli_op(self):
        """test to_pauli_op method"""
        target = X + Y
        self.assertIsInstance(target, PauliSumOp)
        expected = SummedOp([X, Y])
        self.assertEqual(target.to_pauli_op(), expected)

    def test_getitem(self):
        """test get item method"""
        target = X + Z
        self.assertEqual(target[0], PauliSumOp(SparsePauliOp(X.primitive)))
        self.assertEqual(target[1], PauliSumOp(SparsePauliOp(Z.primitive)))

    def test_len(self):
        """test len"""
        target = X + Y + Z
        self.assertEqual(len(target), 3)

    def test_reduce(self):
        """test reduce"""
        target = X + X + Z
        self.assertEqual(len(target.reduce()), 2)

    def test_to_spmatrix(self):
        """test to_spmatrix"""
        target = X + Y
        expected = csr_matrix([[0, 1 - 1j], [1 + 1j, 0]])
        self.assertEqual((target.to_spmatrix() - expected).nnz, 0)

    def test_from_list(self):
        """test from_list"""
        target = PauliSumOp.from_list(
            [
                ("II", -1.052373245772859),
                ("IZ", 0.39793742484318045),
                ("ZI", -0.39793742484318045),
                ("ZZ", -0.01128010425623538),
                ("XX", 0.18093119978423156),
            ]
        )
        expected = (
            -1.052373245772859 * (I ^ I)
            + 0.39793742484318045 * (I ^ Z)
            - 0.39793742484318045 * (Z ^ I)
            - 0.01128010425623538 * (Z ^ Z)
            + 0.18093119978423156 * (X ^ X)
        )
        self.assertEqual(target, expected)

    def test_matrix_iter(self):
        """Test PauliSumOp dense matrix_iter method."""
        labels = ["III", "IXI", "IYY", "YIZ", "XYZ", "III"]
        coeffs = np.array([1, 2, 3, 4, 5, 6])
        table = PauliTable.from_labels(labels)
        coeff = 10
        op = PauliSumOp(SparsePauliOp(table, coeffs), coeff)
        for idx, i in enumerate(op.matrix_iter()):
            self.assertTrue(np.array_equal(i, coeff * coeffs[idx] * Pauli(labels[idx]).to_matrix()))

    def test_matrix_iter_sparse(self):
        """Test PauliSumOp sparse matrix_iter method."""
        labels = ["III", "IXI", "IYY", "YIZ", "XYZ", "III"]
        coeffs = np.array([1, 2, 3, 4, 5, 6])
        coeff = 10
        table = PauliTable.from_labels(labels)
        op = PauliSumOp(SparsePauliOp(table, coeffs), coeff)
        for idx, i in enumerate(op.matrix_iter(sparse=True)):
            self.assertTrue(
                np.array_equal(i.toarray(), coeff * coeffs[idx] * Pauli(labels[idx]).to_matrix())
            )


if __name__ == "__main__":
    unittest.main()

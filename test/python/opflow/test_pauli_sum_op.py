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
from test.aqua import QiskitAquaTestCase

import numpy as np
from scipy.sparse import csr_matrix

from qiskit import QuantumCircuit, transpile
from qiskit.aqua.operators import (
    CX,
    CircuitStateFn,
    DictStateFn,
    H,
    I,
    OperatorStateFn,
    PauliSumOp,
    SummedOp,
    X,
    Y,
    Z,
    Zero,
)
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import Pauli, SparsePauliOp


class TestPauliSumOp(QiskitAquaTestCase):
    """PauliSumOp tests."""

    def test_construct(self):
        """ constructor test """
        sparse_pauli = SparsePauliOp(Pauli(label="XYZX"), coeffs=[2.0])
        coeff = 3.0
        pauli_sum = PauliSumOp(sparse_pauli, coeff=coeff)
        self.assertIsInstance(pauli_sum, PauliSumOp)
        self.assertEqual(pauli_sum.primitive, sparse_pauli)
        self.assertEqual(pauli_sum.coeff, coeff)
        self.assertEqual(pauli_sum.num_qubits, 4)

    def test_add(self):
        """ add test """
        pauli_sum = 3 * X + Y
        self.assertIsInstance(pauli_sum, PauliSumOp)
        expected = PauliSumOp(
            3.0 * SparsePauliOp(Pauli(label="X")) + SparsePauliOp(Pauli(label="Y"))
        )
        self.assertEqual(pauli_sum, expected)

        pauli_sum = X + Y
        summed_op = SummedOp([X, Y])
        self.assertEqual(pauli_sum, summed_op)
        self.assertEqual(summed_op, pauli_sum)

    def test_mul(self):
        """ multiplication test """
        target = 2 * (X + Z)
        self.assertEqual(target.coeff, 1)
        self.assertListEqual(
            target.primitive.to_list(), [("X", (2 + 0j)), ("Z", (2 + 0j))]
        )

        target = 0 * (X + Z)
        self.assertEqual(target.coeff, 0)
        self.assertListEqual(
            target.primitive.to_list(), [("X", (1 + 0j)), ("Z", (1 + 0j))]
        )

        beta = Parameter("β")
        target = beta * (X + Z)
        self.assertEqual(target.coeff, 1.0 * beta)
        self.assertListEqual(
            target.primitive.to_list(), [("X", (1 + 0j)), ("Z", (1 + 0j))]
        )

    def test_adjoint(self):
        """ adjoint test """
        pauli_sum = PauliSumOp(SparsePauliOp(Pauli(label="XYZX"), coeffs=[2]), coeff=3)
        expected = PauliSumOp(SparsePauliOp(Pauli(label="XYZX")), coeff=-6)

        self.assertEqual(pauli_sum.adjoint(), expected)

        pauli_sum = PauliSumOp(SparsePauliOp(Pauli(label="XYZY"), coeffs=[2]), coeff=3j)
        expected = PauliSumOp(SparsePauliOp(Pauli(label="XYZY")), coeff=-6j)
        self.assertEqual(pauli_sum.adjoint(), expected)

    def test_equals(self):
        """ equality test """

        self.assertNotEqual((X ^ X) + (Y ^ Y), X + Y)
        self.assertEqual((X ^ X) + (Y ^ Y), (Y ^ Y) + (X ^ X))

        theta = ParameterVector("theta", 2)
        pauli_sum0 = theta[0] * (X + Z)
        pauli_sum1 = theta[1] * (X + Z)
        expected = PauliSumOp(
            SparsePauliOp(Pauli(label="X")) + SparsePauliOp(Pauli(label="Z")),
            coeff=1.0 * theta[0],
        )
        self.assertEqual(pauli_sum0, expected)
        self.assertNotEqual(pauli_sum1, expected)

    def test_tensor(self):
        """ Test for tensor operation """
        pauli_sum = ((I - Z) ^ (I - Z)) + ((X - Y) ^ (X + Y))
        expected = (
            (I ^ I)
            - (I ^ Z)
            - (Z ^ I)
            + (Z ^ Z)
            + (X ^ X)
            + (X ^ Y)
            - (Y ^ X)
            - (Y ^ Y)
        )
        self.assertEqual(pauli_sum, expected)

    def test_permute(self):
        """ permute test """
        pauli_sum = PauliSumOp(SparsePauliOp((X ^ Y ^ Z).primitive))
        expected = PauliSumOp(SparsePauliOp((X ^ I ^ Y ^ Z ^ I).primitive))

        self.assertEqual(pauli_sum.permute([1, 2, 4]), expected)

    def test_compose(self):
        """ compose test """
        target = (X + Z) @ (Y + Z)
        expected = 1j * Z - 1j * Y - 1j * X + I
        self.assertEqual(target, expected)

        observable = (X ^ X) + (Y ^ Y) + (Z ^ Z)
        state = CircuitStateFn((CX @ (X ^ H @ X)).to_circuit())
        self.assertAlmostEqual((~OperatorStateFn(observable) @ state).eval(), -3)

    def test_to_matrix(self):
        """ test for to_matrix method """
        target = (Z + Y).to_matrix()
        expected = np.array([[1.0, -1j], [1j, -1]])
        np.testing.assert_array_equal(target, expected)

    def test_str(self):
        """ str test """
        target = 3.0 * (X + 2.0 * Y - 4.0 * Z)
        expected = "3.0 * X\n+ 6.0 * Y\n- 12.0 * Z"
        self.assertEqual(str(target), expected)

        alpha = Parameter("α")
        target = alpha * (X + 2.0 * Y - 4.0 * Z)
        expected = "1.0*α * (\n  1.0 * X\n  + 2.0 * Y\n  - 4.0 * Z\n)"
        self.assertEqual(str(target), expected)

    def test_eval(self):
        """ eval test """
        target0 = (2 * (X ^ Y ^ Z) + 3 * (X ^ X ^ Z)).eval("000")
        target1 = (2 * (X ^ Y ^ Z) + 3 * (X ^ X ^ Z)).eval(Zero ^ 3)
        expected = DictStateFn({"011": (2 + 3j)})
        self.assertEqual(target0, expected)
        self.assertEqual(target1, expected)

    def test_exp_i(self):
        """ exp_i test """
        # TODO: add tests when special methods are added
        pass

    def test_to_instruction(self):
        """ test for to_instruction """
        target = ((X + Z) / np.sqrt(2)).to_instruction()
        qc = QuantumCircuit(1)
        qc.u(np.pi / 2, 0, np.pi, 0)
        self.assertEqual(transpile(target.definition, basis_gates=["u"]), qc)

    def test_to_pauli_op(self):
        """ test to_pauli_op method """
        target = X + Y
        self.assertIsInstance(target, PauliSumOp)
        expected = SummedOp([X, Y])
        self.assertEqual(target.to_pauli_op(), expected)

    def test_getitem(self):
        """ test get item method """
        target = X + Z
        self.assertEqual(target[0], PauliSumOp(SparsePauliOp(X.primitive)))
        self.assertEqual(target[1], PauliSumOp(SparsePauliOp(Z.primitive)))

    def test_len(self):
        """ test len """
        target = X + Y + Z
        self.assertEqual(len(target), 3)

    def test_reduce(self):
        """ test reduce """
        target = X + X + Z
        self.assertEqual(len(target.reduce()), 2)

    def test_to_spmatrix(self):
        """ test to_spmatrix """
        target = X + Y
        expected = csr_matrix([[0, 1 - 1j], [1 + 1j, 0]])
        self.assertEqual((target.to_spmatrix() - expected).nnz, 0)

    def test_from_list(self):
        """ test from_list """
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


if __name__ == "__main__":
    unittest.main()

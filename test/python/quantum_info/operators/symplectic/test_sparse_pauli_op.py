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

"""Tests for SparsePauliOp class."""

import unittest
import itertools as it
from test import combine
from ddt import ddt
import numpy as np

from qiskit import QiskitError
from qiskit.test import QiskitTestCase
from qiskit.quantum_info.random import random_unitary
from qiskit.quantum_info.operators import Operator, SparsePauliOp, PauliTable


def pauli_mat(label):
    """Return Pauli matrix from a Pauli label"""
    mat = np.eye(1, dtype=complex)
    for i in label:
        if i == "I":
            mat = np.kron(mat, np.eye(2, dtype=complex))
        elif i == "X":
            mat = np.kron(mat, np.array([[0, 1], [1, 0]], dtype=complex))
        elif i == "Y":
            mat = np.kron(mat, np.array([[0, -1j], [1j, 0]], dtype=complex))
        elif i == "Z":
            mat = np.kron(mat, np.array([[1, 0], [0, -1]], dtype=complex))
        else:
            raise QiskitError(f"Invalid Pauli string {i}")
    return mat


class TestSparsePauliOpInit(QiskitTestCase):
    """Tests for SparsePauliOp initialization."""

    def test_pauli_table_init(self):
        """Test PauliTable initialization."""
        labels = ["I", "X", "Y", "Z"]
        table = PauliTable.from_labels(labels)
        with self.subTest(msg="no coeffs"):
            spp_op = SparsePauliOp(table)
            self.assertTrue(np.array_equal(spp_op.coeffs, np.ones(len(labels))))
            self.assertEqual(spp_op.table, table)
        with self.subTest(msg="no coeffs"):
            coeffs = [1, 2, 3, 4]
            spp_op = SparsePauliOp(table, coeffs)
            self.assertTrue(np.array_equal(spp_op.coeffs, coeffs))
            self.assertEqual(spp_op.table, table)

    def test_str_init(self):
        """Test PauliTable initialization."""
        for label in ["IZ", "XI", "YX", "ZZ"]:
            table = PauliTable(label)
            spp_op = SparsePauliOp(label)
            self.assertEqual(spp_op.table, table)
            self.assertTrue(np.array_equal(spp_op.coeffs, [1]))


class TestSparsePauliOpConversions(QiskitTestCase):
    """Tests SparsePauliOp representation conversions."""

    def test_from_operator(self):
        """Test from_operator methods."""
        for tup in it.product(["I", "X", "Y", "Z"], repeat=2):
            label = "".join(tup)
            with self.subTest(msg=label):
                spp_op = SparsePauliOp.from_operator(Operator(pauli_mat(label)))
                self.assertTrue(np.array_equal(spp_op.coeffs, [1]))
                self.assertEqual(spp_op.table, PauliTable(label))

    def test_from_list(self):
        """Test from_list method."""
        labels = ["XXZ", "IXI", "YZZ", "III"]
        coeffs = [3.0, 5.5, -1j, 23.3333]
        spp_op = SparsePauliOp.from_list(list(zip(labels, coeffs)))
        self.assertTrue(np.array_equal(spp_op.coeffs, coeffs))
        self.assertEqual(spp_op.table, PauliTable.from_labels(labels))

    def test_from_zip(self):
        """Test from_list method for zipped input."""
        labels = ["XXZ", "IXI", "YZZ", "III"]
        coeffs = [3.0, 5.5, -1j, 23.3333]
        spp_op = SparsePauliOp.from_list(zip(labels, coeffs))
        self.assertTrue(np.array_equal(spp_op.coeffs, coeffs))
        self.assertEqual(spp_op.table, PauliTable.from_labels(labels))

    def to_matrix(self):
        """Test to_matrix method."""
        labels = ["XI", "YZ", "YY", "ZZ"]
        coeffs = [-3, 4.4j, 0.2 - 0.1j, 66.12]
        spp_op = SparsePauliOp(PauliTable.from_labels(labels), coeffs)
        target = np.zeros((4, 4), dtype=complex)
        for coeff, label in zip(coeffs, labels):
            target += coeff * pauli_mat(label)
        self.assertTrue(np.array_equal(spp_op.to_matrix(), target))

    def to_operator(self):
        """Test to_operator method."""
        labels = ["XI", "YZ", "YY", "ZZ"]
        coeffs = [-3, 4.4j, 0.2 - 0.1j, 66.12]
        spp_op = SparsePauliOp(PauliTable.from_labels(labels), coeffs)
        target = Operator(np.zeros((4, 4), dtype=complex))
        for coeff, label in zip(coeffs, labels):
            target = target + Operator(coeff * pauli_mat(label))
        self.assertEqual(spp_op.to_operator(), target)

    def to_list(self):
        """Test to_operator method."""
        labels = ["XI", "YZ", "YY", "ZZ"]
        coeffs = [-3, 4.4j, 0.2 - 0.1j, 66.12]
        op = SparsePauliOp(PauliTable.from_labels(labels), coeffs)
        target = list(zip(labels, coeffs))
        self.assertEqual(op.to_list(), target)


class TestSparsePauliOpIteration(QiskitTestCase):
    """Tests for SparsePauliOp iterators class."""

    def test_enumerate(self):
        """Test enumerate with SparsePauliOp."""
        labels = ["III", "IXI", "IYY", "YIZ", "XYZ", "III"]
        coeffs = np.array([1, 2, 3, 4, 5, 6])
        table = PauliTable.from_labels(labels)
        op = SparsePauliOp(table, coeffs)
        for idx, i in enumerate(op):
            self.assertEqual(i, SparsePauliOp(labels[idx], coeffs[[idx]]))

    def test_iter(self):
        """Test iter with PauliTable."""
        labels = ["III", "IXI", "IYY", "YIZ", "XYZ", "III"]
        coeffs = np.array([1, 2, 3, 4, 5, 6])
        table = PauliTable.from_labels(labels)
        op = SparsePauliOp(table, coeffs)
        for idx, i in enumerate(iter(op)):
            self.assertEqual(i, SparsePauliOp(labels[idx], coeffs[[idx]]))

    def test_label_iter(self):
        """Test PauliTable label_iter method."""
        labels = ["III", "IXI", "IYY", "YIZ", "XYZ", "III"]
        coeffs = np.array([1, 2, 3, 4, 5, 6])
        table = PauliTable.from_labels(labels)
        op = SparsePauliOp(table, coeffs)
        for idx, i in enumerate(op.label_iter()):
            self.assertEqual(i, (labels[idx], coeffs[idx]))

    def test_matrix_iter(self):
        """Test PauliTable dense matrix_iter method."""
        labels = ["III", "IXI", "IYY", "YIZ", "XYZ", "III"]
        coeffs = np.array([1, 2, 3, 4, 5, 6])
        table = PauliTable.from_labels(labels)
        op = SparsePauliOp(table, coeffs)
        for idx, i in enumerate(op.matrix_iter()):
            self.assertTrue(np.array_equal(i, coeffs[idx] * pauli_mat(labels[idx])))

    def test_matrix_iter_sparse(self):
        """Test PauliTable sparse matrix_iter method."""
        labels = ["III", "IXI", "IYY", "YIZ", "XYZ", "III"]
        coeffs = np.array([1, 2, 3, 4, 5, 6])
        table = PauliTable.from_labels(labels)
        op = SparsePauliOp(table, coeffs)
        for idx, i in enumerate(op.matrix_iter(sparse=True)):
            self.assertTrue(np.array_equal(i.toarray(), coeffs[idx] * pauli_mat(labels[idx])))


@ddt
class TestSparsePauliOpMethods(QiskitTestCase):
    """Tests for SparsePauliOp operator methods."""

    RNG = np.random.default_rng(1994)

    def random_spp_op(self, num_qubits, num_terms):
        """Generate a pseudo-random SparsePauliOp"""
        coeffs = self.RNG.uniform(-1, 1, size=num_terms) + 1j * self.RNG.uniform(
            -1, 1, size=num_terms
        )
        labels = [
            "".join(self.RNG.choice(["I", "X", "Y", "Z"], size=num_qubits))
            for _ in range(num_terms)
        ]
        return SparsePauliOp(PauliTable.from_labels(labels), coeffs)

    @combine(num_qubits=[1, 2, 3, 4])
    def test_conjugate(self, num_qubits):
        """Test conjugate method for {num_qubits}-qubits."""
        spp_op = self.random_spp_op(num_qubits, 2 ** num_qubits)
        target = Operator(spp_op).conjugate()
        value = spp_op.conjugate().to_operator()
        self.assertEqual(value, target)

    @combine(num_qubits=[1, 2, 3, 4])
    def test_transpose(self, num_qubits):
        """Test transpose method for {num_qubits}-qubits."""
        spp_op = self.random_spp_op(num_qubits, 2 ** num_qubits)
        target = Operator(spp_op).transpose()
        value = spp_op.transpose().to_operator()
        self.assertEqual(value, target)

    @combine(num_qubits=[1, 2, 3, 4])
    def test_adjoint(self, num_qubits):
        """Test adjoint method for {num_qubits}-qubits."""
        spp_op = self.random_spp_op(num_qubits, 2 ** num_qubits)
        target = Operator(spp_op).adjoint()
        value = spp_op.adjoint().to_operator()
        self.assertEqual(value, target)

    @combine(num_qubits=[1, 2, 3, 4])
    def test_compose(self, num_qubits):
        """Test {num_qubits}-qubit compose methods."""
        spp_op1 = self.random_spp_op(num_qubits, 2 ** num_qubits)
        spp_op2 = self.random_spp_op(num_qubits, 2 ** num_qubits)
        target = Operator(spp_op1).compose(Operator(spp_op2))

        value = spp_op1.compose(spp_op2).to_operator()
        self.assertEqual(value, target)

        value = (spp_op1 & spp_op2).to_operator()
        self.assertEqual(value, target)

    @combine(num_qubits=[1, 2, 3, 4])
    def test_dot(self, num_qubits):
        """Test {num_qubits}-qubit compose methods."""
        spp_op1 = self.random_spp_op(num_qubits, 2 ** num_qubits)
        spp_op2 = self.random_spp_op(num_qubits, 2 ** num_qubits)
        target = Operator(spp_op1).dot(Operator(spp_op2))

        value = spp_op1.dot(spp_op2).to_operator()
        self.assertEqual(value, target)

        value = (spp_op1 * spp_op2).to_operator()
        self.assertEqual(value, target)

    @combine(num_qubits=[1, 2, 3])
    def test_qargs_compose(self, num_qubits):
        """Test 3-qubit compose method with {num_qubits}-qubit qargs."""
        spp_op1 = self.random_spp_op(3, 2 ** 3)
        spp_op2 = self.random_spp_op(num_qubits, 2 ** num_qubits)
        qargs = self.RNG.choice(3, size=num_qubits, replace=False).tolist()
        target = Operator(spp_op1).compose(Operator(spp_op2), qargs=qargs)

        value = spp_op1.compose(spp_op2, qargs=qargs).to_operator()
        self.assertEqual(value, target)

        value = (spp_op1 & spp_op2(qargs)).to_operator()
        self.assertEqual(value, target)

    @combine(num_qubits=[1, 2, 3])
    def test_qargs_dot(self, num_qubits):
        """Test 3-qubit dot method with {num_qubits}-qubit qargs."""
        spp_op1 = self.random_spp_op(3, 2 ** 3)
        spp_op2 = self.random_spp_op(num_qubits, 2 ** num_qubits)
        qargs = self.RNG.choice(3, size=num_qubits, replace=False).tolist()
        target = Operator(spp_op1).dot(Operator(spp_op2), qargs=qargs)

        value = spp_op1.dot(spp_op2, qargs=qargs).to_operator()
        self.assertEqual(value, target)

        value = (spp_op1 * spp_op2(qargs)).to_operator()
        self.assertEqual(value, target)

    @combine(num_qubits1=[1, 2, 3], num_qubits2=[1, 2, 3])
    def test_tensor(self, num_qubits1, num_qubits2):
        """Test tensor method for {num_qubits1} and {num_qubits2} qubits."""
        spp_op1 = self.random_spp_op(num_qubits1, 2 ** num_qubits1)
        spp_op2 = self.random_spp_op(num_qubits2, 2 ** num_qubits2)
        target = Operator(spp_op1).tensor(Operator(spp_op2))
        value = spp_op1.tensor(spp_op2).to_operator()
        self.assertEqual(value, target)

    @combine(num_qubits1=[1, 2, 3], num_qubits2=[1, 2, 3])
    def test_expand(self, num_qubits1, num_qubits2):
        """Test expand method for {num_qubits1} and {num_qubits2} qubits."""
        spp_op1 = random_unitary(2 ** num_qubits1, seed=self.RNG)
        spp_op2 = random_unitary(2 ** num_qubits2, seed=self.RNG)
        target = Operator(spp_op1).expand(Operator(spp_op2))
        value = spp_op1.expand(spp_op2).to_operator()
        self.assertEqual(value, target)

    @combine(num_qubits=[1, 2, 3, 4])
    def test_add(self, num_qubits):
        """Test + method for {num_qubits} qubits."""
        spp_op1 = random_unitary(2 ** num_qubits, seed=self.RNG)
        spp_op2 = random_unitary(2 ** num_qubits, seed=self.RNG)
        target = Operator(spp_op1) + Operator(spp_op2)
        value = (spp_op1 + spp_op2).to_operator()
        self.assertEqual(value, target)

    @combine(num_qubits=[1, 2, 3, 4])
    def test_sub(self, num_qubits):
        """Test + method for {num_qubits} qubits."""
        spp_op1 = random_unitary(2 ** num_qubits, seed=self.RNG)
        spp_op2 = random_unitary(2 ** num_qubits, seed=self.RNG)
        target = Operator(spp_op1) - Operator(spp_op2)
        value = (spp_op1 - spp_op2).to_operator()
        self.assertEqual(value, target)

    @combine(num_qubits=[1, 2, 3])
    def test_add_qargs(self, num_qubits):
        """Test + method for 3 qubits with {num_qubits} qubit qargs."""
        spp_op1 = random_unitary(2 ** 3, seed=self.RNG)
        spp_op2 = random_unitary(2 ** num_qubits, seed=self.RNG)
        qargs = self.RNG.choice(3, size=num_qubits, replace=False).tolist()
        target = Operator(spp_op1) + Operator(spp_op2)(qargs)
        value = (spp_op1 + spp_op2(qargs)).to_operator()
        self.assertEqual(value, target)

    @combine(num_qubits=[1, 2, 3])
    def test_sub_qargs(self, num_qubits):
        """Test - method for 3 qubits with {num_qubits} qubit qargs."""
        spp_op1 = random_unitary(2 ** 3, seed=self.RNG)
        spp_op2 = random_unitary(2 ** num_qubits, seed=self.RNG)
        qargs = self.RNG.choice(3, size=num_qubits, replace=False).tolist()
        target = Operator(spp_op1) - Operator(spp_op2)(qargs)
        value = (spp_op1 - spp_op2(qargs)).to_operator()
        self.assertEqual(value, target)

    @combine(num_qubits=[1, 2, 3], value=[0, 1, 1j, -3 + 4.4j, np.int64(2)])
    def test_mul(self, num_qubits, value):
        """Test * method for {num_qubits} qubits and value {value}."""
        spp_op = random_unitary(2 ** num_qubits, seed=self.RNG)
        target = value * Operator(spp_op)
        value = (value * spp_op).to_operator()
        self.assertEqual(value, target)

    @combine(num_qubits=[1, 2, 3], value=[1, 1j, -3 + 4.4j])
    def test_div(self, num_qubits, value):
        """Test / method for {num_qubits} qubits and value {value}."""
        spp_op = random_unitary(2 ** num_qubits, seed=self.RNG)
        target = Operator(spp_op) / value
        value = (spp_op / value).to_operator()
        self.assertEqual(value, target)

    def test_simplify(self):
        """Test simplify method"""
        coeffs = [3 + 1j, -3 - 1j, 0, 4, -5, 2.2, -1.1j]
        labels = ["IXI", "IXI", "ZZZ", "III", "III", "XXX", "XXX"]
        value = SparsePauliOp(PauliTable.from_labels(labels), coeffs).simplify()
        target_coeffs = [-1, 2.2 - 1.1j]
        target_labels = ["III", "XXX"]
        target = SparsePauliOp(PauliTable.from_labels(target_labels), target_coeffs)
        self.assertEqual(value, target)


if __name__ == "__main__":
    unittest.main()

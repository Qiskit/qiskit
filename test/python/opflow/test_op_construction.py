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

""" Test Operator construction, including OpPrimitives and singletons. """


import itertools
import unittest
from math import pi
from test.python.opflow import QiskitOpflowTestCase

import numpy as np
import scipy
from ddt import data, ddt
from scipy.stats import unitary_group

from qiskit import QiskitError
from qiskit.circuit import Instruction, Parameter, ParameterVector, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import CZGate, ZGate
from qiskit.extensions.exceptions import ExtensionError
from qiskit.opflow import (
    CX,
    CircuitOp,
    CircuitStateFn,
    ComposedOp,
    DictStateFn,
    EvolvedOp,
    H,
    I,
    ListOp,
    MatrixOp,
    Minus,
    OperatorBase,
    OperatorStateFn,
    OpflowError,
    PauliOp,
    PrimitiveOp,
    SparseVectorStateFn,
    StateFn,
    SummedOp,
    T,
    TensoredOp,
    VectorStateFn,
    X,
    Y,
    Z,
    Zero,
)
from qiskit.quantum_info import Operator, Pauli, Statevector

# pylint: disable=invalid-name


@ddt
class TestOpConstruction(QiskitOpflowTestCase):
    """Operator Construction tests."""

    def test_pauli_primitives(self):
        """from to file test"""
        newop = X ^ Y ^ Z ^ I
        self.assertEqual(newop.primitive, Pauli("XYZI"))

        kpower_op = (Y ^ 5) ^ (I ^ 3)
        self.assertEqual(kpower_op.primitive, Pauli("YYYYYIII"))

        kpower_op2 = (Y ^ I) ^ 4
        self.assertEqual(kpower_op2.primitive, Pauli("YIYIYIYI"))

        # Check immutability
        self.assertEqual(X.primitive, Pauli("X"))
        self.assertEqual(Y.primitive, Pauli("Y"))
        self.assertEqual(Z.primitive, Pauli("Z"))
        self.assertEqual(I.primitive, Pauli("I"))

    def test_composed_eval(self):
        """Test eval of ComposedOp"""
        self.assertAlmostEqual(Minus.eval("1"), -(0.5 ** 0.5))

    def test_xz_compose_phase(self):
        """Test phase composition"""
        self.assertEqual((-1j * Y).eval("0").eval("0"), 0)
        self.assertEqual((-1j * Y).eval("0").eval("1"), 1)
        self.assertEqual((-1j * Y).eval("1").eval("0"), -1)
        self.assertEqual((-1j * Y).eval("1").eval("1"), 0)
        self.assertEqual((X @ Z).eval("0").eval("0"), 0)
        self.assertEqual((X @ Z).eval("0").eval("1"), 1)
        self.assertEqual((X @ Z).eval("1").eval("0"), -1)
        self.assertEqual((X @ Z).eval("1").eval("1"), 0)
        self.assertEqual((1j * Y).eval("0").eval("0"), 0)
        self.assertEqual((1j * Y).eval("0").eval("1"), -1)
        self.assertEqual((1j * Y).eval("1").eval("0"), 1)
        self.assertEqual((1j * Y).eval("1").eval("1"), 0)
        self.assertEqual((Z @ X).eval("0").eval("0"), 0)
        self.assertEqual((Z @ X).eval("0").eval("1"), -1)
        self.assertEqual((Z @ X).eval("1").eval("0"), 1)
        self.assertEqual((Z @ X).eval("1").eval("1"), 0)

    def test_evals(self):
        """evals test"""
        # TODO: Think about eval names
        self.assertEqual(Z.eval("0").eval("0"), 1)
        self.assertEqual(Z.eval("1").eval("0"), 0)
        self.assertEqual(Z.eval("0").eval("1"), 0)
        self.assertEqual(Z.eval("1").eval("1"), -1)
        self.assertEqual(X.eval("0").eval("0"), 0)
        self.assertEqual(X.eval("1").eval("0"), 1)
        self.assertEqual(X.eval("0").eval("1"), 1)
        self.assertEqual(X.eval("1").eval("1"), 0)
        self.assertEqual(Y.eval("0").eval("0"), 0)
        self.assertEqual(Y.eval("1").eval("0"), -1j)
        self.assertEqual(Y.eval("0").eval("1"), 1j)
        self.assertEqual(Y.eval("1").eval("1"), 0)

        with self.assertRaises(ValueError):
            Y.eval("11")

        with self.assertRaises(ValueError):
            (X ^ Y).eval("1111")

        with self.assertRaises(ValueError):
            Y.eval((X ^ X).to_matrix_op())

        # Check that Pauli logic eval returns same as matrix logic
        self.assertEqual(PrimitiveOp(Z.to_matrix()).eval("0").eval("0"), 1)
        self.assertEqual(PrimitiveOp(Z.to_matrix()).eval("1").eval("0"), 0)
        self.assertEqual(PrimitiveOp(Z.to_matrix()).eval("0").eval("1"), 0)
        self.assertEqual(PrimitiveOp(Z.to_matrix()).eval("1").eval("1"), -1)
        self.assertEqual(PrimitiveOp(X.to_matrix()).eval("0").eval("0"), 0)
        self.assertEqual(PrimitiveOp(X.to_matrix()).eval("1").eval("0"), 1)
        self.assertEqual(PrimitiveOp(X.to_matrix()).eval("0").eval("1"), 1)
        self.assertEqual(PrimitiveOp(X.to_matrix()).eval("1").eval("1"), 0)
        self.assertEqual(PrimitiveOp(Y.to_matrix()).eval("0").eval("0"), 0)
        self.assertEqual(PrimitiveOp(Y.to_matrix()).eval("1").eval("0"), -1j)
        self.assertEqual(PrimitiveOp(Y.to_matrix()).eval("0").eval("1"), 1j)
        self.assertEqual(PrimitiveOp(Y.to_matrix()).eval("1").eval("1"), 0)

        pauli_op = Z ^ I ^ X ^ Y
        mat_op = PrimitiveOp(pauli_op.to_matrix())
        full_basis = list(map("".join, itertools.product("01", repeat=pauli_op.num_qubits)))
        for bstr1, bstr2 in itertools.product(full_basis, full_basis):
            # print('{} {} {} {}'.format(bstr1, bstr2, pauli_op.eval(bstr1, bstr2),
            # mat_op.eval(bstr1, bstr2)))
            np.testing.assert_array_almost_equal(
                pauli_op.eval(bstr1).eval(bstr2), mat_op.eval(bstr1).eval(bstr2)
            )

        gnarly_op = SummedOp(
            [
                (H ^ I ^ Y).compose(X ^ X ^ Z).tensor(Z),
                PrimitiveOp(Operator.from_label("+r0I")),
                3 * (X ^ CX ^ T),
            ],
            coeff=3 + 0.2j,
        )
        gnarly_mat_op = PrimitiveOp(gnarly_op.to_matrix())
        full_basis = list(map("".join, itertools.product("01", repeat=gnarly_op.num_qubits)))
        for bstr1, bstr2 in itertools.product(full_basis, full_basis):
            np.testing.assert_array_almost_equal(
                gnarly_op.eval(bstr1).eval(bstr2), gnarly_mat_op.eval(bstr1).eval(bstr2)
            )

    def test_circuit_construction(self):
        """circuit construction test"""
        hadq2 = H ^ I
        cz = hadq2.compose(CX).compose(hadq2)
        qc = QuantumCircuit(2)
        qc.append(cz.primitive, qargs=range(2))

        ref_cz_mat = PrimitiveOp(CZGate()).to_matrix()
        np.testing.assert_array_almost_equal(cz.to_matrix(), ref_cz_mat)

    def test_io_consistency(self):
        """consistency test"""
        new_op = X ^ Y ^ I
        label = "XYI"
        # label = new_op.primitive.to_label()
        self.assertEqual(str(new_op.primitive), label)
        np.testing.assert_array_almost_equal(
            new_op.primitive.to_matrix(), Operator.from_label(label).data
        )
        self.assertEqual(new_op.primitive, Pauli(label))

        x_mat = X.primitive.to_matrix()
        y_mat = Y.primitive.to_matrix()
        i_mat = np.eye(2, 2)
        np.testing.assert_array_almost_equal(
            new_op.primitive.to_matrix(), np.kron(np.kron(x_mat, y_mat), i_mat)
        )

        hi = np.kron(H.to_matrix(), I.to_matrix())
        hi2 = Operator.from_label("HI").data
        hi3 = (H ^ I).to_matrix()
        np.testing.assert_array_almost_equal(hi, hi2)
        np.testing.assert_array_almost_equal(hi2, hi3)

        xy = np.kron(X.to_matrix(), Y.to_matrix())
        xy2 = Operator.from_label("XY").data
        xy3 = (X ^ Y).to_matrix()
        np.testing.assert_array_almost_equal(xy, xy2)
        np.testing.assert_array_almost_equal(xy2, xy3)

        # Check if numpy array instantiation is the same as from Operator
        matrix_op = Operator.from_label("+r")
        np.testing.assert_array_almost_equal(
            PrimitiveOp(matrix_op).to_matrix(), PrimitiveOp(matrix_op.data).to_matrix()
        )
        # Ditto list of lists
        np.testing.assert_array_almost_equal(
            PrimitiveOp(matrix_op.data.tolist()).to_matrix(),
            PrimitiveOp(matrix_op.data).to_matrix(),
        )

        # TODO make sure this works once we resolve endianness mayhem
        # qc = QuantumCircuit(3)
        # qc.x(2)
        # qc.y(1)
        # from qiskit import BasicAer, QuantumCircuit, execute
        # unitary = execute(qc, BasicAer.get_backend('unitary_simulator')).result().get_unitary()
        # np.testing.assert_array_almost_equal(new_op.primitive.to_matrix(), unitary)

    def test_to_matrix(self):
        """to matrix text"""
        np.testing.assert_array_equal(X.to_matrix(), Operator.from_label("X").data)
        np.testing.assert_array_equal(Y.to_matrix(), Operator.from_label("Y").data)
        np.testing.assert_array_equal(Z.to_matrix(), Operator.from_label("Z").data)

        op1 = Y + H
        np.testing.assert_array_almost_equal(op1.to_matrix(), Y.to_matrix() + H.to_matrix())

        op2 = op1 * 0.5
        np.testing.assert_array_almost_equal(op2.to_matrix(), op1.to_matrix() * 0.5)

        op3 = (4 - 0.6j) * op2
        np.testing.assert_array_almost_equal(op3.to_matrix(), op2.to_matrix() * (4 - 0.6j))

        op4 = op3.tensor(X)
        np.testing.assert_array_almost_equal(
            op4.to_matrix(), np.kron(op3.to_matrix(), X.to_matrix())
        )

        op5 = op4.compose(H ^ I)
        np.testing.assert_array_almost_equal(
            op5.to_matrix(), np.dot(op4.to_matrix(), (H ^ I).to_matrix())
        )

        op6 = op5 + PrimitiveOp(Operator.from_label("+r").data)
        np.testing.assert_array_almost_equal(
            op6.to_matrix(), op5.to_matrix() + Operator.from_label("+r").data
        )

        param = Parameter("α")
        m = np.array([[0, -1j], [1j, 0]])
        op7 = MatrixOp(m, param)
        np.testing.assert_array_equal(op7.to_matrix(), m * param)

        param = Parameter("β")
        op8 = PauliOp(primitive=Pauli("Y"), coeff=param)
        np.testing.assert_array_equal(op8.to_matrix(), m * param)

        param = Parameter("γ")
        qc = QuantumCircuit(1)
        qc.h(0)
        op9 = CircuitOp(qc, coeff=param)
        m = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        np.testing.assert_array_equal(op9.to_matrix(), m * param)

    def test_circuit_op_to_matrix(self):
        """test CircuitOp.to_matrix"""
        qc = QuantumCircuit(1)
        qc.rz(1.0, 0)
        qcop = CircuitOp(qc)
        np.testing.assert_array_almost_equal(
            qcop.to_matrix(), scipy.linalg.expm(-0.5j * Z.to_matrix())
        )

    def test_matrix_to_instruction(self):
        """Test MatrixOp.to_instruction yields an Instruction object."""
        matop = (H ^ 3).to_matrix_op()
        with self.subTest("assert to_instruction returns Instruction"):
            self.assertIsInstance(matop.to_instruction(), Instruction)

        matop = ((H ^ 3) + (Z ^ 3)).to_matrix_op()
        with self.subTest("matrix operator is not unitary"):
            with self.assertRaises(ExtensionError):
                matop.to_instruction()

    def test_adjoint(self):
        """adjoint test"""
        gnarly_op = 3 * (H ^ I ^ Y).compose(X ^ X ^ Z).tensor(T ^ Z) + PrimitiveOp(
            Operator.from_label("+r0IX").data
        )
        np.testing.assert_array_almost_equal(
            np.conj(np.transpose(gnarly_op.to_matrix())), gnarly_op.adjoint().to_matrix()
        )

    def test_primitive_strings(self):
        """get primitives test"""
        self.assertEqual(X.primitive_strings(), {"Pauli"})

        gnarly_op = 3 * (H ^ I ^ Y).compose(X ^ X ^ Z).tensor(T ^ Z) + PrimitiveOp(
            Operator.from_label("+r0IX").data
        )
        self.assertEqual(gnarly_op.primitive_strings(), {"QuantumCircuit", "Matrix"})

    def test_to_pauli_op(self):
        """Test to_pauli_op method"""
        gnarly_op = 3 * (H ^ I ^ Y).compose(X ^ X ^ Z).tensor(T ^ Z) + PrimitiveOp(
            Operator.from_label("+r0IX").data
        )
        mat_op = gnarly_op.to_matrix_op()
        pauli_op = gnarly_op.to_pauli_op()
        self.assertIsInstance(pauli_op, SummedOp)
        for p in pauli_op:
            self.assertIsInstance(p, PauliOp)
        np.testing.assert_array_almost_equal(mat_op.to_matrix(), pauli_op.to_matrix())

    def test_circuit_permute(self):
        r"""Test the CircuitOp's .permute method"""
        perm = range(7)[::-1]
        c_op = (
            ((CX ^ 3) ^ X)
            @ (H ^ 7)
            @ (X ^ Y ^ Z ^ I ^ X ^ X ^ X)
            @ (Y ^ (CX ^ 3))
            @ (X ^ Y ^ Z ^ I ^ X ^ X ^ X)
        )
        c_op_perm = c_op.permute(perm)
        self.assertNotEqual(c_op, c_op_perm)
        c_op_id = c_op_perm.permute(perm)
        self.assertEqual(c_op, c_op_id)

    def test_summed_op_reduce(self):
        """Test SummedOp"""
        sum_op = (X ^ X * 2) + (Y ^ Y)  # type: PauliSumOp
        sum_op = sum_op.to_pauli_op()  # type: SummedOp[PauliOp]
        with self.subTest("SummedOp test 1"):
            self.assertEqual(sum_op.coeff, 1)
            self.assertListEqual([str(op.primitive) for op in sum_op], ["XX", "YY"])
            self.assertListEqual([op.coeff for op in sum_op], [2, 1])

        sum_op = (X ^ X * 2) + (Y ^ Y)
        sum_op += Y ^ Y
        sum_op = sum_op.to_pauli_op()  # type: SummedOp[PauliOp]
        with self.subTest("SummedOp test 2-a"):
            self.assertEqual(sum_op.coeff, 1)
            self.assertListEqual([str(op.primitive) for op in sum_op], ["XX", "YY", "YY"])
            self.assertListEqual([op.coeff for op in sum_op], [2, 1, 1])

        sum_op = sum_op.collapse_summands()
        with self.subTest("SummedOp test 2-b"):
            self.assertEqual(sum_op.coeff, 1)
            self.assertListEqual([str(op.primitive) for op in sum_op], ["XX", "YY"])
            self.assertListEqual([op.coeff for op in sum_op], [2, 2])

        sum_op = (X ^ X * 2) + (Y ^ Y)
        sum_op += (Y ^ Y) + (X ^ X * 2)
        sum_op = sum_op.to_pauli_op()  # type: SummedOp[PauliOp]
        with self.subTest("SummedOp test 3-a"):
            self.assertEqual(sum_op.coeff, 1)
            self.assertListEqual([str(op.primitive) for op in sum_op], ["XX", "YY", "YY", "XX"])
            self.assertListEqual([op.coeff for op in sum_op], [2, 1, 1, 2])

        sum_op = sum_op.reduce().to_pauli_op()
        with self.subTest("SummedOp test 3-b"):
            self.assertEqual(sum_op.coeff, 1)
            self.assertListEqual([str(op.primitive) for op in sum_op], ["XX", "YY"])
            self.assertListEqual([op.coeff for op in sum_op], [4, 2])

        sum_op = SummedOp([X ^ X * 2, Y ^ Y], 2)
        with self.subTest("SummedOp test 4-a"):
            self.assertEqual(sum_op.coeff, 2)
            self.assertListEqual([str(op.primitive) for op in sum_op], ["XX", "YY"])
            self.assertListEqual([op.coeff for op in sum_op], [2, 1])

        sum_op = sum_op.collapse_summands()
        with self.subTest("SummedOp test 4-b"):
            self.assertEqual(sum_op.coeff, 1)
            self.assertListEqual([str(op.primitive) for op in sum_op], ["XX", "YY"])
            self.assertListEqual([op.coeff for op in sum_op], [4, 2])

        sum_op = SummedOp([X ^ X * 2, Y ^ Y], 2)
        sum_op += Y ^ Y
        with self.subTest("SummedOp test 5-a"):
            self.assertEqual(sum_op.coeff, 1)
            self.assertListEqual([str(op.primitive) for op in sum_op], ["XX", "YY", "YY"])
            self.assertListEqual([op.coeff for op in sum_op], [4, 2, 1])

        sum_op = sum_op.collapse_summands()
        with self.subTest("SummedOp test 5-b"):
            self.assertEqual(sum_op.coeff, 1)
            self.assertListEqual([str(op.primitive) for op in sum_op], ["XX", "YY"])
            self.assertListEqual([op.coeff for op in sum_op], [4, 3])

        sum_op = SummedOp([X ^ X * 2, Y ^ Y], 2)
        sum_op += ((X ^ X) * 2 + (Y ^ Y)).to_pauli_op()
        with self.subTest("SummedOp test 6-a"):
            self.assertEqual(sum_op.coeff, 1)
            self.assertListEqual([str(op.primitive) for op in sum_op], ["XX", "YY", "XX", "YY"])
            self.assertListEqual([op.coeff for op in sum_op], [4, 2, 2, 1])

        sum_op = sum_op.collapse_summands()
        with self.subTest("SummedOp test 6-b"):
            self.assertEqual(sum_op.coeff, 1)
            self.assertListEqual([str(op.primitive) for op in sum_op], ["XX", "YY"])
            self.assertListEqual([op.coeff for op in sum_op], [6, 3])

        sum_op = SummedOp([X ^ X * 2, Y ^ Y], 2)
        sum_op += sum_op
        with self.subTest("SummedOp test 7-a"):
            self.assertEqual(sum_op.coeff, 1)
            self.assertListEqual([str(op.primitive) for op in sum_op], ["XX", "YY", "XX", "YY"])
            self.assertListEqual([op.coeff for op in sum_op], [4, 2, 4, 2])

        sum_op = sum_op.collapse_summands()
        with self.subTest("SummedOp test 7-b"):
            self.assertEqual(sum_op.coeff, 1)
            self.assertListEqual([str(op.primitive) for op in sum_op], ["XX", "YY"])
            self.assertListEqual([op.coeff for op in sum_op], [8, 4])

        sum_op = SummedOp([X ^ X * 2, Y ^ Y], 2) + SummedOp([X ^ X * 2, Z ^ Z], 3)
        with self.subTest("SummedOp test 8-a"):
            self.assertEqual(sum_op.coeff, 1)
            self.assertListEqual([str(op.primitive) for op in sum_op], ["XX", "YY", "XX", "ZZ"])
            self.assertListEqual([op.coeff for op in sum_op], [4, 2, 6, 3])

        sum_op = sum_op.collapse_summands()
        with self.subTest("SummedOp test 8-b"):
            self.assertEqual(sum_op.coeff, 1)
            self.assertListEqual([str(op.primitive) for op in sum_op], ["XX", "YY", "ZZ"])
            self.assertListEqual([op.coeff for op in sum_op], [10, 2, 3])

        sum_op = SummedOp([])
        with self.subTest("SummedOp test 9"):
            self.assertEqual(sum_op.reduce(), sum_op)

        sum_op = ((Z + I) ^ Z) + (Z ^ X)
        with self.subTest("SummedOp test 10"):
            expected = SummedOp([PauliOp(Pauli("ZZ")), PauliOp(Pauli("IZ")), PauliOp(Pauli("ZX"))])
            self.assertEqual(sum_op.to_pauli_op(), expected)

    def test_compose_op_of_different_dim(self):
        """
        Test if smaller operator expands to correct dim when composed with bigger operator.
        Test if PrimitiveOps compose methods are consistent.
        """
        # PauliOps of different dim
        xy_p = X ^ Y
        xyz_p = X ^ Y ^ Z

        pauli_op = xy_p @ xyz_p
        expected_result = I ^ I ^ Z
        self.assertEqual(pauli_op, expected_result)

        # MatrixOps of different dim
        xy_m = xy_p.to_matrix_op()
        xyz_m = xyz_p.to_matrix_op()

        matrix_op = xy_m @ xyz_m
        self.assertEqual(matrix_op, expected_result.to_matrix_op())

        # CircuitOps of different dim
        xy_c = xy_p.to_circuit_op()
        xyz_c = xyz_p.to_circuit_op()

        circuit_op = xy_c @ xyz_c

        self.assertTrue(np.array_equal(pauli_op.to_matrix(), matrix_op.to_matrix()))
        self.assertTrue(np.allclose(pauli_op.to_matrix(), circuit_op.to_matrix(), rtol=1e-14))
        self.assertTrue(np.allclose(matrix_op.to_matrix(), circuit_op.to_matrix(), rtol=1e-14))

    def test_permute_on_primitive_op(self):
        """Test if permute methods of PrimitiveOps are consistent and work as expected."""
        indices = [1, 2, 4]

        # PauliOp
        pauli_op = X ^ Y ^ Z
        permuted_pauli_op = pauli_op.permute(indices)
        expected_pauli_op = X ^ I ^ Y ^ Z ^ I

        self.assertEqual(permuted_pauli_op, expected_pauli_op)

        # CircuitOp
        circuit_op = pauli_op.to_circuit_op()
        permuted_circuit_op = circuit_op.permute(indices)
        expected_circuit_op = expected_pauli_op.to_circuit_op()

        self.assertEqual(
            Operator(permuted_circuit_op.primitive), Operator(expected_circuit_op.primitive)
        )

        # MatrixOp
        matrix_op = pauli_op.to_matrix_op()
        permuted_matrix_op = matrix_op.permute(indices)
        expected_matrix_op = expected_pauli_op.to_matrix_op()

        equal = np.allclose(permuted_matrix_op.to_matrix(), expected_matrix_op.to_matrix())
        self.assertTrue(equal)

    def test_permute_on_list_op(self):
        """Test if ListOp permute method is consistent with PrimitiveOps permute methods."""

        op1 = (X ^ Y ^ Z).to_circuit_op()
        op2 = Z ^ X ^ Y

        # ComposedOp
        indices = [1, 2, 0]
        primitive_op = op1 @ op2
        primitive_op_perm = primitive_op.permute(indices)  # CircuitOp.permute

        composed_op = ComposedOp([op1, op2])
        composed_op_perm = composed_op.permute(indices)

        # reduce the ListOp to PrimitiveOp
        to_primitive = composed_op_perm.oplist[0] @ composed_op_perm.oplist[1]
        # compare resulting PrimitiveOps
        equal = np.allclose(primitive_op_perm.to_matrix(), to_primitive.to_matrix())
        self.assertTrue(equal)

        # TensoredOp
        indices = [3, 5, 4, 0, 2, 1]
        primitive_op = op1 ^ op2
        primitive_op_perm = primitive_op.permute(indices)

        tensored_op = TensoredOp([op1, op2])
        tensored_op_perm = tensored_op.permute(indices)

        # reduce the ListOp to PrimitiveOp
        composed_oplist = tensored_op_perm.oplist
        to_primitive = (
            composed_oplist[0]
            @ (composed_oplist[1].oplist[0] ^ composed_oplist[1].oplist[1])
            @ composed_oplist[2]
        )

        # compare resulting PrimitiveOps
        equal = np.allclose(primitive_op_perm.to_matrix(), to_primitive.to_matrix())
        self.assertTrue(equal)

        # SummedOp
        primitive_op = X ^ Y ^ Z
        summed_op = SummedOp([primitive_op])

        indices = [1, 2, 0]
        primitive_op_perm = primitive_op.permute(indices)  # PauliOp.permute
        summed_op_perm = summed_op.permute(indices)

        # reduce the ListOp to PrimitiveOp
        to_primitive = summed_op_perm.oplist[0] @ primitive_op @ summed_op_perm.oplist[2]

        # compare resulting PrimitiveOps
        equal = np.allclose(primitive_op_perm.to_matrix(), to_primitive.to_matrix())
        self.assertTrue(equal)

    def test_expand_on_list_op(self):
        """Test if expanded ListOp has expected num_qubits."""
        add_qubits = 3

        # ComposedOp
        composed_op = ComposedOp([(X ^ Y ^ Z), (H ^ T), (Z ^ X ^ Y ^ Z).to_matrix_op()])
        expanded = composed_op._expand_dim(add_qubits)
        self.assertEqual(composed_op.num_qubits + add_qubits, expanded.num_qubits)

        # TensoredOp
        tensored_op = TensoredOp([(X ^ Y), (Z ^ I)])
        expanded = tensored_op._expand_dim(add_qubits)
        self.assertEqual(tensored_op.num_qubits + add_qubits, expanded.num_qubits)

        # SummedOp
        summed_op = SummedOp([(X ^ Y), (Z ^ I ^ Z)])
        expanded = summed_op._expand_dim(add_qubits)
        self.assertEqual(summed_op.num_qubits + add_qubits, expanded.num_qubits)

    def test_expand_on_state_fn(self):
        """Test if expanded StateFn has expected num_qubits."""
        num_qubits = 3
        add_qubits = 2

        # case CircuitStateFn, with primitive QuantumCircuit
        qc2 = QuantumCircuit(num_qubits)
        qc2.cx(0, 1)

        cfn = CircuitStateFn(qc2, is_measurement=True)

        cfn_exp = cfn._expand_dim(add_qubits)
        self.assertEqual(cfn_exp.num_qubits, add_qubits + num_qubits)

        # case OperatorStateFn, with OperatorBase primitive, in our case CircuitStateFn
        osfn = OperatorStateFn(cfn)
        osfn_exp = osfn._expand_dim(add_qubits)

        self.assertEqual(osfn_exp.num_qubits, add_qubits + num_qubits)

        # case DictStateFn
        dsfn = DictStateFn("1" * num_qubits, is_measurement=True)
        self.assertEqual(dsfn.num_qubits, num_qubits)

        dsfn_exp = dsfn._expand_dim(add_qubits)
        self.assertEqual(dsfn_exp.num_qubits, num_qubits + add_qubits)

        # case VectorStateFn
        vsfn = VectorStateFn(np.ones(2 ** num_qubits, dtype=complex))
        self.assertEqual(vsfn.num_qubits, num_qubits)

        vsfn_exp = vsfn._expand_dim(add_qubits)
        self.assertEqual(vsfn_exp.num_qubits, num_qubits + add_qubits)

    def test_permute_on_state_fn(self):
        """Test if StateFns permute are consistent."""

        num_qubits = 4
        dim = 2 ** num_qubits
        primitive_list = [1.0 / (i + 1) for i in range(dim)]
        primitive_dict = {format(i, "b").zfill(num_qubits): 1.0 / (i + 1) for i in range(dim)}

        dict_fn = DictStateFn(primitive=primitive_dict, is_measurement=True)
        vec_fn = VectorStateFn(primitive=primitive_list, is_measurement=True)

        # check if dict_fn and vec_fn are equivalent
        equivalent = np.allclose(dict_fn.to_matrix(), vec_fn.to_matrix())
        self.assertTrue(equivalent)

        # permute
        indices = [2, 3, 0, 1]
        permute_dict = dict_fn.permute(indices)
        permute_vect = vec_fn.permute(indices)

        equivalent = np.allclose(permute_dict.to_matrix(), permute_vect.to_matrix())
        self.assertTrue(equivalent)

    def test_compose_consistency(self):
        """Test if PrimitiveOp @ ComposedOp is consistent with ComposedOp @ PrimitiveOp."""

        # PauliOp
        op1 = X ^ Y ^ Z
        op2 = X ^ Y ^ Z
        op3 = (X ^ Y ^ Z).to_circuit_op()

        comp1 = op1 @ ComposedOp([op2, op3])
        comp2 = ComposedOp([op3, op2]) @ op1
        self.assertListEqual(comp1.oplist, list(reversed(comp2.oplist)))

        # CircitOp
        op1 = op1.to_circuit_op()
        op2 = op2.to_circuit_op()
        op3 = op3.to_matrix_op()

        comp1 = op1 @ ComposedOp([op2, op3])
        comp2 = ComposedOp([op3, op2]) @ op1
        self.assertListEqual(comp1.oplist, list(reversed(comp2.oplist)))

        # MatrixOp
        op1 = op1.to_matrix_op()
        op2 = op2.to_matrix_op()
        op3 = op3.to_pauli_op()

        comp1 = op1 @ ComposedOp([op2, op3])
        comp2 = ComposedOp([op3, op2]) @ op1
        self.assertListEqual(comp1.oplist, list(reversed(comp2.oplist)))

    def test_compose_with_indices(self):
        """Test compose method using its permutation feature."""

        pauli_op = X ^ Y ^ Z
        circuit_op = T ^ H
        matrix_op = (X ^ Y ^ H ^ T).to_matrix_op()
        evolved_op = EvolvedOp(matrix_op)

        # composition of PrimitiveOps
        num_qubits = 4
        primitive_op = pauli_op @ circuit_op @ matrix_op
        composed_op = pauli_op @ circuit_op @ evolved_op
        self.assertEqual(primitive_op.num_qubits, num_qubits)
        self.assertEqual(composed_op.num_qubits, num_qubits)

        # with permutation
        num_qubits = 5
        indices = [1, 4]
        permuted_primitive_op = evolved_op @ circuit_op.permute(indices) @ pauli_op @ matrix_op
        composed_primitive_op = (
            evolved_op @ pauli_op.compose(circuit_op, permutation=indices, front=True) @ matrix_op
        )

        self.assertTrue(
            np.allclose(permuted_primitive_op.to_matrix(), composed_primitive_op.to_matrix())
        )
        self.assertEqual(num_qubits, permuted_primitive_op.num_qubits)

        # ListOp
        num_qubits = 6
        tensored_op = TensoredOp([pauli_op, circuit_op])
        summed_op = pauli_op + circuit_op.permute([2, 1])
        composed_op = circuit_op @ evolved_op @ matrix_op

        list_op = summed_op @ composed_op.compose(
            tensored_op, permutation=[1, 2, 3, 5, 4], front=True
        )
        self.assertEqual(num_qubits, list_op.num_qubits)

        num_qubits = 4
        circuit_fn = CircuitStateFn(primitive=circuit_op.primitive, is_measurement=True)
        operator_fn = OperatorStateFn(primitive=circuit_op ^ circuit_op, is_measurement=True)

        no_perm_op = circuit_fn @ operator_fn
        self.assertEqual(no_perm_op.num_qubits, num_qubits)

        indices = [0, 4]
        perm_op = operator_fn.compose(circuit_fn, permutation=indices, front=True)
        self.assertEqual(perm_op.num_qubits, max(indices) + 1)

        # StateFn
        num_qubits = 3
        dim = 2 ** num_qubits
        vec = [1.0 / (i + 1) for i in range(dim)]
        dic = {format(i, "b").zfill(num_qubits): 1.0 / (i + 1) for i in range(dim)}

        is_measurement = True
        op_state_fn = OperatorStateFn(matrix_op, is_measurement=is_measurement)  # num_qubit = 4
        vec_state_fn = VectorStateFn(vec, is_measurement=is_measurement)  # 3
        dic_state_fn = DictStateFn(dic, is_measurement=is_measurement)  # 3
        circ_state_fn = CircuitStateFn(circuit_op.to_circuit(), is_measurement=is_measurement)  # 2

        composed_op = op_state_fn @ vec_state_fn @ dic_state_fn @ circ_state_fn
        self.assertEqual(composed_op.num_qubits, op_state_fn.num_qubits)

        # with permutation
        perm = [2, 4, 6]
        composed = (
            op_state_fn
            @ dic_state_fn.compose(vec_state_fn, permutation=perm, front=True)
            @ circ_state_fn
        )
        self.assertEqual(composed.num_qubits, max(perm) + 1)

    def test_summed_op_equals(self):
        """Test corner cases of SummedOp's equals function."""
        with self.subTest("multiplicative factor"):
            self.assertEqual(2 * X, X + X)

        with self.subTest("commutative"):
            self.assertEqual(X + Z, Z + X)

        with self.subTest("circuit and paulis"):
            z = CircuitOp(ZGate())
            self.assertEqual(Z + z, z + Z)

        with self.subTest("matrix op and paulis"):
            z = MatrixOp([[1, 0], [0, -1]])
            self.assertEqual(Z + z, z + Z)

        with self.subTest("matrix multiplicative"):
            z = MatrixOp([[1, 0], [0, -1]])
            self.assertEqual(2 * z, z + z)

        with self.subTest("parameter coefficients"):
            expr = Parameter("theta")
            z = MatrixOp([[1, 0], [0, -1]])
            self.assertEqual(expr * z, expr * z)

        with self.subTest("different coefficient types"):
            expr = Parameter("theta")
            z = MatrixOp([[1, 0], [0, -1]])
            self.assertNotEqual(expr * z, 2 * z)

        with self.subTest("additions aggregation"):
            z = MatrixOp([[1, 0], [0, -1]])
            a = z + z + Z
            b = 2 * z + Z
            c = z + Z + z
            self.assertEqual(a, b)
            self.assertEqual(b, c)
            self.assertEqual(a, c)

    def test_circuit_compose_register_independent(self):
        """Test that CircuitOp uses combines circuits independent of the register.

        I.e. that is uses ``QuantumCircuit.compose`` over ``combine`` or ``extend``.
        """
        op = Z ^ 2
        qr = QuantumRegister(2, "my_qr")
        circuit = QuantumCircuit(qr)
        composed = op.compose(CircuitOp(circuit))

        self.assertEqual(composed.num_qubits, 2)

    def test_matrix_op_conversions(self):
        """Test to reveal QiskitError when to_instruction or to_circuit method is called on
        parameterized matrix op."""
        m = np.array([[0, 0, 1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, -1, 0, 0]])
        matrix_op = MatrixOp(m, Parameter("beta"))
        for method in ["to_instruction", "to_circuit"]:
            with self.subTest(method):
                # QiskitError: multiplication of Operator with ParameterExpression isn't implemented
                self.assertRaises(QiskitError, getattr(matrix_op, method))

    def test_list_op_to_circuit(self):
        """Test if unitary ListOps transpile to circuit."""

        # generate unitary matrices of dimension 2,4,8, seed is fixed
        np.random.seed(233423)
        u2 = unitary_group.rvs(2)
        u4 = unitary_group.rvs(4)
        u8 = unitary_group.rvs(8)

        # pauli matrices as numpy.arrays
        x = np.array([[0.0, 1.0], [1.0, 0.0]])
        y = np.array([[0.0, -1.0j], [1.0j, 0.0]])
        z = np.array([[1.0, 0.0], [0.0, -1.0]])

        # create MatrixOp and CircuitOp out of matrices
        op2 = MatrixOp(u2)
        op4 = MatrixOp(u4)
        op8 = MatrixOp(u8)
        c2 = op2.to_circuit_op()

        # algorithm using only matrix operations on numpy.arrays
        xu4 = np.kron(x, u4)
        zc2 = np.kron(z, u2)
        zc2y = np.kron(zc2, y)
        matrix = np.matmul(xu4, zc2y)
        matrix = np.matmul(matrix, u8)
        matrix = np.kron(matrix, u2)
        operator = Operator(matrix)

        # same algorithm as above, but using PrimitiveOps
        list_op = ((X ^ op4) @ (Z ^ c2 ^ Y) @ op8) ^ op2
        circuit = list_op.to_circuit()

        # verify that ListOp.to_circuit() outputs correct quantum circuit
        self.assertTrue(operator.equiv(circuit), "ListOp.to_circuit() outputs wrong circuit!")

    def test_composed_op_to_circuit(self):
        """
        Test if unitary ComposedOp transpile to circuit and represents expected operator.
        Test if to_circuit on non-unitary ListOp raises exception.
        """

        x = np.array([[0.0, 1.0], [1.0, 0.0]])  # Pauli X as numpy array
        y = np.array([[0.0, -1.0j], [1.0j, 0.0]])  # Pauli Y as numpy array

        m1 = np.array([[0, 0, 1, 0], [0, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0]])  # non-unitary
        m2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, -1, 0, 0]])  # non-unitary

        m_op1 = MatrixOp(m1)
        m_op2 = MatrixOp(m2)

        pm1 = (X ^ Y) ^ m_op1  # non-unitary TensoredOp
        pm2 = (X ^ Y) ^ m_op2  # non-unitary TensoredOp

        self.assertRaises(ExtensionError, pm1.to_circuit)
        self.assertRaises(ExtensionError, pm2.to_circuit)

        summed_op = pm1 + pm2  # unitary SummedOp([TensoredOp, TensoredOp])
        circuit = summed_op.to_circuit()  # should transpile without any exception

        # same algorithm that leads to summed_op above, but using only arrays and matrix operations
        unitary = np.kron(np.kron(x, y), m1 + m2)

        self.assertTrue(Operator(unitary).equiv(circuit))

    def test_pauli_op_to_circuit(self):
        """Test PauliOp.to_circuit()"""
        with self.subTest("single Pauli"):
            pauli = PauliOp(Pauli("Y"))
            expected = QuantumCircuit(1)
            expected.y(0)
            self.assertEqual(pauli.to_circuit(), expected)

        with self.subTest("single Pauli with phase"):
            pauli = PauliOp(Pauli("-iX"))
            expected = QuantumCircuit(1)
            expected.x(0)
            expected.global_phase = -pi / 2
            self.assertEqual(Operator(pauli.to_circuit()), Operator(expected))

        with self.subTest("two qubit"):
            pauli = PauliOp(Pauli("IX"))
            expected = QuantumCircuit(2)
            expected.pauli("IX", range(2))
            self.assertEqual(pauli.to_circuit(), expected)
            expected = QuantumCircuit(2)
            expected.x(0)
            expected.id(1)
            self.assertEqual(pauli.to_circuit().decompose(), expected)

        with self.subTest("two qubit with phase"):
            pauli = PauliOp(Pauli("iXZ"))
            expected = QuantumCircuit(2)
            expected.pauli("XZ", range(2))
            expected.global_phase = pi / 2
            self.assertEqual(pauli.to_circuit(), expected)
            expected = QuantumCircuit(2)
            expected.z(0)
            expected.x(1)
            expected.global_phase = pi / 2
            self.assertEqual(pauli.to_circuit().decompose(), expected)

    def test_op_to_circuit_with_parameters(self):
        """On parameterized SummedOp, to_matrix_op returns ListOp, instead of MatrixOp. To avoid
        the infinite recursion, OpflowError is raised."""
        m1 = np.array([[0, 0, 1, 0], [0, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0]])  # non-unitary
        m2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, -1, 0, 0]])  # non-unitary

        op1_with_param = MatrixOp(m1, Parameter("alpha"))  # non-unitary
        op2_with_param = MatrixOp(m2, Parameter("beta"))  # non-unitary

        summed_op_with_param = op1_with_param + op2_with_param  # unitary
        # should raise OpflowError error
        self.assertRaises(OpflowError, summed_op_with_param.to_circuit)

    def test_permute_list_op_with_inconsistent_num_qubits(self):
        """Test if permute raises error if ListOp contains operators with different num_qubits."""
        list_op = ListOp([X, X ^ X])
        self.assertRaises(OpflowError, list_op.permute, [0, 1])

    @data(Z, CircuitOp(ZGate()), MatrixOp([[1, 0], [0, -1]]))
    def test_op_indent(self, op):
        """Test that indentation correctly adds INDENTATION at the beginning of each line"""
        initial_str = str(op)
        indented_str = op._indent(initial_str)
        starts_with_indent = indented_str.startswith(op.INDENTATION)
        self.assertTrue(starts_with_indent)
        indented_str_content = (indented_str[len(op.INDENTATION) :]).split(f"\n{op.INDENTATION}")
        self.assertListEqual(indented_str_content, initial_str.split("\n"))

    def test_composed_op_immutable_under_eval(self):
        """Test ``ComposedOp.eval`` does not change the operator instance."""
        op = 2 * ComposedOp([X])
        _ = op.eval()
        # previous bug: after op.eval(), op was 2 * ComposedOp([2 * X])
        self.assertEqual(op, 2 * ComposedOp([X]))

    def test_op_parameters(self):
        """Test that Parameters are stored correctly"""
        phi = Parameter("φ")
        theta = ParameterVector(name="θ", length=2)

        qc = QuantumCircuit(2)
        qc.rz(phi, 0)
        qc.rz(phi, 1)
        for i in range(2):
            qc.rx(theta[i], i)
        qc.h(0)
        qc.x(1)

        l = Parameter("λ")
        op = PrimitiveOp(qc, coeff=l)

        params = {phi, l, *theta.params}

        self.assertEqual(params, op.parameters)
        self.assertEqual(params, StateFn(op).parameters)
        self.assertEqual(params, StateFn(qc, coeff=l).parameters)

    def test_list_op_parameters(self):
        """Test that Parameters are stored correctly in a List Operator"""
        lam = Parameter("λ")
        phi = Parameter("φ")
        omega = Parameter("ω")

        mat_op = PrimitiveOp([[0, 1], [1, 0]], coeff=omega)

        qc = QuantumCircuit(1)
        qc.rx(phi, 0)
        qc_op = PrimitiveOp(qc)

        op1 = SummedOp([mat_op, qc_op])

        params = [phi, omega]
        self.assertEqual(op1.parameters, set(params))

        # check list nesting case
        op2 = PrimitiveOp([[1, 0], [0, -1]], coeff=lam)

        list_op = ListOp([op1, op2])

        params.append(lam)
        self.assertEqual(list_op.parameters, set(params))

    @data(
        VectorStateFn([1, 0]),
        CircuitStateFn(QuantumCircuit(1)),
        OperatorStateFn(I),
        OperatorStateFn(MatrixOp([[1, 0], [0, 1]])),
        OperatorStateFn(CircuitOp(QuantumCircuit(1))),
    )
    def test_statefn_eval(self, op):
        """Test calling eval on StateFn returns the statevector."""
        expected = Statevector([1, 0])
        self.assertEqual(op.eval().primitive, expected)

    def test_sparse_eval(self):
        """Test calling eval on a DictStateFn returns a sparse statevector."""
        op = DictStateFn({"0": 1})
        expected = scipy.sparse.csr_matrix([[1, 0]])
        self.assertFalse((op.eval().primitive != expected).toarray().any())

    def test_sparse_to_dict(self):
        """Test converting a sparse vector state function to a dict state function."""
        isqrt2 = 1 / np.sqrt(2)
        sparse = scipy.sparse.csr_matrix([[0, isqrt2, 0, isqrt2]])
        sparse_fn = SparseVectorStateFn(sparse)
        dict_fn = DictStateFn({"01": isqrt2, "11": isqrt2})

        with self.subTest("sparse to dict"):
            self.assertEqual(dict_fn, sparse_fn.to_dict_fn())

        with self.subTest("dict to sparse"):
            self.assertEqual(dict_fn.to_spmatrix_op(), sparse_fn)

    def test_to_circuit_op(self):
        """Test to_circuit_op method."""
        vector = np.array([2, 2])
        vsfn = VectorStateFn([1, 1], coeff=2)
        dsfn = DictStateFn({"0": 1, "1": 1}, coeff=2)

        for sfn in [vsfn, dsfn]:
            np.testing.assert_array_almost_equal(sfn.to_circuit_op().eval().primitive.data, vector)

    def test_invalid_primitive(self):
        """Test invalid MatrixOp construction"""
        msg = (
            "MatrixOp can only be instantiated with "
            "['list', 'ndarray', 'spmatrix', 'Operator'], not "
        )

        with self.assertRaises(TypeError) as cm:
            _ = MatrixOp("invalid")

        self.assertEqual(str(cm.exception), msg + "'str'")

        with self.assertRaises(TypeError) as cm:
            _ = MatrixOp(None)

        self.assertEqual(str(cm.exception), msg + "'NoneType'")

        with self.assertRaises(TypeError) as cm:
            _ = MatrixOp(2.0)

        self.assertEqual(str(cm.exception), msg + "'float'")

    def test_summedop_equals(self):
        """Test SummedOp.equals"""
        ops = [Z, CircuitOp(ZGate()), MatrixOp([[1, 0], [0, -1]]), Zero, Minus]
        sum_op = sum(ops + [ListOp(ops)])
        self.assertEqual(sum_op, sum_op)
        self.assertEqual(sum_op + sum_op, 2 * sum_op)
        self.assertEqual(sum_op + sum_op + sum_op, 3 * sum_op)
        ops2 = [Z, CircuitOp(ZGate()), MatrixOp([[1, 0], [0, 1]]), Zero, Minus]
        sum_op2 = sum(ops2 + [ListOp(ops)])
        self.assertNotEqual(sum_op, sum_op2)
        self.assertEqual(sum_op2, sum_op2)
        sum_op3 = sum(ops)
        self.assertNotEqual(sum_op, sum_op3)
        self.assertNotEqual(sum_op2, sum_op3)
        self.assertEqual(sum_op3, sum_op3)

    def test_empty_listops(self):
        """Test reduce and eval on ListOp with empty oplist."""
        with self.subTest("reduce empty ComposedOp "):
            self.assertEqual(ComposedOp([]).reduce(), ComposedOp([]))
        with self.subTest("reduce empty TensoredOp "):
            self.assertEqual(TensoredOp([]).reduce(), TensoredOp([]))
        with self.subTest("eval empty ComposedOp "):
            self.assertEqual(ComposedOp([]).eval(), 0.0)
        with self.subTest("eval empty TensoredOp "):
            self.assertEqual(TensoredOp([]).eval(), 0.0)


class TestOpMethods(QiskitOpflowTestCase):
    """Basic method tests."""

    def test_listop_num_qubits(self):
        """Test that ListOp.num_qubits checks that all operators have the same number of qubits."""
        op = ListOp([X ^ Y, Y ^ Z])
        with self.subTest("All operators have the same numbers of qubits"):
            self.assertEqual(op.num_qubits, 2)

        op = ListOp([X ^ Y, Y])
        with self.subTest("Operators have different numbers of qubits"):
            with self.assertRaises(ValueError):
                op.num_qubits  # pylint: disable=pointless-statement

            with self.assertRaises(ValueError):
                X @ op  # pylint: disable=pointless-statement


@ddt
class TestListOpMethods(QiskitOpflowTestCase):
    """Test ListOp accessing methods"""

    @data(ListOp, SummedOp, ComposedOp, TensoredOp)
    def test_indexing(self, list_op_type):
        """Test indexing and slicing"""
        coeff = 3 + 0.2j
        states_op = list_op_type([X, Y, Z, I], coeff=coeff)

        single_op = states_op[1]
        self.assertIsInstance(single_op, OperatorBase)
        self.assertNotIsInstance(single_op, ListOp)

        list_one_element = states_op[1:2]
        self.assertIsInstance(list_one_element, list_op_type)
        self.assertEqual(len(list_one_element), 1)
        self.assertEqual(list_one_element[0], Y)

        list_two_elements = states_op[::2]
        self.assertIsInstance(list_two_elements, list_op_type)
        self.assertEqual(len(list_two_elements), 2)
        self.assertEqual(list_two_elements[0], X)
        self.assertEqual(list_two_elements[1], Z)

        self.assertEqual(list_one_element.coeff, coeff)
        self.assertEqual(list_two_elements.coeff, coeff)


class TestListOpComboFn(QiskitOpflowTestCase):
    """Test combo fn is propagated."""

    def setUp(self):
        super().setUp()
        self.combo_fn = lambda x: [x_i ** 2 for x_i in x]
        self.listop = ListOp([X], combo_fn=self.combo_fn)

    def assertComboFnPreserved(self, processed_op):
        """Assert the quadratic combo_fn is preserved."""
        x = [1, 2, 3]
        self.assertListEqual(processed_op.combo_fn(x), self.combo_fn(x))

    def test_at_conversion(self):
        """Test after conversion the combo_fn is preserved."""
        for method in ["to_matrix_op", "to_pauli_op", "to_circuit_op"]:
            with self.subTest(method):
                converted = getattr(self.listop, method)()
                self.assertComboFnPreserved(converted)

    def test_after_mul(self):
        """Test after multiplication the combo_fn is preserved."""
        self.assertComboFnPreserved(2 * self.listop)

    def test_at_traverse(self):
        """Test after traversing the combo_fn is preserved."""

        def traverse_fn(op):
            return -op

        traversed = self.listop.traverse(traverse_fn)
        self.assertComboFnPreserved(traversed)

    def test_after_adjoint(self):
        """Test after traversing the combo_fn is preserved."""
        self.assertComboFnPreserved(self.listop.adjoint())

    def test_after_reduce(self):
        """Test after reducing the combo_fn is preserved."""
        self.assertComboFnPreserved(self.listop.reduce())


if __name__ == "__main__":
    unittest.main()

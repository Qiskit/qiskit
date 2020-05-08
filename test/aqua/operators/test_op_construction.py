# -*- coding: utf-8 -*-

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

import unittest
from test.aqua import QiskitAquaTestCase
import itertools
import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.circuit.library import CZGate

from qiskit.aqua.operators import X, Y, Z, I, CX, T, H, PrimitiveOp, SummedOp, PauliOp, Minus


# pylint: disable=invalid-name

class TestOpConstruction(QiskitAquaTestCase):
    """Operator Construction tests."""

    def test_pauli_primitives(self):
        """ from to file test """
        newop = X ^ Y ^ Z ^ I
        self.assertEqual(newop.primitive, Pauli(label='XYZI'))

        kpower_op = (Y ^ 5) ^ (I ^ 3)
        self.assertEqual(kpower_op.primitive, Pauli(label='YYYYYIII'))

        kpower_op2 = (Y ^ I) ^ 4
        self.assertEqual(kpower_op2.primitive, Pauli(label='YIYIYIYI'))

        # Check immutability
        self.assertEqual(X.primitive, Pauli(label='X'))
        self.assertEqual(Y.primitive, Pauli(label='Y'))
        self.assertEqual(Z.primitive, Pauli(label='Z'))
        self.assertEqual(I.primitive, Pauli(label='I'))

    def test_composed_eval(self):
        """ Test eval of ComposedOp """
        self.assertAlmostEqual(Minus.eval('1'), -.5 ** .5)

    def test_evals(self):
        """ evals test """
        # pylint: disable=no-member
        # TODO: Think about eval names
        self.assertEqual(Z.eval('0').eval('0'), 1)
        self.assertEqual(Z.eval('1').eval('0'), 0)
        self.assertEqual(Z.eval('0').eval('1'), 0)
        self.assertEqual(Z.eval('1').eval('1'), -1)
        self.assertEqual(X.eval('0').eval('0'), 0)
        self.assertEqual(X.eval('1').eval('0'), 1)
        self.assertEqual(X.eval('0').eval('1'), 1)
        self.assertEqual(X.eval('1').eval('1'), 0)
        self.assertEqual(Y.eval('0').eval('0'), 0)
        self.assertEqual(Y.eval('1').eval('0'), -1j)
        self.assertEqual(Y.eval('0').eval('1'), 1j)
        self.assertEqual(Y.eval('1').eval('1'), 0)

        # Check that Pauli logic eval returns same as matrix logic
        self.assertEqual(PrimitiveOp(Z.to_matrix()).eval('0').eval('0'), 1)
        self.assertEqual(PrimitiveOp(Z.to_matrix()).eval('1').eval('0'), 0)
        self.assertEqual(PrimitiveOp(Z.to_matrix()).eval('0').eval('1'), 0)
        self.assertEqual(PrimitiveOp(Z.to_matrix()).eval('1').eval('1'), -1)
        self.assertEqual(PrimitiveOp(X.to_matrix()).eval('0').eval('0'), 0)
        self.assertEqual(PrimitiveOp(X.to_matrix()).eval('1').eval('0'), 1)
        self.assertEqual(PrimitiveOp(X.to_matrix()).eval('0').eval('1'), 1)
        self.assertEqual(PrimitiveOp(X.to_matrix()).eval('1').eval('1'), 0)
        self.assertEqual(PrimitiveOp(Y.to_matrix()).eval('0').eval('0'), 0)
        self.assertEqual(PrimitiveOp(Y.to_matrix()).eval('1').eval('0'), -1j)
        self.assertEqual(PrimitiveOp(Y.to_matrix()).eval('0').eval('1'), 1j)
        self.assertEqual(PrimitiveOp(Y.to_matrix()).eval('1').eval('1'), 0)

        pauli_op = Z ^ I ^ X ^ Y
        mat_op = PrimitiveOp(pauli_op.to_matrix())
        full_basis = list(map(''.join, itertools.product('01', repeat=pauli_op.num_qubits)))
        for bstr1, bstr2 in itertools.product(full_basis, full_basis):
            # print('{} {} {} {}'.format(bstr1, bstr2, pauli_op.eval(bstr1, bstr2),
            # mat_op.eval(bstr1, bstr2)))
            np.testing.assert_array_almost_equal(pauli_op.eval(bstr1).eval(bstr2),
                                                 mat_op.eval(bstr1).eval(bstr2))

        gnarly_op = SummedOp([(H ^ I ^ Y).compose(X ^ X ^ Z).tensor(Z),
                              PrimitiveOp(Operator.from_label('+r0I')),
                              3 * (X ^ CX ^ T)], coeff=3 + .2j)
        gnarly_mat_op = PrimitiveOp(gnarly_op.to_matrix())
        full_basis = list(map(''.join, itertools.product('01', repeat=gnarly_op.num_qubits)))
        for bstr1, bstr2 in itertools.product(full_basis, full_basis):
            np.testing.assert_array_almost_equal(gnarly_op.eval(bstr1).eval(bstr2),
                                                 gnarly_mat_op.eval(bstr1).eval(bstr2))

    def test_circuit_construction(self):
        """ circuit construction test """
        hadq2 = H ^ I
        cz = hadq2.compose(CX).compose(hadq2)
        qc = QuantumCircuit(2)
        qc.append(cz.primitive, qargs=range(2))

        ref_cz_mat = PrimitiveOp(CZGate()).to_matrix()
        np.testing.assert_array_almost_equal(cz.to_matrix(), ref_cz_mat)

    def test_io_consistency(self):
        """ consistency test """
        new_op = X ^ Y ^ I
        label = 'XYI'
        # label = new_op.primitive.to_label()
        self.assertEqual(str(new_op.primitive), label)
        np.testing.assert_array_almost_equal(new_op.primitive.to_matrix(),
                                             Operator.from_label(label).data)
        self.assertEqual(new_op.primitive, Pauli(label=label))

        x_mat = X.primitive.to_matrix()
        y_mat = Y.primitive.to_matrix()
        i_mat = np.eye(2, 2)
        np.testing.assert_array_almost_equal(new_op.primitive.to_matrix(),
                                             np.kron(np.kron(x_mat, y_mat), i_mat))

        hi = np.kron(H.to_matrix(), I.to_matrix())
        hi2 = Operator.from_label('HI').data
        hi3 = (H ^ I).to_matrix()
        np.testing.assert_array_almost_equal(hi, hi2)
        np.testing.assert_array_almost_equal(hi2, hi3)

        xy = np.kron(X.to_matrix(), Y.to_matrix())
        xy2 = Operator.from_label('XY').data
        xy3 = (X ^ Y).to_matrix()
        np.testing.assert_array_almost_equal(xy, xy2)
        np.testing.assert_array_almost_equal(xy2, xy3)

        # Check if numpy array instantiation is the same as from Operator
        matrix_op = Operator.from_label('+r')
        np.testing.assert_array_almost_equal(PrimitiveOp(matrix_op).to_matrix(),
                                             PrimitiveOp(matrix_op.data).to_matrix())
        # Ditto list of lists
        np.testing.assert_array_almost_equal(PrimitiveOp(matrix_op.data.tolist()).to_matrix(),
                                             PrimitiveOp(matrix_op.data).to_matrix())

        # TODO make sure this works once we resolve endianness mayhem
        # qc = QuantumCircuit(3)
        # qc.x(2)
        # qc.y(1)
        # from qiskit import BasicAer, QuantumCircuit, execute
        # unitary = execute(qc, BasicAer.get_backend('unitary_simulator')).result().get_unitary()
        # np.testing.assert_array_almost_equal(new_op.primitive.to_matrix(), unitary)

    def test_to_matrix(self):
        """to matrix text """
        np.testing.assert_array_equal(X.to_matrix(), Operator.from_label('X').data)
        np.testing.assert_array_equal(Y.to_matrix(), Operator.from_label('Y').data)
        np.testing.assert_array_equal(Z.to_matrix(), Operator.from_label('Z').data)

        op1 = Y + H
        np.testing.assert_array_almost_equal(op1.to_matrix(), Y.to_matrix() + H.to_matrix())

        op2 = op1 * .5
        np.testing.assert_array_almost_equal(op2.to_matrix(), op1.to_matrix() * .5)

        op3 = (4 - .6j) * op2
        np.testing.assert_array_almost_equal(op3.to_matrix(), op2.to_matrix() * (4 - .6j))

        op4 = op3.tensor(X)
        np.testing.assert_array_almost_equal(op4.to_matrix(),
                                             np.kron(op3.to_matrix(), X.to_matrix()))

        op5 = op4.compose(H ^ I)
        np.testing.assert_array_almost_equal(op5.to_matrix(), np.dot(op4.to_matrix(),
                                                                     (H ^ I).to_matrix()))

        op6 = op5 + PrimitiveOp(Operator.from_label('+r').data)
        np.testing.assert_array_almost_equal(
            op6.to_matrix(), op5.to_matrix() + Operator.from_label('+r').data)

    def test_adjoint(self):
        """ adjoint test """
        gnarly_op = 3 * (H ^ I ^ Y).compose(X ^ X ^ Z).tensor(T ^ Z) + \
            PrimitiveOp(Operator.from_label('+r0IX').data)
        np.testing.assert_array_almost_equal(np.conj(np.transpose(gnarly_op.to_matrix())),
                                             gnarly_op.adjoint().to_matrix())

    def test_primitive_strings(self):
        """ get primitives test """
        self.assertEqual(X.primitive_strings(), {'Pauli'})

        gnarly_op = 3 * (H ^ I ^ Y).compose(X ^ X ^ Z).tensor(T ^ Z) + \
            PrimitiveOp(Operator.from_label('+r0IX').data)
        self.assertEqual(gnarly_op.primitive_strings(), {'QuantumCircuit', 'Matrix'})

    def test_to_pauli_op(self):
        """ Test to_pauli_op method """
        gnarly_op = 3 * (H ^ I ^ Y).compose(X ^ X ^ Z).tensor(T ^ Z) + \
            PrimitiveOp(Operator.from_label('+r0IX').data)
        mat_op = gnarly_op.to_matrix_op()
        pauli_op = gnarly_op.to_pauli_op()
        self.assertIsInstance(pauli_op, SummedOp)
        for p in pauli_op:
            self.assertIsInstance(p, PauliOp)
        np.testing.assert_array_almost_equal(mat_op.to_matrix(), pauli_op.to_matrix())

    def test_circuit_permute(self):
        r""" Test the CircuitOp's .permute method """
        perm = range(7)[::-1]
        c_op = (((CX ^ 3) ^ X) @
                (H ^ 7) @
                (X ^ Y ^ Z ^ I ^ X ^ X ^ X) @
                (Y ^ (CX ^ 3)) @
                (X ^ Y ^ Z ^ I ^ X ^ X ^ X))
        c_op_perm = c_op.permute(perm)
        self.assertNotEqual(c_op, c_op_perm)
        c_op_id = c_op_perm.permute(perm)
        self.assertEqual(c_op, c_op_id)


if __name__ == '__main__':
    unittest.main()

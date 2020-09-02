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

""" Test Pauli Change of Basis Converter """

import unittest
from test.aqua import QiskitAquaTestCase

import itertools
import numpy as np

from qiskit.aqua.operators import X, Y, Z, I, SummedOp, ComposedOp
from qiskit.aqua.operators.converters import PauliBasisChange


class TestPauliCoB(QiskitAquaTestCase):
    """Pauli Change of Basis Converter tests."""

    def test_pauli_cob_singles(self):
        """ from to file test """
        singles = [X, Y, Z]
        dests = [None, Y]
        for pauli, dest in itertools.product(singles, dests):
            # print(pauli)
            converter = PauliBasisChange(destination_basis=dest)
            inst, dest = converter.get_cob_circuit(pauli.primitive)
            cob = converter.convert(pauli)
            np.testing.assert_array_almost_equal(
                pauli.to_matrix(), inst.adjoint().to_matrix() @ dest.to_matrix() @ inst.to_matrix())
            np.testing.assert_array_almost_equal(pauli.to_matrix(), cob.to_matrix())
            np.testing.assert_array_almost_equal(
                inst.compose(pauli).compose(inst.adjoint()).to_matrix(), dest.to_matrix())

    def test_pauli_cob_two_qubit(self):
        """ pauli cob two qubit test """
        multis = [Y ^ X, Z ^ Y, I ^ Z, Z ^ I, X ^ X, I ^ X]
        for pauli, dest in itertools.product(multis, reversed(multis)):
            converter = PauliBasisChange(destination_basis=dest)
            inst, dest = converter.get_cob_circuit(pauli.primitive)
            cob = converter.convert(pauli)
            np.testing.assert_array_almost_equal(
                pauli.to_matrix(), inst.adjoint().to_matrix() @ dest.to_matrix() @ inst.to_matrix())
            np.testing.assert_array_almost_equal(pauli.to_matrix(), cob.to_matrix())
            np.testing.assert_array_almost_equal(
                inst.compose(pauli).compose(inst.adjoint()).to_matrix(), dest.to_matrix())

    def test_pauli_cob_multiqubit(self):
        """ pauli cob multi qubit test """
        # Helpful prints for debugging commented out below.
        multis = [Y ^ X ^ I ^ I, I ^ Z ^ Y ^ X, X ^ Y ^ I ^ Z, I ^ I ^ I ^ X, X ^ X ^ X ^ X]
        for pauli, dest in itertools.product(multis, reversed(multis)):
            # print(pauli)
            # print(dest)
            converter = PauliBasisChange(destination_basis=dest)
            inst, dest = converter.get_cob_circuit(pauli.primitive)
            cob = converter.convert(pauli)
            # print(inst)
            # print(pauli.to_matrix())
            # print(np.round(inst.adjoint().to_matrix() @ cob.to_matrix()))
            np.testing.assert_array_almost_equal(
                pauli.to_matrix(), inst.adjoint().to_matrix() @ dest.to_matrix() @ inst.to_matrix())
            np.testing.assert_array_almost_equal(pauli.to_matrix(), cob.to_matrix())
            np.testing.assert_array_almost_equal(
                inst.compose(pauli).compose(inst.adjoint()).to_matrix(), dest.to_matrix())

    def test_pauli_cob_traverse(self):
        """ pauli cob traverse test """
        # Helpful prints for debugging commented out below.
        multis = [(X ^ Y) + (I ^ Z) + (Z ^ Z), (Y ^ X ^ I ^ I) + (I ^ Z ^ Y ^ X)]
        dests = [Y ^ Y, I ^ I ^ I ^ Z]
        for pauli, dest in zip(multis, dests):
            # print(pauli)
            # print(dest)
            converter = PauliBasisChange(destination_basis=dest, traverse=True)

            cob = converter.convert(pauli)
            self.assertIsInstance(cob, SummedOp)
            inst = [None] * len(pauli.oplist)
            ret_dest = [None] * len(pauli.oplist)
            cob_mat = [None] * len(pauli.oplist)
            for i in range(len(pauli.oplist)):
                inst[i], ret_dest[i] = converter.get_cob_circuit(pauli.oplist[i].primitive)
                self.assertEqual(dest, ret_dest[i])

                # print(inst[i])
                # print(pauli.oplist[i].to_matrix())
                # print(np.round(inst[i].adjoint().to_matrix() @ cob.oplist[i].to_matrix()))

                self.assertIsInstance(cob.oplist[i], ComposedOp)
                cob_mat[i] = cob.oplist[i].to_matrix()
                np.testing.assert_array_almost_equal(pauli.oplist[i].to_matrix(), cob_mat[i])
            np.testing.assert_array_almost_equal(pauli.to_matrix(), sum(cob_mat))


if __name__ == '__main__':
    unittest.main()

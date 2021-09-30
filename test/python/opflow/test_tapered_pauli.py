# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test TaperedPauliSumOp """

import unittest
from test.python.opflow import QiskitOpflowTestCase

from qiskit.circuit import Parameter
from qiskit.opflow import PauliSumOp, TaperedPauliSumOp, Z2Symmetries
from qiskit.quantum_info import Pauli, SparsePauliOp


class TestZ2Symmetries(QiskitOpflowTestCase):
    """Z2Symmetries tests."""

    def setUp(self):
        super().setUp()
        z2_symmetries = Z2Symmetries(
            [Pauli("IIZI"), Pauli("ZIII")], [Pauli("IIXI"), Pauli("XIII")], [1, 3], [-1, 1]
        )
        self.primitive = SparsePauliOp.from_list(
            [
                ("II", (-1.052373245772859)),
                ("ZI", (-0.39793742484318007)),
                ("IZ", (0.39793742484318007)),
                ("ZZ", (-0.01128010425623538)),
                ("XX", (0.18093119978423142)),
            ]
        )
        self.tapered_qubit_op = TaperedPauliSumOp(self.primitive, z2_symmetries)

    def test_multiply_parameter(self):
        """test for multiplication of parameter"""
        param = Parameter("c")
        expected = PauliSumOp(self.primitive, coeff=param)
        self.assertEqual(param * self.tapered_qubit_op, expected)

    def test_assign_parameters(self):
        """test assign_parameters"""
        param = Parameter("c")
        parameterized_op = param * self.tapered_qubit_op
        expected = PauliSumOp(self.primitive, coeff=46)
        self.assertEqual(parameterized_op.assign_parameters({param: 46}), expected)


if __name__ == "__main__":
    unittest.main()

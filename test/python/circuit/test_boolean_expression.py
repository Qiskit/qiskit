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

"""Test dimacs oracle circuits."""

import unittest

from qiskit.test.base import QiskitTestCase
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.classicalfunction.boolean_expression import BooleanExpression
from qiskit.quantum_info import Operator

EXPRESSION_TEST = "(not v1 or not v2 or not v3) and (v1 or not v2 or v3) and \
    (v1 or v2 or not v3) and (v1 or not v2 or not v3) and (not v1 or v2 or v3)"

DIMACS_TEST = """c This is an example DIMACS CNF file
p cnf 3 5
-1 -2 -3 0
1 -2 3 0
1 2 -3 0
1 -2 -3 0
-1 2 3 0
"""


class TestBooleanExpression(QiskitTestCase):
    """ """

    def test_evaluate(self):
        """ Test the constructor of LogicalExpressionOracle"""
        expression = BooleanExpression(EXPRESSION_TEST)
        result = expression.evaluate()
        self.assertTrue(Operator(le_oracle).equiv(Operator(self._expected_le_oracle)))

    def test_from_dimacs(self):
        """ Test the from_dimacs of LogicalExpressionOracle"""
        le_oracle = BooleanExpression.from_dimacs(DIMACS_TEST)
        self.assertTrue(Operator(le_oracle).equiv(Operator(self._expected_le_oracle)))

        with self.assertRaises(ValueError):
            # Test for a string does not have initialization, "p cnf".
            error_dimacs1 = """c This is an example DIMACS CNF file
            -1 2 3 0
            """
            BooleanExpression.from_dimacs(error_dimacs1)

        with self.assertRaises(ValueError):
            # Test for a line does not end with "0".
            error_dimacs2 = """c This is an example DIMACS CNF file
            p cnf 3 5
            -1 -2 -3
            """
            BooleanExpression.from_dimacs(error_dimacs2)

    def test_evaluate_bitstring(self):
        """ Test the evaluate_bitstring func"""
        le_oracle = BooleanExpression(EXPRESSION_TEST)
        self.assertTrue(le_oracle.evaluate_bitstring('101'))
        self.assertFalse(le_oracle.evaluate_bitstring('001'))


if __name__ == '__main__':
    unittest.main()

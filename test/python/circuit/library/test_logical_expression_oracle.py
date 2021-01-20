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
from qiskit.circuit.library.logical_expression_oracle import LogicalExpressionOracle
from qiskit.quantum_info import Operator

EXPRESSION_TEST = "(~v1 | ~v2 | ~v3) & (v1 | ~v2 | v3) & \
    (v1 | v2 | ~v3) & (v1 | ~v2 | ~v3) & (~v1 | v2 | v3)"

DIMACS_TEST = """c This is an example DIMACS CNF file
p cnf 3 5
-1 -2 -3 0
1 -2 3 0
1 2 -3 0
1 -2 -3 0
-1 2 3 0
"""


class TestLogicalExpressionOracle(QiskitTestCase):
    """Test library of Dimacs oracle circuits."""

    def setUp(self):
        super().setUp()
        self._expected_le_oracle = QuantumCircuit(4)
        self._expected_le_oracle.x(3)
        self._expected_le_oracle.h(3)
        self._expected_le_oracle.barrier()
        self._expected_le_oracle.x([0, 2])
        self._expected_le_oracle.mcx([0, 1, 2], 3)
        self._expected_le_oracle.x([0, 2])
        self._expected_le_oracle.barrier()
        self._expected_le_oracle.cx(1, 3)
        self._expected_le_oracle.barrier()
        self._expected_le_oracle.x([0, 1])
        self._expected_le_oracle.ccx(0, 1, 3)
        self._expected_le_oracle.x([0, 1])
        self._expected_le_oracle.barrier()
        self._expected_le_oracle.cx(2, 3)
        self._expected_le_oracle.barrier()
        self._expected_le_oracle.h(3)
        self._expected_le_oracle.x(3)

    def test_logical_expression_oracle(self):
        """ Test the constructor of LogicalExpressionOracle"""
        le_oracle = LogicalExpressionOracle(EXPRESSION_TEST)
        self.assertTrue(Operator(le_oracle).equiv(Operator(self._expected_le_oracle)))

    def test_from_dimacs(self):
        """ Test the from_dimacs of LogicalExpressionOracle"""
        le_oracle = LogicalExpressionOracle.from_dimacs(DIMACS_TEST)
        self.assertTrue(Operator(le_oracle).equiv(Operator(self._expected_le_oracle)))

    def test_evaluate_bitstring(self):
        """ Test the evaluate_bitstring func"""
        le_oracle = LogicalExpressionOracle(EXPRESSION_TEST)
        self.assertTrue(le_oracle.evaluate_bitstring('101'))
        self.assertFalse(le_oracle.evaluate_bitstring('001'))


if __name__ == '__main__':
    unittest.main()

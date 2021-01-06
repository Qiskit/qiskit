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
from qiskit.circuit.library.dimacs_oracle import DIMACSOracle
from qiskit.quantum_info import Operator


DIMACS_TEST = """c This is an example DIMACS CNF file
p cnf 3 5
-1 -2 -3 0
1 -2 3 0
1 2 -3 0
1 -2 -3 0
-1 2 3 0
"""


class TestDimacsOracle(QiskitTestCase):
    """Test library of Dimacs oracle circuits."""

    def setUp(self):
        super().setUp()
        self._expected_dimacs_oracle = QuantumCircuit(4)
        self._expected_dimacs_oracle.x(3)
        self._expected_dimacs_oracle.h(3)
        self._expected_dimacs_oracle.barrier()
        self._expected_dimacs_oracle.x([0, 2])
        self._expected_dimacs_oracle.mcx([0, 1, 2], 3)
        self._expected_dimacs_oracle.x([0, 2])
        self._expected_dimacs_oracle.barrier()
        self._expected_dimacs_oracle.cx(1, 3)
        self._expected_dimacs_oracle.barrier()
        self._expected_dimacs_oracle.x([0, 1])
        self._expected_dimacs_oracle.ccx(0, 1, 3)
        self._expected_dimacs_oracle.x([0, 1])
        self._expected_dimacs_oracle.barrier()
        self._expected_dimacs_oracle.cx(2, 3)
        self._expected_dimacs_oracle.barrier()
        self._expected_dimacs_oracle.h(3)
        self._expected_dimacs_oracle.x(3)

    def test_dimacs_oracle(self):
        """ Test the constructor of DIMACSOracle"""
        dimacs_oracle = DIMACSOracle(DIMACS_TEST)
        self.assertTrue(Operator(dimacs_oracle).equiv(Operator(self._expected_dimacs_oracle)))

    def test_evaluate_bitstring(self):
        """ Test the evaluate_bitstring func"""
        dimacs_oracle = DIMACSOracle(DIMACS_TEST)
        self.assertTrue(dimacs_oracle.evaluate_bitstring('101'))
        self.assertFalse(dimacs_oracle.evaluate_bitstring('001'))


if __name__ == '__main__':
    unittest.main()

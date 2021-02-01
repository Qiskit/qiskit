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


"""Test Qiskit's Circuit Equivalence Checker"""

import unittest

from qiskit.test import QiskitTestCase

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.tools import EquivalenceChecker


class TestEquivalenceChecker(QiskitTestCase):
    """Test equivalence checker"""

    def verify_result(self, checker, circ1, circ2, equivalent):
        res = checker.run(circ1, circ2)
        self.assertTrue(res._success)
        self.assertEqual(equivalent, res._equivalent)

    def test_basic(self):
        '''Test equivalence chekcer for valid circuits'''
        checker = EquivalenceChecker()
        
        circ1 = QuantumCircuit(2)
        circ1.cx(0, 1)
        circ1.cx(1, 0)
        circ1.cx(0, 1)

        circ2 = QuantumCircuit(2)
        circ2.cx(1, 0)
        circ2.cx(0, 1)
        circ2.cx(1, 0)

        self.verify_result(checker, circ1, circ2, True)
        
        circ1.x(0)
        self.verify_result(checker, circ1, circ2, False)


if __name__ == '__main__':
    unittest.main()

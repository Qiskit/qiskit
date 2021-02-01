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
from qiskit.circuit.tools import UnitaryEquivalenceChecker


class TestEquivalenceChecker(QiskitTestCase):
    """Test equivalence checker"""

    def verify_result(self, checkers, circ1, circ2, success, equivalent):
        for checker in checkers:
            res = checker.run(circ1, circ2)
            checker_msg = "Checker '" + checker.name + "' failed"
            self.assertEqual(success, res.success, checker_msg)
            self.assertEqual(equivalent, res.equivalent, checker_msg)

    def test_equivalence_checkers_basic(self):
        '''Test equivalence chekcer for valid circuits'''
        checkers = [UnitaryEquivalenceChecker()]
        
        circ1 = QuantumCircuit(2)
        circ1.cx(0, 1)
        circ1.cx(1, 0)
        circ1.cx(0, 1)

        circ2 = QuantumCircuit(2)
        circ2.cx(1, 0)
        circ2.cx(0, 1)
        circ2.cx(1, 0)

        self.verify_result(checkers, circ1, circ2, True, True)
        
        circ1.x(0)
        self.verify_result(checkers, circ1, circ2, True, False)

    def test_error_in_unitary_checker(self):
        '''Test error messages for invalid circuits'''
        checker = UnitaryEquivalenceChecker()
        
        circ1 = QuantumCircuit(1, 1)
        circ1.measure(0, 0)

        circ2 = QuantumCircuit(1, 1)

        self.verify_result([checker], circ1, circ2, False, None)
        self.verify_result([checker], circ2, circ1, False, None)

        circ2.measure(0, 0)

        self.verify_result([checker], circ1, circ2, False, None)

if __name__ == '__main__':
    unittest.main()

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
from qiskit.algorithms import UnitaryEquivalenceChecker


class TestEquivalenceChecker(QiskitTestCase):
    """Test equivalence checker"""

    def setUp(self) -> None:
        super().setUp()
        try:
            from qiskit import Aer   # pylint: disable=unused-import
            self.aer_installed = True
        except ImportError:
            self.aer_installed = False

    def verify_result(self, checkers, circ1, circ2,
                      success, equivalent,
                      phase):
        """
        Run the checkers, and verify that their outputs
        match the expected result
        """

        for checker in checkers:
            res = checker.run(circ1, circ2, phase=phase)
            checker_msg = "Checker '" + checker.name + "' failed"
            self.assertEqual(success, res.success, checker_msg)
            self.assertEqual(equivalent, res.equivalent, checker_msg)

    def test_equivalence_checkers_equal_phase(self):
        """Test equivalence chekcers for valid circuits, requiring equal phase"""
        checkers = [
            UnitaryEquivalenceChecker('quantum_info', 'unitary_qi')
            ]
        if self.aer_installed:
            checkers.append(UnitaryEquivalenceChecker('aer', 'unitary_aer'))

        circ1 = QuantumCircuit(2)
        circ1.cx(0, 1)
        circ1.cx(1, 0)
        circ1.cx(0, 1)

        circ2 = QuantumCircuit(2)
        circ2.cx(1, 0)
        circ2.cx(0, 1)
        circ2.cx(1, 0)

        self.verify_result(checkers, circ1, circ2, True, True, 'equal')

        circ1.x(0)
        self.verify_result(checkers, circ1, circ2, True, False, 'equal')

    def test_equivalence_checkers_up_to_global_phase(self):
        """Test equivalence chekcers for valid circuits, requiring up-to-global phase"""
        checkers = [
            UnitaryEquivalenceChecker('quantum_info', 'unitary_qi')
            ]
        if self.aer_installed:
            checkers.append(UnitaryEquivalenceChecker('aer', 'unitary_aer'))

        circ1 = QuantumCircuit(1)
        circ1.x(0)
        circ1.z(0)

        circ2 = QuantumCircuit(1)
        circ2.y(0)

        self.verify_result(checkers, circ1, circ2, True, False, 'equal')
        self.verify_result(checkers, circ1, circ2, True, True, 'up_to_global')

        circ1.x(0)
        self.verify_result(checkers, circ1, circ2, True, False, 'up_to_global')

    def test_error_in_unitary_checker(self):
        """Test error messages for invalid circuits"""
        checkers = [
            UnitaryEquivalenceChecker('quantum_info', 'unitary_qi')
            ]
        if self.aer_installed:
            checkers.append(UnitaryEquivalenceChecker('aer', 'unitary_aer'))

        circ1 = QuantumCircuit(1, 1)
        circ1.measure(0, 0)

        circ2 = QuantumCircuit(1, 1)

        self.verify_result(checkers, circ1, circ2, False, None, 'equal')
        self.verify_result(checkers, circ2, circ1, False, None, 'equal')

        circ2.measure(0, 0)

        self.verify_result(checkers, circ1, circ2, False, None, 'equal')

    def test_error_in_large_circuits(self):
        """Test error messages for large circuits"""
        checkers = [
            UnitaryEquivalenceChecker('quantum_info', 'unitary_qi')
            ]
        if self.aer_installed:
            checkers.append(UnitaryEquivalenceChecker('aer', 'unitary_aer'))

        circ = QuantumCircuit(40)
        self.verify_result(checkers, circ, circ, False, None, 'equal')


if __name__ == '__main__':
    unittest.main()

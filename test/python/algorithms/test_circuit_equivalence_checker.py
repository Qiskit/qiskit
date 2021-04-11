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
from qiskit.algorithms import equivalence_checker


class TestEquivalenceChecker(QiskitTestCase):
    """Test equivalence checker"""

    def setUp(self) -> None:
        super().setUp()
        try:
            from qiskit import Aer   # pylint: disable=unused-import
            self.aer_installed = True
        except ImportError:
            self.aer_installed = False

    def verify_result(self, methods, names, options,
                      circ1, circ2,
                      success, equivalent):
        """
        Run the checkers, and verify that their outputs
        match the expected result
        """

        for method, name, option in zip(methods, names, options):
            res = equivalence_checker(circ1, circ2, method=method, **option)
            checker_msg = "Checker '" + name + "' failed"
            self.assertEqual(success, res.success, checker_msg)
            self.assertEqual(equivalent, res.equivalent, checker_msg)

    def test_equivalence_checkers_equal_phase(self):
        """Test equivalence chekcers for valid circuits, requiring equal phase"""
        methods = ['unitary']
        names = ['unitary_qi']
        options = [{'simulator': 'quantum_info', 'phase': 'equal'}]

        if self.aer_installed:
            methods.append('unitary')
            names.append('unitary_aer')
            options.append({'simulator': 'aer', 'phase': 'equal'})

        circ1 = QuantumCircuit(2)
        circ1.cx(0, 1)
        circ1.cx(1, 0)
        circ1.cx(0, 1)

        circ2 = QuantumCircuit(2)
        circ2.cx(1, 0)
        circ2.cx(0, 1)
        circ2.cx(1, 0)

        self.verify_result(methods, names, options, circ1, circ2, True, True)

        circ1.x(0)
        self.verify_result(methods, names, options, circ1, circ2, True, False)

    def test_equivalence_checkers_up_to_global_phase(self):
        """Test equivalence chekcers for valid circuits, requiring up-to-global phase"""
        
        methods = ['unitary']
        names = ['unitary_qi']
        options = [{'simulator': 'quantum_info', 'phase': 'equal'}]

        if self.aer_installed:
            methods.append('unitary')
            names.append('unitary_aer')
            options.append({'simulator': 'aer', 'phase': 'equal'})
        
        circ1 = QuantumCircuit(1)
        circ1.x(0)
        circ1.z(0)

        circ2 = QuantumCircuit(1)
        circ2.y(0)

        self.verify_result(methods, names, options, circ1, circ2, True, False)

        for option in options:
            option['phase'] = 'up_to_global'
        self.verify_result(methods, names, options, circ1, circ2, True, True)

        circ1.x(0)
        self.verify_result(methods, names, options, circ1, circ2, True, False)

    def test_error_in_unitary_checker(self):
        """Test error messages for invalid circuits"""

        methods = ['unitary']
        names = ['unitary_qi']
        options = [{'simulator': 'quantum_info', 'phase': 'equal'}]

        if self.aer_installed:
            methods.append('unitary')
            names.append('unitary_aer')
            options.append({'simulator': 'aer', 'phase': 'equal'})
            
        circ1 = QuantumCircuit(1, 1)
        circ1.measure(0, 0)

        circ2 = QuantumCircuit(1, 1)

        self.verify_result(methods, names, options, circ1, circ2, False, None)
        self.verify_result(methods, names, options, circ2, circ1, False, None)

        circ2.measure(0, 0)

        self.verify_result(methods, names, options, circ1, circ2, False, None)

    def test_error_in_large_circuits(self):
        """Test error messages for large circuits"""

        methods = ['unitary']
        names = ['unitary_qi']
        options = [{'simulator': 'quantum_info', 'phase': 'equal'}]

        if self.aer_installed:
            methods.append('unitary')
            names.append('unitary_aer')
            options.append({'simulator': 'aer', 'phase': 'equal'})
        
        circ = QuantumCircuit(40)
        self.verify_result(methods, names, options, circ, circ, False, None)


if __name__ == '__main__':
    unittest.main()

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
from ddt import ddt, data, unpack

from qiskit.test import QiskitTestCase

from qiskit.circuit import QuantumCircuit
from qiskit.algorithms import equivalence_checker


@ddt
class TestEquivalenceChecker(QiskitTestCase):
    """Test equivalence checker"""

    def setUp(self) -> None:
        super().setUp()
        try:
            from qiskit import Aer

            Aer.get_backend("qasm_simulator")
            self.aer_installed = True
        except ImportError:
            self.aer_installed = False

    def verify_result(self, method, options, circ1, circ2, success, equivalent):
        """
        Run the checker, and verify that its output
        matches the expected result
        """

        res = equivalence_checker(circ1, circ2, method=method, **options)
        self.assertEqual(success, res.success)
        self.assertEqual(equivalent, res.equivalent)

    @data(("unitary", {"simulator": "quantum_info"}), ("unitary", {"simulator": "aer"}))
    @unpack
    def test_equivalence_checkers_equal_phase(self, method, options):
        """Test equivalence checkers for valid circuits, requiring equal phase"""

        options["phase"] = "equal"
        if not self.aer_installed and options["simulator"] == "aer":
            return

        circ1 = QuantumCircuit(2)
        circ1.cx(0, 1)
        circ1.cx(1, 0)
        circ1.cx(0, 1)

        circ2 = QuantumCircuit(2)
        circ2.cx(1, 0)
        circ2.cx(0, 1)
        circ2.cx(1, 0)

        self.verify_result(method, options, circ1, circ2, True, True)

        circ1.x(0)
        self.verify_result(method, options, circ1, circ2, True, False)

    @data(("unitary", {"simulator": "quantum_info"}), ("unitary", {"simulator": "aer"}))
    @unpack
    def test_equivalence_checkers_up_to_global_phase(self, method, options):
        """Test equivalence chekcers for valid circuits, requiring up-to-global phase"""

        options["phase"] = "equal"
        if not self.aer_installed and options["simulator"] == "aer":
            return

        circ1 = QuantumCircuit(1)
        circ1.x(0)
        circ1.z(0)

        circ2 = QuantumCircuit(1)
        circ2.y(0)

        self.verify_result(method, options, circ1, circ2, True, False)

        options["phase"] = "up_to_global"
        self.verify_result(method, options, circ1, circ2, True, True)

        circ1.x(0)
        self.verify_result(method, options, circ1, circ2, True, False)

    @data(("unitary", {"simulator": "quantum_info"}), ("unitary", {"simulator": "aer"}))
    @unpack
    def test_error_in_unitary_checker(self, method, options):
        """Test error messages for invalid circuits"""

        options["phase"] = "equal"
        if not self.aer_installed and options["simulator"] == "aer":
            return

        circ1 = QuantumCircuit(1, 1)
        circ1.measure(0, 0)

        circ2 = QuantumCircuit(1, 1)

        self.verify_result(method, options, circ1, circ2, False, None)
        self.verify_result(method, options, circ2, circ1, False, None)

        circ2.measure(0, 0)

        self.verify_result(method, options, circ1, circ2, False, None)

    @data(("unitary", {"simulator": "quantum_info"}), ("unitary", {"simulator": "aer"}))
    @unpack
    def test_error_in_large_circuits(self, method, options):
        """Test error messages for large circuits"""

        options["phase"] = "equal"
        if not self.aer_installed and options["simulator"] == "aer":
            return

        circ = QuantumCircuit(40)
        self.verify_result(method, options, circ, circ, False, None)

    def test_automatic_simulator(self):
        """
        Verify that Aer is auotmatically selected if it is installed
        """

        circ = QuantumCircuit(1)
        res = equivalence_checker(circ, circ, "unitary", simulator="automatic", phase="equal")

        if self.aer_installed:
            self.assertEqual(res.simulator, "aer")
        else:
            self.assertEqual(res.simulator, "quantum_info")


if __name__ == "__main__":
    unittest.main()

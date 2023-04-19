# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the expectation factory."""

import unittest
from test.python.opflow import QiskitOpflowTestCase

from qiskit.opflow import PauliExpectation, AerPauliExpectation, ExpectationFactory, Z, I, X
from qiskit.utils import optionals


class TestExpectationFactory(QiskitOpflowTestCase):
    """Tests for the expectation factory."""

    @unittest.skipUnless(optionals.HAS_AER, "qiskit-aer is required to run this test")
    def test_aer_simulator_pauli_sum(self):
        """Test expectation selection with Aer's qasm_simulator."""
        from qiskit_aer import AerSimulator

        backend = AerSimulator()
        op = 0.2 * (X ^ X) + 0.1 * (Z ^ I)
        with self.assertWarns(DeprecationWarning):
            with self.subTest("Defaults"):
                expectation = ExpectationFactory.build(op, backend, include_custom=False)
                self.assertIsInstance(expectation, PauliExpectation)

            with self.subTest("Include custom"):
                expectation = ExpectationFactory.build(op, backend, include_custom=True)
                self.assertIsInstance(expectation, AerPauliExpectation)

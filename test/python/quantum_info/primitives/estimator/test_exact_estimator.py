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

"""Tests for ExactExpectationValue."""

import unittest

from qiskit.circuit.library import RealAmplitudes
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info.primitives import ExactEstimator
from qiskit.test import QiskitTestCase
from qiskit.utils import has_aer

if has_aer():
    from qiskit import Aer


class TestExactEstimator(QiskitTestCase):
    """Test ExactEstimator"""

    @unittest.skipUnless(has_aer(), "qiskit-aer doesn't appear to be installed.")
    def test_evaluate(self):
        """test for evaluate"""
        observable = PauliSumOp.from_list([("XX", 1), ("YY", 2), ("ZZ", 3)])
        ansatz = RealAmplitudes(num_qubits=2, reps=2)
        with ExactEstimator(
            [ansatz], [observable], backend=Aer.get_backend("aer_simulator")
        ) as est:
            result = est([0, 1, 1, 2, 3, 5])
        self.assertIsInstance(result.values[0], float)
        self.assertAlmostEqual(result.values[0], 1.84209213)
        self.assertIsInstance(result.variances[0], float)
        self.assertAlmostEqual(result.variances[0], 6.43276352)

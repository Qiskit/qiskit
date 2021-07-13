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
""" Test QDrift. """
import unittest

from qiskit.algorithms.quantum_time_evolution.builders.implementations.trotterizations.qdrift import (
    QDrift,
)
from test.python.opflow import QiskitOpflowTestCase
from qiskit.opflow import (
    CircuitOp,
    EvolvedOp,
    I,
    SummedOp,
    X,
    Y,
    Z,
)


class TestQDrift(QiskitOpflowTestCase):
    """QDrift tests."""

    def test_qdrift(self):
        """QDrift test."""
        op = (2 * Z ^ Z) + (3 * X ^ X) - (4 * Y ^ Y) + (0.5 * Z ^ I)
        trotterization = QDrift().build(op)
        self.assertGreater(len(trotterization.oplist), 150)
        last_coeff = None
        # Check that all types are correct and all coefficients are equals
        for op in trotterization.oplist:
            self.assertIsInstance(op, (EvolvedOp, CircuitOp))
            if isinstance(op, EvolvedOp):
                if last_coeff:
                    self.assertEqual(op.primitive.coeff, last_coeff)
                else:
                    last_coeff = op.primitive.coeff

    def test_qdrift_summed_op(self):
        """QDrift test for SummedOp."""
        op = SummedOp(
            [
                (2 * Z ^ Z),
                (3 * X ^ X),
                (-4 * Y ^ Y),
                (0.5 * Z ^ I),
            ]
        )
        trotterization = QDrift().build(op)
        self.assertGreater(len(trotterization.oplist), 150)
        last_coeff = None
        # Check that all types are correct and all coefficients are equals
        for op in trotterization.oplist:
            self.assertIsInstance(op, (EvolvedOp, CircuitOp))
            if isinstance(op, EvolvedOp):
                if last_coeff:
                    self.assertEqual(op.primitive.coeff, last_coeff)
                else:
                    last_coeff = op.primitive.coeff


if __name__ == "__main__":
    unittest.main()

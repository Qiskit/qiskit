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
""" Test Suzuki. """
import unittest

import numpy as np

from qiskit.algorithms.quantum_time_evolution.builders.implementations.trotterizations.suzuki \
    import \
    Suzuki
from test.python.opflow import QiskitOpflowTestCase
from qiskit.opflow import (
    X,
    Z,
)


class TestSuzuki(QiskitOpflowTestCase):
    """Suzuki tests."""
    def test_suzuki_directly(self):
        """Test for Suzuki converter"""
        operator = X + Z

        evo = Suzuki()
        evolution = evo.build(operator)

        matrix = np.array(
            [[0.29192658 - 0.45464871j, -0.84147098j], [-0.84147098j, 0.29192658 + 0.45464871j]]
        )
        np.testing.assert_array_almost_equal(evolution.to_matrix(), matrix)


if __name__ == "__main__":
    unittest.main()
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

"""Tests QRTE operators validation methods."""

import unittest

from ddt import ddt, data, unpack
import numpy as np

from qiskit.algorithms.quantum_time_evolution.real.implementations.trotterization.trotter_ops_validator import (
    _validate_hamiltonian_form,
)
from test.python.opflow import QiskitOpflowTestCase
from qiskit.circuit import Parameter
from qiskit.opflow import (
    X,
    Z,
    Y,
)


@ddt
class TestTrotterQrte(QiskitOpflowTestCase):
    """Trotter QRTE operators validation tests."""

    @data(
        (X, True),
        (Parameter("theta") * Y, True),
        (Parameter("theta") * Parameter("gamma") * Z, False),
        (Parameter("theta") * X + Parameter("gamma") * Y, True),
        (Parameter("theta1") * Parameter("theta2") * X + Parameter("gamma") * Y, False),
    )
    @unpack
    def test_validate_hamiltonian_form(self, hamiltonian, expected):
        valid = True
        try:
            _validate_hamiltonian_form(hamiltonian)
        except ValueError:
            valid = False

        np.testing.assert_equal(valid, expected)


if __name__ == "__main__":
    unittest.main()

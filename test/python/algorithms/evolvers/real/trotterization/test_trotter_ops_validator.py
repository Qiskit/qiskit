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

from test.python.opflow import QiskitOpflowTestCase
from ddt import ddt, data, unpack
import numpy as np

from qiskit.algorithms.evolvers.real.trotterization.trotter_ops_validator import (
    validate_hamiltonian_form,
    _is_pauli_lin_single_param,
)
from qiskit.circuit.library import EfficientSU2
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
        (Parameter("theta") * X + Y, True),
        (X + Parameter("gamma") * Y, True),
        (Parameter("theta1") * Parameter("theta2") * X + Parameter("gamma") * Y, False),
        (EfficientSU2, False),
    )
    @unpack
    def test_validate_hamiltonian_form(self, hamiltonian, expected):
        """Tests that a Hamiltonian is of a valid form supported by the TrotterQrte algorithm."""
        valid = True
        try:
            validate_hamiltonian_form(hamiltonian)
        except ValueError:
            valid = False

        np.testing.assert_equal(valid, expected)

    @data(
        (X, True),
        (X + Y, False),
        (-5 * X, True),
        (5j * Y, True),
        (X + Parameter("theta1") * Parameter("theta2") * Y, False),
        (Parameter("theta") * Y, True),
        (Parameter("theta") * Parameter("gamma") * Z, False),
        (5 * Parameter("theta") * X, True),
        (Parameter("theta1") * Parameter("theta2") * X, False),
    )
    @unpack
    def test_is_pauli_lin_single_param(self, operator, expected):
        """Tests that operator is validated to be linearly dependent on a single parameter."""
        try:
            linear_with_single_param = _is_pauli_lin_single_param(operator)
        except ValueError:
            linear_with_single_param = False

        np.testing.assert_equal(linear_with_single_param, expected)


if __name__ == "__main__":
    unittest.main()

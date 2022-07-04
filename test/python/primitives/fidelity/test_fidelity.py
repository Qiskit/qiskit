# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for Sampler."""

import unittest
from test import combine

import numpy as np
from functools import partial
from ddt import ddt

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import Sampler
from qiskit.primitives.fidelity import Fidelity
from qiskit.test import QiskitTestCase


@ddt
class TestFidelity(QiskitTestCase):
    """Test Fidelity"""

    def setUp(self):
        super().setUp()
        parameters = ParameterVector("x", 2)

        rx_rotations = QuantumCircuit(2)
        rx_rotations.rx(parameters[0], 0)
        rx_rotations.rx(parameters[1], 1)

        ry_rotations = QuantumCircuit(2)
        ry_rotations.ry(parameters[0], 0)
        ry_rotations.ry(parameters[1], 1)
        self._circuit = [rx_rotations, ry_rotations]
        self._sampler_factory = partial(Sampler)
        self._params_left = np.array([[0, 0], [np.pi / 2, 0], [0, np.pi / 2], [np.pi, np.pi]])
        self._params_right = np.array([[0, 0], [0, 0], [np.pi / 2, 0], [0, 0]])

    def test_fidelity_1param_pair(self):
        """test for fidelity with one pai"""

        with Fidelity(self._circuit[0], self._circuit[1], self._sampler_factory) as fidelity:
            results = fidelity.compute(self._params_left[0], self._params_right[0])
            np.testing.assert_allclose(results, np.array([1.0]))

    def test_fidelity_4param_pairs(self):
        """test for fidelity with one pai"""

        with Fidelity(self._circuit[0], self._circuit[1], self._sampler_factory) as fidelity:
            results = fidelity.compute(self._params_left, self._params_right)
            np.testing.assert_allclose(
                results,
                np.array([1.0, 0.5, 0.25, 0.0]),
            )


if __name__ == "__main__":
    unittest.main()

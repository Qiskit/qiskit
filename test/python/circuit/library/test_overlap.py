# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test unitary overlap function"""
import unittest
import numpy as np

from qiskit.test.base import QiskitTestCase
from qiskit.circuit.library import EfficientSU2, UnitaryOverlap
from qiskit.quantum_info import Statevector
from qiskit.circuit.exceptions import CircuitError


class TestUnitaryOverlap(QiskitTestCase):
    """Test the unitary overlap circuit class."""

    def test_identity(self):
        """Test identity is returned"""
        U = EfficientSU2(2)
        U.assign_parameters(np.random.random(size=U.num_parameters), inplace=True)

        overlap = UnitaryOverlap(U, U)
        self.assertTrue(abs(Statevector.from_instruction(overlap)[0] - 1) < 1e-15)

    def test_parameterized_identity(self):
        """Test identity is returned"""
        U = EfficientSU2(2)

        overlap = UnitaryOverlap(U, U)
        rands = np.random.random(size=U.num_parameters)
        double_rands = np.hstack((rands, rands))
        overlap.assign_parameters(double_rands, inplace=True)
        self.assertTrue(abs(Statevector.from_instruction(overlap)[0] - 1) < 1e-15)

    def test_two_parameterized_inputs(self):
        """Test two parameterized inputs"""
        U = EfficientSU2(2)
        V = EfficientSU2(2)

        overlap = UnitaryOverlap(U, V)
        self.assertEqual(overlap.num_parameters, U.num_parameters + V.num_parameters)

    def test_partial_parameterized_inputs(self):
        """Test one parameterized inputs (1)"""
        U = EfficientSU2(2)
        U.assign_parameters(np.random.random(size=U.num_parameters), inplace=True)

        V = EfficientSU2(2, reps=5)

        overlap = UnitaryOverlap(U, V)
        self.assertEqual(overlap.num_parameters, V.num_parameters)

    def test_partial_parameterized_inputs2(self):
        """Test one parameterized inputs (2)"""
        U = EfficientSU2(2)
        V = EfficientSU2(2, reps=5)
        V.assign_parameters(np.random.random(size=V.num_parameters), inplace=True)

        overlap = UnitaryOverlap(U, V)
        self.assertEqual(overlap.num_parameters, U.num_parameters)

    def test_measurements(self):
        """Test that exception is thrown for measurements"""
        U = EfficientSU2(2)
        U.measure_all()
        V = EfficientSU2(2)

        with self.assertRaises(CircuitError):
            _ = UnitaryOverlap(U, V)


if __name__ == "__main__":
    unittest.main()

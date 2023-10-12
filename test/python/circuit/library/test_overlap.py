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
from qiskit.circuit import QuantumCircuit, Qubit, Clbit
from qiskit.circuit.library import EfficientSU2, UnitaryOverlap
from qiskit.quantum_info import Statevector
from qiskit.circuit.exceptions import CircuitError


class TestUnitaryOverlap(QiskitTestCase):
    """Test the unitary overlap circuit class."""

    def test_identity(self):
        """Test identity is returned"""
        unitary = EfficientSU2(2)
        unitary.assign_parameters(np.random.random(size=unitary.num_parameters), inplace=True)

        overlap = UnitaryOverlap(unitary, unitary)
        self.assertTrue(abs(Statevector.from_instruction(overlap)[0] - 1) < 1e-15)

    def test_parameterized_identity(self):
        """Test identity is returned"""
        unitary = EfficientSU2(2)

        overlap = UnitaryOverlap(unitary, unitary)
        rands = np.random.random(size=unitary.num_parameters)
        double_rands = np.hstack((rands, rands))
        overlap.assign_parameters(double_rands, inplace=True)
        self.assertTrue(abs(Statevector.from_instruction(overlap)[0] - 1) < 1e-15)

    def test_two_parameterized_inputs(self):
        """Test two parameterized inputs"""
        unitary1 = EfficientSU2(2)
        unitary2 = EfficientSU2(2)

        overlap = UnitaryOverlap(unitary1, unitary2)
        self.assertEqual(overlap.num_parameters, unitary1.num_parameters + unitary2.num_parameters)

    def test_parameter_prefixes(self):
        """Test two parameterized inputs"""
        unitary1 = EfficientSU2(2)
        unitary2 = EfficientSU2(2)

        overlap = UnitaryOverlap(unitary1, unitary2, prefix1="a", prefix2="b")
        self.assertEqual(overlap.num_parameters, unitary1.num_parameters + unitary2.num_parameters)

        expected_names = [f"a[{i}]" for i in range(unitary1.num_parameters)]
        expected_names += [f"b[{i}]" for i in range(unitary2.num_parameters)]

        self.assertListEqual([p.name for p in overlap.parameters], expected_names)

    def test_partial_parameterized_inputs(self):
        """Test one parameterized inputs (1)"""
        unitary1 = EfficientSU2(2)
        unitary1.assign_parameters(np.random.random(size=unitary1.num_parameters), inplace=True)

        unitary2 = EfficientSU2(2, reps=5)

        overlap = UnitaryOverlap(unitary1, unitary2)
        self.assertEqual(overlap.num_parameters, unitary2.num_parameters)

    def test_partial_parameterized_inputs2(self):
        """Test one parameterized inputs (2)"""
        unitary1 = EfficientSU2(2)
        unitary2 = EfficientSU2(2, reps=5)
        unitary2.assign_parameters(np.random.random(size=unitary2.num_parameters), inplace=True)

        overlap = UnitaryOverlap(unitary1, unitary2)
        self.assertEqual(overlap.num_parameters, unitary1.num_parameters)

    def test_measurements(self):
        """Test that exception is thrown for measurements"""
        unitary1 = EfficientSU2(2)
        unitary1.measure_all()
        unitary2 = EfficientSU2(2)

        with self.assertRaises(CircuitError):
            _ = UnitaryOverlap(unitary1, unitary2)

    def test_rest(self):
        """Test that exception is thrown for rest"""
        unitary1 = EfficientSU2(1, reps=0)
        unitary1.reset(0)
        unitary2 = EfficientSU2(1, reps=1)

        with self.assertRaises(CircuitError):
            _ = UnitaryOverlap(unitary1, unitary2)

    def test_controlflow(self):
        """Test that exception is thrown for controlflow"""
        bit = Clbit()
        unitary1 = QuantumCircuit([Qubit(), bit])
        unitary1.h(0)
        with unitary1.if_test((bit, 0)):
            unitary1.x(0)

        unitary2 = QuantumCircuit(1)
        unitary2.rx(0.2, 0)

        with self.assertRaises(CircuitError):
            _ = UnitaryOverlap(unitary1, unitary2)

    def test_mismatching_qubits(self):
        """Test that exception is thrown for mismatching circuit"""
        unitary1 = EfficientSU2(2)
        unitary2 = EfficientSU2(1)

        with self.assertRaises(CircuitError):
            _ = UnitaryOverlap(unitary1, unitary2)


if __name__ == "__main__":
    unittest.main()

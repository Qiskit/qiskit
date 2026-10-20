# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for PhasedUGate."""

import io
import unittest

import numpy as np
from ddt import ddt, data

from qiskit import QuantumCircuit, qpy
from qiskit.circuit import ControlledGate, Parameter
from qiskit.circuit.annotated_operation import AnnotatedOperation
from qiskit.circuit.library import CUGate, PhasedUGate, UGate
from qiskit.quantum_info import Operator
from test import QiskitTestCase


@ddt
class TestPhasedUGate(QiskitTestCase):
    """Tests for PhasedUGate."""

    def test_matrix_is_phased_u(self):
        """Test that the matrix of PhasedUGate is e^{i gamma} times the matrix of UGate."""
        theta, phi, lam, gamma = 0.3, 0.4, 0.5, 0.6
        gate = PhasedUGate(theta, phi, lam, gamma)
        expected = np.exp(1j * gamma) * UGate(theta, phi, lam).to_matrix()
        np.testing.assert_allclose(gate.to_matrix(), expected, atol=1e-12)
        np.testing.assert_allclose(Operator(gate.definition).data, expected, atol=1e-12)

    def test_zero_phase_matches_u(self):
        """Test that PhasedUGate with zero phase has the same matrix as UGate."""
        np.testing.assert_allclose(
            PhasedUGate(0.3, 0.4, 0.5, 0).to_matrix(), UGate(0.3, 0.4, 0.5).to_matrix()
        )

    def test_inverse(self):
        """Test that the inverse is a PhasedUGate and is the exact adjoint, including the phase."""
        gate = PhasedUGate(0.3, 0.4, 0.5, 0.6)
        inverse = gate.inverse()
        self.assertIsInstance(inverse, PhasedUGate)
        self.assertEqual(inverse.params, [-0.3, -0.5, -0.4, -0.6])
        self.assertEqual(Operator(inverse), Operator(gate).adjoint())

    def test_equality(self):
        """Test that PhasedUGate compares by parameters and is not equal to UGate."""
        self.assertEqual(PhasedUGate(0.3, 0.4, 0.5, 0.6), PhasedUGate(0.3, 0.4, 0.5, 0.6))
        self.assertNotEqual(PhasedUGate(0.3, 0.4, 0.5, 0.6), PhasedUGate(0.3, 0.4, 0.5, 0.7))
        self.assertNotEqual(PhasedUGate(0.3, 0.4, 0.5, 0), UGate(0.3, 0.4, 0.5))

    @data(None, False, True)
    def test_control_one_is_cu(self, annotated):
        """Test that a single control yields a CUGate with the same parameters, for all values
        of ``annotated``."""
        gate = PhasedUGate(0.3, 0.4, 0.5, 0.6, label="base")
        controlled = gate.control(1, label="controlled", ctrl_state=0, annotated=annotated)
        self.assertIsInstance(controlled, CUGate)
        self.assertEqual(controlled.params, [0.3, 0.4, 0.5, 0.6])
        self.assertEqual(controlled.ctrl_state, 0)
        self.assertEqual(controlled.label, "controlled")
        self.assertEqual(controlled.base_gate.label, "base")
        self.assertEqual(controlled.base_gate, PhasedUGate(0.3, 0.4, 0.5, 0.6))

    @data(None, False)
    def test_control_many_not_annotated(self, annotated):
        """Test that multiple controls yield a ControlledGate when ``annotated`` is not ``True``,
        also for parameterized gates."""
        theta = Parameter("theta")
        for gate in (PhasedUGate(0.3, 0.4, 0.5, 0.6), PhasedUGate(theta, 0.4, 0.5, theta)):
            with self.subTest(gate=gate):
                controlled = gate.control(2, annotated=annotated)
                self.assertIsInstance(controlled, ControlledGate)
                self.assertNotIsInstance(controlled, AnnotatedOperation)

    def test_control_many_annotated(self):
        """Test that multiple controls with ``annotated=True`` yield an equivalent
        AnnotatedOperation."""
        gate = PhasedUGate(0.3, 0.4, 0.5, 0.6)
        annotated = gate.control(2, annotated=True)
        self.assertIsInstance(annotated, AnnotatedOperation)
        self.assertEqual(Operator(annotated), Operator(gate.control(2, annotated=False)))

    def test_assign_parameters(self):
        """Test binding a parameter that appears in both the rotation and the phase."""
        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.append(PhasedUGate(theta, 0.4, 0.5, theta), [0])
        bound = qc.assign_parameters({theta: 0.3})

        expected = QuantumCircuit(1)
        expected.append(PhasedUGate(0.3, 0.4, 0.5, 0.3), [0])
        self.assertEqual(bound, expected)
        self.assertEqual(Operator(bound), Operator(expected))

    def test_qpy_roundtrip(self):
        """Test that PhasedUGate survives a QPY roundtrip, including its label."""
        qc = QuantumCircuit(1)
        qc.append(PhasedUGate(0.3, 0.4, 0.5, 0.6), [0])
        qc.append(PhasedUGate(0.1, 0.2, 0.3, 0.4, label="my_u4"), [0])
        with io.BytesIO() as buf:
            qpy.dump(qc, buf)
            buf.seek(0)
            new_qc = qpy.load(buf)[0]
        self.assertEqual(new_qc, qc)
        self.assertIsInstance(new_qc.data[0].operation, PhasedUGate)
        self.assertEqual(new_qc.data[1].operation.label, "my_u4")


if __name__ == "__main__":
    unittest.main()

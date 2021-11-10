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

# pylint: disable=missing-class-docstring,missing-function-docstring
# pylint: disable=missing-module-docstring

import math
import statistics

from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.test.base import QiskitTestCase
from qiskit.test.mock.fake_backend_v2 import FakeBackendV2


class TestBackendV2(QiskitTestCase):
    def setUp(self):
        super().setUp()
        self.backend = FakeBackendV2()

    def test_qubit_properties(self):
        """Test that qubit properties are returned as expected."""
        props = self.backend.qubits([1, 0])
        self.assertEqual([73.09352e-6, 63.48783e-6], [x.t1 for x in props])
        self.assertEqual([126.83382e-6, 112.23246e-6], [x.t2 for x in props])
        self.assertEqual([5.26722e9, 5.17538e9], [x.frequency for x in props])

    def test_t1(self):
        """Test that the t1 helper method works as expected."""
        t1 = self.backend.t1([0, 1])
        expected_mean = statistics.mean([73.09352e-6, 63.48783e-6])
        self.assertAlmostEqual(expected_mean, t1.mean())

    def test_t1_single_not_defined(self):
        """Test that t1 correctly errors for a qubit doesn't have t1 defined."""
        self.backend.qubit_properties[0].t1 = None
        with self.assertRaises(ValueError):
            self.backend.t1(0)

    def test_t1_list_not_definedinvalid_qubit(self):
        """Test that t1 correctly errors if an qubit which doesn't have t1
        defined is in the input list."""
        self.backend.qubit_properties[1].t1 = None
        with self.assertRaises(ValueError):
            self.backend.t1([0, 1])

    def test_t2(self):
        """Test that the t2 helper method works as expected."""
        t2 = self.backend.t2([0, 1])
        expected_mean = statistics.mean([126.83382e-6, 112.23246e-6])
        self.assertAlmostEqual(expected_mean, t2.mean())

    def test_t2_single_not_defined(self):
        """Test that t2 correctly errors for a qubit which doesn't have t2 defined."""
        self.backend.qubit_properties[1].t2 = None
        with self.assertRaises(ValueError):
            self.backend.t2(1)

    def test_t2_list_invalid_qubit(self):
        """Test that t2 correctly errors if a qubit which doesn't have t2
        defined is in the input list."""
        self.backend.qubit_properties[1].t2 = None
        with self.assertRaises(ValueError):
            self.backend.t2([0, 1])

    def test_option_bounds(self):
        """Test that option bounds are enforced."""
        with self.assertRaises(ValueError) as cm:
            self.backend.set_options(shots=8192)
        self.assertEqual(
            str(cm.exception),
            "Specified value for 'shots' is not a valid value, must be >=1 or <=4096",
        )

    def test_transpile(self):
        """Test that transpile() works with a BackendV2 backend."""
        qc = QuantumCircuit(2)
        qc.h(1)
        qc.cz(1, 0)
        qc.measure_all()
        with self.assertLogs("qiskit.providers.backend", level="WARN") as log:
            tqc = transpile(qc, self.backend)
        self.assertEqual(
            log.output,
            [
                "WARNING:qiskit.providers.backend:This backend's operations: "
                "ecr only apply to a subset of qubits. Using this property to "
                "get 'basis_gates' for the transpiler may potentially create "
                "invalid output"
            ],
        )
        expected = QuantumCircuit(2)
        expected.u(math.pi / 2, 0, -math.pi, 0)
        expected.u(math.pi / 2, 0, -math.pi, 1)
        expected.cx(1, 0)
        expected.u(math.pi / 2, 0, -math.pi, 0)
        expected.measure_all()
        self.assertEqual(tqc, expected)

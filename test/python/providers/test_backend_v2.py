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

from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.test.base import QiskitTestCase
from qiskit.test.mock.fake_backend_v2 import FakeBackendV2


class TestBackendV2(QiskitTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.backend = FakeBackendV2()

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
                "WARNING:qiskit.providers.backend:This backend's instructions: "
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

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

"""Test of qasm fake backends from qiskit.mock package."""
import unittest
from qiskit import QuantumCircuit, transpile
from qiskit.test import QiskitTestCase
from qiskit.providers.fake_provider import FakeBogota
from qiskit.utils import optionals as _optionals


class FakeQasmBackendsTest(QiskitTestCase):
    """Tests for FakeQasmBackend"""

    @unittest.skipUnless(_optionals.HAS_AER, "qiskit-aer is required to run this test")
    def test_fake_qasm_backend_configured(self):
        """Fake backends honor kwargs passed."""
        backend = FakeBogota()  # this is a FakePulseBackend implementation

        qc = QuantumCircuit(2)
        qc.x(range(0, 2))
        qc.measure_all()

        trans_qc = transpile(qc, backend)
        raw_counts = backend.run(trans_qc, shots=1000).result().get_counts()

        self.assertEqual(sum(raw_counts.values()), 1000)

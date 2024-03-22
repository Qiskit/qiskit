# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test of generated fake backends."""
import math
import unittest

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, transpile
from qiskit.providers.fake_provider import Fake5QV1, GenericBackendV2
from qiskit.utils import optionals
from test import QiskitTestCase  # pylint: disable=wrong-import-order


def get_test_circuit():
    """Generates simple circuit for tests."""
    desired_vector = [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)]
    qr = QuantumRegister(2, "qr")
    cr = ClassicalRegister(2, "cr")
    qc = QuantumCircuit(qr, cr)
    qc.initialize(desired_vector, [qr[0], qr[1]])
    qc.measure(qr[0], cr[0])
    qc.measure(qr[1], cr[1])
    return qc


class FakeBackendsTest(QiskitTestCase):
    """fake backends test."""

    @unittest.skipUnless(optionals.HAS_AER, "qiskit-aer is required to run this test")
    def test_fake_backends_get_kwargs(self):
        """Fake backends honor kwargs passed."""
        backend = Fake5QV1()

        qc = QuantumCircuit(2)
        qc.x(range(0, 2))
        qc.measure_all()

        trans_qc = transpile(qc, backend)
        raw_counts = backend.run(trans_qc, shots=1000).result().get_counts()

        self.assertEqual(sum(raw_counts.values()), 1000)

    @unittest.skipUnless(optionals.HAS_AER, "qiskit-aer is required to run this test")
    def test_fake_backend_v2_noise_model_always_present(self):
        """Test that FakeBackendV2 instances always run with noise."""
        backend = GenericBackendV2(num_qubits=5, seed=42)
        backend.set_options(seed_simulator=42)
        qc = QuantumCircuit(1)
        qc.x(0)
        qc.measure_all()
        res = backend.run(qc, shots=1000).result().get_counts()
        # Assert noise was present and result wasn't ideal
        self.assertNotEqual(res, {"1": 1000})

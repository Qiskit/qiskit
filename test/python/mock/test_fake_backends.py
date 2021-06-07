# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
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

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, schedule, transpile, assemble
from qiskit.pulse import Schedule
from qiskit.qobj import PulseQobj
from qiskit.test import QiskitTestCase
from qiskit.test.mock.utils import ConfigurableFakeBackend

from qiskit.test.mock.fake_backend import HAS_AER


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


class GeneratedFakeBackendsTest(QiskitTestCase):
    """Generated fake backends test."""

    def setUp(self) -> None:
        self.backend = ConfigurableFakeBackend("Tashkent", n_qubits=4)

    @unittest.skip("Skipped until qiskit-aer#741 is fixed and released")
    @unittest.skipUnless(HAS_AER, "qiskit-aer is required to run this test")
    def test_transpile_schedule_and_assemble(self):
        """Test transpile, schedule and assemble on generated backend."""
        qc = get_test_circuit()

        circuit = transpile(qc, backend=self.backend)
        self.assertTrue(isinstance(circuit, QuantumCircuit))
        self.assertEqual(circuit.num_qubits, 4)

        experiments = schedule(circuits=circuit, backend=self.backend)
        self.assertTrue(isinstance(experiments, Schedule))
        self.assertGreater(experiments.duration, 0)

        qobj = assemble(experiments, backend=self.backend)
        self.assertTrue(isinstance(qobj, PulseQobj))
        self.assertEqual(qobj.header.backend_name, "Tashkent")
        self.assertEqual(len(qobj.experiments), 1)

        job = self.backend.run(qobj)
        result = job.result()
        self.assertTrue(result.success)
        self.assertEqual(len(result.results), 1)

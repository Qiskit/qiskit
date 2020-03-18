# -*- coding: utf-8 -*-

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

from qiskit import QuantumRegister, QuantumCircuit, execute, schedule, transpile, assemble, BasicAer
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeOpenPulse2Q
from qiskit.test.mock.utils.fake_backend_builder import FakeBackendBuilder


class GeneratedFakeBackendsTest(QiskitTestCase):
    """Generated fake backends test."""

    def setUp(self) -> None:
        self.backend = FakeBackendBuilder("Tashkent", n_qubits=4).build()

    def test_not_even_came_up_with_name_yet(self):
        pass
        # desired_vector = [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)]
        # qr = QuantumRegister(2, "qr")
        # qc = QuantumCircuit(qr)
        # qc.initialize(desired_vector, [qr[0], qr[1]])
        # # job = execute(qc, self.backend)
        # # result = job.result()
        # # print(result)
        #
        # experiments = transpile(qc, backend=self.backend)
        # experiments = schedule(circuits=experiments, backend=self.backend)
        # qobj = assemble(experiments, backend=self.backend)
        #
        # job = self.backend.run(qobj)
        #
        # result = job.result()
        #
        # print(qobj)

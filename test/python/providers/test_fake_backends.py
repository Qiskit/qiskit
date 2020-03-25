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

# pylint: disable=missing-class-docstring,missing-function-docstring
# pylint: disable=missing-module-docstring

import operator

from test import combine
from ddt import ddt

from qiskit.circuit import QuantumCircuit
from qiskit.execute import execute
from qiskit.test.base import QiskitTestCase
from qiskit.test.mock import backends
from qiskit.test.mock.fake_backend import HAS_AER


FAKE_BACKENDS = [
    x for x in dir(backends) if x.startswith('Fake')]


@ddt
class TestFakeBackends(QiskitTestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.circuit = QuantumCircuit(2)
        cls.circuit.h(0)
        cls.circuit.h(1)
        cls.circuit.h(0)
        cls.circuit.h(1)
        cls.circuit.x(0)
        cls.circuit.x(1)
        cls.circuit.measure_all()

    @combine(fake_backends=FAKE_BACKENDS,
             optimization_level=[0, 1, 2, 3])
    def test_circuit_on_fake_backend(self, fake_backends, optimization_level):
        backend = getattr(backends, fake_backends)()
        if not HAS_AER and backend.configuration().n_qubits > 20:
            self.skipTest(
                'Unable to run fake_backend %s without qiskit-aer' %
                backend.configuration().backend_name)
        job = execute(self.circuit, backend,
                      optimization_level=optimization_level,
                      seed_simulator=42, seed_transpiler=42)
        result = job.result()
        counts = result.get_counts()
        max_count = max(counts.items(), key=operator.itemgetter(1))[0]
        self.assertEqual(max_count, '11')

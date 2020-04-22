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
from ddt import ddt, data

from qiskit.circuit import QuantumCircuit
from qiskit.execute import execute
from qiskit.test.base import QiskitTestCase
from qiskit.test.mock import FakeProvider
from qiskit.test.mock.fake_backend import HAS_AER


FAKE_PROVIDER = FakeProvider()


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

    @combine(backend=[be for be in FAKE_PROVIDER.backends()
                      if be.configuration().num_qubits > 1],
             optimization_level=[0, 1, 2, 3])
    def test_circuit_on_fake_backend(self, backend, optimization_level):
        if not HAS_AER and backend.configuration().num_qubits > 20:
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

    @data(*FAKE_PROVIDER.backends())
    def test_to_dict_properties(self, backend):
        properties = backend.properties()
        if properties:
            self.assertIsInstance(backend.properties().to_dict(), dict)
        else:
            self.assertTrue(backend.configuration().simulator)

    @data(*FAKE_PROVIDER.backends())
    def test_to_dict_configuration(self, backend):
        configuration = backend.configuration()
        self.assertIsInstance(configuration.to_dict(), dict)

    @data(*FAKE_PROVIDER.backends())
    def test_defaults_to_dict(self, backend):
        if hasattr(backend, 'defaults'):
            self.assertIsInstance(backend.defaults().to_dict(), dict)
        else:
            self.skipTest('Backend %s does not have defaults' % backend)

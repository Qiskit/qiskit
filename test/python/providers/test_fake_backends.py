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
from qiskit.compiler import assemble
from qiskit.compiler import transpile
from qiskit.exceptions import QiskitError
from qiskit.execute_function import execute
from qiskit.test.base import QiskitTestCase
from qiskit.providers.fake_provider import (
    FakeProviderForBackendV2,
    FakeProvider,
    FakeMumbaiV2,
    FakeYorktown,
    FakeMumbai,
)
from qiskit.providers.backend_compat import BackendV2Converter
from qiskit.providers.backend import BackendV2
from qiskit.utils import optionals

FAKE_PROVIDER_FOR_BACKEND_V2 = FakeProviderForBackendV2()
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

    @combine(
        backend=[be for be in FAKE_PROVIDER_FOR_BACKEND_V2.backends() if be.num_qubits > 1],
        optimization_level=[0, 1, 2, 3],
    )
    def test_circuit_on_fake_backend_v2(self, backend, optimization_level):
        if not optionals.HAS_AER and backend.num_qubits > 20:
            self.skipTest("Unable to run fake_backend %s without qiskit-aer" % backend.backend_name)
        job = execute(
            self.circuit,
            backend,
            optimization_level=optimization_level,
            seed_simulator=42,
            seed_transpiler=42,
        )
        result = job.result()
        counts = result.get_counts()
        max_count = max(counts.items(), key=operator.itemgetter(1))[0]
        self.assertEqual(max_count, "11")

    @combine(
        backend=[be for be in FAKE_PROVIDER.backends() if be.configuration().num_qubits > 1],
        optimization_level=[0, 1, 2, 3],
        dsc="Test execution path on {backend} with optimization level {optimization_level}",
        name="{backend}_opt_level_{optimization_level}",
    )
    def test_circuit_on_fake_backend(self, backend, optimization_level):
        if not optionals.HAS_AER and backend.configuration().num_qubits > 20:
            self.skipTest(
                "Unable to run fake_backend %s without qiskit-aer"
                % backend.configuration().backend_name
            )
        job = execute(
            self.circuit,
            backend,
            optimization_level=optimization_level,
            seed_simulator=42,
            seed_transpiler=42,
        )
        result = job.result()
        counts = result.get_counts()
        max_count = max(counts.items(), key=operator.itemgetter(1))[0]
        self.assertEqual(max_count, "11")

    def test_qobj_failure(self):
        backend = FAKE_PROVIDER.backends()[-1]
        tqc = transpile(self.circuit, backend)
        qobj = assemble(tqc, backend)
        with self.assertRaises(QiskitError):
            backend.run(qobj)

    @data(*FAKE_PROVIDER.backends())
    def test_to_dict_properties(self, backend):
        properties = backend.properties()
        if properties:
            self.assertIsInstance(backend.properties().to_dict(), dict)
        else:
            self.assertTrue(backend.configuration().simulator)

    @data(*FAKE_PROVIDER_FOR_BACKEND_V2.backends())
    def test_convert_to_target(self, backend):
        target = backend.target
        if target.dt is not None:
            self.assertLess(target.dt, 1e-6)

    @data(*FAKE_PROVIDER_FOR_BACKEND_V2.backends())
    def test_backend_v2_dtm(self, backend):
        if backend.dtm:
            self.assertLess(backend.dtm, 1e-6)

    @data(*FAKE_PROVIDER.backends())
    def test_to_dict_configuration(self, backend):
        configuration = backend.configuration()
        if configuration.open_pulse:
            self.assertLess(configuration.dt, 1e-6)
            self.assertLess(configuration.dtm, 1e-6)
            for i in configuration.qubit_lo_range:
                self.assertGreater(i[0], 1e6)
                self.assertGreater(i[1], 1e6)
                self.assertLess(i[0], i[1])

            for i in configuration.meas_lo_range:
                self.assertGreater(i[0], 1e6)
                self.assertGreater(i[0], 1e6)
                self.assertLess(i[0], i[1])

            for i in configuration.rep_times:
                self.assertGreater(i, 0)
                self.assertLess(i, 1)

        self.assertIsInstance(configuration.to_dict(), dict)

    @data(*FAKE_PROVIDER.backends())
    def test_defaults_to_dict(self, backend):
        if hasattr(backend, "defaults"):
            defaults = backend.defaults()
            self.assertIsInstance(defaults.to_dict(), dict)

            for i in defaults.qubit_freq_est:
                self.assertGreater(i, 1e6)
                self.assertGreater(i, 1e6)

            for i in defaults.meas_freq_est:
                self.assertGreater(i, 1e6)
                self.assertGreater(i, 1e6)
        else:
            self.skipTest("Backend %s does not have defaults" % backend)

    def test_delay_circuit(self):
        backend = FakeMumbaiV2()
        qc = QuantumCircuit(2)
        qc.delay(502, 0, unit="ns")
        qc.x(1)
        qc.delay(250, 1, unit="ns")
        qc.measure_all()
        res = transpile(qc, backend)
        self.assertIn("delay", res.count_ops())

    @data(0, 1, 2, 3)
    def test_converter(self, opt_level):
        backend = FakeYorktown()
        backend_v2 = BackendV2Converter(backend)
        self.assertIsInstance(backend_v2, BackendV2)
        res = transpile(self.circuit, backend_v2, optimization_level=opt_level)
        job = backend_v2.run(res)
        result = job.result()
        counts = result.get_counts()
        max_count = max(counts.items(), key=operator.itemgetter(1))[0]
        self.assertEqual(max_count, "11")

    def test_converter_delay_circuit(self):
        backend = FakeMumbai()
        backend_v2 = BackendV2Converter(backend, add_delay=True)
        self.assertIsInstance(backend_v2, BackendV2)
        qc = QuantumCircuit(2)
        qc.delay(502, 0, unit="ns")
        qc.x(1)
        qc.delay(250, 1, unit="ns")
        qc.measure_all()
        res = transpile(qc, backend_v2)
        self.assertIn("delay", res.count_ops())

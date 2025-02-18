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

# pylint: disable=missing-class-docstring,missing-function-docstring
# pylint: disable=missing-module-docstring

import operator
from test import combine
from ddt import ddt, data

from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.utils import optionals

from qiskit.transpiler.coupling import CouplingMap
from test.utils.base import QiskitTestCase  # pylint: disable=wrong-import-order

BACKENDS_V2 = []
for n in [5, 7, 16, 20, 27, 65, 127]:
    cmap = CouplingMap.from_ring(n)
    BACKENDS_V2.append(GenericBackendV2(num_qubits=n, coupling_map=cmap, seed=42))


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
        backend=BACKENDS_V2,
        optimization_level=[0, 1, 2, 3],
    )
    def test_circuit_on_fake_backend_v2(self, backend, optimization_level):
        if not optionals.HAS_AER and backend.num_qubits > 20:
            self.skipTest(f"Unable to run fake_backend {backend.name} without qiskit-aer")
        job = backend.run(
            transpile(
                self.circuit, backend, seed_transpiler=42, optimization_level=optimization_level
            ),
            seed_simulator=42,
        )
        result = job.result()
        counts = result.get_counts()
        max_count = max(counts.items(), key=operator.itemgetter(1))[0]
        self.assertEqual(max_count, "11")

    @combine(
        backend=BACKENDS_V2,
        optimization_level=[0, 1, 2, 3],
        dsc="Test execution path on {backend} with optimization level {optimization_level}",
        name="{backend}_opt_level_{optimization_level}",
    )
    def test_circuit_on_fake_backend(self, backend, optimization_level):
        if not optionals.HAS_AER and backend.num_qubits > 20:
            self.skipTest(f"Unable to run fake_backend {backend.name} without qiskit-aer")
        with self.assertWarnsRegex(
            DeprecationWarning,
            expected_regex="The property ``qiskit.dagcircuit.dagcircuit.DAGCircuit.duration`` "
            "is deprecated as of Qiskit 1.3.0. It will be removed in Qiskit 2.0.0.",
        ):
            transpiled = transpile(
                self.circuit, backend, seed_transpiler=42, optimization_level=optimization_level
            )
        job = backend.run(transpiled, seed_simulator=42)
        result = job.result()
        counts = result.get_counts()
        max_count = max(counts.items(), key=operator.itemgetter(1))[0]
        self.assertEqual(max_count, "11")

    @data(*BACKENDS_V2)
    def test_convert_to_target(self, backend):
        target = backend.target
        if target.dt is not None:
            self.assertLess(target.dt, 1e-6)

    @data(*BACKENDS_V2)
    def test_backend_v2_dtm(self, backend):
        if backend.dtm:
            self.assertLess(backend.dtm, 1e-6)

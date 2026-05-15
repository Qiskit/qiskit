# This code is part of Qiskit.
#
# (C) Copyright IBM 2024
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.

"""Test SABRE layout with control flow operations."""

import unittest
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from test import QiskitTestCase


class TestSabreControlFlow(QiskitTestCase):
    def setUp(self):
        super().setUp()
        cmap = CouplingMap.from_heavy_hex(5)
        self.backend = GenericBackendV2(
            cmap.size(),
            control_flow=True,
            coupling_map=cmap,
            seed=42,
        )

    def test_if_else_layout_improvement(self):
        pm = generate_preset_pass_manager(
            optimization_level=1,
            backend=self.backend,
            seed_transpiler=42,
        )
        qc = QuantumCircuit(10, 1)
        for i in range(0, 9, 2):
            qc.cx(i, i + 1)
        for i in range(1, 8, 2):
            qc.cx(i, i + 1)
        qc.measure(0, 0)
        with qc.if_test((0, 1)):
            for i in range(0, 9, 2):
                qc.swap(i, i + 1)
            for i in range(1, 8, 2):
                qc.swap(i, i + 1)
        result = pm.run(qc)
        depth = result.depth(lambda g: g.operation.num_qubits == 2)
        self.assertLess(depth, 20)


if __name__ == "__main__":
    unittest.main()

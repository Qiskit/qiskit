# This code is part of Qiskit.
#
# (C) Copyright IBM 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=missing-function-docstring

"""Tests the layout object"""

import numpy as np

from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.transpiler.layout import Layout, TranspileLayout
from qiskit.transpiler.coupling import CouplingMap
from qiskit.compiler import transpile
from qiskit.test import QiskitTestCase
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import BackendEstimator
from qiskit.providers.fake_provider import FakeNairobiV2


class TranspileLayoutTest(QiskitTestCase):
    """Test the methods in the TranspileLayout object."""

    def test_full_layout_full_path(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        cmap = CouplingMap.from_line(3, bidirectional=False)
        tqc = transpile(qc, coupling_map=cmap, initial_layout=[2, 1, 0], seed_transpiler=42)
        res = tqc.layout.full_layout()
        self.assertEqual(res, [2, 0, 1])

    def test_permute_sparse_pauli_op(self):
        psi = EfficientSU2(4, reps=4, entanglement="circular")
        op = SparsePauliOp.from_list([("IIII", 1), ("IZZZ", 2), ("XXXI", 3)])
        backend = FakeNairobiV2()
        transpiled_psi = transpile(psi, backend, optimization_level=3, seed_transpiler=12345)
        permuted_op = transpiled_psi.layout.permute_sparse_pauli_op(op)
        identity_op = SparsePauliOp("I" * 7)
        initial_layout = transpiled_psi.layout.initial_layout_list(filter_ancillas=True)
        final_layout = transpiled_psi.layout.final_layout_list()
        qargs = [final_layout[x] for x in initial_layout]
        expected_op = identity_op.compose(op, qargs=qargs)
        self.assertNotEqual(op, permuted_op)
        self.assertEqual(permuted_op, expected_op)

    def test_permute_sparse_pauli_op_estimator_example(self):
        psi = EfficientSU2(4, reps=4, entanglement="circular")
        op = SparsePauliOp.from_list([("IIII", 1), ("IZZZ", 2), ("XXXI", 3)])
        backend = FakeNairobiV2()
        backend.set_options(seed_simulator=123)
        estimator = BackendEstimator(backend=backend, skip_transpilation=True)
        thetas = list(range(len(psi.parameters)))
        transpiled_psi = transpile(psi, backend, optimization_level=3)
        permuted_op = transpiled_psi.layout.permute_sparse_pauli_op(op)
        job = estimator.run(transpiled_psi, permuted_op, thetas)
        res = job.result().values
        np.testing.assert_allclose(res, [1.35351562], rtol=0.5, atol=0.2)

    def test_final_layout_list(self):
        qr = QuantumRegister(5)
        final_layout = Layout(
            {
                qr[0]: 2,
                qr[1]: 4,
                qr[2]: 1,
                qr[3]: 0,
                qr[4]: 3,
            }
        )
        layout_obj = TranspileLayout(
            initial_layout=Layout.generate_trivial_layout(qr),
            input_qubit_mapping={v: k for k, v in enumerate(qr)},
            final_layout=final_layout,
            _input_qubit_count=5,
            _output_qubit_list=list(qr),
        )
        res = layout_obj.final_layout_list()
        self.assertEqual(res, [2, 4, 1, 0, 3])

    def test_final_layout_list_no_final_layout(self):
        qr = QuantumRegister(5)
        layout_obj = TranspileLayout(
            initial_layout=Layout.generate_trivial_layout(qr),
            input_qubit_mapping={v: k for k, v in enumerate(qr)},
            final_layout=None,
            _input_qubit_count=5,
            _output_qubit_list=list(qr),
        )
        res = layout_obj.final_layout_list()
        self.assertEqual(res, list(range(5)))

    def test_initial_layout_list(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        cmap = CouplingMap.from_line(3, bidirectional=False)
        tqc = transpile(qc, coupling_map=cmap, initial_layout=[2, 1, 0], seed_transpiler=42)
        self.assertEqual(tqc.layout.initial_layout_list(), [2, 1, 0])

    def test_initial_layout_list_with_ancillas(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        cmap = CouplingMap.from_line(6, bidirectional=False)
        tqc = transpile(qc, coupling_map=cmap, initial_layout=[2, 1, 0], seed_transpiler=42)
        self.assertEqual(tqc.layout.initial_layout_list(), [2, 1, 0, 3, 4, 5])

    def test_initial_layout_list_filter_ancillas(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        cmap = CouplingMap.from_line(6, bidirectional=False)
        tqc = transpile(qc, coupling_map=cmap, initial_layout=[5, 2, 1], seed_transpiler=42)
        self.assertEqual(tqc.layout.initial_layout_list(True), [5, 2, 1])

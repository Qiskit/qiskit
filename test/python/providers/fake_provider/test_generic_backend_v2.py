# This code is part of Qiskit.
#
# (C) Copyright IBM 2023, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test of GenericBackendV2 backend"""

import math

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit import transpile
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import CouplingMap
from qiskit.exceptions import QiskitError
from qiskit.test import QiskitTestCase


class TestGenericBackendV2(QiskitTestCase):
    """Test class for GenericBackendV2 backend"""

    def setUp(self):
        super().setUp()
        self.cmap = CouplingMap(
            [(0, 2), (0, 1), (1, 3), (2, 4), (2, 3), (3, 5), (4, 6), (4, 5), (5, 7), (6, 7)]
        )

    def test_supported_basis_gates(self):
        """Test that target raises error if basis_gate not in ``supported_names``."""
        with self.assertRaises(QiskitError):
            GenericBackendV2(num_qubits=8, basis_gates=["cx", "id", "rz", "sx", "zz"])

    def test_operation_names(self):
        """Test that target basis gates include "delay", "measure" and "reset" even
        if not provided by user."""
        target = GenericBackendV2(num_qubits=8)
        op_names = list(target.operation_names)
        op_names.sort()
        self.assertEqual(op_names, ["cx", "delay", "id", "measure", "reset", "rz", "sx", "x"])

        target = GenericBackendV2(num_qubits=8, basis_gates=["ecr", "id", "rz", "sx", "x"])
        op_names = list(target.operation_names)
        op_names.sort()
        self.assertEqual(op_names, ["delay", "ecr", "id", "measure", "reset", "rz", "sx", "x"])

    def test_incompatible_coupling_map(self):
        """Test that the size of the coupling map must match num_qubits."""
        with self.assertRaises(QiskitError):
            GenericBackendV2(num_qubits=5, coupling_map=self.cmap)

    def test_control_flow_operation_names(self):
        """Test that control flow instructions are added to the target if control_flow is True."""
        target = GenericBackendV2(
            num_qubits=8,
            basis_gates=["ecr", "id", "rz", "sx", "x"],
            coupling_map=self.cmap,
            control_flow=True,
        ).target
        op_names = list(target.operation_names)
        op_names.sort()
        reference = [
            "break",
            "continue",
            "delay",
            "ecr",
            "for_loop",
            "id",
            "if_else",
            "measure",
            "reset",
            "rz",
            "switch_case",
            "sx",
            "while_loop",
            "x",
        ]
        self.assertEqual(op_names, reference)

    def test_default_coupling_map(self):
        """Test that fully-connected coupling map is generated correctly."""

        # fmt: off
        reference_cmap = [(0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 0), (0, 4), (4, 0), (1, 2), (2, 1),
                          (1, 3), (3, 1), (1, 4), (4, 1), (2, 3), (3, 2), (2, 4), (4, 2), (3, 4), (4, 3)]
        # fmt: on
        self.assertEqual(
            list(GenericBackendV2(num_qubits=5).coupling_map.get_edges()),
            reference_cmap,
        )

    def test_run(self):
        """Test run method, confirm correct noisy simulation if Aer is installed."""

        qr = QuantumRegister(5)
        cr = ClassicalRegister(5)
        qc = QuantumCircuit(qr, cr)
        qc.h(qr[0])
        for k in range(1, 4):
            qc.cx(qr[0], qr[k])
        qc.measure(qr, cr)

        backend = GenericBackendV2(num_qubits=5, basis_gates=["cx", "id", "rz", "sx", "x"])
        tqc = transpile(qc, backend=backend, optimization_level=3, seed_transpiler=42)
        result = backend.run(tqc, seed_simulator=42, shots=1000).result()
        counts = result.get_counts()

        self.assertTrue(math.isclose(counts["00000"], 500, rel_tol=0.1))
        self.assertTrue(math.isclose(counts["01111"], 500, rel_tol=0.1))

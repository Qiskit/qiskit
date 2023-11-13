# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test of FakeGeneric backend"""
from qiskit.providers.fake_provider import FakeGeneric
from qiskit.providers.fake_provider.fake_generic import GenericTarget
from qiskit.transpiler import CouplingMap
from qiskit.exceptions import QiskitError
from qiskit.test import QiskitTestCase


class TestGenericTarget(QiskitTestCase):
    """Test class for GenericTarget"""

    def setUp(self):
        super().setUp()
        self.cmap = CouplingMap(
            [(0, 2), (0, 1), (1, 3), (2, 4), (2, 3), (3, 5), (4, 6), (4, 5), (5, 7), (6, 7)]
        )
        self.basis_gates = ["cx", "id", "rz", "sx", "x"]

    def test_supported_basis_gates(self):
        """Test that target raises error if basis_gate not in ``supported_names``."""
        with self.assertRaises(QiskitError):
            GenericTarget(
                num_qubits=8, basis_gates=["cx", "id", "rz", "sx", "zz"], coupling_map=self.cmap
            )

    def test_operation_names(self):
        """Test that target basis gates include "delay", "measure" and "reset" even
        if not provided by user."""
        target = GenericTarget(
            num_qubits=8, basis_gates=["ecr", "id", "rz", "sx", "x"], coupling_map=self.cmap
        )
        op_names = list(target.operation_names)
        op_names.sort()
        self.assertEqual(op_names, ["delay", "ecr", "id", "measure", "reset", "rz", "sx", "x"])

    def test_incompatible_coupling_map(self):
        """Test that the size of the coupling map must match num_qubits."""
        with self.assertRaises(QiskitError):
            FakeGeneric(
                num_qubits=5, basis_gates=["cx", "id", "rz", "sx", "x"], coupling_map=self.cmap
            )

    def test_control_flow_operation_names(self):
        """Test that control flow instructions are added to the target if control_flow is True."""
        target = GenericTarget(
            num_qubits=8,
            basis_gates=["ecr", "id", "rz", "sx", "x"],
            coupling_map=self.cmap,
            control_flow=True,
        )
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


class TestFakeGeneric(QiskitTestCase):
    """Test class for FakeGeneric backend"""

    def test_default_coupling_map(self):
        """Test that fully-connected coupling map is generated correctly."""

        # fmt: off
        reference_cmap = [(0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 0), (0, 4), (4, 0), (1, 2), (2, 1),
                          (1, 3), (3, 1), (1, 4), (4, 1), (2, 3), (3, 2), (2, 4), (4, 2), (3, 4), (4, 3)]
        # fmt: on

        self.assertEqual(
            list(
                FakeGeneric(
                    num_qubits=5, basis_gates=["cx", "id", "rz", "sx", "x"]
                ).coupling_map.get_edges()
            ),
            reference_cmap,
        )

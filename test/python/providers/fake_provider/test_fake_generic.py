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
from qiskit.test import QiskitTestCase


class TestFakeGeneric(QiskitTestCase):
    def test_heavy_hex_num_qubits(self):
        """Test if num_qubits=5 and coupling_map_type is heavy_hex the number of qubits generated is 19"""
        self.assertEqual(FakeGeneric(num_qubits=5, coupling_map_type="heavy_hex").num_qubits, 19)

    def test_heavy_hex_coupling_map(self):
        """Test if coupling_map of heavy_hex is generated right"""
        cp_mp = [
            (0, 13),
            (1, 13),
            (1, 14),
            (2, 14),
            (3, 15),
            (4, 15),
            (4, 16),
            (5, 16),
            (6, 17),
            (7, 17),
            (7, 18),
            (8, 18),
            (0, 9),
            (3, 9),
            (5, 12),
            (8, 12),
            (10, 14),
            (10, 16),
            (11, 15),
            (11, 17),
        ]
        self.assertEqual(
            list(
                FakeGeneric(num_qubits=19, coupling_map_type="heavy_hex").coupling_map.get_edges()
            ),
            cp_mp,
        )

    def test_grid_coupling_map(self):
        """Test if grid coupling map is generated correct.
        In this test num_qubits=8, so a grid of 2x4 qubits need to be constructed"""
        cp_mp = [(0, 2), (0, 1), (1, 3), (2, 4), (2, 3), (3, 5), (4, 6), (4, 5), (5, 7), (6, 7)]
        self.assertEqual(
            list(FakeGeneric(num_qubits=8, coupling_map_type="grid").coupling_map.get_edges()),
            cp_mp,
        )

    def test_basis_gates(self):
        """Test if the backend has a default basis gates, that includes delay and measure"""
        self.assertEqual(
            FakeGeneric(num_qubits=8).operation_names,
            ["ecr", "id", "rz", "sx", "x", "delay", "measure", "reset"],
        )

    def test_if_cx_replaced_with_ecr(self):
        """Test if cx is not replaced with ecr"""
        self.assertEqual(
            FakeGeneric(num_qubits=8, replace_cx_with_ecr=False).operation_names,
            ["cx", "id", "rz", "sx", "x", "delay", "measure", "reset"],
        )

    def test_dynamic_true_basis_gates(self):
        """Test if basis_gates includes ControlFlowOps when dynamic is set to True"""
        self.assertEqual(
            FakeGeneric(num_qubits=9, dynamic=True).operation_names,
            [
                "ecr",
                "id",
                "rz",
                "sx",
                "x",
                "delay",
                "measure",
                "if_else",
                "while_loop",
                "for_loop",
                "switch_case",
                "break",
                "continue",
                "reset",
            ],
        )

    def test_if_excludes_reset(self):
        """Test if reset is excluded from the operation names when set enable_reset False"""
        self.assertEqual(
            FakeGeneric(num_qubits=9, enable_reset=False).operation_names,
            ["ecr", "id", "rz", "sx", "x", "delay", "measure"],
        )

    def test_if_dt_is_set_correctly(self):
        """Test if dt is set correctly"""
        self.assertEqual(FakeGeneric(num_qubits=4, dt=0.5e-9).dt, 0.5e-9)

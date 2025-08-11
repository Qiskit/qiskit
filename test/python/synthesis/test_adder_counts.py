# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test gate counts of synthesis algorithms for adder circuits."""

from __future__ import annotations
import unittest
from ddt import ddt, data

from qiskit.synthesis.arithmetic.adders import (
    adder_modular_v17,
)
from qiskit.transpiler import generate_preset_pass_manager

from test import QiskitTestCase


@ddt
class TestAdderSynthesisCounts(QiskitTestCase):
    """Test gate counts of synthesis algorithms for adder circuits."""

    def setUp(self):
        super().setUp()
        self.pm = generate_preset_pass_manager(
            optimization_level=0, basis_gates=["u", "cx"], seed_transpiler=12345
        )

    @data(*range(2, 12, 2))
    def test_vrg_modular_adder_counts(self, num_qubits):
        """Test gate counts of VRG modular adder."""
        qc = adder_modular_v17(num_qubits)
        transpiled = self.pm.run(qc)
        cx_count = transpiled.count_ops().get("cx", 0)
        self.assertLessEqual(cx_count, 16 * num_qubits - 13)
        self.assertEqual(transpiled.num_qubits, 2 * num_qubits)


if __name__ == "__main__":
    unittest.main()

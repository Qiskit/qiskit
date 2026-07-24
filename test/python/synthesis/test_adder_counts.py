# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test gate counts of synthesis algorithms for adder circuits."""

from __future__ import annotations
import unittest
from test import QiskitTestCase
from ddt import ddt, data, unpack

from qiskit.synthesis.arithmetic.adders import (
    adder_modular_v17,
    adder_ripple_c04,
)
from qiskit.transpiler import (
    generate_preset_clifford_t_pass_manager,
    generate_preset_pass_manager,
)
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library.arithmetic.adders import ModularAdderGate


@ddt
class TestAdderSynthesisCounts(QiskitTestCase):
    """Test gate counts of synthesis algorithms for adder circuits."""

    def setUp(self):
        super().setUp()
        # Need optimization level 2 for small modular adder counts
        self.pm = generate_preset_pass_manager(
            optimization_level=2, basis_gates=["u", "cx"], seed_transpiler=12345
        )
        self.clifford_t_pm = generate_preset_clifford_t_pass_manager(optimization_level=0)

    @data(*range(1, 6))
    def test_small_modular_adder_cx_count(self, num_ctrl_gates: int):
        """Test gate counts of small modular adder."""

        qc = QuantumCircuit(2 * num_ctrl_gates)
        qc.append(ModularAdderGate(num_ctrl_gates), range(2 * num_ctrl_gates))
        transpiled = self.pm.run(qc)
        cx_count = transpiled.count_ops().get("cx", 0)

        expected = {1: 1, 2: 10, 3: 21, 4: 40, 5: 65}
        self.assertLessEqual(cx_count, expected[num_ctrl_gates])

    @data(*range(2, 15, 2))
    def test_vrg_modular_adder_counts(self, num_qubits):
        """Test gate counts of VRG modular adder."""
        qc = adder_modular_v17(num_qubits)
        transpiled = self.pm.run(qc)
        cx_count = transpiled.count_ops().get("cx", 0)
        self.assertLessEqual(cx_count, 16 * num_qubits - 13)
        self.assertEqual(transpiled.num_qubits, 2 * num_qubits)

    @data(("fixed", -8), ("half", 1), ("full", 1))
    @unpack
    def test_cdkm_adder_counts(self, kind, cx_offset):
        """Test the exact CX and T counts of the optimized CDKM adder."""
        for num_qubits in (1, 3, 5):
            circuit = adder_ripple_c04(num_qubits, kind=kind)
            cx_count = self.pm.run(circuit).count_ops().get("cx", 0)
            clifford_t_ops = self.clifford_t_pm.run(circuit).count_ops()
            t_count = clifford_t_ops.get("t", 0) + clifford_t_ops.get("tdg", 0)
            self.assertEqual(cx_count, 10 * num_qubits + cx_offset)
            self.assertEqual(t_count, 8 * num_qubits)

    @data((4, 32), (5, 42))
    @unpack
    def test_modular_adder_with_clean_ancilla_counts(self, num_qubits, expected_cx):
        """Test metric-aware default synthesis when a clean ancilla is available."""
        circuit = QuantumCircuit(2 * num_qubits + 1)
        circuit.append(ModularAdderGate(num_qubits), range(2 * num_qubits))

        cx_count = self.pm.run(circuit).count_ops().get("cx", 0)
        clifford_t_ops = self.clifford_t_pm.run(circuit).count_ops()
        t_count = clifford_t_ops.get("t", 0) + clifford_t_ops.get("tdg", 0)

        self.assertEqual(cx_count, expected_cx)
        self.assertEqual(t_count, 8 * (num_qubits - 1))


if __name__ == "__main__":
    unittest.main()

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

"""Test Clifford+T transpilation pipeline"""

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import QFTGate, iqp, GraphStateGate
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from test import QiskitTestCase  # pylint: disable=wrong-import-order


class TestCliffordTPassManager(QiskitTestCase):
    """Test Clifford+T transpilation pipeline."""

    def setUp(self):
        super().setUp()
        self.basis_gates = ["cx", "s", "sdg", "h", "t", "tdg"]
        self.pm = generate_preset_pass_manager(basis_gates=self.basis_gates)

    def test_cliffords_1q(self):
        """Clifford+T transpilation of a circuit with single-qubit Clifford gates."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.s(1)
        qc.h(2)
        qc.sdg(2)
        qc.h(2)
        qc.h(2)

        transpiled = self.pm.run(qc)

        # The single 1q Clifford gates on qubits 0 and 1 should remain, the multiple 1q Clifford
        # gates on qubit 2 should be resynthesized as Clifford gates.
        expected = QuantumCircuit(3)
        expected.h(0)
        expected.s(1)
        expected.x(2)
        expected.h(2)
        expected.s(2)

        self.assertEqual(transpiled, expected)

    def test_complex_clifford(self):
        """Clifford+T transpilation of a more complex Clifford circuit."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.s(0)
        qc.cx(1, 0)
        qc.cx(0, 1)
        qc.s(1)
        qc.sdg(0)
        qc.cx(1, 0)

        transpiled = self.pm.run(qc)
        transpiled_ops = transpiled.count_ops()

        # We should not have "t", "tdg" or "u" gates
        self.assertNotIn("t", transpiled_ops)
        self.assertNotIn("tdg", transpiled_ops)
        self.assertNotIn("u", transpiled_ops)

    def test_t_gates(self):
        """Clifford+T transpilation of a circuit with T/Tdg-gates."""
        qc = QuantumCircuit(3)
        qc.t(0)
        qc.tdg(1)
        qc.t(2)
        qc.tdg(2)

        transpiled = self.pm.run(qc)

        # The single T/Tdg gates on qubits 0 and 1 should remain, the T/Tdg pair on qubit 2
        # should cancel out.
        expected = QuantumCircuit(3)
        expected.t(0)
        expected.tdg(1)

        self.assertEqual(transpiled, expected)

    def test_ccx(self):
        """Clifford+T transpilation of the Toffoli circuit."""
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)

        transpiled = self.pm.run(qc)

        # Should get the efficient decomposition of the Toffoli gates into Clifford+T.
        self.assertEqual(transpiled.count_ops(), {"cx": 6, "t": 4, "tdg": 3, "h": 2})

    def test_rx(self):
        """Clifford+T transpilation of a circuit with a single-qubit rotation gate,
        requiring the usage of the Solovay-Kitaev decomposition.
        """
        qc = QuantumCircuit(1)
        qc.rx(0.8, 0)

        transpiled = self.pm.run(qc)
        self.assertLessEqual(
            set(transpiled.count_ops().keys()), {"cx", "h", "s", "sdg", "t", "tdg", "z"}
        )

    def test_qft(self):
        """Clifford+T transpilation of a more complex circuit, requiring the usage of the
        Solovay-Kitaev decomposition.
        """
        qc = QuantumCircuit(4)
        qc.append(QFTGate(4), [0, 1, 2, 3])

        transpiled = self.pm.run(qc)
        self.assertLessEqual(
            set(transpiled.count_ops().keys()), {"cx", "h", "s", "sdg", "t", "tdg", "z"}
        )

    def test_iqp(self):
        """Clifford+T transpilation of IQP circuits."""
        interactions = np.array([[6, 5, 1], [5, 4, 3], [1, 3, 2]])
        qc = iqp(interactions)

        transpiled = self.pm.run(qc)
        transpiled_ops = transpiled.count_ops()

        self.assertLessEqual(set(transpiled_ops.keys()), {"cx", "h", "s", "sdg", "t", "tdg", "z"})

        # The transpiled circuit should be fairly efficient in terms of gates.
        self.assertLessEqual(transpiled.size(), 30)

    def test_graph_state(self):
        """Clifford+T transpilation of graph-state circuits."""
        adjacency_matrix = [
            [0, 1, 0, 0, 1],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [1, 0, 0, 1, 0],
        ]
        graph_state_gate = GraphStateGate(adjacency_matrix)

        qc = QuantumCircuit(5)
        qc.append(graph_state_gate, [0, 1, 2, 3, 4])

        transpiled = self.pm.run(qc)
        transpiled_ops = transpiled.count_ops()

        # The resulting circuit should not have any T/Tdg-gates.
        self.assertEqual(transpiled_ops, {"h": 5, "cx": 5})

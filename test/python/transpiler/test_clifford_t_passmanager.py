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
from ddt import ddt, data, unpack

from qiskit.converters import circuit_to_dag
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import QFTGate, iqp, GraphStateGate
from qiskit.transpiler.passes.utils import CheckGateDirection
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler import CouplingMap
from qiskit.providers.fake_provider import GenericBackendV2

from test import QiskitTestCase  # pylint: disable=wrong-import-order


@ddt
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
        self.assertLessEqual(set(transpiled.count_ops()), set(self.basis_gates))

        # Note:
        # The transpilation calls SolovayKitaevSynthesis plugin with basis_gates=['h', 't', 'tdg'].
        # This is why the transpiled circuit contains T/Tdg gates even though they could be replaced
        # by Clifford gates within the basis.
        expected = QuantumCircuit(3)
        expected.h(0)
        expected.t(1)
        expected.t(1)
        expected.h(2)
        expected.tdg(2)
        expected.tdg(2)

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
        self.assertLessEqual(set(transpiled.count_ops()), set(self.basis_gates))

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
        self.assertLessEqual(set(transpiled.count_ops()), set(self.basis_gates))

    def test_qft(self):
        """Clifford+T transpilation of a more complex circuit, requiring the usage of the
        Solovay-Kitaev decomposition.
        """
        qc = QuantumCircuit(4)
        qc.append(QFTGate(4), [0, 1, 2, 3])

        transpiled = self.pm.run(qc)
        self.assertLessEqual(set(transpiled.count_ops()), set(self.basis_gates))

    def test_iqp(self):
        """Clifford+T transpilation of IQP circuits."""
        interactions = np.array([[6, 5, 1], [5, 4, 3], [1, 3, 2]])
        qc = iqp(interactions)

        transpiled = self.pm.run(qc)
        transpiled_ops = transpiled.count_ops()

        self.assertLessEqual(set(transpiled_ops), set(self.basis_gates))

        # The transpiled circuit should be fairly efficient in terms of T/Tdg gates:
        # the circuit contains 3 powers of CZ-gates, each leading to at most 9 T/Tdg gates,
        # and 3 powers of T-gate, each leading to at most 4 T/Tdg gate.
        max_t_size = 3 * 9 + 3 * 4
        transpiled_ops = transpiled.count_ops()
        t_size = transpiled_ops.get("t", 0) + transpiled_ops.get("tdg", 0)
        self.assertLessEqual(t_size, max_t_size)

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

    def test_gate_direction_remapped(self):
        """Test that gate directions are correct."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1)

        basis_gates = ["cx", "h", "t", "tdg"]
        coupling_map = CouplingMap([[1, 0]])

        pm = generate_preset_pass_manager(
            basis_gates=basis_gates, coupling_map=coupling_map, optimization_level=0
        )

        transpiled = pm.run(qc)
        self.assertLessEqual(set(transpiled.count_ops()), set(basis_gates))

        # Make sure gate direction is correct
        pass_ = CheckGateDirection(coupling_map=coupling_map)
        pass_.run(circuit_to_dag(transpiled))
        self.assertTrue(pass_.property_set["is_direction_mapped"])

    @data(["t", "h"], ["tdg", "h"], ["t", "sx"], ["t", "sxdg"], ["tdg", "sx"], ["tdg", "sxdg"])
    def test_clifford_t_bases(self, basis_gates):
        """Test transpiling into various Clifford+T basis sets."""
        qc = QuantumCircuit(2)
        qc.rz(0.8, 0)
        pm = generate_preset_pass_manager(basis_gates=basis_gates, optimization_level=0)
        transpiled = pm.run(qc)
        self.assertLessEqual(set(transpiled.count_ops()), set(basis_gates))

    def test_target(self):
        """Clifford+T transpilation given Target."""
        qc = QuantumCircuit(2)
        qc.rz(0.8, 0)

        basis_gates = ["cx", "t", "tdg", "h"]
        backend = GenericBackendV2(5, basis_gates=basis_gates)
        pm = generate_preset_pass_manager(target=backend.target, optimization_level=0)
        transpiled = pm.run(qc)
        self.assertLessEqual(set(transpiled.count_ops()), set(basis_gates))

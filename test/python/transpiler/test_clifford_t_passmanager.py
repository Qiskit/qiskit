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
from ddt import ddt, data

from qiskit.converters import circuit_to_dag
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import (
    QFTGate,
    iqp,
    GraphStateGate,
    MCXGate,
    MultiplierGate,
    ModularAdderGate,
)
from qiskit.transpiler.passes.utils import CheckGateDirection
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler import CouplingMap
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import get_clifford_gate_names

from test import QiskitTestCase  # pylint: disable=wrong-import-order


@ddt
class TestCliffordTPassManager(QiskitTestCase):
    """Test Clifford+T transpilation pipeline."""

    @data(0, 1, 2, 3)
    def test_cliffords_1q(self, optimization_level):
        """Clifford+T transpilation of a circuit with single-qubit Clifford gates."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.s(1)
        qc.h(2)
        qc.sdg(2)
        qc.h(2)
        qc.h(2)

        basis_gates = ["cx", "s", "sdg", "h", "t", "tdg"]
        pm = generate_preset_pass_manager(
            basis_gates=basis_gates, optimization_level=optimization_level
        )
        transpiled = pm.run(qc)
        self.assertLessEqual(set(transpiled.count_ops()), set(basis_gates))
        # The resulting circuit should not have any T/Tdg-gates.
        self.assertEqual(_get_t_count(transpiled), 0)

    @data(0, 1, 2, 3)
    def test_complex_clifford(self, optimization_level):
        """Clifford+T transpilation of a more complex Clifford circuit."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.s(0)
        qc.cx(1, 0)
        qc.cx(0, 1)
        qc.s(1)
        qc.sdg(0)
        qc.cx(1, 0)

        basis_gates = ["cx", "s", "sdg", "h", "t", "tdg"]
        pm = generate_preset_pass_manager(
            basis_gates=basis_gates, optimization_level=optimization_level
        )
        transpiled = pm.run(qc)
        self.assertLessEqual(set(transpiled.count_ops()), set(basis_gates))
        # The resulting circuit should not have any T/Tdg-gates.
        self.assertEqual(_get_t_count(transpiled), 0)

    @data(0, 1, 2, 3)
    def test_rx(self, optimization_level):
        """Clifford+T transpilation of a circuit with a single-qubit rotation gate,
        requiring the usage of the Solovay-Kitaev decomposition.
        """
        qc = QuantumCircuit(1)
        qc.rx(0.8, 0)

        basis_gates = ["cx", "s", "sdg", "h", "t", "tdg"]
        pm = generate_preset_pass_manager(
            basis_gates=basis_gates, optimization_level=optimization_level
        )
        transpiled = pm.run(qc)
        self.assertLessEqual(set(transpiled.count_ops()), set(basis_gates))

    @data(0, 1, 2, 3)
    def test_rx_pi2(self, optimization_level):
        """Clifford+T transpilation of a circuit with a single-qubit rotation gate
        that corresponds to a Clifford gate.
        """
        qc = QuantumCircuit(1)
        qc.rx(np.pi / 2, 0)

        basis_gates = get_clifford_gate_names() + ["t", "tdg"]
        pm = generate_preset_pass_manager(
            basis_gates=basis_gates, optimization_level=optimization_level
        )
        transpiled = pm.run(qc)
        self.assertLessEqual(set(transpiled.count_ops()), set(basis_gates))
        # The resulting circuit should not have any T/Tdg-gates.
        self.assertEqual(_get_t_count(transpiled), 0)

    @data(0, 1, 2, 3)
    def test_rx_pi4(self, optimization_level):
        """Clifford+T transpilation of a circuit with a single-qubit rotation gate
        with a "nice" angle.
        """
        qc = QuantumCircuit(1)
        qc.rx(np.pi / 4, 0)

        basis_gates = ["cx", "h", "t", "tdg"]
        pm = generate_preset_pass_manager(
            basis_gates=basis_gates, optimization_level=optimization_level
        )
        transpiled = pm.run(qc)
        self.assertLessEqual(set(transpiled.count_ops()), set(basis_gates))
        self.assertEqual(transpiled.count_ops(), {"h": 2, "t": 1})

    @data(0, 1, 2, 3)
    def test_qft(self, optimization_level):
        """Clifford+T transpilation of a more complex circuit, requiring the usage of the
        Solovay-Kitaev decomposition.
        """
        qc = QuantumCircuit(4)
        qc.append(QFTGate(4), [0, 1, 2, 3])

        basis_gates = ["cx", "s", "sdg", "h", "t", "tdg"]
        pm = generate_preset_pass_manager(
            basis_gates=basis_gates, optimization_level=optimization_level
        )
        transpiled = pm.run(qc)
        self.assertLessEqual(set(transpiled.count_ops()), set(basis_gates))

    @data(0, 1, 2, 3)
    def test_multiplier(self, optimization_level):
        """Clifford+T transpilation of a multiplier gate, using different optimization levels."""
        gate = MultiplierGate(4)
        qc = QuantumCircuit(gate.num_qubits)
        qc.append(gate, qc.qubits)

        # Transpile to a Clifford+T basis set
        basis_gates = get_clifford_gate_names() + ["t", "tdg"]
        pm = generate_preset_pass_manager(
            basis_gates=basis_gates, optimization_level=optimization_level, seed_transpiler=0
        )
        transpiled = pm.run(qc)

        self.assertLessEqual(set(transpiled.count_ops()), set(basis_gates))
        t_count = _get_t_count(transpiled)

        # This is the T-count with optimization level 0.
        # We should not expect to see more T-gates with higher optimization levels
        # (while this is technically possible, it means that Clifford+T transpiler
        # pipeline is not setup correctly).
        expected_t_count = 1085
        self.assertLessEqual(t_count, expected_t_count)

    @data(0, 1, 2, 3)
    def test_iqp(self, optimization_level):
        """Clifford+T transpilation of IQP circuits."""
        interactions = np.array([[6, 5, 1], [5, 4, 3], [1, 3, 2]])
        qc = iqp(interactions)

        basis_gates = ["cx", "s", "sdg", "h", "t", "tdg"]
        pm = generate_preset_pass_manager(
            basis_gates=basis_gates, optimization_level=optimization_level
        )
        transpiled = pm.run(qc)
        self.assertLessEqual(set(transpiled.count_ops()), set(basis_gates))

        # The transpiled circuit should be fairly efficient in terms of T/Tdg gates:
        # the circuit contains 3 powers of CZ-gates, each leading to at most 9 T/Tdg gates,
        # and 3 powers of T-gates, each leading to at most 4 T/Tdg gates.
        # Importantly, the transpilation should not make this worse.
        max_t_size = 3 * 9 + 3 * 4
        self.assertLessEqual(_get_t_count(transpiled), max_t_size)

    @data(0, 1, 2, 3)
    def test_graph_state(self, optimization_level):
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

        basis_gates = ["cx", "s", "sdg", "h", "t", "tdg"]
        pm = generate_preset_pass_manager(
            basis_gates=basis_gates, optimization_level=optimization_level
        )
        transpiled = pm.run(qc)
        transpiled_ops = transpiled.count_ops()
        self.assertLessEqual(set(transpiled_ops), set(basis_gates))
        # The resulting circuit should not have any T/Tdg-gates.
        self.assertEqual(_get_t_count(transpiled), 0)

    @data(0, 1, 2, 3)
    def test_ccx(self, optimization_level):
        """Clifford+T transpilation of the Toffoli circuit."""
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)

        basis_gates = ["cx", "h", "t", "tdg"]
        pm = generate_preset_pass_manager(
            basis_gates=basis_gates, optimization_level=optimization_level
        )
        transpiled = pm.run(qc)

        # Should get the efficient decomposition of the Toffoli gates into Clifford+T.
        self.assertEqual(transpiled.count_ops(), {"cx": 6, "t": 4, "tdg": 3, "h": 2})

    def test_t_gates(self):
        """Clifford+T transpilation of a circuit with T/Tdg-gates."""
        qc = QuantumCircuit(3)
        qc.t(0)
        qc.tdg(1)
        qc.t(2)
        qc.tdg(2)

        basis_gates = ["h", "t", "tdg"]
        pm = generate_preset_pass_manager(basis_gates=basis_gates)
        transpiled = pm.run(qc)

        # The single T/Tdg gates on qubits 0 and 1 should remain, the T/Tdg pair on qubit 2
        # should cancel out.
        expected = QuantumCircuit(3)
        expected.t(0)
        expected.tdg(1)

        self.assertEqual(transpiled, expected)

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

    def test_get_clifford_gate_names(self):
        """Test transpiling with get_clifford_gate_names."""
        qc = QuantumCircuit(1)
        qc.rx(0.8, 0)

        basis_gates = get_clifford_gate_names() + ["t", "tdg"]
        pm = generate_preset_pass_manager(basis_gates=basis_gates)
        transpiled = pm.run(qc)
        self.assertLessEqual(set(transpiled.count_ops()), set(basis_gates))

    @data(2, 3, 4, 5, 6, 7)
    def test_mcx_gate_1_clean_ancilla(self, n):
        """Clifford+T transpilation of a circuit with an mcx gate."""
        # Create a circuit with an mcx gate and 1 additional clean ancilla qubit
        gate = MCXGate(n)
        nq = gate.num_qubits
        qc = QuantumCircuit(nq + 1)
        qc.append(gate, qc.qubits[0:nq])

        # Transpile to a Clifford+T basis set
        basis_gates = get_clifford_gate_names() + ["t", "tdg"]
        pm = generate_preset_pass_manager(basis_gates=basis_gates, optimization_level=0)
        transpiled = pm.run(qc)
        self.assertLessEqual(set(transpiled.count_ops()), set(basis_gates))

        # The resulting decomposition should be efficient in terms of T-count
        # provided 1 ancilla qubit is available
        t_count = _get_t_count(transpiled)
        expected_t_count = {1: 0, 2: 7, 3: 15, 4: 27, 5: 39, 6: 51, 7: 63}
        self.assertLessEqual(t_count, expected_t_count[n])

    @data(2, 3, 4, 5, 6, 7)
    def test_mcx_gate_many_dirty_ancillas(self, n):
        """Clifford+T transpilation of a circuit with an mcx gate."""
        # Create a circuit with an mcx gate and many dirty ancillas
        nqc = 20  # number of qubits in the circuit
        qc = QuantumCircuit(nqc)
        for i in range(nqc):  # make all the qubits dirty
            qc.x(i)
        gate = MCXGate(n)
        nq = gate.num_qubits  # number of qubits in the gate
        qc.append(gate, qc.qubits[0:nq])

        # Transpile to a Clifford+T basis set
        basis_gates = get_clifford_gate_names() + ["t", "tdg"]
        pm = generate_preset_pass_manager(basis_gates=basis_gates, optimization_level=0)
        transpiled = pm.run(qc)
        self.assertLessEqual(set(transpiled.count_ops()), set(basis_gates))

        # The resulting decomposition should be efficient in terms of T-count
        # provided 1 ancilla qubit is available
        t_count = _get_t_count(transpiled)
        expected_t_count = {1: 0, 2: 7, 3: 17, 4: 29, 5: 41, 6: 51, 7: 63}
        self.assertLessEqual(t_count, expected_t_count[n])

    @data(2, 3, 4, 5, 6, 7)
    def test_multiplier_gate(self, n):
        """Clifford+T transpilation of a circuit with a multiplier gate."""
        # Create a circuit with a multiplier gate
        gate = MultiplierGate(n)
        qc = QuantumCircuit(gate.num_qubits)
        qc.append(gate, qc.qubits)

        # Transpile to a Clifford+T basis set
        basis_gates = get_clifford_gate_names() + ["t", "tdg"]
        pm = generate_preset_pass_manager(basis_gates=basis_gates, optimization_level=0)
        transpiled = pm.run(qc)
        self.assertLessEqual(set(transpiled.count_ops()), set(basis_gates))

        # The resulting decomposition should be efficient in terms of T-count,
        # except surprisingly for the case n=1 (which is why it is not used in this test)
        t_count = _get_t_count(transpiled)
        expected_t_count = {2: 153, 3: 501, 4: 1114, 5: 2005, 6: 2596, 7: 3850}
        self.assertLessEqual(t_count, expected_t_count[n])

    @data(1, 2, 3, 4, 5, 6, 7)
    def test_modular_adder_gate(self, n):
        """Clifford+T transpilation of a circuit with a modular adder gate."""
        # Create a circuit with a modular adder gate
        gate = ModularAdderGate(n)
        qc = QuantumCircuit(gate.num_qubits)
        qc.append(gate, qc.qubits)

        # Transpile to a Clifford+T basis set
        basis_gates = get_clifford_gate_names() + ["t", "tdg"]
        pm = generate_preset_pass_manager(basis_gates=basis_gates, optimization_level=0)
        transpiled = pm.run(qc)
        self.assertLessEqual(set(transpiled.count_ops()), set(basis_gates))

        # The resulting decomposition should be efficient in terms of T-count,
        t_count = _get_t_count(transpiled)
        expected_t_count = {1: 0, 2: 8, 3: 16, 4: 24, 5: 32, 6: 40, 7: 48}
        self.assertLessEqual(t_count, expected_t_count[n])


def _get_t_count(qc):
    """Returns the number of T/Tdg gates in a circuit."""
    ops = qc.count_ops()
    return ops.get("t", 0) + ops.get("tdg", 0)

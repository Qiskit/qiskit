# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the evolution gate."""

import scipy
from ddt import ddt, data

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter, SuzukiTrotter, MatrixExponential
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase
from qiskit.opflow import I, X, Y, Z, PauliSumOp
from qiskit.quantum_info import Operator, SparsePauliOp, Pauli


@ddt
class TestEvolutionGate(QiskitTestCase):
    """Test the evolution gate."""

    def test_matrix_decomposition(self):
        """Test the default decomposition."""
        op = (X ^ 3) + (Y ^ 3) + (Z ^ 3)
        time = 0.123

        matrix = op.to_matrix()
        evolved = scipy.linalg.expm(-1j * time * matrix)

        evo_gate = PauliEvolutionGate(op, time, synthesis=MatrixExponential())

        self.assertTrue(Operator(evo_gate).equiv(evolved))

    def test_lie_trotter(self):
        """Test constructing the circuit with Lie Trotter decomposition."""
        op = (X ^ 3) + (Y ^ 3) + (Z ^ 3)
        time = 0.123
        reps = 4
        evo_gate = PauliEvolutionGate(op, time, synthesis=LieTrotter(reps=reps))
        decomposed = evo_gate.definition.decompose()
        self.assertEqual(decomposed.count_ops()["cx"], reps * 3 * 4)

    def test_rzx_order(self):
        """Test ZX is mapped onto the correct qubits."""
        evo_gate = PauliEvolutionGate(X ^ Z)
        decomposed = evo_gate.definition.decompose()

        ref = QuantumCircuit(2)
        ref.h(1)
        ref.cx(1, 0)
        ref.rz(2.0, 0)
        ref.cx(1, 0)
        ref.h(1)

        # don't use circuit equality since RZX here decomposes with RZ on the bottom
        self.assertTrue(Operator(decomposed).equiv(ref))

    def test_suzuki_trotter(self):
        """Test constructing the circuit with Lie Trotter decomposition."""
        op = (X ^ 3) + (Y ^ 3) + (Z ^ 3)
        time = 0.123
        reps = 4
        for order in [2, 4, 5]:
            if order == 2:
                expected_cx = reps * 5 * 4
            elif order % 2 == 0:
                # recurse (order - 2) / 2 times, base case has 5 blocks with 4 CX each
                expected_cx = reps * 5 ** ((order - 2) / 2) * 5 * 4
            else:
                # recurse (order - 1) / 2 times, base case has 3 blocks with 4 CX each
                expected_cx = reps * 5 ** ((order - 1) / 2) * 3 * 4

            with self.subTest(order=order):
                evo_gate = PauliEvolutionGate(
                    op, time, synthesis=SuzukiTrotter(order=order, reps=reps)
                )
                decomposed = evo_gate.definition.decompose()
                self.assertEqual(decomposed.count_ops()["cx"], expected_cx)

    def test_suzuki_trotter_manual(self):
        """Test the evolution circuit of Suzuki Trotter against a manually constructed circuit."""
        op = X + Y
        time = 0.1
        reps = 1
        evo_gate = PauliEvolutionGate(op, time, synthesis=SuzukiTrotter(order=4, reps=reps))

        # manually construct expected evolution
        expected = QuantumCircuit(1)
        p_4 = 1 / (4 - 4 ** (1 / 3))  # coefficient for reduced time from Suzuki paper
        for _ in range(2):
            # factor of 1/2 already included with factor 1/2 in rotation gate
            expected.rx(p_4 * time, 0)
            expected.ry(2 * p_4 * time, 0)
            expected.rx(p_4 * time, 0)

        expected.rx((1 - 4 * p_4) * time, 0)
        expected.ry(2 * (1 - 4 * p_4) * time, 0)
        expected.rx((1 - 4 * p_4) * time, 0)

        for _ in range(2):
            expected.rx(p_4 * time, 0)
            expected.ry(2 * p_4 * time, 0)
            expected.rx(p_4 * time, 0)

        self.assertEqual(evo_gate.definition.decompose(), expected)

    def test_passing_grouped_paulis(self):
        """Test passing a list of already grouped Paulis."""
        grouped_ops = [(X ^ Y) + (Y ^ X), (Z ^ I) + (Z ^ Z) + (I ^ Z), (X ^ X)]
        evo_gate = PauliEvolutionGate(grouped_ops, time=0.12, synthesis=LieTrotter())
        decomposed = evo_gate.definition.decompose()
        self.assertEqual(decomposed.count_ops()["rz"], 4)
        self.assertEqual(decomposed.count_ops()["rzz"], 1)
        self.assertEqual(decomposed.count_ops()["rxx"], 1)

    def test_list_from_grouped_paulis(self):
        """Test getting a string representation from grouped Paulis."""
        grouped_ops = [(X ^ Y) + (Y ^ X), (Z ^ I) + (Z ^ Z) + (I ^ Z), (X ^ X)]
        evo_gate = PauliEvolutionGate(grouped_ops, time=0.12, synthesis=LieTrotter())

        pauli_strings = []
        for op in evo_gate.operator:
            if isinstance(op, SparsePauliOp):
                pauli_strings.append(op.to_list())
            else:
                pauli_strings.append([(str(op), 1 + 0j)])

        expected = [
            [("XY", 1 + 0j), ("YX", 1 + 0j)],
            [("ZI", 1 + 0j), ("ZZ", 1 + 0j), ("IZ", 1 + 0j)],
            [("XX", 1 + 0j)],
        ]
        self.assertListEqual(pauli_strings, expected)

    def test_dag_conversion(self):
        """Test constructing a circuit with evolutions yields a DAG with evolution blocks."""
        time = Parameter("t")
        evo = PauliEvolutionGate((Z ^ 2) + (X ^ 2), time=time)

        circuit = QuantumCircuit(2)
        circuit.h(circuit.qubits)
        circuit.append(evo, circuit.qubits)
        circuit.cx(0, 1)

        dag = circuit_to_dag(circuit)

        expected_ops = {"HGate", "CXGate", "PauliEvolutionGate"}
        ops = {node.op.__class__.__name__ for node in dag.op_nodes()}

        self.assertEqual(ops, expected_ops)

    @data("chain", "fountain")
    def test_cnot_chain_options(self, option):
        """Test selecting different kinds of CNOT chains."""

        op = Z ^ Z ^ Z
        synthesis = LieTrotter(reps=1, cx_structure=option)
        evo = PauliEvolutionGate(op, synthesis=synthesis)

        expected = QuantumCircuit(3)
        if option == "chain":
            expected.cx(2, 1)
            expected.cx(1, 0)
        else:
            expected.cx(1, 0)
            expected.cx(2, 0)

        expected.rz(2, 0)

        if option == "chain":
            expected.cx(1, 0)
            expected.cx(2, 1)
        else:
            expected.cx(2, 0)
            expected.cx(1, 0)

        self.assertEqual(expected, evo.definition)

    @data(
        Pauli("XI"),
        X ^ I,  # PauliOp
        SparsePauliOp(Pauli("XI")),
        PauliSumOp(SparsePauliOp("XI")),
    )
    def test_different_input_types(self, op):
        """Test all different supported input types and that they yield the same."""
        expected = QuantumCircuit(2)
        expected.rx(4, 1)

        with self.subTest(msg="plain"):
            evo = PauliEvolutionGate(op, time=2, synthesis=LieTrotter())
            self.assertEqual(evo.definition, expected)

        with self.subTest(msg="wrapped in list"):
            evo = PauliEvolutionGate([op], time=2, synthesis=LieTrotter())
            self.assertEqual(evo.definition, expected)

    def test_pauliop_coefficients_respected(self):
        """Test that global ``PauliOp`` coefficients are being taken care of."""
        evo = PauliEvolutionGate(5 * (Z ^ I), time=1, synthesis=LieTrotter())
        circuit = evo.definition.decompose()
        rz_angle = circuit.data[0][0].params[0]
        self.assertEqual(rz_angle, 10)

    def test_paulisumop_coefficients_respected(self):
        """Test that global ``PauliSumOp`` coefficients are being taken care of."""
        evo = PauliEvolutionGate(5 * (2 * X + 3 * Y - Z), time=1, synthesis=LieTrotter())
        circuit = evo.definition.decompose()
        rz_angles = [
            circuit.data[0][0].params[0],  # X
            circuit.data[1][0].params[0],  # Y
            circuit.data[2][0].params[0],  # Z
        ]
        self.assertListEqual(rz_angles, [20, 30, -10])

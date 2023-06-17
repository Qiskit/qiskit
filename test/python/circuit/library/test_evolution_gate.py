# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the evolution gate."""

import unittest
import numpy as np
import scipy
from ddt import ddt, data, unpack

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter, SuzukiTrotter, MatrixExponential, QDrift
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase
from qiskit.opflow import I, X, Y, Z, PauliSumOp
from qiskit.quantum_info import Operator, SparsePauliOp, Pauli, Statevector
from qiskit.utils import algorithm_globals


@ddt
class TestEvolutionGate(QiskitTestCase):
    """Test the evolution gate."""

    def setUp(self):
        super().setUp()
        # fix random seed for reproducibility (used in QDrift)
        algorithm_globals.random_seed = 2

    def test_matrix_decomposition(self):
        """Test the default decomposition."""
        with self.assertWarns(DeprecationWarning):
            op = (X ^ 3) + (Y ^ 3) + (Z ^ 3)
        time = 0.123

        matrix = op.to_matrix()
        evolved = scipy.linalg.expm(-1j * time * matrix)

        evo_gate = PauliEvolutionGate(op, time, synthesis=MatrixExponential())

        self.assertTrue(Operator(evo_gate).equiv(evolved))

    def test_lie_trotter(self):
        """Test constructing the circuit with Lie Trotter decomposition."""
        with self.assertWarns(DeprecationWarning):
            op = (X ^ 3) + (Y ^ 3) + (Z ^ 3)
        time = 0.123
        reps = 4
        evo_gate = PauliEvolutionGate(op, time, synthesis=LieTrotter(reps=reps))
        decomposed = evo_gate.definition.decompose()
        self.assertEqual(decomposed.count_ops()["cx"], reps * 3 * 4)

    def test_rzx_order(self):
        """Test ZX and XZ is mapped onto the correct qubits."""
        with self.assertWarns(DeprecationWarning):
            op = (X ^ 3) + (Y ^ 3) + (Z ^ 3)
            for op, indices in zip([X ^ Z, Z ^ X], [(0, 1), (1, 0)]):
                with self.subTest(op=op, indices=indices):
                    evo_gate = PauliEvolutionGate(op)
                    decomposed = evo_gate.definition.decompose()

                    #           ┌───┐┌───────┐┌───┐
                    # q_0: ─────┤ X ├┤ Rz(2) ├┤ X ├─────
                    #      ┌───┐└─┬─┘└───────┘└─┬─┘┌───┐
                    # q_1: ┤ H ├──■─────────────■──┤ H ├
                    #      └───┘                   └───┘
                    ref = QuantumCircuit(2)
                    ref.h(indices[1])
                    ref.cx(indices[1], indices[0])
                    ref.rz(2.0, indices[0])
                    ref.cx(indices[1], indices[0])
                    ref.h(indices[1])

                    # don't use circuit equality since RZX here decomposes with RZ on the bottom
                    self.assertTrue(Operator(decomposed).equiv(ref))

    def test_suzuki_trotter(self):
        """Test constructing the circuit with Lie Trotter decomposition."""
        with self.assertWarns(DeprecationWarning):
            op = (X ^ 3) + (Y ^ 3) + (Z ^ 3)
        time = 0.123
        reps = 4
        for order in [2, 4, 6]:
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
        with self.assertWarns(DeprecationWarning):
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

    @data(
        (X + Y, 0.5, 1, [(Pauli("X"), 0.5), (Pauli("X"), 0.5)]),
        (X, 0.238, 2, [(Pauli("X"), 0.238)]),
    )
    @unpack
    def test_qdrift_manual(self, op, time, reps, sampled_ops):
        """Test the evolution circuit of Suzuki Trotter against a manually constructed circuit."""
        qdrift = QDrift(reps=reps)
        evo_gate = PauliEvolutionGate(op, time, synthesis=qdrift)
        evo_gate.definition.decompose()

        # manually construct expected evolution
        expected = QuantumCircuit(1)
        for pauli in sampled_ops:
            if pauli[0].to_label() == "X":
                expected.rx(2 * pauli[1], 0)
            elif pauli[0].to_label() == "Y":
                expected.ry(2 * pauli[1], 0)

        self.assertTrue(Operator(evo_gate.definition).equiv(expected))

    def test_qdrift_evolution(self):
        """Test QDrift on an example."""
        with self.assertWarns(DeprecationWarning):
            op = 0.1 * (Z ^ Z) + (X ^ I) + (I ^ X) + 0.2 * (X ^ X)
        reps = 20
        qdrift = PauliEvolutionGate(op, time=0.5 / reps, synthesis=QDrift(reps=reps)).definition
        exact = scipy.linalg.expm(-0.5j * op.to_matrix()).dot(np.eye(4)[0, :])

        def energy(evo):
            return Statevector(evo).expectation_value(op.to_matrix())

        self.assertAlmostEqual(energy(exact), energy(qdrift), places=2)

    def test_passing_grouped_paulis(self):
        """Test passing a list of already grouped Paulis."""
        with self.assertWarns(DeprecationWarning):
            grouped_ops = [(X ^ Y) + (Y ^ X), (Z ^ I) + (Z ^ Z) + (I ^ Z), (X ^ X)]
        evo_gate = PauliEvolutionGate(grouped_ops, time=0.12, synthesis=LieTrotter())
        decomposed = evo_gate.definition.decompose()
        self.assertEqual(decomposed.count_ops()["rz"], 4)
        self.assertEqual(decomposed.count_ops()["rzz"], 1)
        self.assertEqual(decomposed.count_ops()["rxx"], 1)

    def test_list_from_grouped_paulis(self):
        """Test getting a string representation from grouped Paulis."""
        with self.assertWarns(DeprecationWarning):
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
        with self.assertWarns(DeprecationWarning):
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

        with self.assertWarns(DeprecationWarning):
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
        with self.assertWarns(DeprecationWarning):
            evo = PauliEvolutionGate(5 * (Z ^ I), time=1, synthesis=LieTrotter())
        circuit = evo.definition.decompose()
        rz_angle = circuit.data[0].operation.params[0]
        self.assertEqual(rz_angle, 10)

    def test_paulisumop_coefficients_respected(self):
        """Test that global ``PauliSumOp`` coefficients are being taken care of."""
        with self.assertWarns(DeprecationWarning):
            evo = PauliEvolutionGate(5 * (2 * X + 3 * Y - Z), time=1, synthesis=LieTrotter())
        circuit = evo.definition.decompose()
        rz_angles = [
            circuit.data[0].operation.params[0],  # X
            circuit.data[1].operation.params[0],  # Y
            circuit.data[2].operation.params[0],  # Z
        ]
        self.assertListEqual(rz_angles, [20, 30, -10])

    def test_lie_trotter_two_qubit_correct_order(self):
        """Test that evolutions on two qubit operators are in the right order.

        Regression test of Qiskit/qiskit-terra#7544.
        """
        with self.assertWarns(DeprecationWarning):
            operator = I ^ Z ^ Z
        time = 0.5
        exact = scipy.linalg.expm(-1j * time * operator.to_matrix())
        lie_trotter = PauliEvolutionGate(operator, time, synthesis=LieTrotter())

        self.assertTrue(Operator(lie_trotter).equiv(exact))

    def test_complex_op_raises(self):
        """Test an operator with complex coefficient raises an error."""
        with self.assertRaises(ValueError):
            _ = PauliEvolutionGate(Pauli("iZ"))

    def test_paramtrized_op_raises(self):
        """Test an operator with parametrized coefficient raises an error."""
        with self.assertRaises(ValueError):
            _ = PauliEvolutionGate(SparsePauliOp("Z", np.array(Parameter("t"))))

    @data(LieTrotter, MatrixExponential)
    def test_inverse(self, synth_cls):
        """Test calculating the inverse is correct."""
        with self.assertWarns(DeprecationWarning):
            evo = PauliEvolutionGate(X + Y, time=0.12, synthesis=synth_cls())

        circuit = QuantumCircuit(1)
        circuit.append(evo, circuit.qubits)
        circuit.append(evo.inverse(), circuit.qubits)

        self.assertTrue(Operator(circuit).equiv(np.identity(2**circuit.num_qubits)))

    def test_labels_and_name(self):
        """Test the name and labels are correct."""
        with self.assertWarns(DeprecationWarning):
            operators = [X, (X + Y), ((I ^ Z) + (Z ^ I) - 0.2 * (X ^ X))]

        # note: the labels do not show coefficients!
        expected_labels = ["X", "(X + Y)", "(IZ + ZI + XX)"]
        for op, label in zip(operators, expected_labels):
            with self.subTest(op=op, label=label):
                evo = PauliEvolutionGate(op)
                self.assertEqual(evo.name, "PauliEvolution")
                self.assertEqual(evo.label, f"exp(-it {label})")


if __name__ == "__main__":
    unittest.main()

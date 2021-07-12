# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the grover operator."""

import unittest
import numpy as np

from qiskit.test.base import QiskitTestCase
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import GroverOperator
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Operator, Statevector, DensityMatrix


class TestGroverOperator(QiskitTestCase):
    """Test the Grover operator."""

    def assertGroverOperatorIsCorrect(self, grover_op, oracle, state_in=None, zero_reflection=None):
        """Test that ``grover_op`` implements the correct Grover operator."""

        oracle = Operator(oracle)

        if state_in is None:
            state_in = QuantumCircuit(oracle.num_qubits)
            state_in.h(state_in.qubits)
        state_in = Operator(state_in)

        if zero_reflection is None:
            zero_reflection = np.eye(2 ** oracle.num_qubits)
            zero_reflection[0][0] = -1
        zero_reflection = Operator(zero_reflection)

        expected = state_in.dot(zero_reflection).dot(state_in.adjoint()).dot(oracle)
        self.assertTrue(Operator(grover_op).equiv(expected))

    def test_grover_operator(self):
        """Test the base case for the Grover operator."""
        with self.subTest("single Z oracle"):
            oracle = QuantumCircuit(3)
            oracle.z(2)  # good state if last qubit is 1
            grover_op = GroverOperator(oracle)
            self.assertGroverOperatorIsCorrect(grover_op, oracle)

        with self.subTest("target state x0x1"):
            oracle = QuantumCircuit(4)
            oracle.x(1)
            oracle.z(1)
            oracle.x(1)
            oracle.z(3)
            grover_op = GroverOperator(oracle)
            self.assertGroverOperatorIsCorrect(grover_op, oracle)

    def test_quantum_info_input(self):
        """Test passing quantum_info.Operator and Statevector as input."""
        mark = Statevector.from_label("001")
        diffuse = 2 * DensityMatrix.from_label("000") - Operator.from_label("III")
        grover_op = GroverOperator(oracle=mark, zero_reflection=diffuse)
        self.assertGroverOperatorIsCorrect(
            grover_op, oracle=np.diag((-1) ** mark.data), zero_reflection=diffuse.data
        )

    def test_stateprep_contains_instruction(self):
        """Test wrapping works if the state preparation is not unitary."""
        oracle = QuantumCircuit(1)
        oracle.z(0)

        instr = QuantumCircuit(1)
        instr.s(0)
        instr = instr.to_instruction()

        stateprep = QuantumCircuit(1)
        stateprep.append(instr, [0])

        grover_op = GroverOperator(oracle, stateprep)
        self.assertEqual(grover_op.num_qubits, 1)

    def test_reflection_qubits(self):
        """Test setting idle qubits doesn't apply any operations on these qubits."""
        oracle = QuantumCircuit(4)
        oracle.z(3)
        grover_op = GroverOperator(oracle, reflection_qubits=[0, 3])
        dag = circuit_to_dag(grover_op.decompose())
        self.assertEqual(set(dag.idle_wires()), {dag.qubits[1], dag.qubits[2]})

    def test_custom_state_in(self):
        """Test passing a custom state_in operator."""
        oracle = QuantumCircuit(1)
        oracle.z(0)

        bernoulli = QuantumCircuit(1)
        sampling_probability = 0.2
        bernoulli.ry(2 * np.arcsin(np.sqrt(sampling_probability)), 0)

        grover_op = GroverOperator(oracle, bernoulli)
        self.assertGroverOperatorIsCorrect(grover_op, oracle, bernoulli)

    def test_custom_zero_reflection(self):
        """Test passing in a custom zero reflection."""
        oracle = QuantumCircuit(1)
        oracle.z(0)

        zero_reflection = QuantumCircuit(1)
        zero_reflection.x(0)
        zero_reflection.rz(np.pi, 0)
        zero_reflection.x(0)

        grover_op = GroverOperator(oracle, zero_reflection=zero_reflection)

        with self.subTest("zero reflection up to phase works"):
            self.assertGroverOperatorIsCorrect(grover_op, oracle)

        with self.subTest("circuits match"):
            expected = QuantumCircuit(*grover_op.qregs, global_phase=np.pi)
            expected.compose(oracle, inplace=True)
            expected.h(0)  # state_in is H
            expected.compose(zero_reflection, inplace=True)
            expected.h(0)
            self.assertEqual(expected, grover_op.decompose())

    def test_num_mcx_ancillas(self):
        """Test the number of ancilla bits for the mcx gate in zero_reflection."""
        oracle = QuantumCircuit(7)
        oracle.x(6)
        oracle.h(6)
        oracle.ccx(0, 1, 4)
        oracle.ccx(2, 3, 5)
        oracle.ccx(4, 5, 6)
        oracle.h(6)
        oracle.x(6)
        grover_op = GroverOperator(oracle, reflection_qubits=[0, 1])
        self.assertEqual(grover_op.width(), 7)


if __name__ == "__main__":
    unittest.main()

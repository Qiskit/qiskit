# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the DenseLayout pass"""

import unittest

import numpy as np

from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.circuit import Parameter, Qubit
from qiskit.circuit.library import CXGate, UGate, ECRGate, RZGate
from qiskit.transpiler import CouplingMap, Target, InstructionProperties, TranspilerError
from qiskit.transpiler.passes import DenseLayout
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase
from qiskit.providers.fake_provider import FakeTokyo
from qiskit.transpiler.passes.layout.dense_layout import _build_error_matrix


class TestDenseLayout(QiskitTestCase):
    """Tests the DenseLayout pass"""

    def setUp(self):
        super().setUp()
        self.cmap20 = FakeTokyo().configuration().coupling_map
        self.target_19 = Target()
        rng = np.random.default_rng(12345)
        instruction_props = {
            edge: InstructionProperties(
                duration=rng.uniform(1e-7, 1e-6), error=rng.uniform(1e-4, 1e-3)
            )
            for edge in CouplingMap.from_heavy_hex(3).get_edges()
        }
        self.target_19.add_instruction(CXGate(), instruction_props)

    def test_5q_circuit_20q_coupling(self):
        """Test finds dense 5q corner in 20q coupling map."""
        qr = QuantumRegister(5, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[3])
        circuit.cx(qr[3], qr[4])
        circuit.cx(qr[3], qr[1])
        circuit.cx(qr[0], qr[2])

        dag = circuit_to_dag(circuit)
        pass_ = DenseLayout(CouplingMap(self.cmap20))
        pass_.run(dag)

        layout = pass_.property_set["layout"]
        self.assertEqual(layout[qr[0]], 11)
        self.assertEqual(layout[qr[1]], 10)
        self.assertEqual(layout[qr[2]], 6)
        self.assertEqual(layout[qr[3]], 5)
        self.assertEqual(layout[qr[4]], 0)

    def test_6q_circuit_20q_coupling(self):
        """Test finds dense 5q corner in 20q coupling map."""
        qr0 = QuantumRegister(3, "q0")
        qr1 = QuantumRegister(3, "q1")
        circuit = QuantumCircuit(qr0, qr1)
        circuit.cx(qr0[0], qr1[2])
        circuit.cx(qr1[1], qr0[2])

        dag = circuit_to_dag(circuit)
        pass_ = DenseLayout(CouplingMap(self.cmap20))
        pass_.run(dag)

        layout = pass_.property_set["layout"]

        self.assertEqual(layout[qr0[0]], 11)
        self.assertEqual(layout[qr0[1]], 10)
        self.assertEqual(layout[qr0[2]], 6)
        self.assertEqual(layout[qr1[0]], 5)
        self.assertEqual(layout[qr1[1]], 1)
        self.assertEqual(layout[qr1[2]], 0)

    def test_5q_circuit_19q_target_with_noise(self):
        """Test layout works finds a dense 5q subgraph in a 19q heavy hex target."""
        qr = QuantumRegister(5, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[3])
        circuit.cx(qr[3], qr[4])
        circuit.cx(qr[3], qr[1])
        circuit.cx(qr[0], qr[2])
        dag = circuit_to_dag(circuit)
        pass_ = DenseLayout(target=self.target_19)
        pass_.run(dag)
        layout = pass_.property_set["layout"]
        self.assertEqual(layout[qr[0]], 9)
        self.assertEqual(layout[qr[1]], 3)
        self.assertEqual(layout[qr[2]], 11)
        self.assertEqual(layout[qr[3]], 15)
        self.assertEqual(layout[qr[4]], 4)

    def test_5q_circuit_19q_target_without_noise(self):
        """Test layout works finds a dense 5q subgraph in a 19q heavy hex target with no noise."""
        qr = QuantumRegister(5, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[3])
        circuit.cx(qr[3], qr[4])
        circuit.cx(qr[3], qr[1])
        circuit.cx(qr[0], qr[2])
        dag = circuit_to_dag(circuit)
        instruction_props = {edge: None for edge in CouplingMap.from_heavy_hex(3).get_edges()}
        noiseless_target = Target()
        noiseless_target.add_instruction(CXGate(), instruction_props)
        pass_ = DenseLayout(target=noiseless_target)
        pass_.run(dag)
        layout = pass_.property_set["layout"]
        self.assertEqual(layout[qr[0]], 1)
        self.assertEqual(layout[qr[1]], 13)
        self.assertEqual(layout[qr[2]], 0)
        self.assertEqual(layout[qr[3]], 9)
        self.assertEqual(layout[qr[4]], 3)

    def test_ideal_target_no_coupling(self):
        """Test pass fails as expected if a target without edge constraints exists."""
        qr = QuantumRegister(5, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[3])
        circuit.cx(qr[3], qr[4])
        circuit.cx(qr[3], qr[1])
        circuit.cx(qr[0], qr[2])
        dag = circuit_to_dag(circuit)
        target = Target(num_qubits=19)
        target.add_instruction(CXGate())
        layout_pass = DenseLayout(target=target)
        with self.assertRaises(TranspilerError):
            layout_pass.run(dag)

    def test_target_too_small_for_circuit(self):
        """Test error is raised when target is too small for circuit."""
        target = Target()
        target.add_instruction(
            CXGate(), {edge: None for edge in CouplingMap.from_line(3).get_edges()}
        )
        dag = circuit_to_dag(QuantumCircuit(5))
        layout_pass = DenseLayout(target=target)
        with self.assertRaises(TranspilerError):
            layout_pass.run(dag)

    def test_19q_target_with_noise_error_matrix(self):
        """Test the error matrix construction works for a just cx target."""
        expected_error_mat = np.zeros((19, 19))
        for qargs, props in self.target_19["cx"].items():
            error = props.error
            expected_error_mat[qargs[0]][qargs[1]] = error
        error_mat = _build_error_matrix(
            self.target_19.num_qubits,
            {i: i for i in range(self.target_19.num_qubits)},
            target=self.target_19,
        )[0]
        np.testing.assert_array_equal(expected_error_mat, error_mat)

    def test_multiple_gate_error_matrix(self):
        """Test error matrix ona small target with multiple gets on each qubit generates"""
        target = Target(num_qubits=3)
        phi = Parameter("phi")
        lam = Parameter("lam")
        theta = Parameter("theta")
        target.add_instruction(
            RZGate(phi), {(i,): InstructionProperties(duration=0, error=0) for i in range(3)}
        )
        target.add_instruction(
            UGate(theta, phi, lam),
            {(i,): InstructionProperties(duration=1e-7, error=1e-2) for i in range(3)},
        )
        cx_props = {
            (0, 1): InstructionProperties(error=1e-3),
            (0, 2): InstructionProperties(error=1e-3),
            (1, 0): InstructionProperties(error=1e-3),
            (1, 2): InstructionProperties(error=1e-3),
            (2, 0): InstructionProperties(error=1e-3),
            (2, 1): InstructionProperties(error=1e-3),
        }
        target.add_instruction(CXGate(), cx_props)
        ecr_props = {
            (0, 1): InstructionProperties(error=2e-2),
            (1, 2): InstructionProperties(error=2e-2),
            (2, 0): InstructionProperties(error=2e-2),
        }
        target.add_instruction(ECRGate(), ecr_props)
        expected_error_matrix = np.array(
            [
                [1e-2, 2e-2, 1e-3],
                [1e-3, 1e-2, 2e-2],
                [2e-2, 1e-3, 1e-2],
            ]
        )
        error_mat = _build_error_matrix(
            target.num_qubits, {i: i for i in range(target.num_qubits)}, target=target
        )[0]
        np.testing.assert_array_equal(expected_error_matrix, error_mat)

    def test_5q_circuit_20q_with_if_else(self):
        """Test layout works finds a dense 5q subgraph in a 19q heavy hex target."""
        qr = QuantumRegister(5, "q")
        cr = ClassicalRegister(5)
        circuit = QuantumCircuit(qr, cr)
        true_body = QuantumCircuit(qr, cr)
        false_body = QuantumCircuit(qr, cr)
        true_body.cx(qr[0], qr[3])
        true_body.cx(qr[3], qr[4])
        false_body.cx(qr[3], qr[1])
        false_body.cx(qr[0], qr[2])
        circuit.if_else((cr, 0), true_body, false_body, qr, cr)
        circuit.cx(0, 4)

        dag = circuit_to_dag(circuit)
        pass_ = DenseLayout(CouplingMap(self.cmap20))
        pass_.run(dag)
        layout = pass_.property_set["layout"]
        self.assertEqual(layout[qr[0]], 11)
        self.assertEqual(layout[qr[1]], 10)
        self.assertEqual(layout[qr[2]], 6)
        self.assertEqual(layout[qr[3]], 5)
        self.assertEqual(layout[qr[4]], 0)

    def test_loose_bit_circuit(self):
        """Test dense layout works with loose bits outside a register."""
        bits = [Qubit() for _ in range(5)]
        circuit = QuantumCircuit()
        circuit.add_bits(bits)
        circuit.h(3)
        circuit.cx(3, 4)
        circuit.cx(3, 2)
        circuit.cx(3, 0)
        circuit.cx(3, 1)
        dag = circuit_to_dag(circuit)
        pass_ = DenseLayout(CouplingMap(self.cmap20))
        pass_.run(dag)
        layout = pass_.property_set["layout"]
        self.assertEqual(layout[bits[0]], 11)
        self.assertEqual(layout[bits[1]], 10)
        self.assertEqual(layout[bits[2]], 6)
        self.assertEqual(layout[bits[3]], 5)
        self.assertEqual(layout[bits[4]], 0)


if __name__ == "__main__":
    unittest.main()

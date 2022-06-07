# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Test Qiskit's Arguments Broadcaster."""

import unittest
import math

from qiskit.test import QiskitTestCase
from qiskit.circuit.library import StatePreparation
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info.operators import Clifford


class TestBroadcast(QiskitTestCase):
    """Testing qiskit.circuit.broadcast"""

    def test_barrier(self):
        """Test adding a barrier to a quantum circuit."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.barrier([0, 1, 2])
        qc.s(1)
        ops = qc.count_ops()
        self.assertIn("barrier", ops.keys())
        self.assertEqual(ops["barrier"], 1)

    def test_delay(self):
        """Test adding a delay to a quantum circuit."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.delay(1)
        qc.s(1)
        ops = qc.count_ops()
        self.assertIn("delay", ops.keys())
        self.assertEqual(ops["delay"], 3)

    def test_gates_1q_1(self):
        """Test the basic way of adding single-qubit gates to a quantum circuit."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.s(1)
        qc.h(2)
        ops = qc.count_ops()
        self.assertIn("h", ops.keys())
        self.assertEqual(ops["h"], 2)
        self.assertIn("s", ops.keys())
        self.assertEqual(ops["s"], 1)

    def test_gates_1q_2(self):
        """Test the alternative way of adding single-qubit gates to a quantum circuit."""
        qc = QuantumCircuit(3)
        qc.h([0, 1])
        qc.s(range(3))
        qc.sdg([0, 2])
        ops = qc.count_ops()
        self.assertIn("h", ops.keys())
        self.assertEqual(ops["h"], 2)
        self.assertIn("s", ops.keys())
        self.assertEqual(ops["s"], 3)
        self.assertIn("sdg", ops.keys())
        self.assertEqual(ops["sdg"], 2)

    def test_gates_2q_1(self):
        """Test the basic way of adding two-qubit gates to a quantum circuit."""
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.swap(1, 2)
        ops = qc.count_ops()
        self.assertIn("cx", ops.keys())
        self.assertEqual(ops["cx"], 1)
        self.assertIn("swap", ops.keys())
        self.assertEqual(ops["swap"], 1)

    def test_gates_2q_2(self):
        """Test an alternative way of adding two-qubit gates to a quantum circuit."""
        qc = QuantumCircuit(4)
        qc.cx(0, [1, 2])
        qc.swap([0, 1, 2], 3)
        qc.cz(3, range(3))
        qc.cz(range(3), 3)
        ops = qc.count_ops()
        self.assertIn("cx", ops.keys())
        self.assertEqual(ops["cx"], 2)
        self.assertIn("swap", ops.keys())
        self.assertEqual(ops["swap"], 3)
        self.assertIn("cz", ops.keys())
        self.assertEqual(ops["cz"], 6)

    def test_gates_2q_3(self):
        """Test another alternative way of adding two-qubit gates to a quantum circuit."""
        qc = QuantumCircuit(4)
        qc.cx([0, 0, 0], [1, 1, 1])
        ops = qc.count_ops()
        self.assertIn("cx", ops.keys())
        self.assertEqual(ops["cx"], 3)

    def test_gates_3q_1(self):
        """Test the basic way of adding three-qubit+ gates to a quantum circuit."""
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)
        ops = qc.count_ops()
        self.assertIn("ccx", ops.keys())
        self.assertEqual(ops["ccx"], 1)

    def test_gates_3q_2(self):
        """Test an alternative way of adding three-qubit+ gates to a quantum circuit."""
        qc = QuantumCircuit(4)
        qc.ccx([0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 1, 2])
        ops = qc.count_ops()
        self.assertIn("ccx", ops.keys())
        self.assertEqual(ops["ccx"], 4)

    def test_measure_1(self):
        """Test adding a measure to a quantum circuit."""
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.s(1)
        qc.cx(0, 2)
        qc.measure([0, 1, 2], [0, 1, 2])
        ops = qc.count_ops()
        self.assertIn("measure", ops.keys())
        self.assertEqual(ops["measure"], 3)

    def test_measure_2(self):
        """Test an alternative way of adding a measure to a quantum circuit."""
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.s(1)
        qc.cx(0, 2)
        qc.measure(0, [0, 1, 2])
        qc.measure([1], [0, 1, 2])
        ops = qc.count_ops()
        self.assertIn("measure", ops.keys())
        self.assertEqual(ops["measure"], 6)

    def test_initializer(self):
        """Test adding an initializer to a quantum circuit."""
        desired_vector_1 = [1.0 / math.sqrt(2), 1.0 / math.sqrt(2)]
        qr = QuantumRegister(1, "qr")
        cr = ClassicalRegister(1, "cr")
        qc = QuantumCircuit(qr, cr)
        qc.initialize(desired_vector_1, [qr[0]])
        ops = qc.count_ops()
        self.assertIn("initialize", ops.keys())
        self.assertEqual(ops["initialize"], 1)

    def test_reset(self):
        """Test adding a reset to a quantum circuit."""
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.reset(0)
        qc.cx(0, 2)
        ops = qc.count_ops()
        self.assertIn("reset", ops.keys())
        self.assertEqual(ops["reset"], 1)

    def test_quantum_circuit(self):
        """Test adding a quantum circuit to a quantum circuit."""
        qc1 = QuantumCircuit(2)
        qc1.cx(0, 1)
        qc1.h(1)
        qc = QuantumCircuit(3)
        qc.append(qc1, [1, 2])
        ops = qc.count_ops()
        self.assertEqual(len(ops), 1)

    def test_clifford(self):
        """Test adding a clifford to a quantum circuit."""
        qc1 = QuantumCircuit(3)
        qc1.h(0)
        qc1.s(1)
        qc1.cx(0, 2)
        cliff1 = Clifford(qc1)
        qc = QuantumCircuit(5)
        qc.append(cliff1, [1, 3, 4])
        ops = qc.count_ops()
        self.assertIn("clifford", ops.keys())
        self.assertEqual(ops["clifford"], 1)

    def test_state_preparation(self):
        """Test adding a state preparation to a quantum circuit."""
        qc = QuantumCircuit(2)
        stateprep = StatePreparation([1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)])
        qc.append(stateprep.inverse().inverse(), [0, 1])
        ops = qc.count_ops()
        self.assertIn("state_preparation", ops.keys())
        self.assertEqual(ops["state_preparation"], 1)


if __name__ == "__main__":
    unittest.main()

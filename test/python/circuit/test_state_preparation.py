# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
StatePreparation test.
"""

import math
import unittest

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.test import QiskitTestCase
from qiskit.converters import circuit_to_dag

class TestStatePreparation(QiskitTestCase):
    """Test initialization with StatePreparation class"""

    def test_prepare_from_label(self):
        """Prepare state from label."""
        desired_sv = Statevector.from_label("01+-lr")
        qc = QuantumCircuit(6)
        qc.prepare_state("01+-lr", range(6))
        actual_sv = Statevector.from_instruction(qc)
        self.assertTrue(desired_sv == actual_sv)

    def test_prepare_from_int(self):
        """Prepare state from int."""
        desired_sv = Statevector.from_label("110101")
        qc = QuantumCircuit(6)
        qc.prepare_state(53, range(6))
        actual_sv = Statevector.from_instruction(qc)
        self.assertTrue(desired_sv == actual_sv)
    
    def test_prepare_from_list(self):
        """Prepare state from list."""
        desired_sv = Statevector([1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)])
        qc = QuantumCircuit(2)
        qc.prepare_state([1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)])
        actual_sv = Statevector.from_instruction(qc)
        self.assertTrue(desired_sv == actual_sv)
    
    def test_decompose_with_int(self):
        """Test prepare_state with int arg decomposes to a StatePreparation and reset"""
        qc = QuantumCircuit(2)
        qc.prepare_state(2)
        decom_circ = qc.decompose()
        dag = circuit_to_dag(decom_circ)

        self.assertEqual(len(dag.op_nodes()), 1)
        self.assertIsNot(dag.op_nodes()[0].name, "reset")
        self.assertEqual(dag.op_nodes()[0].name, "x")

    def test_decompose_with_string(self):
        """Test prepare_state with string arg decomposes to a StatePreparation without resets"""
        qc = QuantumCircuit(2)
        qc.prepare_state("11")
        decom_circ = qc.decompose()
        dag = circuit_to_dag(decom_circ)

        self.assertEqual(len(dag.op_nodes()), 2)
        self.assertIsNot(dag.op_nodes()[0].name, "reset")
        self.assertEqual(dag.op_nodes()[0].name, "x")
        self.assertIsNot(dag.op_nodes()[1].name, "reset")
        self.assertEqual(dag.op_nodes()[1].name, "x")

    def test_decompose_with_statevector(self):
        """Test prepare_state with statevector arg decomposes to a StatePreparation without resets"""
        qc = QuantumCircuit(2)
        qc.prepare_state([1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)])
        decom_circ = qc.decompose()
        dag = circuit_to_dag(decom_circ)

        self.assertEqual(len(dag.op_nodes()), 1)
        self.assertIsNot(dag.op_nodes()[0].name, "reset")
        self.assertEqual(dag.op_nodes()[0].name, "disentangler_dg")

if __name__ == "__main__":
    unittest.main()
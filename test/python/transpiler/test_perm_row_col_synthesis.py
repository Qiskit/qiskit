"""Test PermRowColSynthesis"""

import unittest
import numpy as np
from qiskit.test import QiskitTestCase
from qiskit.transpiler.passes.synthesis.perm_row_col_synthesis import PermRowColSynthesis
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.transpiler import CouplingMap


class TestPermRowColSynthesis(QiskitTestCase):
    """Test PermRowColSynthesis"""

    def test_run_returns_a_dag(self):
        """Test the output type of run"""
        coupling = CouplingMap()
        circuit = QuantumCircuit()
        dag = circuit_to_dag(circuit)
        synthesis = PermRowColSynthesis(coupling)

        instance = synthesis.run(dag)

        self.assertIsInstance(instance, DAGCircuit)

    def test_perm_row_col_returns_circuit(self):
        """Test the output type of perm_row_col"""
        coupling = CouplingMap()
        synthesis = PermRowColSynthesis(coupling)
        parity_mat = np.identity(3)

        instance = synthesis.perm_row_col(parity_mat, coupling)

        self.assertIsInstance(instance, QuantumCircuit)

    def test_choose_row_returns_int(self):
        """Test the output type of choose_row"""
        coupling = CouplingMap()
        synthesis = PermRowColSynthesis(coupling)
        parity_mat = np.array([[1, 0], [0, 1]])
        vertices = np.array([0, 1])

        instance = synthesis.choose_row(vertices, parity_mat)

        self.assertIsInstance(instance, np.int64)

    def test_choose_row_returns_correct_index(self):
        """
        Test method to test the correctness of the choose_row method
        """
        coupling = CouplingMap()
        synthesis = PermRowColSynthesis(coupling)
        parity_mat = np.array(
            [
                [0, 1, 0, 0, 0, 1, 0, 0, 0],
                [1, 0, 1, 0, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0, 1, 0, 1, 0],
                [1, 0, 0, 0, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1, 0, 1],
                [0, 1, 0, 1, 0, 1, 0, 1, 0],
            ]
        )
        vertices = np.array([1, 3, 4, 5, 6, 8])

        index = synthesis.choose_row(vertices, parity_mat)

        self.assertEqual(index, 6)

    def test_choose_column_returns_int(self):
        """Test the output type of choose_column"""
        coupling = CouplingMap()
        synthesis = PermRowColSynthesis(coupling)
        parity_mat = np.ndarray(0)
        cols = np.ndarray(0)

        instance = synthesis.choose_column(parity_mat, cols, 0)

        self.assertIsInstance(instance, int)

    def test_eliminate_column_returns_int(self):
        """Test the output type of eliminate_column"""
        coupling = CouplingMap()
        synthesis = PermRowColSynthesis(coupling)
        parity_mat = np.ndarray(0)
        terminals = np.ndarray(0)

        instance = synthesis.eliminate_column(parity_mat, coupling, 0, terminals)

        self.assertIsInstance(instance, np.ndarray)

    def test_eliminate_row_returns_int(self):
        """Test the output type of eliminate_row"""
        coupling = CouplingMap()
        synthesis = PermRowColSynthesis(coupling)
        parity_mat = np.ndarray(0)
        terminals = np.ndarray(0)

        instance = synthesis.eliminate_row(parity_mat, coupling, 0, terminals)

        self.assertIsInstance(instance, np.ndarray)


if __name__ == "__main__":
    unittest.main()

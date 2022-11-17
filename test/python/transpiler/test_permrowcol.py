"""Test PermRowCol"""

import unittest
import numpy as np
import retworkx as rx

from qiskit.test import QiskitTestCase
from qiskit.transpiler.synthesis.permrowcol import PermRowCol
from qiskit.circuit.library import LinearFunction
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.transpiler import CouplingMap


class TestPermRowCol(QiskitTestCase):
    """Test PermRowCol"""

    def test_perm_row_col_returns_circuit(self):
        """Test the output type of perm_row_col"""
        coupling = CouplingMap()
        permrowcol = PermRowCol(coupling)
        parity_mat = np.identity(3)

        instance = permrowcol.perm_row_col(parity_mat)

        self.assertIsInstance(instance, QuantumCircuit)

    def test_choose_row_returns_np_int64(self):
        """Test the output type of choose_row"""
        coupling = CouplingMap()
        permrowcol = PermRowCol(coupling)
        parity_mat = np.array([[1, 0], [0, 1]])
        vertices = np.array([0, 1])

        instance = permrowcol.choose_row(vertices, parity_mat)

        self.assertIsInstance(instance, np.int64)

    def test_choose_row_returns_correct_index(self):
        """
        Test method to test the correctness of the choose_row method
        """
        coupling = CouplingMap()
        permrowcol = PermRowCol(coupling)
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

        index = permrowcol.choose_row(vertices, parity_mat)

        self.assertEqual(index, 6)

    def test_choose_column_returns_np_int64(self):
        """Test the output type of choose_column"""
        coupling = CouplingMap()
        permrowcol = PermRowCol(coupling)
        parity_mat = np.array([[1, 0], [0, 1]])
        cols = np.array([0, 1])

        instance = permrowcol.choose_column(parity_mat, cols, 0)

        self.assertIsInstance(instance, np.int64)

    def test_choose_column_returns_correct_index(self):
        """Test choose_colum method for correctness"""
        coupling = CouplingMap()
        permrowcol = PermRowCol(coupling)

        parity_mat = np.array(
            [
                [0, 1, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0, 1, 0, 1, 0],
                [1, 0, 0, 0, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1, 0, 1],
                [0, 1, 0, 1, 0, 1, 0, 1, 0],
            ]
        )
        vertices = np.array([1, 3, 4, 5, 6, 7])

        index = permrowcol.choose_column(parity_mat, vertices, 4)

        self.assertEqual(index, 3)

    def test_choose_column_returns_correct_index_with_similar_col_sums(self):
        """Test choose_column method for correctness in case of col_sums having same integers"""
        coupling = CouplingMap()
        permrowcol = PermRowCol(coupling)

        parity_mat = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]])

        vertices = np.array([0, 1, 2])

        index = permrowcol.choose_column(parity_mat, vertices, 2)

        self.assertEqual(index, 2)

    def test_eliminate_column_returns_list(self):
        """Test the output type of eliminate_column"""
        coupling = CouplingMap()
        permrowcol = PermRowCol(coupling)
        parity_mat = np.ndarray(0)
        terminals = np.ndarray(0)

        instance = permrowcol._eliminate_column(parity_mat, 0, 0, terminals)

        self.assertIsInstance(instance, list)

    def test_eliminate_column_returns_correct_list_of_tuples_with_given_input(self):
        """Test eliminate_column method for correctness in case of example parity_matrix and coupling map"""
        coupling_list = [(0, 1), (0, 3), (1, 2), (1, 4), (2, 5), (3, 4), (4, 5)]
        coupling = CouplingMap(coupling_list)
        permrowcol = PermRowCol(coupling)
        parity_mat = np.array(
            [
                [0, 1, 0, 1, 1, 0],
                [1, 1, 1, 1, 1, 0],
                [1, 0, 0, 0, 1, 1],
                [1, 1, 1, 0, 1, 0],
                [1, 0, 1, 0, 1, 0],
                [1, 0, 1, 0, 1, 1],
            ]
        )
        root = 0
        column = 3
        terminals = np.array([1, 0])
        ret = permrowcol._eliminate_column(parity_mat, root, column, terminals)

        self.assertEqual(ret, [(1, 0)])

    def test_eliminate_column_eliminates_selected_column(self):
        """Test eliminate_column for eliminating selected column in case of example parity_matrix and coupling map"""
        coupling_list = [(0, 1), (0, 3), (1, 2), (1, 4), (2, 5), (3, 4), (4, 5)]
        coupling = CouplingMap(coupling_list)
        permrowcol = PermRowCol(coupling)
        parity_mat = np.array(
            [
                [0, 1, 0, 1, 1, 0],
                [1, 1, 1, 1, 1, 0],
                [1, 0, 0, 0, 1, 1],
                [1, 1, 1, 0, 1, 0],
                [1, 0, 1, 0, 1, 0],
                [1, 0, 1, 0, 1, 1],
            ]
        )

        root = 0
        column = 3
        terminals = np.array([1, 0])
        ret = permrowcol._eliminate_column(parity_mat, root, column, terminals)

        self.assertEqual(1, sum(parity_mat[:, column]))
        self.assertEqual(1, parity_mat[0, column])

    def test_eliminate_row_returns_list(self):
        """Test the output type of eliminate_row"""
        coupling = CouplingMap()
        permrowcol = PermRowCol(coupling)
        parity_mat = np.ndarray(0)
        terminals = np.ndarray(0)

        instance = permrowcol._eliminate_row(parity_mat, 0, terminals)

        self.assertIsInstance(instance, list)
        self.assertEqual(instance, [])

    def test_eliminate_row_returns_correct_list_of_tuples_with_given_input(self):
        """Test eliminate_row method for correctness in case of example parity_matrix and coupling map"""
        coupling_list = [(0, 1), (0, 3), (1, 2), (1, 4), (2, 5), (3, 4), (4, 5)]
        coupling = CouplingMap(coupling_list)
        permrowcol = PermRowCol(coupling)
        parity_mat = np.array(
            [
                [0, 1, 0, 1, 1, 0],
                [1, 0, 1, 0, 0, 0],
                [1, 0, 0, 0, 1, 1],
                [1, 1, 1, 0, 1, 0],
                [1, 0, 1, 0, 1, 0],
                [1, 0, 1, 0, 1, 1],
            ]
        )
        root = 0
        terminals = np.array([0, 1, 3])
        ret = permrowcol._eliminate_row(parity_mat, root, terminals)

        self.assertEqual(ret, [(0, 1), (0, 3)])

    def test_eliminate_row__eliminates_selected_row(self):
        """Test eliminate_row method for correctness in case of example parity_matrix and coupling map"""
        coupling_list = [(0, 1), (0, 3), (1, 2), (1, 4), (2, 5), (3, 4), (4, 5)]
        coupling = CouplingMap(coupling_list)
        permrowcol = PermRowCol(coupling)
        parity_mat = np.array(
            [
                [0, 1, 0, 1, 1, 0],
                [1, 0, 1, 0, 0, 0],
                [1, 0, 0, 0, 1, 1],
                [1, 1, 1, 0, 1, 0],
                [1, 0, 1, 0, 1, 0],
                [1, 0, 1, 0, 1, 1],
            ]
        )
        root = 0
        terminals = np.array([0, 1, 3])
        ret = permrowcol._eliminate_row(parity_mat, root, terminals)

        self.assertEqual(1, sum(parity_mat[0]))
        self.assertEqual(1, parity_mat[0, 3])

    def test_eliminate_row__eliminates_selected_row_2(self):
        """Test eliminate_row method for correctness in case of example parity_matrix and coupling map"""
        coupling_list = [(1, 2), (1, 4), (2, 5), (3, 4), (4, 5)]
        coupling = CouplingMap(coupling_list)
        permrowcol = PermRowCol(coupling)
        parity_mat = np.array(
            [
                [0, 0, 0, 1, 0, 0],
                [1, 0, 1, 0, 0, 0],
                [1, 0, 0, 0, 1, 1],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
        root = 1
        terminals = np.array([1, 2, 4, 5])
        ret = permrowcol._eliminate_row(parity_mat, root, terminals)
        self.assertEqual(1, sum(parity_mat[1]))
        self.assertEqual(1, parity_mat[1, 2])

<<<<<<< HEAD
    def test_if_matrix_edit_returns_circuit(self):
        """Tests if matrix_edit returns circuit"""

        coupling = CouplingMap()
        permrowcol = PermRowCol(coupling)
        parity_mat = np.identity(3)
        chosen_column = 1
        chosen_row = 1
        circuit = QuantumCircuit(QuantumRegister(0))
        instance = permrowcol._matrix_edit(parity_mat, chosen_column, chosen_row, circuit)

        self.assertIsInstance(instance, QuantumCircuit)

    def test_matrix_edit_returns_circuit_with_eliminated_row_if_the_row_is_not_already_eliminated(
        self,
    ):
        """Test matrix_edit for correctnes"""

        coupling_list = [(0, 1), (0, 3), (1, 2), (1, 4), (2, 5), (3, 4), (4, 5)]
        coupling = CouplingMap(coupling_list)
        permrowcol = PermRowCol(coupling)
        parity_mat = np.array(
            [
                [0, 1, 0, 1, 1, 0],
                [1, 0, 1, 0, 0, 0],
                [1, 0, 0, 0, 1, 1],
                [1, 1, 1, 0, 1, 0],
                [1, 0, 1, 0, 1, 0],
                [1, 0, 1, 0, 1, 1],
            ]
        )
        chosen_column = 3
        chosen_row = 0
        circuit = QuantumCircuit(len(coupling.graph))
        edge = (1, 0)
        circuit.cx(edge[0], edge[1])

        instance = permrowcol._matrix_edit(parity_mat, chosen_column, chosen_row, circuit)

        result_circuit = [
            [True, False, False, True, False, False],
            [False, True, False, False, False, False],
            [False, False, True, False, False, False],
            [False, False, False, True, False, False],
            [False, False, False, False, True, False],
            [False, False, False, False, False, True],
        ]

        result_parity_matrix = [
            [0, 0, 0, 1, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 1, 1],
            [1, 1, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 1],
        ]

        self.assertEqual(np.array_equal(LinearFunction(instance).linear, result_circuit), True)
        self.assertEqual(np.array_equal(parity_mat, result_parity_matrix), True)
=======
    def test_return_columns_return_list(self):
        """Test the output type of return_columns"""
        coupling = CouplingMap()
        permrowcol = PermRowCol(coupling)

        instance = permrowcol._return_columns([-1, -1, -1])

        self.assertIsInstance(instance, list)

    def test_return_columns_return_list_of_indices(self):
        """Test the correctness of return_columns output"""
        coupling = CouplingMap()
        permrowcol = PermRowCol(coupling)

        instance = permrowcol._return_columns([-1, -1, -1])

        self.assertCountEqual(instance, [0, 1, 2])

        instance = permrowcol._return_columns([-2, -1, 0, 1, 2, -1])
        self.assertCountEqual(instance, [1, 5])

        instance = permrowcol._return_columns([])
        self.assertCountEqual(instance, [])

        instance = permrowcol._return_columns([1, 2, 3, 4, 5, 6])
        self.assertCountEqual(instance, [])

    def test_get_nodes_returns_list(self):
        """Test the output type of get_nodes"""
        coupling = CouplingMap()
        coupling.add_physical_qubit(0)
        coupling.add_physical_qubit(1)
        coupling.add_physical_qubit(2)
        permrowcol = PermRowCol(coupling)

        instance = permrowcol._get_nodes(np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]]), 0)
        self.assertIsInstance(instance, list)

    def test_get_nodes_returns_correct_list(self):
        """Test the correctness of the get_nodes"""
        coupling = CouplingMap()
        coupling.add_physical_qubit(0)
        coupling.add_physical_qubit(1)
        coupling.add_physical_qubit(2)
        permrowcol = PermRowCol(coupling)

        instance = permrowcol._get_nodes(np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]]), 2)
        self.assertCountEqual(instance, [0, 1, 2])

        instance = permrowcol._get_nodes(np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]]), 1)
        self.assertCountEqual(instance, [1])

        instance = permrowcol._get_nodes(np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]]), 0)
        self.assertCountEqual(instance, [0])

        permrowcol._reduce_graph(1)

        instance = permrowcol._get_nodes(np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]]), 2)
        self.assertCountEqual(instance, [0, 2])

        permrowcol._reduce_graph(0)

        instance = permrowcol._get_nodes(np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]]), 0)
        self.assertCountEqual(instance, [])

    def test_reduce_graph_reduces_graph(self):
        """Test that reduce_graph reduces the graph"""
        coupling = CouplingMap([[0, 1], [0, 2], [1, 2], [0, 3]])
        permrowcol = PermRowCol(coupling)

        permrowcol._reduce_graph(0)

        self.assertEqual(len(permrowcol._graph.node_indexes()), 3)

    def test_reduce_graph_removes_correct_node(self):
        """Test reduce_graph removes correct node"""
        coupling = CouplingMap([[0, 1], [0, 2], [1, 2], [0, 3]])
        permrowcol = PermRowCol(coupling)

        permrowcol._reduce_graph(0)
        self.assertCountEqual(permrowcol._graph.node_indexes(), [1, 2, 3])

        permrowcol._reduce_graph(2)
        self.assertCountEqual(permrowcol._graph.node_indexes(), [1, 3])

        permrowcol._reduce_graph(1)
        self.assertCountEqual(permrowcol._graph.node_indexes(), [3])

        permrowcol._reduce_graph(3)
        self.assertCountEqual(permrowcol._graph.node_indexes(), [])

    def test_reduce_graph_does_not_change_graph_with_wrong_index(self):
        """Test that graph does not change when reduce_graph uses an
        index that does not exist"""
        coupling = CouplingMap([[0, 1], [0, 2], [1, 2], [0, 3]])
        permrowcol = PermRowCol(coupling)

        permrowcol._reduce_graph(4)
        self.assertCountEqual(permrowcol._graph.node_indexes(), [0, 1, 2, 3])

    def test_reduce_graph_removes_edges_from_graph(self):
        """Test that reduce graph removes edges from the graph"""
        coupling = CouplingMap([[0, 1], [0, 2], [1, 2], [0, 3]])
        permrowcol = PermRowCol(coupling)

        permrowcol._reduce_graph(3)
        self.assertCountEqual(permrowcol._graph.edge_list(), [(0, 1), (0, 2), (1, 2)])

        permrowcol._reduce_graph(0)
        self.assertCountEqual(permrowcol._graph.edge_list(), [(1, 2)])

        permrowcol._reduce_graph(2)
        self.assertCountEqual(permrowcol._graph.edge_list(), [])
>>>>>>> perm-row-col


if __name__ == "__main__":
    unittest.main()

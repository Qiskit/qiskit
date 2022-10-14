"""Test PermRowCol"""

import unittest
import numpy as np

from qiskit.test import QiskitTestCase
from qiskit.transpiler.synthesis.permrowcol import PermRowCol
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
import retworkx as rx



class TestPermRowCol(QiskitTestCase):
    """Test PermRowCol"""

    def test_perm_row_col_returns_circuit(self):
        """Test the output type of perm_row_col"""
        coupling = CouplingMap()
        permrowcol = PermRowCol(coupling)
        parity_mat = np.identity(3)

        instance = permrowcol.perm_row_col(parity_mat, coupling)

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

    def test_eliminate_column_returns_np_ndarray(self):
        """Test the output type of eliminate_column"""
        coupling = CouplingMap()
        permrowcol = PermRowCol(coupling)
        parity_mat = np.ndarray(0)
        terminals = np.ndarray(0)

        instance = permrowcol.eliminate_column(parity_mat, coupling, 0,0, terminals)

        self.assertIsInstance(instance, np.ndarray)

    def test_eliminate_column_returns_correct_list_of_tuples_with_given_input(self):
        """Test eliminate_column method for correctness in case of example parity_matrix and couplingmap"""
        coupling_list = [(0, 1),(0, 3),(1, 2),(1, 4),(2, 5),(3, 4),(4, 5)]
        coupling = CouplingMap(coupling_list)
        permrowcol = PermRowCol(coupling)
        parity_mat = np.array(
        [
            [0, 1, 0, 1, 1, 0],
            [1, 1, 1, 1, 1, 0],
            [1, 0, 0, 0, 1, 1],
            [1, 1, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 1]
        ])

        terminals = np.array([1,0])
        ret = permrowcol.eliminate_column(parity_mat, coupling, 0,3, terminals)

        self.assertEqual(ret.all(), np.array([1,0]).all())

    # def test_eliminate_column_eliminates_selected:column:(self):
    #     """Test eliminate_column method for correctness in case of example parity_matrix and couplingmap"""
    #     coupling_list = [(0, 1),(0, 3),(1, 2),(1, 4),(2, 5),(3, 4),(4, 5)]
    #     coupling = CouplingMap(coupling_list)
    #     permrowcol = PermRowCol(coupling)
    #     parity_mat = np.array(
    #     [
    #         [0, 1, 0, 1, 1, 0],
    #         [1, 1, 1, 1, 1, 0],
    #         [1, 0, 0, 0, 1, 1],
    #         [1, 1, 1, 0, 1, 0],
    #         [1, 0, 1, 0, 1, 0],
    #         [1, 0, 1, 0, 1, 1]
    #     ])

    #     terminals = np.array([1,0])
    #     ret = permrowcol.eliminate_column(parity_mat, coupling, 0,3, terminals)

    #     self.assertEqual(ret.all(), np.array([1,0]).all())

    def test_eliminate_row_returns_np_ndarray(self):
        """Test the output type of eliminate_row"""
        coupling = CouplingMap()
        permrowcol = PermRowCol(coupling)
        parity_mat = np.ndarray(0)
        terminals = np.ndarray(0)

        instance = permrowcol.eliminate_row(parity_mat, coupling, 0, terminals)

        self.assertIsInstance(instance, np.ndarray)


if __name__ == "__main__":
    unittest.main()

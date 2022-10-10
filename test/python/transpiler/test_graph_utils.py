"""Test graph utils"""

import unittest
import numpy as np
import retworkx as rx

from qiskit.test import QiskitTestCase
from qiskit.transpiler.synthesis.graph_utils import noncutting_vertices, pydigraph_to_pygraph
from qiskit.transpiler import CouplingMap


class TestGraphUtils(QiskitTestCase):
    """Test graph utils"""

    def test_noncutting_vertices_returns_np_ndarray(self):
        """Test the output type of noncutting_vertices"""
        coupling = CouplingMap()

        instance = noncutting_vertices(coupling)

        self.assertIsInstance(instance, np.ndarray)

    def test_noncutting_vertices_returns_an_ndarray_with_noncutting_vertices(self):
        """Test noncutting_vertices method for correctness"""
        coupling_list = [[0, 2], [1, 2], [2, 3], [2, 4], [3, 6], [4, 5], [4, 6]]
        coupling = CouplingMap(couplinglist=coupling_list)

        instance = noncutting_vertices(coupling)
        expected = np.array([0, 1, 3, 5, 6])

        self.assertCountEqual(instance, expected)

    def test_pydigraph_to_pygraph_returns_pygraph(self):
        """Test the output type of pydigraph_to_pygraph"""
        coupling = CouplingMap()

        instance = pydigraph_to_pygraph(coupling.graph)

        self.assertIsInstance(instance, rx.PyGraph)


if __name__ == "__main__":
    unittest.main()

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

"""Test library of graph state circuits."""

import unittest

from qiskit.test.base import QiskitTestCase
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.library import GraphState
from qiskit.quantum_info import Clifford


class TestGraphStateLibrary(QiskitTestCase):
    """Test the graph state circuit."""

    def assertGraphStateIsCorrect(self, adjacency_matrix, graph_state):
        """Check the stabilizers of the graph state against the expected stabilizers.
        Based on https://arxiv.org/pdf/quant-ph/0307130.pdf, Eq. (6).
        """

        stabilizers = Clifford(graph_state).stabilizer.pauli.to_labels()

        expected_stabilizers = []  # keep track of all expected stabilizers
        num_vertices = len(adjacency_matrix)
        for vertex_a in range(num_vertices):
            stabilizer = [None] * num_vertices  # Paulis must be put into right place
            for vertex_b in range(num_vertices):
                if vertex_a == vertex_b:  # self-connection --> 'X'
                    stabilizer[vertex_a] = "X"
                elif adjacency_matrix[vertex_a][vertex_b] != 0:  # vertices connected --> 'Z'
                    stabilizer[vertex_b] = "Z"
                else:  # else --> 'I'
                    stabilizer[vertex_b] = "I"

            # need to reverse for Qiskit's tensoring order
            expected_stabilizers.append("".join(stabilizer)[::-1])

        self.assertListEqual(expected_stabilizers, stabilizers)

    def test_graph_state(self):
        """Verify the GraphState by checking if the circuit has the expected stabilizers."""
        adjacency_matrix = [
            [0, 1, 0, 0, 1],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [1, 0, 0, 1, 0],
        ]
        graph_state = GraphState(adjacency_matrix)
        self.assertGraphStateIsCorrect(adjacency_matrix, graph_state)

    def test_non_symmetric_raises(self):
        """Test that adjacency matrix is required to be symmetric."""
        adjacency_matrix = [[1, 1, 0], [1, 0, 1], [1, 1, 1]]
        with self.assertRaises(CircuitError):
            GraphState(adjacency_matrix)


if __name__ == "__main__":
    unittest.main()

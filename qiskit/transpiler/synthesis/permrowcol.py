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

"""Permrowcol-algorithm functionality implementation"""

import numpy as np
import retworkx as rx

from qiskit.transpiler import CouplingMap
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.transpiler.synthesis.graph_utils import (
    postorder_traversal,
    preorder_traversal,
    pydigraph_to_pygraph,
    noncutting_vertices,
)
from qiskit.circuit.library import LinearFunction


class PermRowCol:
    """Permrowcol algorithm"""

    def __init__(self, coupling_map: CouplingMap):
        self._coupling_map = coupling_map
        self._graph = pydigraph_to_pygraph(self._coupling_map.graph)

    def perm_row_col(self, parity_mat: np.ndarray) -> QuantumCircuit:
        """Run permrowcol algorithm on the given parity matrix

        Args:
            parity_mat (np.ndarray): parity matrix representing a circuit

        Returns:
            QuantumCircuit: synthesized circuit
        """
        qubit_alloc = [-1] * len(self._graph.node_indexes())

        circuit = QuantumCircuit(len(self._graph.node_indexes()))

        while len(self._graph.node_indexes()) > 1:
            n_vertices = noncutting_vertices(self._graph)
            row = self.choose_row(n_vertices, parity_mat)

            cols = self._return_columns(qubit_alloc)
            column = self.choose_column(parity_mat, cols, row)
            nodes = self._get_nodes(parity_mat, column)
            for edge in self._eliminate_column(parity_mat, row, column, nodes):
                circuit.cx(edge[0], edge[1])

            if sum(parity_mat[row]) > 1:
                A = np.delete(np.delete(parity_mat, row, 0), column, 1)
                B = np.delete(parity_mat, column, 1)[row]
                # X = inv(A) * B
                # Eliminate row will go here

            qubit_alloc[column] = row

            self._reduce_graph(column)

        if len(qubit_alloc) != 0:
            qubit_alloc[qubit_alloc.index(-1)] = self._graph.node_indexes()[0]

        return circuit  # Supposed to also return qubit_alloc?

    def _reduce_graph(self, node: int):
        """Removes a node from pydigraph

        Args:
            node (int): index of node to remove
        """
        self._graph.remove_node(node)

    def _get_nodes(self, parity_mat: np.ndarray, column: int) -> list:
        """Returns a list of nodes that have 1s in the chosen column in the parity matrix

        Args:
            parity_mat (np.ndarray): parity matrix
            column (int): column index

        Returns:
            list: list of nodes

        """
        return [node for node in self._graph.node_indexes() if parity_mat[node, column] == 1]

    def _return_columns(self, qubit_alloc: list) -> list:
        """Returns list of indices of not yet processed columns in parity matrix

        Args:
            qubit_alloc (list): qubit allocation list

        Returns:
            list: list of indices of yet to be processed columns in the parity matrix

        """
        return [i for i in range(len(qubit_alloc)) if qubit_alloc[i] == -1]

    def choose_row(self, vertices: np.ndarray, parity_mat: np.ndarray) -> np.int64:
        """Choose row to eliminate and return the index.

        Args:
            vertices (np.ndarray): vertices (corresponding to rows) to choose from
            parity_mat (np.ndarray): parity matrix

        Returns:
            int: vertex/row index
        """
        return vertices[np.argmin([sum(parity_mat[i]) for i in vertices])]

    def choose_column(self, parity_mat: np.ndarray, cols: np.ndarray, chosen_row: int) -> np.int64:
        """Choose column to eliminate and return the index.

        Args:
            parity_mat (np.ndarray): parity matrix
            cols (np.ndarray): column indices to choose from
            chosen_row (int): row index that has been eliminated

        Returns:
            int: column index
        """
        col_sum = [
            sum(parity_mat[:, i]) if parity_mat[chosen_row][i] == 1 else len(parity_mat) + 1
            for i in cols
        ]
        return cols[np.argmin(col_sum)]

    def _eliminate_column(
        self,
        parity_mat: np.ndarray,
        root: int,
        col: int,
        terminals: np.ndarray,
    ) -> list:
        """Eliminates the selected column from the parity matrix and returns the operations.

        Args:
            parity_mat (np.ndarray): parity matrix
            coupling (CouplingMap): topology
            root (int): root of the steiner tree
            terminals (np.ndarray): terminals of the steiner tree

        Returns:
            list: list of tuples represents control and target qubits with a cnot gate between them.
        """
        C = []
        tree = rx.steiner_tree(self._graph, terminals, weight_fn=lambda x: 1)
        post_edges = []
        postorder_traversal(tree, root, post_edges)

        for edge in post_edges:
            if parity_mat[edge[0], col] == 0:
                C.append((edge[0], edge[1]))
                parity_mat[edge[0], :] = (parity_mat[edge[0], :] + parity_mat[edge[1], :]) % 2

        for edge in post_edges:
            C.append((edge[1], edge[0]))
            parity_mat[edge[1], :] = (parity_mat[edge[0], :] + parity_mat[edge[1], :]) % 2

        return C

    def _eliminate_row(self, parity_mat: np.ndarray, root: int, terminals: np.ndarray) -> list:
        """Eliminates the selected row from the parity matrix and returns the operations as a list of tuples.

        Args:
            parity_mat (np.ndarray): parity matrix
            coupling (CouplingMap): topology
            root (int): root of the steiner tree
            terminals (np.ndarray): terminals of the steiner tree

        Returns:
            list of tuples represents control and target qubits with a cnot gate between them.
        """
        C = []
        tree = rx.steiner_tree(self._graph, terminals, weight_fn=lambda x: 1)

        pre_edges = []
        preorder_traversal(tree, root, pre_edges)
        post_edges = []
        postorder_traversal(tree, root, post_edges)

        for edge in pre_edges:

            if edge[1] not in terminals:
                C.append((edge[0], edge[1]))
                parity_mat[edge[0], :] = (parity_mat[edge[0], :] + parity_mat[edge[1], :]) % 2

        for edge in post_edges:
            C.append((edge[0], edge[1]))
            parity_mat[edge[0], :] = (parity_mat[edge[0], :] + parity_mat[edge[1], :]) % 2

        return C

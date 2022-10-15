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
from qiskit.transpiler.synthesis.graph_utils import postorder_traversal, pydigraph_to_pygraph


class PermRowCol:
    """Permrowcol algorithm"""

    def __init__(self, coupling_map: CouplingMap):
        self._coupling_map = coupling_map

    def perm_row_col(self, parity_mat: np.ndarray, coupling_map: CouplingMap) -> QuantumCircuit:
        """Run permrowcol algorithm on the given parity matrix

        Args:
            parity_mat (np.ndarray): parity matrix representing a circuit
            coupling_map (CouplingMap): topology constraint

        Returns:
            QuantumCircuit: synthesized circuit
        """
        # TODO
        circuit = QuantumCircuit(QuantumRegister(0))
        return circuit

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

    def eliminate_column(
        self,
        parity_mat: np.ndarray,
        coupling: CouplingMap,
        root: int,
        col: int,
        terminals: np.ndarray,
    ) -> np.ndarray:
        """Eliminates the selected column from the parity matrix and returns the operations.

        Args:
            parity_mat (np.ndarray): parity matrix
            coupling (CouplingMap): topology
            root (int): root of the steiner tree
            col (int): selected column to eliminate
            terminals (np.ndarray): terminals of the steiner tree

        Returns:
            np.ndarray: list of operations
        """
        C = []
        tree = rx.steiner_tree(
            pydigraph_to_pygraph(coupling.graph), terminals, weight_fn=lambda x: 1
        )
        post_edges = []
        postorder_traversal(tree, root, post_edges)

        for edge in post_edges:
            if parity_mat[edge[0], col] == 0:
                C.append((edge[0], edge[1]))
                parity_mat[edge[0], :] = (parity_mat[edge[0], :] + parity_mat[edge[1], :]) % 2

        for edge in post_edges:
            C.append((edge[1], edge[0]))
            parity_mat[edge[1], :] = (parity_mat[edge[0], :] + parity_mat[edge[1], :]) % 2

        return np.array(C, dtype=object)

    def eliminate_row(
        self, parity_mat: np.ndarray, coupling: CouplingMap, root: int, terminals: np.ndarray
    ) -> np.ndarray:
        """Eliminates the selected row from the parity matrix and returns the operations.

        Args:
            parity_mat (np.ndarray): parity matrix
            coupling (CouplingMap): topology
            root (int): root of the steiner tree
            terminals (np.ndarray): terminals of the steiner tree

        Returns:
            np.ndarray: list of operations
        """
        return np.ndarray(0)

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

"""Perm_row_col function implementation"""

import numpy as np
from qiskit.transpiler.passes.synthesis.high_level_synthesis import HighLevelSynthesis
from qiskit.circuit.library.generalized_gates.linear_function import LinearFunction
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.transpiler import CouplingMap
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit import QuantumRegister, QuantumCircuit


class PermRowColSynthesis(HighLevelSynthesis):
    """Synthesize high-level objects by using permrowcol algorithm"""

    def __init__(self, coupling_map: CouplingMap):
        self._coupling_map = coupling_map

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Perform the synthesization and return the synthesized
        circuit as a dag

        Args:
            dag (DAGCircuit): dag circuit to re-synthesize

        Returns:
            DAGCircuit: re-synthesized dag circuit
        """
        for node in dag.named_nodes("cx"):
            # TODO: do something to the nodes
            pass

        # alt parity matrix of dag circuit, dtype=bool
        # parity_mat = LinearFunction(dag_to_circuit(dag)).linear

        parity_mat = np.identity(3)
        res_circuit = self.perm_row_col(parity_mat, self._coupling_map)
        return circuit_to_dag(res_circuit)

    def perm_row_col(self, parity_mat: np.ndarray, coupling_map: CouplingMap) -> QuantumCircuit:
        """Run permrowcol algorithm on the given parity matrix

        Args:
            parity_mat (np.ndarray): parity matrix representing a circuit
            coupling_map (CouplingMap): topology constraint

        Returns:
            QuantumCircuit: synthesized circuit
        """
        circuit = QuantumCircuit(QuantumRegister(0))
        return circuit

    def choose_row(self, vertices: np.ndarray, parity_mat: np.ndarray) -> int:
        """Choose row to eliminate and return the index.

        Args:
            vertices (np.ndarray): vertices (corresponding to rows) to choose from
            parity_mat (np.ndarray): parity matrix

        Returns:
            int: vertex/row index
        """
        return vertices[np.argmin([sum(parity_mat[i]) for i in vertices])]

    def choose_column(self, parity_mat: np.ndarray, cols: np.ndarray, chosen_row: int) -> int:
        """Choose column to eliminate and return the index.

        Args:
            parity_mat (np.ndarray): parity matrix
            cols (np.ndarray): column indices to choose from
            chosen_row (int): row index that has been eliminated

        Returns:
            int: column index
        """
        return 0

    def eliminate_column(
        self, parity_mat: np.ndarray, coupling: CouplingMap, root: int, terminals: np.ndarray
    ) -> np.ndarray:
        """Eliminates the selected column from the parity matrix and returns the operations.

        Args:
            parity_mat (np.ndarray): parity matrix
            coupling (CouplingMap): topology
            root (int): root of the steiner tree
            terminals (np.ndarray): terminals of the steiner tree

        Returns:
            np.ndarray: list of operations
        """
        return np.ndarray(0)

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

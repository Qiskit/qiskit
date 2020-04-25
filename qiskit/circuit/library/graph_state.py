# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=no-member

"""Graph State circuit."""

from typing import Union, List

import numpy as np
from qiskit import QuantumCircuit


class GraphState(QuantumCircuit):
    r"""Graph state circuit.

    The circuit prepares a graph state with the given adjacency
    matrix and measures the state in the product basis specified
    by the list of measurement angles. The angles specify the
    theta, phi, and lambda parameters of u3 gates acting before
    each measurement.
    """

    def __init__(self,
                 adjacency_matrix: Union[List, np.array]) -> None:
        """Make graph state and measure in product basis.

        Args:
            adjacency_matrix: input graph as n-by-n list of 0-1 lists

        The circuit prepares a graph state with the given adjacency
        matrix and measures the state in the product basis specified
        by the list of measurement angles. The angles specify the
        theta, phi, and lambda parameters of u3 gates acting before
        each measurement.

        Reference Circuit:
            .. jupyter-execute::
                :hide-code:

                from qiskit.circuit.library import GraphState
                import qiskit.tools.jupyter
                import networkx as nx
                G = nx.Graph()
                G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)])
                adjmat = nx.adjacency_matrix(G)
                circuit = GraphState(adjmat.toarray())
                %circuit_library_info circuit
        """
        num_qubits = len(adjacency_matrix)
        super().__init__(num_qubits, name=f"graph: %s" % (adjacency_matrix)

        self.h(range(num_qubits))
        for i in range(num_qubits):
            for j in range(i+1, num_qubits):
                if adjacency_matrix[i][j] == 1:
                    self.cz(i, j)
        for i in range(num_qubits):
            self.u3(measurement_angles[3*i],
                    measurement_angles[3*i+1],
                    measurement_angles[3*i+2],
                    i)

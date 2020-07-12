# -*- coding: utf-8 -*-

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

# pylint: disable=no-member

"""Graph State circuit."""

from typing import Union, List

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.exceptions import CircuitError


class GraphState(QuantumCircuit):
    r"""Circuit to prepare a graph state.

    Given a graph G = (V, E), with the set of vertices V and the set of edges E,
    the corresponding graph state is defined as

    .. math::

        |G\rangle = \prod_{(a,b) \in E} CZ_{(a,b)} {|+\rangle}^{\otimes V}

    Such a state can be prepared by first preparing all qubits in the :math:`+`
    state, then applying a :math:`CZ` gate for each corresponding graph edge.

    Graph state preparation circuits are Clifford circuits, and thus
    easy to simulate classically. However, by adding a layer of measurements
    in a product basis at the end, there is evidence that the circuit becomes
    hard to simulate [2].

    **Reference Circuit:**

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

    **References:**

    [1] M. Hein, J. Eisert, H.J. Briegel, Multi-party Entanglement in Graph States,
        `arXiv:0307130 <https://arxiv.org/pdf/quant-ph/0307130.pdf>`_
    [2] D. Koh, Further Extensions of Clifford Circuits & their Classical Simulation Complexities.
        `arXiv:1512.07892 <https://arxiv.org/pdf/1512.07892.pdf>`_
    """

    def __init__(self,
                 adjacency_matrix: Union[List, np.array]) -> None:
        """Create graph state preparation circuit.

        Args:
            adjacency_matrix: input graph as n-by-n list of 0-1 lists

        Raises:
            CircuitError: If adjacency_matrix is not symmetric.

        The circuit prepares a graph state with the given adjacency
        matrix.
        """

        adjacency_matrix = np.asarray(adjacency_matrix)
        if not np.allclose(adjacency_matrix, adjacency_matrix.transpose()):
            raise CircuitError("The adjacency matrix must be symmetric.")

        num_qubits = len(adjacency_matrix)
        super().__init__(num_qubits, name="graph: %s" % (adjacency_matrix))

        self.h(range(num_qubits))
        for i in range(num_qubits):
            for j in range(i+1, num_qubits):
                if adjacency_matrix[i][j] == 1:
                    self.cz(i, j)

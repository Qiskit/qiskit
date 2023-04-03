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

"""Graph State circuit."""

from __future__ import annotations

import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
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

    .. plot::

       from qiskit.circuit.library import GraphState
       from qiskit.tools.jupyter.library import _generate_circuit_library_visualization
       import rustworkx as rx
       G = rx.generators.cycle_graph(5)
       circuit = GraphState(rx.adjacency_matrix(G))
       _generate_circuit_library_visualization(circuit)

    **References:**

    [1] M. Hein, J. Eisert, H.J. Briegel, Multi-party Entanglement in Graph States,
        `arXiv:0307130 <https://arxiv.org/pdf/quant-ph/0307130.pdf>`_
    [2] D. Koh, Further Extensions of Clifford Circuits & their Classical Simulation Complexities.
        `arXiv:1512.07892 <https://arxiv.org/pdf/1512.07892.pdf>`_
    """

    def __init__(self, adjacency_matrix: list | np.ndarray) -> None:
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
        circuit = QuantumCircuit(num_qubits, name="graph: %s" % (adjacency_matrix))

        circuit.h(range(num_qubits))
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                if adjacency_matrix[i][j] == 1:
                    circuit.cz(i, j)

        super().__init__(*circuit.qregs, name=circuit.name)
        self.compose(circuit.to_gate(), qubits=self.qubits, inplace=True)

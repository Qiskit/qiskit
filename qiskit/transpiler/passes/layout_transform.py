# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Map (with minimum effort) a DAGCircuit onto a `coupling_map` adding swap gates."""

from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.routing import util
from qiskit.transpiler.routing.general import ApproximateTokenSwapper


class LayoutTransformation(TransformationPass):
    """ Adds a Swap circuit for a given (partial) permutation to the circuit.

    This circuit is found by a 4-approximation algorithm for Token Swapping.
    More details are available in the routing code.
    """

    def __init__(self, coupling_map, initial_layout, final_layout, trials=4):
        """LayoutTransformation initializer.

        Args:
            coupling_map (CouplingMap): Directed graph represented a coupling map.
            initial_layout (Layout): The starting layout of qubits onto physical qubits.
            final_layout (Layout): The final layout of qubits on phyiscal qubits.
            trials (int): How many randomized trials to perform, taking the best circuit as output.
        """
        super().__init__()
        self.coupling_map = coupling_map
        self.initial_layout = initial_layout
        self.final_layout = final_layout
        graph = coupling_map.graph.to_undirected
        token_swapper = ApproximateTokenSwapper(graph)

        # Find the permutation that between the initial physical qubits and final physical qubits.
        permutation = {pqubit: final_layout.get_virtual_bits()[vqubit]
                       for vqubit, pqubit in initial_layout.get_virtual_bits().items()}
        swaps = token_swapper.map(permutation, trials)
        # None of the swaps are guaranteed to be disjoint so we perform one swap every layer.
        parallel_swaps = [[swap] for swap in swaps]
        self.permutation_circuit = util.circuit(parallel_swaps)

    def run(self, dag):
        """Apply the specified partial permutation to the circuit.

        Args:
            dag (DAGCircuit): DAG to transform the layout of.

        Returns:
            DAGCircuit: The DAG with transformed layout.

        Raises:
            TranspilerError: if the coupling map or the layout are not
            compatible with the DAG.
        """
        if len(dag.qregs) != 1 or dag.qregs.get('q', None) is None:
            raise TranspilerError('LayoutTransform runs on physical circuits only')

        if len(dag.qubits()) > len(self.coupling_map.physical_qubits):
            raise TranspilerError('The layout does not match the amount of qubits in the DAG')

        edge_map = {self.initial_layout.get_physical_bits()[pqubit]: vqubit
                    for (pqubit, vqubit) in self.permutation_circuit.inputmap}
        return dag.extend_back(self.permutation_circuit.circuit, edge_map=edge_map)

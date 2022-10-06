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

"""Utility functions for routing"""

from qiskit.transpiler import CouplingMap
from qiskit.transpiler.exceptions import TranspilerError
from .algorithms import ApproximateTokenSwapper


def combine_permutations(*permutations):
    """
    Chain a series of permutations.

    Args:
        *permutations (list(int)): permutations to combine

    Returns:
        list: combined permutation
    """
    order = permutations[0]
    for this_order in permutations[1:]:
        order = [order[i] for i in this_order]
    return order


def get_swap_map_dag(dag, coupling_map, from_layout, to_layout, seed, trials=4):
    """Get the circuit of swaps to go from from_layout to to_layout, and the qubit ordering of the
    qubits in that circuit."""

    if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
        raise TranspilerError("layout transformation runs on physical circuits only")

    if len(dag.qubits) > len(coupling_map.physical_qubits):
        raise TranspilerError("The layout does not match the amount of qubits in the DAG")

    if coupling_map:
        graph = coupling_map.graph.to_undirected()
    else:
        coupling_map = CouplingMap.from_full(len(to_layout))
        graph = coupling_map.graph.to_undirected()

    token_swapper = ApproximateTokenSwapper(graph, seed)
    # Find the permutation between the initial physical qubits and final physical qubits.
    permutation = {
        pqubit: to_layout.get_virtual_bits()[vqubit]
        for vqubit, pqubit in from_layout.get_virtual_bits().items()
    }
    permutation_circ = token_swapper.permutation_circuit(permutation, trials)
    permutation_qubits = [dag.qubits[i] for i in sorted(permutation_circ.inputmap.keys())]
    return permutation_circ.circuit, permutation_qubits

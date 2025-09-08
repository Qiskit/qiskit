# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""This module contains common utils for vf2 layout passes."""

__all__ = ["ErrorMap"]

from collections import defaultdict

from rustworkx import PyDiGraph, PyGraph, connected_components, weakly_connected_components

from qiskit._accelerate.error_map import ErrorMap
from qiskit.circuit import ForLoopOp
from qiskit.converters import circuit_to_dag


# This function is (unfortunately) used by non-VF2 places in Qiskit.
def build_interaction_graph(dag, strict_direction=True):
    """Build an interaction graph from a dag."""
    im_graph = PyDiGraph(multigraph=False) if strict_direction else PyGraph(multigraph=False)
    im_graph_node_map = {}
    reverse_im_graph_node_map = {}

    class MultiQEncountered(Exception):
        """Used to singal an error-status return from the DAG visitor."""

    def _visit(dag, weight, wire_map):
        for node in dag.op_nodes(include_directives=False):
            if node.is_control_flow():
                if isinstance(node.op, ForLoopOp):
                    inner_weight = len(node.op.params[0]) * weight
                else:
                    inner_weight = weight
                for block in node.op.blocks:
                    inner_wire_map = {
                        inner: wire_map[outer] for outer, inner in zip(node.qargs, block.qubits)
                    }
                    _visit(circuit_to_dag(block), inner_weight, inner_wire_map)
                continue
            len_args = len(node.qargs)
            qargs = [wire_map[q] for q in node.qargs]
            if len_args == 1:
                if qargs[0] not in im_graph_node_map:
                    weights = defaultdict(int)
                    weights[node.name] += weight
                    im_graph_node_map[qargs[0]] = im_graph.add_node(weights)
                    reverse_im_graph_node_map[im_graph_node_map[qargs[0]]] = qargs[0]
                else:
                    im_graph[im_graph_node_map[qargs[0]]][node.name] += weight
            if len_args == 2:
                if qargs[0] not in im_graph_node_map:
                    im_graph_node_map[qargs[0]] = im_graph.add_node(defaultdict(int))
                    reverse_im_graph_node_map[im_graph_node_map[qargs[0]]] = qargs[0]
                if qargs[1] not in im_graph_node_map:
                    im_graph_node_map[qargs[1]] = im_graph.add_node(defaultdict(int))
                    reverse_im_graph_node_map[im_graph_node_map[qargs[1]]] = qargs[1]
                edge = (im_graph_node_map[qargs[0]], im_graph_node_map[qargs[1]])
                if im_graph.has_edge(*edge):
                    im_graph.get_edge_data(*edge)[node.name] += weight
                else:
                    weights = defaultdict(int)
                    weights[node.name] += weight
                    im_graph.add_edge(*edge, weights)
            if len_args > 2:
                raise MultiQEncountered()

    try:
        _visit(dag, 1, {bit: bit for bit in dag.qubits})
    except MultiQEncountered:
        return None
    # Remove components with no 2q interactions from interaction graph
    # these will be evaluated separately independently of scoring isomorphic
    # mappings. This is not done for strict direction because for post layout
    # we need to factor in local operation constraints when evaluating a graph
    free_nodes = {}
    if not strict_direction:
        conn_comp = connected_components(im_graph)
    else:
        conn_comp = weakly_connected_components(im_graph)
    for comp in conn_comp:
        if len(comp) == 1:
            index = comp.pop()
            free_nodes[index] = im_graph[index]
            if not strict_direction:
                im_graph.remove_node(index)

    return im_graph, im_graph_node_map, reverse_im_graph_node_map, free_nodes

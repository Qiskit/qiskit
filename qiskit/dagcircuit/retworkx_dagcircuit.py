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

"""
Object to represent a quantum circuit as a directed acyclic graph (DAG).

The nodes in the graph are either input/output nodes or operation nodes.
The edges correspond to qubits or bits in the circuit. A directed edge
from node A to node B means that the (qu)bit passes from the output of A
to the input of B. The object's methods allow circuits to be constructed,
composed, and modified. Some natural properties like depth can be computed
directly from the graph.
"""

import copy

import retworkx as rx

from .dagnode import DAGNode
from .dagcircuit import DAGCircuit


class RetworkxDAGCircuit(DAGCircuit):
    """
    Quantum circuit as a directed acyclic graph.

    There are 3 types of nodes in the graph: inputs, outputs, and operations.
    The nodes are connected by directed edges that correspond to qubits and
    bits.
    """

    def __init__(self):
        super().__init__()

        self._USE_RX = True
        self._gx = 'rx'
        self._multi_graph = rx.PyDAG()

    def _add_multi_graph_node(self, node):
        # nx: requires manual node id handling.
        # rx: provides defined ids for added nodes.

        node_id = self._multi_graph.add_node(node)
        node._node_id = node_id
        self._id_to_node[node_id] = node
        return node_id

    def _get_multi_graph_nodes(self):
        return iter(self._multi_graph.nodes())

    def _add_multi_graph_edge(self, src_id, dest_id, data):
        # nx: accepts edge data as kwargs.
        # rx: accepts edge data as a dict arg.

        self._multi_graph.add_edge(src_id, dest_id, data)

    def _get_all_multi_graph_edges(self, src_id, dest_id):
        # nx: edge enumeration through indexing multigraph
        # rx: edge enumeration through method get_all_edge_data

        return self._multi_graph.get_all_edge_data(src_id, dest_id)

    def _get_multi_graph_edges(self):
        # nx: Includes edge data in return only when data kwarg = True
        # rx: Always includes edge data in return

        return [(src, dest, data)
                for src_node in self._multi_graph.nodes()
                for (src, dest, data)
                in self._multi_graph.out_edges(src_node._node_id)]

    def _get_multi_graph_in_edges(self, node_id):
        # nx: Includes edge data in return only when data kwarg = True
        # rx: Always includes edge data in return
        return self._multi_graph.in_edges(node_id)

    def _get_multi_graph_out_edges(self, node_id):
        # nx: Includes edge data in return only when data kwarg = True
        # rx: Always includes edge data in return
        return self._multi_graph.out_edges(node_id)

    def __eq__(self, other):
        # TODO this works but is a horrible way to do this
        slf = copy.deepcopy(self._multi_graph)
        oth = copy.deepcopy(other._multi_graph)

        return rx.is_isomorphic_node_match(
            slf, oth,
            DAGNode.semantic_eq)

    def topological_nodes(self):
        """
        Yield nodes in topological order.

        Returns:
            generator(DAGNode): node in topological order
        """
        def _key(x):
            return str(x.qargs)

        return iter(rx.lexicographical_topological_sort(
            self._multi_graph,
            key=_key))

    def successors(self, node):
        """Returns iterator of the successors of a node as DAGNodes."""
        return iter(self._multi_graph.successors(node._node_id))

    def predecessors(self, node):
        """Returns iterator of the predecessors of a node as DAGNodes."""
        return iter(self._multi_graph.predecessors(node._node_id))

    def bfs_successors(self, node):
        """
        Returns an iterator of tuples of (DAGNode, [DAGNodes]) where the DAGNode is the current node
        and [DAGNode] is its successors in  BFS order.
        """
        return iter(rx.bfs_successors(self._multi_graph, node._node_id))

    def multigraph_layers(self):
        """Yield layers of the multigraph."""
        first_layer = [x._node_id for x in self.input_map.values()]
        yield from rx.layers(self._multi_graph, first_layer)

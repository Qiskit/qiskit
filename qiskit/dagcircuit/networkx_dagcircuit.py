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

import networkx as nx

from .dagnode import DAGNode
from .dagcircuit import DAGCircuit


class NetworkxDAGCircuit(DAGCircuit):
    """
    Quantum circuit as a directed acyclic graph.

    There are 3 types of nodes in the graph: inputs, outputs, and operations.
    The nodes are connected by directed edges that correspond to qubits and
    bits.
    """

    def __init__(self):
        super().__init__()

        self._USE_RX = False
        self._gx = 'nx'
        self._multi_graph = nx.MultiDiGraph()

    def _add_multi_graph_node(self, node):
        # nx: requires manual node id handling.
        # rx: provides defined ids for added nodes.
        self._max_node_id += 1
        node._node_id = self._max_node_id
        self._id_to_node[node._node_id] = node

        self._multi_graph.add_node(node._node_id)

        return node._node_id

    def _get_multi_graph_nodes(self):
        return (self._id_to_node[node_index]
                for node_index in self._multi_graph.nodes())

    def _add_multi_graph_edge(self, src_id, dest_id, data):
        # nx: accepts edge data as kwargs.
        # rx: accepts edge data as a dict arg.
        self._multi_graph.add_edge(src_id, dest_id, **data)

    def _get_all_multi_graph_edges(self, src_id, dest_id):
        # nx: edge enumeration through indexing multigraph
        # rx: edge enumeration through method get_all_edge_data
        return list(self._multi_graph[src_id][dest_id].values())

    def _get_multi_graph_edges(self):
        # nx: Includes edge data in return only when data kwarg = True
        # rx: Always includes edge data in return
        return self._multi_graph.edges(data=True)

    def _get_multi_graph_in_edges(self, node_id):
        # nx: Includes edge data in return only when data kwarg = True
        # rx: Always includes edge data in return
        return self._multi_graph.in_edges(node_id, data=True)

    def _get_multi_graph_out_edges(self, node_id):
        # nx: Includes edge data in return only when data kwarg = True
        # rx: Always includes edge data in return
        return self._multi_graph.out_edges(node_id, data=True)

    def __eq__(self, other):
        slf = copy.deepcopy(self._multi_graph)
        oth = copy.deepcopy(other._multi_graph)

        for node_id in slf.nodes:
            slf.nodes[node_id]['node'] = self._id_to_node[node_id]
        for node_id in oth.nodes:
            oth.nodes[node_id]['node'] = other._id_to_node[node_id]

        return nx.is_isomorphic(
            slf, oth,
            node_match=lambda x, y: DAGNode.semantic_eq(x['node'], y['node']))

    def topological_nodes(self):
        """
        Yield nodes in topological order.

        Returns:
            generator(DAGNode): node in topological order
        """
        def _key(x):
            return str(self._id_to_node[x].qargs)

        return (self._id_to_node[idx]
                for idx in nx.lexicographical_topological_sort(
                    self._multi_graph,
                    key=_key))

    def successors(self, node):
        """Returns iterator of the successors of a node as DAGNodes."""
        return (self._id_to_node[idx]
                for idx in self._multi_graph.successors(node._node_id))

    def predecessors(self, node):
        """Returns iterator of the predecessors of a node as DAGNodes."""
        return (self._id_to_node[idx]
                for idx in self._multi_graph.predecessors(node._node_id))

    def bfs_successors(self, node):
        """
        Returns an iterator of tuples of (DAGNode, [DAGNodes]) where the DAGNode is the current node
        and [DAGNode] is its successors in  BFS order.
        """
        return ((self._id_to_node[idx], [self._id_to_node[succ] for succ in succ_list])
                for idx, succ_list in nx.bfs_successors(self._multi_graph, node._node_id))

    def multigraph_layers(self):
        """Yield layers of the multigraph."""
        predecessor_count = dict()  # Dict[node, predecessors not visited]
        cur_layer = self.input_map.values()
        yield cur_layer
        next_layer = []
        while cur_layer:
            for node in cur_layer:
                # Count multiedges with multiplicity.
                for successor in self.successors(node):
                    multiplicity = self._multi_graph.number_of_edges(
                        node._node_id,
                        successor._node_id)
                    if successor in predecessor_count:
                        predecessor_count[successor] -= multiplicity
                    else:
                        predecessor_count[successor] = \
                            self._multi_graph.in_degree(successor._node_id) - multiplicity

                    if predecessor_count[successor] == 0:
                        next_layer.append(successor)
                        del predecessor_count[successor]

            yield next_layer
            cur_layer = next_layer
            next_layer = []

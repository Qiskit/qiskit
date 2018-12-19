# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=no-member

"""
Directed graph object for representing coupling between physical qubits.

The nodes of the graph correspond to physical qubits (represented as integers) and the
directed edges indicate which physical qubits are coupled and the permitted direction of
CNOT gates. The object has a distance function that can be used to map quantum circuits
onto a device with this coupling.
"""

import warnings
import networkx as nx
from ._couplingerror import CouplingError


class CouplingMap:
    """
    Directed graph specifying fixed coupling.

    Nodes correspond to physical qubits (integers) and directed edges correspond
    to permitted CNOT gates
    """

    def __init__(self, couplingdict=None, couplinglist=None):
        """
        Create coupling graph. By default, the generated coupling has no nodes.

        Args:
            couplinglist (list or None): An initial coupling graph, specified as
                an adjacency list, e.g. [[0,1], [0,2], [1,2]].
            couplingdict (dict or None): DEPRECATED An initial coupling graph
                specified as an adjacency dict, e.g. {0: [1, 2], 1: [2]}.
        Raises:
            CouplingError: If both couplinglist and couplingdict are supplied.
        """
        if couplingdict is not None and couplinglist is not None:
            raise CouplingError('Cannot specify both couplingdict and couplinglist')

        if couplingdict is not None:
            if isinstance(couplingdict, list):
                couplinglist = couplingdict
                couplingdict = None
            else:
                warnings.warn('Initializing a coupling object through a couplingdict is '
                              'deprecated. Use a couplinglist instead.', DeprecationWarning,
                              stacklevel=2)

        # the coupling map graph
        self.graph = nx.DiGraph()
        # a dict of dicts from node pairs to distances
        self._dist_matrix = None
        # a sorted list of physical qubits (integers) in this coupling map
        self._qubit_list = None

        if couplingdict is not None:
            for origin, destinations in couplingdict.items():
                for destination in destinations:
                    self.add_edge(origin, destination)

        if couplinglist is not None:
            for source, target in couplinglist:
                self.add_edge(source, target)

    def size(self):
        """Return the number of physical qubits in this graph."""
        return len(self.graph.nodes)

    def get_edges(self):
        """
        Gets the list of edges in the coupling graph.

        Returns:
            Tuple(int,int): Each edge is a pair of physical qubits.
        """
        return [edge for edge in self.graph.edges()]

    def add_physical_qubit(self, physical_qubit):
        """Add a physical qubit to the coupling graph as a node.

        physical_qubit (int): An integer representing a physical qubit.

        Raises:
            CouplingError: if trying to add duplicate qubit
        """
        if not isinstance(physical_qubit, int):
            raise CouplingError("Physical qubits should be integers.")
        if physical_qubit in self.physical_qubits:
            raise CouplingError(
                "The physical qubit %s is already in the coupling graph" % physical_qubit)
        self.graph.add_node(physical_qubit)
        self._dist_matrix = None  # invalidate
        self._qubit_list = None  # invalidate

    def add_edge(self, src, dst):
        """
        Add directed edge to coupling graph.

        src (int): source physical qubit
        dst (int): destination physical qubit
        """
        if src not in self.physical_qubits:
            self.add_physical_qubit(src)
        if dst not in self.physical_qubits:
            self.add_physical_qubit(dst)
        self.graph.add_edge(src, dst)
        self._dist_matrix = None  # invalidate

    def subgraph(self, nodelist):
        """Return a CouplingMap object for a subgraph of self.

        nodelist (list): list of integer node labels
        """
        subcoupling = CouplingMap()
        subcoupling.graph = self.graph.subgraph(nodelist)
        for node in nodelist:
            if node not in subcoupling.physical_qubits:
                subcoupling.add_physical_qubit(node)
        return subcoupling

    @property
    def physical_qubits(self):
        """Returns a sorted list of physical_qubits"""
        if self._qubit_list is None:
            self._qubit_list = sorted([pqubit for pqubit in self.graph.nodes])
        return self._qubit_list

    def is_connected(self):
        """
        Test if the graph is connected.

        Return True if connected, False otherwise
        """
        try:
            return nx.is_weakly_connected(self.graph)
        except nx.exception.NetworkXException:
            return False

    def _compute_distance_matrix(self):
        """Compute the full distance matrix on pairs of nodes.

        The distance map self._dist_matrix is computed from the graph using
        all_pairs_shortest_path_length.
        """
        if not self.is_connected():
            raise CouplingError("coupling graph not connected")
        lengths = nx.all_pairs_shortest_path_length(self.graph.to_undirected(as_view=True))
        self._dist_matrix = dict(lengths)

    def distance(self, physical_qubit1, physical_qubit2):
        """Returns the undirected distance between physical_qubit1 and physical_qubit2.

        Args:
            physical_qubit1 (int): A physical qubit
            physical_qubit2 (int): Another physical qubit

        Returns:
            int: The undirected distance

        Raises:
            CouplingError: if the qubits do not exist in the CouplingMap
        """
        if physical_qubit1 not in self.physical_qubits:
            raise CouplingError("%s not in coupling graph" % (physical_qubit1,))
        if physical_qubit2 not in self.physical_qubits:
            raise CouplingError("%s not in coupling graph" % (physical_qubit2,))
        if self._dist_matrix is None:
            self._compute_distance_matrix()
        return self._dist_matrix[physical_qubit1][physical_qubit2]

    def shortest_undirected_path(self, physical_qubit1, physical_qubit2):
        """Returns the shortest undirected path between physical_qubit1 and physical_qubit2.
        Args:
            physical_qubit1 (int): A physical qubit
            physical_qubit2 (int): Another physical qubit
        Returns:
            List: The shortest undirected path
        Raises:
            CouplingError: When there is no path between physical_qubit1, physical_qubit2.
        """
        try:
            return nx.shortest_path(self.graph.to_undirected(as_view=True), source=physical_qubit1,
                                    target=physical_qubit2)
        except nx.exception.NetworkXNoPath:
            raise CouplingError(
                "Nodes %s and %s are not connected" % (str(physical_qubit1), str(physical_qubit2)))

    def __str__(self):
        """Return a string representation of the coupling graph."""
        string = ""
        if self.get_edges():
            string += "["
            string += ", ".join(["(%s, %s)" % (src, dst) for (src, dst) in self.get_edges()])
            string += "]"
        return string

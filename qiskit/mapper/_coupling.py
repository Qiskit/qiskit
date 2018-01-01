# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=no-member

"""
Directed graph object for representing coupling between qubits.

The nodes of the graph correspond to named qubits and the directed edges
indicate which qubits are coupled and the permitted direction of CNOT gates.
The object has a distance_qubits function that can be used to map quantum circuits
onto a device with this coupling.
"""
import warnings
from collections import OrderedDict
import networkx as nx
from qiskit import _quantumregister
from ._couplingerror import CouplingError


class Coupling:
    """
    Directed graph specifying fixed coupling.

    Nodes correspond to qubits and directed edges correspond to permitted
    CNOT gates
    """

    # pylint: disable=invalid-name

    @staticmethod
    def coupling_dict2list(couplingdict):
        """Convert coupling map dictionary into list.

        Example dictionary format: {0: [1, 2], 1: [2]}
        Example list format: [[0, 1], [0, 2], [1, 2]]

        We do not do any checking of the input.

        Return coupling map in list format.
        """
        if not couplingdict:
            return None
        couplinglist = []
        for ctl, tgtlist in couplingdict.items():
            for tgt in tgtlist:
                couplinglist.append([ctl, tgt])
        return couplinglist

    @staticmethod
    def coupling_list2dict(couplinglist):
        """Convert coupling map list into dictionary.

        Example list format: [[0, 1], [0, 2], [1, 2]]
        Example dictionary format: {0: [1, 2], 1: [2]}

        We do not do any checking of the input.

        Return coupling map in dict format.
        """
        if not couplinglist:
            return None
        couplingdict = {}
        for pair in couplinglist:
            if pair[0] in couplingdict:
                couplingdict[pair[0]].append(pair[1])
            else:
                couplingdict[pair[0]] = [pair[1]]
        return couplingdict

    def __init__(self, couplingdict=None):
        """
        Create coupling graph.

        By default, the coupling graph has no nodes. The optional couplingdict
        specifies the graph as an adjacency list. For example,
        couplingdict = {0: [1, 2], 1: [2]}.
        """
        self.graph = nx.DiGraph()
        if isinstance(couplingdict, dict):
            for origin, destinations in couplingdict.items():
                for destination in destinations:
                    self.add_edge(origin, destination)

    def size(self):
        """Return the number of wires in this graph."""
        return len(self.graph.nodes)

    def get_edges(self):
        """Return a list of edges in the coupling graph.

        Each edge is a pair of wires.
        """
        return [edge for edge in self.graph.edges()]

    def add_wire(self, wire):
        """
        Add a wire to the coupling graph as a node.

        wire (int): A wire
        """
        if not isinstance(wire, int):
            raise CouplingError("Wires should be numbers.")
        if wire in self.wires:
            raise CouplingError("The wire %s is already in the coupling graph" % wire)

        self.graph.add_node(wire)

    def add_edge(self, src_wire, dst_wire):
        """
        Add directed edge to coupling graph.

        src_wire (int): source wire
        dst_wire (int): destination wire
        """
        if src_wire not in self.wires:
            self.add_wire(src_wire)
        if dst_wire not in self.wires:
            self.add_wire(dst_wire)
        self.graph.add_edge(src_wire, dst_wire)

    @property
    def wires(self):
        return sorted([wire for wire in self.graph.nodes])


    def is_connected(self):
        """
        Test if the graph is connected.

        Return True if connected, False otherwise
        """
        try:
            return nx.is_weakly_connected(self.graph)
        except nx.exception.NetworkXException:
            return False


    def compute_distance(self):
        """
        Compute the undirected distance function on pairs of nodes.

        The distance_qubits map self.dist is computed from the graph using
        all_pairs_shortest_path_length.
        """
        if not self.connected():
            raise CouplingError("coupling graph not connected")
        lengths = dict(nx.all_pairs_shortest_path_length(self.G.to_undirected()))
        self.dist = {}
        for i in self.qubits.keys():
            self.dist[i] = {}
            for j in self.qubits.keys():
                self.dist[i][j] = lengths[self.qubits[i]][self.qubits[j]]

    def distance(self, wire1, wire2):
        """Return the undirected distance between wire1 and wire2."""
        try:
            return len(nx.shortest_path(self.graph.to_undirected(), source=wire1, target=wire2))-1
        except nx.exception.NetworkXNoPath:
            raise CouplingError("Nodes %s and %s are not connected" % (str(wire1), str(wire2)))

    def __str__(self):
        """Return a string representation of the coupling graph."""
        s = ""
        if self.get_edges():
            s += "["
            s += ", ".join([ "(%s, %s)" % (src,dst) for (src,dst) in self.get_edges()])
            s += "]"
        return s

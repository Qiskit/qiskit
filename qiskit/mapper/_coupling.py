# -*- coding: utf-8 -*-

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""
Directed graph object for representing coupling between qubits.

The nodes of the graph correspond to named qubits and the directed edges
indicate which qubits are coupled and the permitted direction of CNOT gates.
The object has a distance function that can be used to map quantum circuits
onto a device with this coupling.
"""
from collections import OrderedDict
import networkx as nx
from ._couplingerror import CouplingError


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


class Coupling:
    """
    Directed graph specifying fixed coupling.

    Nodes correspond to qubits and directed edges correspond to permitted
    CNOT gates
    """
    # pylint: disable=invalid-name

    def __init__(self, couplingdict=None):
        """
        Create coupling graph.

        By default, the coupling graph has no nodes. The optional couplingdict
        specifies the graph as an adjacency list. For example,
        couplingdict = {0: [1, 2], 1: [2]}.
        """
        # self.qubits is dict from qubit (regname,idx) tuples to node indices
        self.qubits = OrderedDict()
        # self.index_to_qubit is a dict from node indices to qubits
        self.index_to_qubit = {}
        # self.node_counter is integer counter for labeling nodes
        self.node_counter = 0
        # self.G is the coupling digraph
        self.G = nx.DiGraph()
        # self.dist is a dict of dicts from node pairs to distances
        # it must be computed, it is the distance on the digraph
        self.dist = None
        # Add edges to the graph if the couplingdict is present
        if couplingdict is not None:
            for v0, alist in couplingdict.items():
                for v1 in alist:
                    regname = "q"
                    self.add_edge((regname, v0), (regname, v1))
            self.compute_distance()

    def size(self):
        """Return the number of qubits in this graph."""
        return len(self.qubits)

    def get_qubits(self):
        """Return the qubits in this graph as (qreg, index) tuples."""
        return list(self.qubits.keys())

    def get_edges(self):
        """Return a list of edges in the coupling graph.

        Each edge is a pair of qubits and each qubit is a tuple (qreg, index).
        """
        return list(map(lambda x: (self.index_to_qubit[x[0]],
                                   self.index_to_qubit[x[1]]), self.G.edges()))

    def add_qubit(self, name):
        """
        Add a qubit to the coupling graph.

        name = tuple (regname, idx) for qubit
        """
        if name in self.qubits:
            raise CouplingError("%s already in coupling graph" % name)

        self.node_counter += 1
        self.G.add_node(self.node_counter)
        self.G.node[self.node_counter]["name"] = name
        self.qubits[name] = self.node_counter
        self.index_to_qubit[self.node_counter] = name

    def add_edge(self, s_name, d_name):
        """
        Add directed edge to coupling graph.

        s_name = source qubit tuple
        d_name = destination qubit tuple
        """
        if s_name not in self.qubits:
            self.add_qubit(s_name)
        if d_name not in self.qubits:
            self.add_qubit(d_name)
        self.G.add_edge(self.qubits[s_name], self.qubits[d_name])

    def connected(self):
        """
        Test if the graph is connected.

        Return True if connected, False otherwise
        """
        return nx.is_weakly_connected(self.G)

    def compute_distance(self):
        """
        Compute the distance function on pairs of nodes.

        The distance map self.dist is computed from the graph using
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

    def distance(self, q1, q2):
        """Return the distance between qubit q1 to qubit q2."""
        if self.dist is None:
            raise CouplingError("distance has not been computed")
        if q1 not in self.qubits:
            raise CouplingError("%s not in coupling graph" % q1)
        if q2 not in self.qubits:
            raise CouplingError("%s not in coupling graph" % q2)
        return self.dist[q1][q2]

    def __str__(self):
        """Return a string representation of the coupling graph."""
        s = "qubits: "
        s += ", ".join(["%s[%d] @ %d" % (k[0], k[1], v)
                        for k, v in self.qubits.items()])
        s += "\nedges: "
        s += ", ".join(["%s[%d]-%s[%d]" % (e[0][0], e[0][1], e[1][0], e[1][1])
                        for e in self.get_edges()])
        return s

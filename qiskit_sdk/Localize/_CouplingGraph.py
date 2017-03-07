"""
Directed graph object for representing coupling between qubits.

The nodes of the graph correspond to named qubits and the directed edges
indicate which qubits are coupled and the permitted direction of CNOT gates.
The object has a distance function that can be used to map quantum circuits
onto a device with this coupling.

Author: Andrew Cross
"""
import networkx as nx
import numpy as np
from ._CouplingGraphError import CouplingGraphError


class CouplingGraph:
    """
    Directed graph specifying fixed coupling.

    Nodes correspond to qubits and directed edges correspond to permitted
    CNOT gates
    """

    def __init__(self, couplingstr=None):
        """
        Create coupling graph.

        By default, the coupling graph has no nodes. The optional couplingstr
        has the form "qreg,idx:qreg,idx;..." where each (qreg,idx) pair is a
        qubit, ":" separates qubits in a directed edge "src:dest", and ";"
        separates directed edges "dedge;dedge;...".
        """
        # self.qubits is dict from qubit (regname,idx) tuples to node indices
        self.qubits = {}
        # self.node_counter is integer counter for labeling nodes
        self.node_counter = 0
        # self.G is the coupling digraph
        self.G = nx.DiGraph()
        # self.dist is a dict of dicts from node pairs to distances
        # it must be computed, it is the distance on the digraph
        self.dist = None
        # self.hdist is a dict of dicts from node pairs to distances
        # it must be computed, it is a heuristic distance function
        self.hdist = None
        # Add edges to the graph if the couplingstr is present
        if couplingstr is not None:
            edge_list = couplingstr.split(';')
            for e in edge_list:
                vertex_pair = e.split(':')
                vertex0 = vertex_pair[0].split(',')
                vertex1 = vertex_pair[1].split(',')
                vtuple0 = (vertex0[0], int(vertex0[1]))
                vtuple1 = (vertex1[0], int(vertex1[1]))
                self.add_edge(vtuple0, vtuple1)

    def add_qubit(self, name):
        """
        Add a qubit to the coupling graph.

        name = tuple (regname,idx) for qubit
        """
        if name in self.qubits:
            raise CouplingGraphError("%s already in coupling graph" % name)

        self.node_counter += 1
        self.G.add_node(self.node_counter)
        self.G.node[self.node_counter]["name"] = name
        self.qubits[name] = self.node_counter

    def add_edge(self, s_name, d_name):
        """
        Add directed edge to connectivity graph.

        s_name = source qubit tuple
        d_name = destination qubit tuple
        """
        if s_name not in self.qubits:
            self.node_counter += 1
            self.G.add_node(self.node_counter)
            self.G.node[self.node_counter]["name"] = s_name
            self.qubits[s_name] = self.node_counter
        if d_name not in self.qubits:
            self.node_counter += 1
            self.G.add_node(self.node_counter)
            self.G.node[self.node_counter]["name"] = d_name
            self.qubits[d_name] = self.node_counter
        self.G.add_edge(self.qubits[s_name], self.qubits[d_name])

    def connected(self):
        """
        Test if the graph is connected.

        Return True if connected, False otherwise
        """
        return nx.is_weakly_connected(self.G)

    def compute_distance(self, randomize=False):
        """
        Compute the distance function on pairs of nodes.

        The distance map self.dist is computed from the graph using
        all_pairs_shortest_path_length. The distance map self.hdist is also
        computed. If randomize is False, we use self.dist. Otherwise, we use
        Sergey Bravyi's randomization heuristic.
        """
        if not self.connected():
            raise CouplingGraphError("coupling graph not connected")
        lengths = nx.all_pairs_shortest_path_length(self.G.to_undirected())
        self.dist = {}
        self.hdist = {}
        for i in self.qubits.keys():
            self.dist[i] = {}
            self.hdist[i] = {}
            for j in self.qubits.keys():
                self.dist[i][j] = lengths[self.qubits[i]][self.qubits[j]]
                self.hdist[i][j] = self.dist[i][j]
        if randomize:
            for i in self.qubits.keys():
                for j in self.qubits.keys():
                    scale = (1.0 + np.random.normal(0.0, 1.0/len(self.qubits)))
                    self.hdist[i][j] = scale * self.dist[i][j]**2
                    self.hdist[j][i] = self.hdist[i][j]

    def distance(self, q1, q2, h=False):
        """
        Return the distance between qubit q1 to qubit q2.

        We look this up in self.dist if h is False and in self.hdist
        if h is True.
        """
        if self.dist is None:
            raise CouplingGraphError("distance has not been computed")
        if q1 not in self.qubits:
            raise CouplingGraphError("%s not in coupling graph" % q1)
        if q2 not in self.qubits:
            raise CouplingGraphError("%s not in coupling graph" % q2)
        if h:
            return self.hdist[q1][q2]
        else:
            return self.dist[q1][q2]

    def __str__(self):
        """Return a string representation of the coupling graph."""
        s = "%s" % self.qubits
        s += "\n%s" % self.G.edges()
        return s

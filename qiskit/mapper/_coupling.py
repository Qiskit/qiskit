"""
Directed graph object for representing coupling between qubits.

The nodes of the graph correspond to named qubits and the directed edges
indicate which qubits are coupled and the permitted direction of CNOT gates.
The object has a distance function that can be used to map quantum circuits
onto a device with this coupling.

Author: Andrew Cross
"""
import networkx as nx
from ._couplingerror import CouplingError


class Coupling:
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
        # self.index_to_qubit is a dict from node indices to qubits
        self.index_to_qubit = {}
        # self.node_counter is integer counter for labeling nodes
        self.node_counter = 0
        # self.G is the coupling digraph
        self.G = nx.DiGraph()
        # self.dist is a dict of dicts from node pairs to distances
        # it must be computed, it is the distance on the digraph
        self.dist = None
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

        name = tuple (regname,idx) for qubit
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
        lengths = nx.all_pairs_shortest_path_length(self.G.to_undirected())
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

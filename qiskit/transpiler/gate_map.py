# This code is part of Qiskit.
#
# (C) Copyright IBM 2021
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Directed graph object for representing basis gates on physical qubits.

The nodes of the graph correspond to physical qubits (represented as integers) and the
directed edges indicate which physical qubits are coupled and the permitted direction of
CNOT gates. The object has a distance function that can be used to map quantum circuits
onto a device with this coupling.
"""

import retworkx as rx

from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.exceptions import CouplingError
from qiskit.exceptions import MissingOptionalLibraryError


class GateWeight:
    __slots__ = ("gate", "length", "error", "properties")

    def __init__(self, gate_cls, length=None, error=None, properties=None):
        self.gate = gate_cls
        self.length = length
        self.error = error
        self.properties = properties


class GateMap:
    """
    Directed graph specifying fixed coupling.

    Nodes correspond to physical qubits (integers) and directed edges correspond
    to permitted CNOT gates
    """

    def __init__(self, gate_map, coupling_map=None, description=None):
        """
        Create coupling graph. By default, the generated coupling has no nodes.

        Args:
            gate_map (dict): A dictionary of gate_weight classes for keys and a list
                qargs for
            couplinglist (list or None): An initial coupling graph, specified as
                an adjacency list containing couplings, e.g. [[0,1], [0,2], [1,2]].
            description (str): A string to describe the coupling map.
        """
        self.description = description
        self.graph = rx.PyDiGraph(multigraph=False)
        if coupling_map is not None:
            self.graph.extend_from_edge_list(
                [(x[0], x[1], list()) for x in coupling_map.get_edges()]
            )
            for node in self.graph.node_indexes():
                self.graph[node] = []
        for gate, qarg_list in gate_map.items():
            for qargs in qarg_list:
                for qarg in qargs:
                    if len(self.graph) < qarg:
                        self.graph.add_nodes_from([list() for _ in range(qarg[0] - len(qarg_list))])
                if len(qargs) == 2:
                    if self.graph.has_edge(*qargs):
                        self.graph.get_edge_data(*qargs).append(gate)
                    else:
                        self.graph.add_edge(qargs[0], qargs[1], gate)
                if len(qargs) == 1:
                    self.graph[qargs[0]].append(gate)

        # a dict of dicts from node pairs to distances
        self._unweighted_dist_matrix = None
        self._length_distance_matrix = None
        self._error_distance_matrix = None

    def size(self):
        """Return the number of physical qubits in this graph."""
        return len(self.graph)

    def get_edges(self):
        """
        Gets the list of edges in the coupling graph.

        Returns:
            Tuple(int,int): Each edge is a pair of physical qubits.
        """
        return self.graph.edge_list()

    def gate_weights(self):
        """Return a list of the gate weight objects in the GateMap."""
        out = [x for node in self.graph.nodes() for x in node]
        out.extend([x for edge in self.graph.edges() for x in edge])
        return out

    @property
    def physical_qubits(self):
        """Returns a sorted list of physical_qubits"""
        return self.graph.node_indexes()

    def is_connected(self):
        """
        Test if the graph is connected.

        Return True if connected, False otherwise
        """
        try:
            return rx.is_weakly_connected(self.graph)
        except rx.NullGraph:
            return False

    def neighbors(self, physical_qubit):
        """Return the nearest neighbors of a physical qubit.

        Directionality matters, i.e. a neighbor must be reachable
        by going one hop in the direction of an edge.
        """
        return self.graph.neighbors(physical_qubit)

    def distance_matrix(self, weight=None):
        """Return the distance matrix for the coupling map."""
        if weight is None:
            if self._unweighted_dist_matrix is None:
                self._compute_distance_matrix()
            return self._unweighted_dist_matrix
        elif weight == "error":
            if self._error_distance_matrix is None:
                self._error_distance_matrix = rx.digraph_floyd_warshall_numpy(
                    self.graph,
                    weight_fn=lambda edge: min(edge, key=lambda x: x.error),
                    as_undirected=True,
                )
            return self._error_distance_matrix
        elif weight == "length":
            if self._length_distance_matrix is None:
                self._length_distance_matrix = rx.digraph_floyd_warshall_numpy(
                    self.graph,
                    weight_fn=lambda edge: min(edge, key=lambda x: x.error),
                    as_undirected=True,
                )
            return self._error_distance_matrix
        else:
            raise TypeError("Invalid weight type %s" % weight)

    def _compute_distance_matrix(self):
        """Compute the full distance matrix on pairs of nodes.

        The distance map self._dist_matrix is computed from the graph using
        all_pairs_shortest_path_length.
        """
        if not self.is_connected():
            raise CouplingError("coupling graph not connected")
        self._dist_matrix = rx.digraph_distance_matrix(self.graph, as_undirected=True)

    def to_coupling_map(self):
        """Convert to a new CouplingMap object."""
        out_graph = rx.PyDiGraph()
        out_graph.extend_from_edge_list(list(self.graph.edge_list()))
        out_map = CouplingMap(description=self.description)
        out_map.graph = out_graph
        return out_map

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
        if physical_qubit1 >= self.size():
            raise CouplingError("%s not in coupling graph" % physical_qubit1)
        if physical_qubit2 >= self.size():
            raise CouplingError("%s not in coupling graph" % physical_qubit2)
        if self._dist_matrix is None:
            self._compute_distance_matrix()
        return int(self._dist_matrix[physical_qubit1, physical_qubit2])

    def shortest_undirected_path(self, physical_qubit1, physical_qubit2, weight=None):
        """Returns the shortest undirected path between physical_qubit1 and physical_qubit2.

        Args:
            physical_qubit1 (int): A physical qubit
            physical_qubit2 (int): Another physical qubit
        Returns:
            List: The shortest undirected path
        Raises:
            CouplingError: When there is no path between physical_qubit1, physical_qubit2.
        """
        if weight is None:
            weight_fn = None
        elif weight == "error":

            def weight_fn(edge):
                return min(edge, key=lambda x: x.error)

        elif weight == "length":

            def weight_fn(edge):
                return min(edge, key=lambda x: x.length)

        paths = rx.digraph_dijkstra_shortest_paths(
            self.graph,
            source=physical_qubit1,
            target=physical_qubit2,
            as_undirected=True,
            weight_fn=weight_fn,
        )
        if not paths:
            raise CouplingError(
                f"Nodes {str(physical_qubit1)} and {str(physical_qubit2)} are not connected"
            )
        return paths[physical_qubit2]

    @property
    def is_symmetric(self):
        """
        Test if the graph is symmetric.

        Return True if symmetric, False otherwise
        """
        return self.graph.is_symmetric()

    def largest_connected_component(self):
        """Return a set of qubits in the largest connected component."""
        return max(rx.weakly_connected_components(self.graph), key=len)

    def __str__(self):
        """Return a string representation of the coupling graph."""
        string = ""
        if self.get_edges():
            string += "["
            string += ", ".join([f"[{src}, {dst}]" for (src, dst) in self.get_edges()])
            string += "]"
        return string

    def draw(self):
        """Draws the coupling map.

        This function needs `matplotlib <https://matplotlib.org>`__

        Returns:
            matplotlib.figure.Figure: Drawn coupling map.

        Raises:
            MissingOptionalLibraryError: when pydot or pillow are not installed.
        """
        try:
            from retworkx.visualization import mpl_draw
        except ImportError as ex:
            raise MissingOptionalLibraryError(
                libname="matplotlib",
                name="gate map drawer",
                pip_install="pip install matplotlib",
            ) from ex

        def node_fn(node):
            return {"label": "\n".join([str(x.gate.__name__) for x in node])}

        def edge_fn(edge):
            return {"label": "\n".join([str(x.gate.__name__) for x in edge])}

        return mpl_draw(self.graph, with_labels=True, labels=node_fn, edge_labels=edge_fn)

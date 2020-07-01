# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Directed graph object for representing coupling between physical qubits.

The nodes of the graph correspond to physical qubits (represented as integers) and the
directed edges indicate which physical qubits are coupled and the permitted direction of
CNOT gates. The object has a distance function that can be used to map quantum circuits
onto a device with this coupling.
"""
import io
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as cs
import networkx as nx
from qiskit.transpiler.exceptions import CouplingError


class CouplingMap:
    """
    Directed graph specifying fixed coupling.

    Nodes correspond to physical qubits (integers) and directed edges correspond
    to permitted CNOT gates
    """

    def __init__(self, couplinglist=None, description=None):
        """
        Create coupling graph. By default, the generated coupling has no nodes.

        Args:
            couplinglist (list or None): An initial coupling graph, specified as
                an adjacency list containing couplings, e.g. [[0,1], [0,2], [1,2]].
            description (str): A string to describe the coupling map.
        """
        self.description = description
        # the coupling map graph
        self.graph = nx.DiGraph()
        # a dict of dicts from node pairs to distances
        self._dist_matrix = None
        # a sorted list of physical qubits (integers) in this coupling map
        self._qubit_list = None
        # a sorted list of physical qubits (integers) in this coupling map
        self._is_symmetric = None

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
        return list(self.graph.edges())

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
        self._is_symmetric = None  # invalidate

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
            self._qubit_list = sorted(self.graph.nodes)
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

    def neighbors(self, physical_qubit):
        """Return the nearest neighbors of a physical qubit.

        Directionality matters, i.e. a neighbor must be reachable
        by going one hop in the direction of an edge.
        """
        return self.graph.neighbors(physical_qubit)

    def _compute_distance_matrix(self):
        """Compute the full distance matrix on pairs of nodes.

        The distance map self._dist_matrix is computed from the graph using
        all_pairs_shortest_path_length.
        """
        if not self.is_connected():
            raise CouplingError("coupling graph not connected")
        lengths = nx.all_pairs_shortest_path_length(self.graph.to_undirected(as_view=True))
        lengths = dict(lengths)
        size = len(lengths)
        cmap = np.zeros((size, size))
        for idx in range(size):
            cmap[idx, np.fromiter(lengths[idx].keys(), dtype=int)] = np.fromiter(
                lengths[idx].values(), dtype=int)
        self._dist_matrix = cmap

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
        return int(self._dist_matrix[physical_qubit1, physical_qubit2])

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

    @property
    def is_symmetric(self):
        """
        Test if the graph is symmetric.

        Return True if symmetric, False otherwise
        """
        if self._is_symmetric is None:
            self._is_symmetric = self._check_symmetry()
        return self._is_symmetric

    def make_symmetric(self):
        """
        Convert uni-directional edges into bi-directional.
        """
        edges = self.get_edges()
        for src, dest in edges:
            if (dest, src) not in edges:
                self.add_edge(dest, src)
        self._dist_matrix = None  # invalidate
        self._is_symmetric = None  # invalidate

    def _check_symmetry(self):
        """
        Calculates symmetry

        Returns:
            Bool: True if symmetric, False otherwise
        """
        mat = nx.adjacency_matrix(self.graph)
        return (mat - mat.T).nnz == 0

    def reduce(self, mapping):
        """Returns a reduced coupling map that
        corresponds to the subgraph of qubits
        selected in the mapping.

        Args:
            mapping (list): A mapping of reduced qubits to device
                            qubits.

        Returns:
            CouplingMap: A reduced coupling_map for the selected qubits.

        Raises:
            CouplingError: Reduced coupling map must be connected.
        """
        reduced_qubits = len(mapping)
        inv_map = [None] * (max(mapping) + 1)
        for idx, val in enumerate(mapping):
            inv_map[val] = idx

        reduced_cmap = []

        for edge in self.get_edges():
            if edge[0] in mapping and edge[1] in mapping:
                reduced_cmap.append([inv_map[edge[0]], inv_map[edge[1]]])

        # Verify coupling_map is connected
        rows = np.array([edge[0] for edge in reduced_cmap], dtype=int)
        cols = np.array([edge[1] for edge in reduced_cmap], dtype=int)
        data = np.ones_like(rows)

        mat = sp.coo_matrix((data, (rows, cols)),
                            shape=(reduced_qubits, reduced_qubits)).tocsr()

        if cs.connected_components(mat)[0] != 1:
            raise CouplingError('coupling_map must be connected.')

        return CouplingMap(reduced_cmap)

    @classmethod
    def from_full(cls, num_qubits, bidirectional=True):
        """Return a fully connected coupling map on n qubits."""
        cmap = cls(description='full')
        for i in range(num_qubits):
            for j in range(i):
                cmap.add_edge(j, i)
                if bidirectional:
                    cmap.add_edge(i, j)
        return cmap

    @classmethod
    def from_line(cls, num_qubits, bidirectional=True):
        """Return a fully connected coupling map on n qubits."""
        cmap = cls(description='line')
        for i in range(num_qubits-1):
            cmap.add_edge(i, i+1)
            if bidirectional:
                cmap.add_edge(i+1, i)
        return cmap

    @classmethod
    def from_ring(cls, num_qubits, bidirectional=True):
        """Return a fully connected coupling map on n qubits."""
        cmap = cls(description='ring')
        for i in range(num_qubits):
            if i == num_qubits - 1:
                k = 0
            else:
                k = i + 1
            cmap.add_edge(i, k)
            if bidirectional:
                cmap.add_edge(k, i)
        return cmap

    @classmethod
    def from_grid(cls, num_rows, num_columns, bidirectional=True):
        """Return qubits connected on a grid of num_rows x num_columns."""
        cmap = cls(description='grid')
        for i in range(num_rows):
            for j in range(num_columns):
                node = i * num_columns + j

                up = (node-num_columns) if i > 0 else None  # pylint: disable=invalid-name
                down = (node+num_columns) if i < num_rows-1 else None
                left = (node-1) if j > 0 else None
                right = (node+1) if j < num_columns-1 else None

                if up is not None and bidirectional:
                    cmap.add_edge(node, up)
                if left is not None and bidirectional:
                    cmap.add_edge(node, left)
                if down is not None:
                    cmap.add_edge(node, down)
                if right is not None:
                    cmap.add_edge(node, right)
        return cmap

    def largest_connected_component(self):
        """Return a set of qubits in the largest connected component."""
        return max(nx.strongly_connected_components(self.graph), key=len)

    def __str__(self):
        """Return a string representation of the coupling graph."""
        string = ""
        if self.get_edges():
            string += "["
            string += ", ".join(["[%s, %s]" % (src, dst) for (src, dst) in self.get_edges()])
            string += "]"
        return string

    def draw(self):
        """Draws the coupling map.

        This function needs `pydot <https://github.com/erocarrera/pydot>`_,
        which in turn needs `Graphviz <https://www.graphviz.org/>`_ to be
        installed. Additionally, `pillow <https://python-pillow.org/>`_ will
        need to be installed.

        Returns:
            PIL.Image: Drawn coupling map.

        Raises:
            ImportError: when pydot or pillow are not installed.
        """

        try:
            import pydot  # pylint: disable=unused-import
            from PIL import Image
        except ImportError:
            raise ImportError("CouplingMap.draw requires pydot and pillow. "
                              "Run 'pip install pydot pillow'.")

        dot = nx.drawing.nx_pydot.to_pydot(self.graph)
        png = dot.create_png(prog='neato')

        return Image.open(io.BytesIO(png))

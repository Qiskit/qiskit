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

"""
Subgraph of CouplingMap...

The nodes of the graph correspond to physical qubits (represented as integers) and the
directed edges indicate which physical qubits are coupled and the permitted direction of
CNOT gates. The object has a distance function that can be used to map quantum circuits
onto a device with this coupling.
"""

import copy
import io
import itertools
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import retworkx as rx

from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.transpiler.exceptions import CouplingError
from .coupling import CouplingMap


class SubCouplingMap(CouplingMap):
    """
    Directed graph specifying fixed coupling.

    Nodes correspond to physical qubits (integers) and directed edges correspond
    to permitted CNOT gates, with source and destination corresponding to control
    and target qubits, respectively.
    """

    __slots__ = ("_qubit_to_idx",)

    def __init__(
        self,
        coupling: Optional[Union[List[Sequence[int]], CouplingMap]] = None,
        qubit_list: Optional[Sequence[int]] = None,
        description: Optional[str] = None,
    ):
        """
        Create a coupling subgraph. By default, it is the same as `coupling`.

        Args:
            coupling: An original coupling graph, specified as
                an adjacency list containing couplings, e.g. [[0,1], [0,2], [1,2]].
            qubit_list: Sub-sequence of device qubits corresponding to the nodes of the subgraph.
                If `None`, all device qubits in the original coupling are used.
            description: A string to describe the coupling map.
        """
        super().__init__(couplinglist=None, description=description)

        if coupling is None:
            edge_list = []
        elif isinstance(coupling, CouplingMap):
            edge_list = coupling.get_edges()
        else:
            edge_list = coupling

        if qubit_list is None:
            if isinstance(coupling, CouplingMap):
                qubit_list = copy.deepcopy(coupling.physical_qubits)
            else:
                qubit_list = list(set(itertools.chain.from_iterable(edge_list)))

        self._qubit_list = list(qubit_list)  # node index -> qubit
        self._qubit_to_idx = {qubit: idx for idx, qubit in enumerate(qubit_list)}

        reduced_edge_list = []
        for edge in edge_list:
            if edge[0] in qubit_list and edge[1] in qubit_list:
                reduced_edge_list.append((self._qubit_to_idx[edge[0]], self._qubit_to_idx[edge[1]]))

        self.graph.add_nodes_from(qubit_list)
        self.graph.add_edges_from_no_data(reduced_edge_list)

    def get_edges(self) -> List[Tuple[int, int]]:
        """
        Gets the list of edges in the coupling graph.

        Returns:
            Each edge is a pair of physical qubits.
        """
        return [(self._qubit_list[i], self._qubit_list[j]) for i, j in self.graph.edge_list()]

    def add_physical_qubit(self, physical_qubit: int):
        """Add a physical qubit to the sub-coupling graph as a node.

        Args:
            physical_qubit: An integer representing a physical qubit to be added.

        Raises:
            CouplingError: if trying to add duplicate qubit
        """
        if not isinstance(physical_qubit, int):
            raise CouplingError("Physical qubits should be integers.")
        if physical_qubit in self.physical_qubits:
            raise CouplingError(f"The physical qubit {physical_qubit} is already in the graph")
        new_node_idx = self.size()
        self.graph.add_node(physical_qubit)
        self._qubit_to_idx[physical_qubit] = new_node_idx
        self._qubit_list.append(physical_qubit)
        self._dist_matrix = None  # invalidate
        self._size = None  # invalidate

    def remove_physical_qubit(self, physical_qubit: int):
        """Remove a physical qubit (node) from the sub-coupling graph.

        Args:
            physical_qubit: An integer representing a physical qubit to be removed.

        Raises:
            CouplingError: if trying to remove qubit not in the graph
        """
        if not isinstance(physical_qubit, int):
            raise CouplingError("Physical qubits should be integers.")
        if physical_qubit not in self.physical_qubits:
            raise CouplingError(f"The physical qubit {physical_qubit} is not in the graph")
        self.graph.remove_node(self._qubit_to_idx.pop(physical_qubit))
        self._qubit_list.remove(physical_qubit)
        self._dist_matrix = None  # invalidate
        self._size = None  # invalidate

    def add_edge(self, src: int, dst: int):
        """
        Add directed edge to sub-coupling graph.

        Args:
            src: source physical qubit
            dst: destination physical qubit
        """
        if src not in self.physical_qubits:
            self.add_physical_qubit(src)
        if dst not in self.physical_qubits:
            self.add_physical_qubit(dst)
        self.graph.add_edge(self._qubit_to_idx[src], self._qubit_to_idx[dst], None)
        self._dist_matrix = None  # invalidate
        self._is_symmetric = None  # invalidate

    def remove_edge(self, src: int, dst: int):
        """
        Remove directed edge from sub-coupling graph.
        Note that the removal may produce unconnected components (e.g. isolated note).

        Args:
            src: source physical qubit
            dst: destination physical qubit

        Raises:
            CouplingError: if either of physical qubits of the edge is not in the graph
            CouplingError: if the edge (src, dst) is not in the graph
        """
        if src not in self.physical_qubits:
            raise CouplingError(f"The physical qubit {src} is not in the graph")
        if dst not in self.physical_qubits:
            raise CouplingError(f"The physical qubit {dst} is not in the graph")
        edge_indices = self._qubit_to_idx[src], self._qubit_to_idx[dst]
        if not self.graph.has_edge(*edge_indices):
            raise CouplingError(f"The edge ({src}, {dst}) is not in the graph")
        self.graph.remove_edge(*edge_indices)
        self._dist_matrix = None  # invalidate
        self._is_symmetric = None  # invalidate

    @property
    def physical_qubits(self) -> List[int]:
        """Returns a list of physical_qubits"""
        return self._qubit_list

    def neighbors(self, physical_qubit: int) -> List[int]:
        """Return the nearest neighbor qubits of a physical qubit.

        Directionality matters, i.e. a neighbor must be reachable
        by going one hop in the direction of an edge.

        Returns:
            The nearest neighbor quibts of a physical qubit.

        Raises:
            CouplingError: if the `physical_qubit` is not in this graph
        """
        if physical_qubit not in self.physical_qubits:
            raise CouplingError(f"physical qubit {physical_qubit} is not in the graph")

        return [
            self._qubit_list[i] for i in self.graph.neighbors(self._qubit_to_idx[physical_qubit])
        ]

    def compute_distance_matrix(self):
        """Compute the full distance matrix on pairs of nodes.

        The distance map self._dist_matrix is computed from the graph using
        all_pairs_shortest_path_length. This is normally handled internally
        by the :attr:`~qiskit.transpiler.CouplingMap.distance_matrix`
        attribute or the :meth:`~qiskit.transpiler.CouplingMap.distance` method
        but can be called if you're accessing the distance matrix outside of
        those or want to pre-generate it.
        """
        if self._dist_matrix is None:
            if not self.is_connected():
                raise CouplingError("coupling graph not connected")
            self._dist_matrix = rx.digraph_distance_matrix(self.graph, as_undirected=True)

            # enlarge matrix by changing matrix index from node index to qubit
            size = max(self.physical_qubits) + 1
            new_matrix = np.full((size, size), np.inf)
            for i in self.physical_qubits:
                for j in self.physical_qubits:
                    new_matrix[i, j] = self._dist_matrix[
                        self._qubit_to_idx[i], self._qubit_to_idx[j]
                    ]

            self._dist_matrix = new_matrix

    def shortest_undirected_path(self, physical_qubit1: int, physical_qubit2: int) -> List[int]:
        """Returns the shortest undirected path between physical_qubit1 (source) and
        physical_qubit2 (target).

        Args:
            physical_qubit1: A physical qubit to be a source of the path
            physical_qubit2: Another physical qubit to be a target of the path
        Returns:
            The shortest undirected path
        Raises:
            CouplingError: If source or target qubit is not in this graph.
            CouplingError: When there is no path between source and target.
        """
        if physical_qubit1 not in self.physical_qubits:
            raise CouplingError(f"Source qubit {physical_qubit1} is not in the graph")
        if physical_qubit2 not in self.physical_qubits:
            raise CouplingError(f"Target qubit {physical_qubit2} is not in the graph")

        paths = rx.digraph_dijkstra_shortest_paths(
            self.graph,
            source=self._qubit_to_idx[physical_qubit1],
            target=self._qubit_to_idx[physical_qubit2],
            as_undirected=True,
        )
        if not paths:
            raise CouplingError(
                f"Nodes {str(physical_qubit1)} and {str(physical_qubit2)} are not connected"
            )
        return paths[self._qubit_to_idx[physical_qubit2]]

    def reduce(self, mapping: List[int]) -> "SubCouplingMap":
        """Returns a reduced sub-coupling map that
        corresponds to the subgraph of qubits selected in the mapping.

        Args:
            mapping: A mapping of reduced qubits to device qubits.

        Returns:
            A reduced coupling_map for the selected qubits.

        Raises:
            CouplingError: If the reduced coupling map is not connected.
        """
        reduced_graph = SubCouplingMap(coupling=self, qubit_list=mapping)

        if not reduced_graph.is_connected():
            raise CouplingError("Reduced coupling map must be connected.")

        return reduced_graph

    def largest_connected_component(self) -> List[int]:
        """Return a set of qubits in the largest connected component."""
        return [
            self._qubit_list[i] for i in max(rx.weakly_connected_components(self.graph), key=len)
        ]

    def __str__(self):
        """Return a string representation of the sub-coupling graph."""
        string = "["
        string += ", ".join([f"[{src}, {dst}]" for (src, dst) in self.get_edges()])
        string += "]"
        return string

    def draw(self):
        """Draws the sub-coupling map.

        This function needs `pydot <https://github.com/erocarrera/pydot>`_,
        which in turn needs `Graphviz <https://www.graphviz.org/>`_ to be
        installed. Additionally, `pillow <https://python-pillow.org/>`_ will
        need to be installed.

        Returns:
            PIL.Image: Drawn sub-coupling map.

        Raises:
            MissingOptionalLibraryError: when pydot or pillow are not installed.
        """
        try:
            import pydot
        except ImportError as ex:
            raise MissingOptionalLibraryError(
                libname="pydot",
                name="coupling map drawer",
                pip_install="pip install pydot",
            ) from ex

        try:
            from PIL import Image
        except ImportError as ex:
            raise MissingOptionalLibraryError(
                libname="pillow",
                name="coupling map drawer",
                pip_install="pip install pillow",
            ) from ex
        dot_str = self.graph.to_dot(node_attr=lambda node: {"label": str(node)})
        dot = pydot.graph_from_dot_data(dot_str)[0]
        png = dot.create_png(prog="neato")

        return Image.open(io.BytesIO(png))

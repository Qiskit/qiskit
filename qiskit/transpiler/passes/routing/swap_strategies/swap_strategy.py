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

"""Defines a swap strategy class."""

from typing import Any, List, Optional, Set, Tuple
import copy
import numpy as np

from qiskit import QiskitError
from qiskit.transpiler import CouplingMap


class SwapStrategy:
    """A class representing swap strategies for coupling maps.

    A swap strategy is a list of swap layers to apply to the physical coupling map. Each swap layer
    is specified by a set of tuples which correspond to the edges of the physical coupling map that
    are swapped. At each swap layer SWAP gates are applied to the corresponding edges. This class
    stores the permutations of the qubits resulting from the swap strategy. Swap strategies are
    intended to be used on blocks of commuting gates which are often present in variational
    algorithms.
    """

    def __init__(self, coupling_map: CouplingMap, swap_layers: List[List[Tuple[int, int]]]) -> None:
        """
        Args:
            coupling_map: The coupling map the strategy is implemented for.
            swap_layers: The swap layers of the strategy, specified as a list of sets of
                edges (edges can be represented as lists, sets or tuples containing two integers).

        Raises:
            QiskitError: If the coupling map is not specified.
            QiskitError: if the swap strategy is not valid. A swap strategy is valid if all
                swap gates, specified as tuples, are contained in the edge set of the coupling map.
        """
        self._coupling_map = coupling_map
        self._num_vertices = coupling_map.size()
        self._swap_layers = swap_layers
        self._distance_matrix = None
        self._possible_edges = None
        self._inverse_composed_permutation = {0: list(range(self._num_vertices))}

        edge_set = set(self._coupling_map.get_edges())

        for i, layer in enumerate(self._swap_layers):
            for edge in layer:
                if edge not in edge_set:
                    raise QiskitError(
                        f"The {i}th SWAP layer contains the edge {edge} which is not "
                        f"part of the underlying coupling map with {edge_set} edges."
                    )

    def __len__(self) -> int:
        """Return the length of the strategy as the number of layers.

        Returns:
            The number of layers of the swap strategy.
        """
        return len(self._swap_layers)

    def __repr__(self) -> str:
        """Representation of the swap strategy.

        Returns:
            The represenation of the swap strategy.
        """
        description = [f"{self.__class__.__name__} with swap layers:\n"]

        for layer in self._swap_layers:
            description.append(f"{layer},\n")

        description.append(f"on {self._coupling_map} coupling map.")
        description = "".join(description)

        return description

    def swap_layer(self, idx: int) -> List[Tuple[int, int]]:
        """Return the layer of swaps at the given index.

        Args:
            idx: The index of the returned swap layer.

        Returns:
            The swap layer at ``idx``.
        """
        return list(swap for swap in self._swap_layers[idx])

    @property
    def distance_matrix(self) -> List[List[int]]:
        """A matrix describing when qubits become adjacent in the swap strategy.

        Returns:
            The distance matrix for the SWAP strategy as a nested list. Here the entry (i,j)
            corresponds to the number of SWAP layers that need to be applied to obtain a connection
            between physical qubits i and j.
        """
        if self._distance_matrix is None:
            self._distance_matrix = [[None] * self._num_vertices for _ in range(self._num_vertices)]

            for i in range(self._num_vertices):
                self._distance_matrix[i][i] = 0

            for i in range(len(self._swap_layers) + 1):
                for j, k in self.swapped_coupling_map(i).get_edges():

                    # This if ensures that the smallest distance is used.
                    if self._distance_matrix[j][k] is None:
                        self._distance_matrix[j][k] = i
                        self._distance_matrix[k][j] = i

        return self._distance_matrix

    def new_connections(self, idx: int) -> List[Set[int]]:
        """
        Returns the new connections obtained after applying the SWAP layer specified by idx, i.e.
        a list of qubit pairs that are adjacent to one another after idx steps of the SWAP strategy.

        Args:
            idx: The index of the SWAP layer. 1 refers to the first SWAP layer
                whereas an ``idx`` of 0 will return the connections present in the original coupling
                map.

        Returns:
            A list of edges representing the new qubit connections.
        """
        connections = []
        for i in range(self._num_vertices):
            for j in range(i):
                if self.distance_matrix[i][j] == idx:
                    connections.append({i, j})

        return connections

    def _build_edges(self) -> Set[Tuple[int, int]]:
        """Build the possible edges that the swap strategy accommodates."""

        possible_edges = set()
        for swap_layer_idx in range(len(self) + 1):
            for edge in self.swapped_coupling_map(swap_layer_idx).get_edges():
                possible_edges.add(edge)
                possible_edges.add(edge[::-1])

        return possible_edges

    @property
    def possible_edges(self) -> Set[Tuple[int, int]]:
        """Return the qubit connections that can be generated.

        Returns:
            The qubit connections that can be accomodated by the swap strategy.
        """
        if self._possible_edges is None:
            self._possible_edges = self._build_edges()

        return self._possible_edges

    def missing_couplings(self) -> Set[Tuple[int, int]]:
        """Compute the set of couplings that cannot be reached.

        Returns:
            The couplings that cannot be reached as a set of Tuples of int. Here,
            each int corresponds to a qubit in the coupling map.
        """
        physical_qubits = list(set(sum(self._coupling_map.get_edges(), ())))
        missed_edges = set()
        for i, physical_qubit_i in enumerate(physical_qubits):
            for j in range(i + 1, len(physical_qubits)):
                missed_edges.add((physical_qubit_i, physical_qubits[j]))
                missed_edges.add((physical_qubits[j], physical_qubit_i))

        for layer_idx in range(len(self) + 1):
            for edge in self.new_connections(layer_idx):
                for edge_tuple in [tuple(edge), tuple(edge)[::-1]]:
                    missed_edges.discard(edge_tuple)

        return missed_edges

    @property
    def reaches_full_connectivity(self) -> bool:
        """Returns whether the swap strategy reaches full connectivity.

        Returns:
            True if the swap strategy reaches full connectivity and False otherwise.
        """
        return len(self.missing_couplings()) == 0

    def swapped_coupling_map(self, idx: int) -> CouplingMap:
        """Returns the coupling map after applying ``idx`` swap layers of strategy.

        Args:
            idx: The number of swap layers to apply. For idx = 0, the original coupling
                map is returned.

        Returns:
            The swapped coupling map.
        """
        permutation = self.inverse_composed_permutation(idx)

        edges = [[permutation[i], permutation[j]] for i, j in self._coupling_map.get_edges()]

        return CouplingMap(couplinglist=edges)

    def apply_swap_layer(self, list_to_swap: List[Any], idx: int) -> List[Any]:
        """Permute the elements of ``list_to_swap`` based on layer indexed by ``idx``.

        Args:
            list_to_swap: The list of elements to swap.
            idx: The index of the swap layer to apply.

        Returns:
            The list with swapped elements
        """
        x = copy.copy(list_to_swap)

        for i, j in self._swap_layers[idx]:
            x[i], x[j] = x[j], x[i]

        return x

    def composed_permutation(self, idx: int) -> List[int]:
        """Returns the composed permutation of all swap layers applied up to index ``idx``.

        Permutations are represented by list of integers where the ith element
        corresponds to the mapping of i under the permutation.

        Args:
            idx: The number of SWAP layers to apply.

        Returns:
            The permutation as a list of integer values.
        """
        return list(np.argsort(self.inverse_composed_permutation(idx)))

    def inverse_composed_permutation(self, idx: int) -> List[int]:
        """
        Returns the inversed composed permutation of all swap layers applied up to layer
        ``idx``. Permutations are represented by list of integers where the ith element
        corresponds to the mapping of i under the permutation.

        Args:
            idx: The number of swap layers to apply.

        Returns:
            The inversed permutation as a list of integer values.
        """
        # Only compute the inverse permutation if it has not been computed before
        if idx not in self._inverse_composed_permutation:
            self._inverse_composed_permutation[idx] = self.apply_swap_layer(
                self.inverse_composed_permutation(idx - 1), idx - 1
            )

        return self._inverse_composed_permutation[idx]


class LineSwapStrategy(SwapStrategy):
    """An optimal SWAP strategy for a line."""

    def __init__(self, line: List[int], num_swap_layers: Optional[int] = None) -> None:
        """
        Creates a swap strategy for a line graph with the specified number of SWAP layers.
        This SWAP strategy will use the full line if instructed to do so (i.e. num_variables
        is None or equal to num_vertices). If instructed otherwise then the first num_variables
        nodes of the line will be used in the swap strategy.

        Args:
            line: A line given as a list of nodes, e.g. ``[0, 2, 3, 4]``.
            num_swap_layers: Number of swap layers the swap manager should be initialized with.

        Raises:
            ValueError: If the ``num_swap_layers`` is negative.
            ValueError: If the ``line`` has less than 2 elements and no swap strategy can be applied.
        """
        if len(line) < 2:
            raise ValueError(f"The line cannot have less than two elements, but is {line}")

        if num_swap_layers is None:
            num_swap_layers = len(line) - 2

        elif num_swap_layers < 0:
            raise ValueError(f"Negative number {num_swap_layers} passed for number of swap layers.")

        swap_layer0 = [(line[i], line[i + 1]) for i in range(0, len(line) - 1, 2)]
        swap_layer1 = [(line[i], line[i + 1]) for i in range(1, len(line) - 1, 2)]

        base_layers = [swap_layer0, swap_layer1]

        swap_layers = [base_layers[i % 2] for i in range(num_swap_layers)]

        couplings = []
        for idx in range(len(line) - 1):
            couplings.append((line[idx], line[idx + 1]))
            couplings.append((line[idx + 1], line[idx]))

        super().__init__(coupling_map=CouplingMap(couplings), swap_layers=swap_layers)

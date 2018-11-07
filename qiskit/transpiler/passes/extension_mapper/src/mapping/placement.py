"""Placement is the process of placing gates at a location.

It includes functionality to compute permutations needed for a certain placement of gates
and also circuits to implement the Placement."""
from typing import TypeVar, Generic, Mapping, Callable, Iterable, List, Union, Any, Tuple

import networkx as nx

from .. import permutation as pm
from ..permutation import Swap

Reg = TypeVar("Reg")
ArchNode = TypeVar('ArchNode')
_V = TypeVar("_V")


class Placement(Generic[Reg, ArchNode]):  # pylint: disable=unsubscriptable-object
    """A Placement represents a placement of quantum registers on architecture graph nodes.

    IDEA: Replace with NamedTuple once it supports constructors/preconditions.
    TODO: Get rid of current_mapping and mapped_to fields.
    """

    def __init__(self, current_mapping: Mapping[Reg, ArchNode],
                 mapped_to: Mapping[Reg, ArchNode]) -> None:
        """Construct a Placement object.


        :param current_mapping: A mapping of qubits that are being placed.
        :param mapped_to: A Map specifying where (some of) the qubits are being mapped to.
        """
        assert set(current_mapping.keys()).issuperset(set(mapped_to.keys())), \
            "The qubits in the current mapping must be a superset of the mapped qubits."
        self.current_mapping = current_mapping
        self.mapped_to = mapped_to
        # The mapping that is defined by this placement.
        # It is used to define the hash and equality,
        # since a same arch_mapping would generate the same mapping circuit.
        self.arch_mapping = {self.current_mapping[k]: v for k, v in mapped_to.items()}

        # The class fields are immutable so we precompute a fixed hash.
        self._hash = hash(frozenset(self.arch_mapping.items()))

    def is_local(self, arch_graph: nx.Graph) -> bool:
        """Checks if the current placement only has local gates and does not need a permutation"""
        if len(self.current_mapping) < 2:
            return True
        return list(self.current_mapping.values()) in arch_graph.edges

    def __add__(self, other: 'Placement[Reg, ArchNode]') -> 'Placement[Reg, ArchNode]':
        """Construct a placement that combines this placement and the other placement.

        The other permutation must not also place the same registers as this placement."""
        # Make sure each placement does not intersect.
        assert all(v == other.current_mapping[k]
                   for k, v in self.current_mapping.items() if k in other.current_mapping), \
            "The other placement's current_mapping is different from self."
        assert self.mapped_to.keys().isdisjoint(other.mapped_to.keys()), \
            "Placements are not distinct."
        assert set(self.mapped_to.values()).isdisjoint(other.mapped_to.values()), \
            "Placements are not distinct."
        return Placement({**self.current_mapping, **other.current_mapping},
                         {**self.mapped_to, **other.mapped_to})

    def place(self, permutation: pm.Permutation[ArchNode]) -> None:
        """Place this placement in the permutation.

            To place the qubits in the mapping and keep the mapping consistent,
            we need to place the qubits on their mapped architecture nodes
            and then remap the previously mapped qubits to those nodes
            to the spots that have opened up.

            IDEA: Implement as sympy Permutations: (succ(current),mapped_to)*C1*C2

            :param permutation: A permutation of nodes in the architecture graph.
        """
        inv_permutation = {v: k for k, v in permutation.items()}
        for current_map, mapped_to in self.arch_mapping.items():
            prev_to = inv_permutation[mapped_to]
            succ_current = permutation[current_map]
            # prev(mapped_to) -> succ(current)
            permutation[prev_to] = succ_current
            # current -> mapped_to
            permutation[current_map] = mapped_to
            # Now we have: .... -> current -> mapped_to -> succ(mapped_to) -> ... -> prev(mapped_to)
            # -> succ(current) -> ...
            # Also update inverse permutation
            inv_permutation[mapped_to] = current_map
            inv_permutation[succ_current] = prev_to

    def mapping_circuit(self,
                        arch_permuter: Callable[[Mapping[ArchNode, ArchNode]],
                                                Iterable[List[pm.Swap[ArchNode]]]],
                        allow_swaps: bool = False) \
            -> Tuple[Iterable[List[Swap[ArchNode]]], 'pm.util.PermutationCircuit']:
        """Construct a circuit that implements this placement as a partial mapping.

        :param arch_permuter: The permuter for the architecture graph, must support partial
            mappings.
        :param allow_swaps:
        :return: Tuple of swaps and mapping_circuit implementing the placement."""
        # Construct the partial mapping of architecture nodes.
        swaps = list(arch_permuter(self.arch_mapping))
        mapping_circuit = pm.util.circuit(swaps, allow_swaps=allow_swaps)
        return swaps, mapping_circuit

    def permutation_circuit(self,
                            arch_permuter: Callable[[pm.Permutation[ArchNode]],
                                                    Iterable[List[pm.Swap[ArchNode]]]],
                            arch_graph: Union[nx.Graph, nx.DiGraph],
                            allow_swaps: bool = False) -> 'pm.util.PermutationCircuit':
        """Construct a circuit that implements this placement as a (complete) permutation.

        :param arch_permuter: The permuter of this circuit.
            Also could support partial mappings. Function arguments are contravariant.
        :param arch_graph: The architecture graph. Used to complete the mapping.
        :param allow_swaps:
        :return: Mapping circuit."""
        arch_perm = {i: i for i in arch_graph.nodes}
        self.place(arch_perm)
        swaps = arch_permuter(arch_perm)
        mapping_circuit = pm.util.circuit(swaps, allow_swaps=allow_swaps)
        return mapping_circuit

    def __str__(self) -> str:
        return f"Placement(permutation:{self.arch_mapping}, " \
               f"mapped_to:{self.mapped_to}, current_mapping:{self.current_mapping})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Placement):
            return False
        return self.arch_mapping == other.arch_mapping

    def __hash__(self) -> int:
        return self._hash

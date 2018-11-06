"""Utility functions shared between permutation functionality."""
import itertools
from typing import List, Tuple, Dict, TypeVar, Iterable, Iterator, NamedTuple, Any, Callable, \
    Mapping, MutableMapping

import networkx as nx
from qiskit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.mapper._mapping import cx_data, swap_data

from .. import mapping
from .. import permutation as pm

_K = TypeVar('_K')
_V = TypeVar('_V')


def cycles(permutation: pm.Permutation[_V]) -> List[pm.Permutation[_V]]:
    """
    Break down the given permutation into cyclic permutations.

    :param permutation:
    :return:
    """

    # Keep track of which items we haven't entered into a cycle yet, in order.
    todo = permutation.copy()
    found_cycles: List[pm.Permutation[_V]] = []
    while todo:
        # Take the first of the next cycle and follow permutations until the cycle has completed.
        first, mapped_to = todo.popitem()
        current = first
        current_cycle = {current: mapped_to}
        while mapped_to != first:
            current = mapped_to
            mapped_to = todo.pop(current)  # Remove the item from todos
            current_cycle[current] = mapped_to

        found_cycles.append(current_cycle)

    return found_cycles


def flatten_swaps(swaps: Iterable[Iterable[List[pm.Swap[_V]]]]) -> Iterable[List[pm.Swap[_V]]]:
    """
    Transform multiple sequences of swaps that are disjoint into one sequence of swaps.

    :param swaps:
    :return:
    """

    parallel = itertools.zip_longest(*swaps, fillvalue=[])  # Each time step zipped together
    # Make one list for each timestep.
    return map(lambda el: list(itertools.chain.from_iterable(el)), parallel)


def optimize_swaps(swaps: Iterable[Iterable[pm.Swap[_V]]]) -> Iterator[List[pm.Swap[_V]]]:
    """
    Remove empty steps in the given sequence of swaps.

    :param swaps:
    :return:
    """
    return filter(lambda el: len(el) > 0, map(list, swaps))


def swap_permutation(swaps: Iterable[Iterable[pm.Swap[_K]]], mapping: MutableMapping[_K, _V],
                     allow_missing_keys: bool = False) -> None:
    """
    Given a circuit of swaps, apply them to the permutation (in-place).

    :param swaps:
    :param mapping: A mapping of Keys to Values, where the Keys are being swapped.
    :param allow_missing_keys:
    """
    for swap_step in swaps:
        for swap in swap_step:
            sw1, sw2 = swap
            # Take into account non-existent keys.
            if allow_missing_keys:
                val1, val2 = mapping.pop(sw1, None), mapping.pop(sw2, None)
            else:
                val1, val2 = mapping.pop(sw1), mapping.pop(sw2)

            if val1 is not None:
                mapping[sw2] = val1
            if val2 is not None:
                mapping[sw1] = val2


class PermutationCircuit(NamedTuple):
    """Represents a circuit for permuting to a mapping with the associated cost."""
    circuit: DAGCircuit
    inputmap: Dict[Any, Tuple[Any, int]]  # A mapping from architecture nodes to circuit registers.


def circuit(swaps: Iterable[List[pm.Swap[_V]]], allow_swaps: bool = False) \
        -> PermutationCircuit:
    """Produce a circuit description of a list of swaps.

        With a given permutation and permuter you can compute the swaps using the permuter function
        then feed it into this circuit function to obtain a circuit description.

        :param swaps: An iterable of swaps to perform.
        :param allow_swaps: Will use "cx" in basis of circuit by default,
            but if swaps are allowed, will use those instead.
        :return: A MappingCircuit with the circuit and a mapping of node to qubit in the circuit.
    """
    # Construct a circuit with each unique node id becoming a quantum register of size 1.
    dag = DAGCircuit()
    dag.add_basis_element('cx', 2)
    dag.add_gate_data('cx', cx_data)
    if allow_swaps:
        dag.add_basis_element('swap', 2)
        dag.add_gate_data('swap', swap_data)

    swap_list = list(swaps)

    # Set of unique nodes used in the swaps.
    nodes = {
        swap_node
        for swap_step in swap_list
        for swap_nodes in swap_step
        for swap_node in swap_nodes
        }

    inputmap: Dict[_V, Tuple[_V, int]] = {}

    for node in nodes:
        dag.add_qreg(QuantumRegister(name='q%d' % node, size=1))
        inputmap[node] = ('q%d' % node, 0)

    def swap(node0: Tuple[_V, int], node1: Tuple[_V, int]) \
            -> List[Tuple[str, List[Tuple[_V, int]]]]:
        """A circuit implementing a SWAP in the basis."""
        if "swap" in dag.basis:
            return [("swap", [node0, node1])]

        return [
            ("cx", [node0, node1]),
            ("cx", [node1, node0]),
            ("cx", [node0, node1])
            ]

    # Apply swaps to the circuit.
    for swap_step in swap_list:
        for swap0, swap1 in swap_step:
            for gate in swap(('q%d' % swap0, 0), ('q%d' % swap1, 0)):
                dag.apply_operation_back(*gate)

    return PermutationCircuit(dag, inputmap)


def sequential_permuter(permuter: Callable[[pm.Permutation[_V]], Iterable[List[pm.Swap]]],
                        arch_graph: nx.Graph) \
        -> Callable[[Mapping[_V, _V]], Iterable[pm.Swap[_V]]]:
    """Construct a sequential permuter from a parallel permuter.

    A sequential permuter takes in a (partial) mapping instead of a permutation of all nodes."""

    def mapping_permuter(mapping: Mapping[_V, _V]) -> Iterable[pm.Swap[_V]]:
        # Extend mapping to permutation
        id_mapping = {i: i for i in mapping}
        placement = src.mapping.placement.Placement(id_mapping, mapping)
        permutation = {i: i for i in arch_graph.nodes()}
        placement.place(permutation)
        parallel_swaps = permuter(permutation)
        # Make sequential
        return (swap for swap_step in parallel_swaps for swap in swap_step)

    return mapping_permuter

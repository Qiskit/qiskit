"""Utility functions shared between permutation functionality."""
#  arct performs circuit transformations of quantum circuit for architectures
#  Copyright (C) 2019  Andrew M. Childs, Eddie Schoute, Cem M. Unsal
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import itertools
from collections import defaultdict
from typing import List, Tuple, Dict, TypeVar, Iterable, Iterator, NamedTuple, Any, Callable, \
    Mapping, MutableMapping, Optional, Generic
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.extensions import SwapGate

import qiskit.transpiler.routing as rt

_K = TypeVar('_K')
_V = TypeVar('_V')
Reg = Tuple[str, int]


def cycles(permutation: rt.Permutation[_V]) -> List[rt.Permutation[_V]]:
    """
    Break down the given permutation into cyclic permutations.

    :param permutation:
    :return:
    """

    # Keep track of which items we haven't entered into a cycle yet, in order.
    todo = permutation.copy()
    found_cycles: List[rt.Permutation[_V]] = []
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


def flatten_swaps(swaps: Iterable[Iterable[List[rt.Swap[_V]]]]) -> Iterable[List[rt.Swap[_V]]]:
    """
    Transform multiple sequences of swaps that are disjoint into one sequence of swaps.

    :param swaps:
    :return:
    """

    parallel = itertools.zip_longest(*swaps, fillvalue=[])  # Each time step zipped together
    # Make one list for each timestep.
    return (list(itertools.chain.from_iterable(time_step)) for time_step in parallel)


def optimize_swaps(swaps: Iterable[Iterable[rt.Swap[_V]]]) -> Iterator[List[rt.Swap[_V]]]:
    """
    Remove empty steps in the given sequence of swaps.

    :param swaps:
    :return:
    """
    return (el for el in (list(swap) for swap in swaps) if len(el) > 0)


def swap_permutation(swaps: Iterable[Iterable[rt.Swap[_K]]], mapping: MutableMapping[_K, _V],
                     allow_missing_keys: bool = False) -> None:
    """
    Given a circuit of swaps, apply them to the permutation (in-place).

    :param swaps:
    :param mapping: A mapping of Keys to Values, where the Keys are being swapped.
    :param allow_missing_keys:
    """
    for swap_step in swaps:
        for sw1, sw2 in swap_step:
            # Take into account non-existent keys.
            val1: Optional[_V]
            val2: Optional[_V]
            if allow_missing_keys:
                val1 = mapping.pop(sw1, None)
                val2 = mapping.pop(sw2, None)
            else:
                # Asserts that both keys exist
                val1, val2 = mapping.pop(sw1), mapping.pop(sw2)

            if val1 is not None:
                mapping[sw2] = val1
            if val2 is not None:
                mapping[sw1] = val2


@dataclass(frozen=True)
class PermutationCircuit(Generic[_V]):
    """Represents a circuit for permuting to a mapping with the associated cost."""
    circuit: DAGCircuit
    inputmap: Dict[_V, Reg]  # A mapping from architecture nodes to circuit registers.


def circuit(swaps: Iterable[List[rt.Swap[_V]]]) \
        -> PermutationCircuit[_V]:
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
    swap_list = list(swaps)

    # Set of unique nodes used in the swaps.
    nodes = {
        swap_node
        for swap_step in swap_list
        for swap_nodes in swap_step
        for swap_node in swap_nodes
        }

    node_qargs = {node: QuantumRegister(1) for node in nodes}
    for qubit in node_qargs.values():
        dag.add_qreg(qubit)

    inputmap = {node: q[0] for node, q in node_qargs.items()}

    # Apply swaps to the circuit.
    for swap_step in swap_list:
        for swap0, swap1 in swap_step:
            dag.apply_operation_back(SwapGate(), [inputmap[swap0], inputmap[swap1]])

    return PermutationCircuit(dag, inputmap)


def sequential_permuter(permuter: Callable[[Mapping[_V, _V]], Iterable[List[rt.Swap[_V]]]]) \
        -> Callable[[Mapping[_V, _V]], Iterable[rt.Swap[_V]]]:
    """Construct a partial sequential permuter from a partial parallel permuter."""

    def mapping_permuter(mapping: Mapping[_V, _V]) -> Iterable[rt.Swap[_V]]:
        return (swap for swap_step in permuter(mapping) for swap in swap_step)

    return mapping_permuter


def longest_path(swaps: Iterable[List[rt.Swap[_V]]]) -> int:
    """Compute the longest path in the DAG defined by the sequence of swaps."""
    length: Dict[_V, int] = defaultdict(lambda: 0)
    for time_step in swaps:
        for sw0, sw1 in time_step:
            length[sw0] = length[sw1] = max(length[sw0], length[sw1]) + 1
    return max(length.values(), default=0)


def random_partial_permutation(domain: List[_V],
                               nr_elements: Optional[int] = None) -> Dict[_V, _V]:
    """Construct a random partial permutation

    :param domain: All elements
    :param nr_elements: Number of mappings in the partial permutation. If None, will pick include
    a mapping of a node with probability 0.5
    """
    if nr_elements is None:
        nr_elements = np.random.binomial(len(domain), 0.5)
    elif nr_elements > len(domain):
        raise ValueError("Too many elements for the partial permutation")

    sampled = np.random.choice(domain, size=nr_elements, replace=False)
    destinations = np.random.choice(domain, size=nr_elements, replace=False)

    return dict(zip(sampled, destinations))

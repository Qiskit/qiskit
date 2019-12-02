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

# Copyright 2019 Andrew M. Childs, Eddie Schoute, Cem M. Unsal
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

"""Utility functions shared between permutation functionality."""

import itertools
from collections import defaultdict
from typing import List, Tuple, Dict, TypeVar, Iterable, Iterator, NamedTuple, Any, Callable, \
    Mapping, MutableMapping, Optional

import numpy as np

import qiskit.transpiler.routing as rt
from qiskit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.extensions import SwapGate

_K = TypeVar('_K')
_V = TypeVar('_V')
Reg = Tuple[str, int]


def cycles(permutation: rt.Permutation[_V]) -> List[rt.Permutation[_V]]:
    """Break down the given permutation into cyclic permutations."""

    # Keep track of which items we haven't entered into a cycle yet, in order.
    todo = permutation.copy()
    found_cycles = []  # type: List[rt.Permutation[_V]]
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
    """Transform multiple sequences of swaps that are disjoint into one sequence of swaps."""

    parallel = itertools.zip_longest(*swaps, fillvalue=[])  # Each time step zipped together
    # Make one list for each timestep.
    return (list(itertools.chain.from_iterable(time_step)) for time_step in parallel)


def optimize_swaps(swaps: Iterable[Iterable[rt.Swap[_V]]]) -> Iterator[List[rt.Swap[_V]]]:
    """Remove empty steps in the given sequence of swaps."""
    return (el for el in (list(swap) for swap in swaps) if len(el) > 0)


def swap_permutation(swaps: Iterable[Iterable[rt.Swap[_K]]],
                     mapping: MutableMapping[_K, _V],
                     allow_missing_keys: bool = False) -> None:
    """Given a circuit of swaps, apply them to the permutation (in-place).

    Args:
      swaps: param mapping: A mapping of Keys to Values, where the Keys are being swapped.
      mapping: The permutation to have swaps applied to.
      allow_missing_keys: Whether to allow swaps of missing keys in mapping.
    """
    for swap_step in swaps:
        for sw1, sw2 in swap_step:
            # Take into account non-existent keys.
            val1 = None  # type: Optional[_V]
            val2 = None  # type: Optional[_V]
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


# Represents a circuit for permuting to a mapping with the associated cost.
PermutationCircuit = NamedTuple('PermutationCircuit',
                                [('circuit', DAGCircuit),
                                 ('inputmap', Dict[Any, Reg])
                                 # A mapping from architecture nodes to circuit registers.
                                 ])


def circuit(swaps: Iterable[List[rt.Swap[_V]]]) -> PermutationCircuit:
    """Produce a circuit description of a list of swaps.

        With a given permutation and permuter you can compute the swaps using the permuter function
        then feed it into this circuit function to obtain a circuit description.

    Args:
      swaps: An iterable of swaps to perform.

    Returns:
      A MappingCircuit with the circuit and a mapping of node to qubit in the circuit.

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
    length = defaultdict(lambda: 0)  # type: Dict[_V, int]
    for time_step in swaps:
        for sw0, sw1 in time_step:
            length[sw0] = length[sw1] = max(length[sw0], length[sw1]) + 1
    return max(length.values(), default=0)


def random_partial_permutation(domain: List[_V],
                               nr_elements: Optional[int] = None) -> Dict[_V, _V]:
    """Construct a random partial permutation

    Args:
      domain: All elements
      nr_elements: Number of mappings in the partial permutation. If None, will pick include
    a mapping of a node with probability 0.5

    Returns:
      A partial permutation

    Raises:
      ValueError: If the domain is too smal to contain the number of elements.

    """
    if nr_elements is None:
        nr_elements = np.random.binomial(len(domain), 0.5)
    elif nr_elements > len(domain):
        raise ValueError("Too many elements for the partial permutation")

    sampled = np.random.choice(domain, size=nr_elements, replace=False)
    destinations = np.random.choice(domain, size=nr_elements, replace=False)

    return dict(zip(sampled, destinations))

"""Utility functions shared between permutation functionality."""
import itertools
from collections import namedtuple

from qiskit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.mapper._mapping import cx_data, swap_data


def cycles(permutation):
    """
    Break down the given permutation into cyclic permutations.

    :param permutation:
    :type permutation: pm.Permutation[_V]
    :return:
    :rtype: List[pm.Permutation[_V]]
    """

    # Keep track of which items we haven't entered into a cycle yet, in order.
    todo = permutation.copy()
    found_cycles = []
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


def flatten_swaps(swaps):
    """
    Transform multiple sequences of swaps that are disjoint into one sequence of swaps.

    :param swaps:
    :type swaps: Iterable[Iterable[List[pm.Swap[_V]]]]
    :return:
    :rtype: Iterable[List[pm.Swap[_V]]]
    """

    parallel = itertools.zip_longest(*swaps, fillvalue=[])  # Each time step zipped together
    # Make one list for each timestep.
    return map(lambda el: list(itertools.chain.from_iterable(el)), parallel)


def optimize_swaps(swaps):
    """
    Remove empty steps in the given sequence of swaps.

    :param swaps:
    :type swaps: Iterable[Iterable[pm.Swap[_V]]]
    :return:
    :rtype: Iterator[List[pm.Swap[_V]]]
    """
    return filter(lambda el: len(el) > 0, map(list, swaps))


def swap_permutation(swaps, mapping, allow_missing_keys=False):
    """
    Given a circuit of swaps, apply them to the permutation (in-place).

    :param swaps:
    :param mapping: A mapping of Keys to Values, where the Keys are being swapped.
    :param allow_missing_keys:
    :type swaps: Iterable[Iterable[pm.Swap[_K]]]
    :type mapping: MutableMapping[_K, _V]
    :type allow_missing_keys: bool
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


# Represents a circuit for permuting to a mapping with the associated cost.
# inputmap: A mapping from architecture nodes to circuit registers.
PermutationCircuit = namedtuple('PermutationCircuit', ['circuit', 'inputmap'])


def circuit(swaps, allow_swaps=False):
    """Produce a circuit description of a list of swaps.

        With a given permutation and permuter you can compute the swaps using the permuter function
        then feed it into this circuit function to obtain a circuit description.

        :param swaps: An iterable of swaps to perform.
        :param allow_swaps: Will use "cx" in basis of circuit by default,
            but if swaps are allowed, will use those instead.
        :type swaps: Iterable[List[pm.Swap[_V]]]
        :type allow_swaps: bool
        :return: A MappingCircuit with the circuit and a mapping of node to qubit in the circuit.
        :rtype: PermutationCircuit
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

    inputmap = {}

    for node in nodes:
        dag.add_qreg(QuantumRegister(name='q%d' % node, size=1))
        inputmap[node] = ('q%d' % node, 0)

    def swap(node0, node1):
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

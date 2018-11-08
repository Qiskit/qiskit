"""Compile a circuit to a given architecture.

The compiler takes in a given circuit and architecture graph to produce a circuit which can be
run on that architecture."""
import copy
import logging

import networkx as nx
from qiskit import QuantumRegister

from . import mapping as mp, util
from .mapping.placement import Placement

logger = logging.getLogger(__name__)


def compile_to_arch(circuit, arch_graph, arch_mapper):
    """
    Takes a circuit and compiles it to a circuit that can be run on the architecture.

    This compilation function optimizes for circuit depth.

    :param circuit: A DAGCircuit object containing the circuit to compile to the architecture.
    :param arch_graph: A graph that models the quantum architecture structure.
    :param arch_mapper: A function that given a (simple) circuit outputs a mapping
        and a PermutationCircuit for that mapping.
    :type circuit: DAGCircuit
    :type arch_graph: Union[nx.Graph, nx.DiGraph]
    :type arch_mapper: Callable[[DAGCircuit, Mapping[Reg[_V], ArchNode]],
                                Tuple[Mapping[Reg[_V], ArchNode], PermutationCircuit]]
    :return: The compiled circuit and a mapping of its registers to architecture graph qubits.
    :rtype: Tuple[DAGCircuit, Mapping[ArchReg, ArchNode]]
    """

    assert len(circuit.qregs) <= len(arch_graph.nodes), \
        "There must be more qubits in the architecture (%s) " \
        "than circuit (%s)." % (len(arch_graph.nodes), len(circuit.qregs))

    todo_circuit = copy.deepcopy(circuit)
    arch_qreg = QuantumRegister(len(arch_graph.nodes))
    arch_circuit = util.empty_circuit(circuit, qregs={arch_qreg.name: arch_qreg})
    # A fixed mapping from the graph nodes to the circuit registers.
    # FIXME: sorting is not necessary in general! But IBM expects this kind of mapping.
    arch_mapping = dict(zip(
        sorted(arch_graph.nodes),
        ((qreg.name, i) for qname, qreg in arch_circuit.qregs.items() for i in range(qreg.size))
        ))
    logger.debug("Arch mapping: %s", arch_mapping)

    ###
    # We now have an empty DAGCircuit with the same parameters as the original circuit,
    # but instead the nodes have the same ids as the architecture graph.
    # It remains to add gates to the circuit.
    ###
    # Get the first layer if it exists.
    layer = util.first_layer(todo_circuit)
    if layer is None:
        return arch_circuit, {v: k for k, v in arch_mapping.items()}

    # Find a good mapping and ignore the circuit, since it is free anyway.
    # Assign arbitrarily to have a valid initial state,
    # the mapper will find a good initial mapping.
    current_mapping = dict(zip(
        [(qname, i) for qname, qreg in circuit.qregs.items() for i in range(qreg.size)],
        arch_graph.nodes()))
    logger.debug("Initial current mapping: %s", current_mapping)
    # IDEA: Allow the initial mapper to be configurable
    # Only consider CNOTs for the intial placement.
    # This will allow the greedy mapper to ignore any 1-qubit gates when finding
    # an initial placement
    cnot_circuit = util.empty_circuit(todo_circuit)
    for node in nx.topological_sort(cnot_circuit.multi_graph):
        cnot_node = cnot_circuit.multi_graph.node[node]
        if cnot_node["type"] == "op" and cnot_node["name"] == "cx":
            cnot_circuit.apply_operation_back("cx", cnot_node["qargs"])

    # Permutations are free.
    size_mapper = mp.size.SizeMapper(arch_graph, lambda p: [])
    partial_mapping = size_mapper.greedy(cnot_circuit, current_mapping)
    logger.debug("Initial placement partial mapping: %s", partial_mapping)
    # Find the complete mapping after applying the partial mapping
    permutation = {i: i for i in arch_graph.nodes()}
    Placement(current_mapping, partial_mapping).place(permutation)
    current_mapping = mp.util.new_mapping(current_mapping, permutation)
    logger.debug("Current mapping after intial placement: %s", current_mapping)

    # Begin placing circuits.
    counter = 0
    while layer:
        logger.debug("Iteration %s", counter)
        counter += 1

        mapping, mapping_circuit = arch_mapper(todo_circuit, current_mapping)

        # Always perform whatever the mapper tells the compiler is a good mapping,
        # it will return an empty mapping if no movement is needed.
        # Assume there are no classical registers
        wire_map = {qubit: arch_mapping[node] for node, qubit in mapping_circuit.inputmap.items()}
        arch_circuit.compose_back(mapping_circuit.circuit, wire_map=wire_map)
        if current_mapping != mapping:
            logger.debug("New current mapping: %s", mapping)
        current_mapping = mapping

        # Given the current mapping, apply as much of a layer as possible.
        node_data = map(lambda n: n[1], layer.multi_graph.nodes(data=True))
        op_nodes = list(filter(lambda n: n["type"] == "op", node_data))

        for op_node in op_nodes:
            success = _apply_operation(op_node, current_mapping,
                                       # Ignore directionality
                                       arch_graph.to_undirected(as_view=True),
                                       arch_circuit,
                                       arch_mapping)
            if success:
                # Assume that an op node always has at least 1 qarg.
                _remove_successor(op_node["qargs"][0], todo_circuit)

        # The layers may have changed depending on which operations were applied.
        layer = util.first_layer(todo_circuit)

    return arch_circuit, {v: k for k, v in arch_mapping.items()}


def _remove_successor(node, todo_circuit):
    """Remove the successor node in the todo_circuit of a register."""
    input_node = todo_circuit.input_map[node]
    # There should be exactly one edge
    successors = todo_circuit.multi_graph.edges(nbunch=input_node)
    assert len(successors) == 1, "The input node is not adjacent to exactly one operation."
    successor_node = list(successors)[0][1]
    # There is no other way to remove a specific node from the circuit.
    # See: https://github.com/QISKit/qiskit-sdk-py/issues/289
    todo_circuit._remove_op_node(successor_node)  # pylint: disable=protected-access


def _apply_operation(operation, current_mapping, arch_graph, arch_circuit, arch_mapping):
    """Apply the given operation to the architecture circuit.

    :return: A boolean indicating if the operation was applied or not.
    :rtype: bool
    """

    # Find the architecture nodes associated with the operation, given the current mapping.
    arch_nodes = [current_mapping[n] for n in operation["qargs"]]
    # Check connectivity for 2-qubit operations
    if operation["name"] != "barrier" \
            and len(arch_nodes) == 2 \
            and (arch_nodes[0], arch_nodes[1]) not in arch_graph.edges:
        return False

    arch_regs = [arch_mapping[node] for node in arch_nodes]
    permuted_node = dict(operation)
    permuted_node["qargs"] = arch_regs
    del permuted_node["type"]
    arch_circuit.apply_operation_back(**permuted_node)
    return True

"""Util provides methods that are used throughout the source code without internal dependencies."""
import copy

import networkx as nx
from qiskit.dagcircuit import DAGCircuit


def empty_circuit(circuit, qregs=None):
    """
    Copy the parameters of the given circuit, but exclude its contents such as operations.

    The basis, gates and wires are all copied.
    :param circuit:
    :param qregs: The quantum registers that the empty circuit should have.
    :type circuit: DAGCircuit
    :type qregs: List[Tuple[str, int]] = None)
    :return:
    :rtype: DAGCircuit
    """
    qregs = qregs or circuit.qregs
    copied_circuit = DAGCircuit()
    # Re-add all basis elements
    copied_circuit.basis = circuit.basis.copy()
    # And gates
    copied_circuit.gates = circuit.gates.copy()
    # Also copy over classical wires, since we do not care about placing them.
    for creg in circuit.cregs.values():
        copied_circuit.add_creg(creg)
    for qreg in qregs.values():
        copied_circuit.add_qreg(qreg)
    return copied_circuit


def first_layer(circuit):
    """Take the first layer of the DAGCircuit and return it."""
    try:
        return next(dagcircuit_layers(circuit))["graph"]
    except StopIteration:
        return None


def dagcircuit_layers(circuit):
    """Yield a shallow view on a layer of this DAGCircuit for all d layers of this circuit.

    A layer is a circuit whose gates act on disjoint qubits, i.e.
    a layer has depth 1. The total number of layers equals the
    circuit depth d. The layers are indexed from 0 to d-1 with the
    earliest layer at index 0. The layers are constructed using a
    greedy algorithm. Each returned layer is a dict containing
    {"graph": circuit graph, "partition": list of qubit lists}.


    TODO: Gates that use the same cbits will end up in different
    layers as this is currently implemented. This may not be
    the desired behavior.
    """
    graph_layers = dagcircuit_multigraph_layers(circuit)
    try:
        next(graph_layers)  # Remove input nodes
    except StopIteration:
        return

    def nodes_data(nodes):
        """Construct full nodes from just node ids."""
        return (
            (node_id, circuit.multi_graph.nodes[node_id]) for node_id in nodes
            )

    for graph_layer in graph_layers:
        # Get the op nodes from the layer, removing any input and output nodes.
        op_nodes = list(filter(lambda node: node[1]["type"] == "op",
                               nodes_data(graph_layer)))

        # Stop yielding once there are no more op_nodes in a layer.
        if not op_nodes:
            return

        # Construct a shallow copy of self
        new_layer = copy.copy(circuit)
        new_layer.multi_graph = nx.MultiDiGraph()

        new_layer.multi_graph.add_nodes_from(nodes_data(circuit.input_map.values()))
        new_layer.multi_graph.add_nodes_from(nodes_data(circuit.output_map.values()))

        # The quantum registers that have an operation in this layer.
        support_list = [
            op_node[1]["qargs"]
            for op_node in op_nodes if op_node[1]["name"] != "barrier"
            ]
        new_layer.multi_graph.add_nodes_from(op_nodes)

        # Now add the edges to the multi_graph
        # By default we just wire inputs to the outputs.
        wires = {circuit.input_map[register]: circuit.output_map[register]
                 for register in circuit.wire_type}
        # Wire inputs to op nodes, and op nodes to outputs.
        for op_node in op_nodes:
            args = circuit._bits_in_condition(op_node[1]["condition"]) \
                   + op_node[1]["cargs"] + op_node[1]["qargs"]
            arg_ids = map(lambda arg: circuit.input_map[arg], args)  # map from ("q",0) to node id.
            for arg_id in arg_ids:
                wires[arg_id], wires[op_node[0]] = op_node[0], wires[arg_id]

        # Add wiring to/from the operations and between unused inputs & outputs.
        new_layer.multi_graph.add_edges_from(wires.items())
        yield {"graph": new_layer, "partition": support_list}


def dagcircuit_multigraph_layers(circuit):
    """Yield layers of a DAGCircuit multigraph."""
    predecessor_count = dict()
    cur_layer = [node for node in circuit.input_map.values()]
    yield cur_layer
    next_layer = []
    while cur_layer:
        for node in cur_layer:
            # Count multiedges with multiplicity.
            for successor in circuit.multi_graph.successors(node):
                multiplicity = circuit.multi_graph.number_of_edges(node, successor)
                if successor in predecessor_count:
                    predecessor_count[successor] -= multiplicity
                else:
                    predecessor_count[successor] = \
                        circuit.multi_graph.in_degree(successor) - multiplicity

                if predecessor_count[successor] == 0:
                    next_layer.append(successor)
                    del predecessor_count[successor]

        yield next_layer
        cur_layer = next_layer
        next_layer = []


def dagcircuit_get_named_nodes(circuit, name):
    """Return an iterator over "op" nodes with the given name."""
    for node in circuit.multi_graph.nodes:
        node_data = circuit.multi_graph.node[node]
        if node_data["type"] == "op" and node_data["name"] == name:
            yield node

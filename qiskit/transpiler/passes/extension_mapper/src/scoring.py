"""Scoring assigns a score or cost to a mapping.

For a mapping we can calculate the cost to permute to that mapping.
"""
from typing import Dict, NamedTuple, Any, TypeVar, Mapping, Union

import networkx as nx
from qiskit.dagcircuit import DAGCircuit
from . import util

Reg = TypeVar('Reg')
ArchNode = TypeVar('ArchNode')


class Cost(NamedTuple):
    """A cost object that wraps the outputs of this package descriptively.

    depth: The number of timesteps in the circuit.
    synchronous_cost - The total cost when performing gates in lock-step.
    asynchronous_cost - The total cost when performing gates as soon as possible.
    cumulative_cost - The weighted total circuit size.
    """
    depth: int
    synchronous_cost: int
    asynchronous_cost: int
    cumulative_cost: int


def default_gate_costs() -> Mapping[str, int]:
    """The default costs of gates as proscribed by IBM.

    TODO: modify per cost function, since SWAP costs differ per cost function."""
    return {
        'id': 0, 'u1': 0, 'measure': 0, 'reset': 0, 'barrier': 0,
        'u2': 1, 'u3': 1, 'U': 1,
        'cx': 10, 'CX': 10,
        'swap': 34  # symmetric so this always possible.
        }


def cost(circuit: DAGCircuit,
         current_mapping: Mapping[Reg, ArchNode],
         arch_graph: Union[nx.Graph, nx.DiGraph],
         gate_costs: Dict[str, int] = None,
         allow_missing_edge: bool = False) -> Cost:
    """Calculates all costs for the given circuit and mapping on the architecture graph.

    This will compute all cost fields for a Cost object:
    The circuit depth, the synchronous cost and the asynchronous cost.

    :raises: KeyError when a 2-qubit operation is between nodes
        that does not exist in the arch_graph (by the permutation).
    :param circuit: The circuit to perform on the hardware.
    :param current_mapping: A mapping from circuit nodes to arch_graph nodes.
    :param arch_graph: The architecture graph representing the underlying hardware.
        Optionally weighted.
    :param gate_costs: A mapping from basis operation name to cost.
    :param allow_missing_edge: Will ignore the existence of edges, and fix edge weight to 1.
    :return:
    """

    sync_cost = synchronous_cost(circuit, current_mapping, arch_graph, gate_costs,
                                 allow_missing_edge=allow_missing_edge)
    async_cost = asynchronous_cost(circuit, current_mapping, arch_graph, gate_costs,
                                   allow_missing_edge=allow_missing_edge)
    cum_cost = cumulative_cost(circuit, current_mapping, arch_graph, gate_costs,
                               allow_missing_edge=allow_missing_edge)

    return Cost(
        depth=sync_cost.depth,
        synchronous_cost=sync_cost.synchronous_cost,
        asynchronous_cost=async_cost,
        cumulative_cost=cum_cost
        )


def synchronous_cost(circuit: DAGCircuit,
                     current_mapping: Mapping[Reg, ArchNode],
                     arch_graph: Union[nx.Graph, nx.DiGraph],
                     gate_costs: Dict[str, int] = None,
                     allow_missing_edge: bool = False) -> Cost:
    """Compute the cost of executing the circuit on the architecture graph in 'lock-step'.

    'lock-step' means that at every time step one gate can be performed per qubit (single-qubit)
    or per edge. The longest gate defines the cost/length of the timestep.

    :raises: KeyError when a 2-qubit operation is between nodes
        that does not exist in the arch_graph (by the permutation).
    :param circuit: The circuit to perform on the hardware.
    :param current_mapping: A mapping from circuit nodes to arch_graph nodes.
    :param arch_graph: The architecture graph representing the underlying hardware.
        Optionally weighted.
    :param gate_costs: A mapping from basis operation name to cost.
    :param allow_missing_edge: Will ignore the existence of edges, and fix edge weight to 1.
    :return: Cost object with synchronous cost and depth fields filled.
    """
    sync_cost = 0
    depth = 0
    full_nodes = circuit.multi_graph.nodes(data=True)
    # Synchronous ("lock-step") operation cost
    for layer_nodes in util.dagcircuit_multigraph_layers(circuit):
        node_data = map(lambda node_id: full_nodes[node_id], layer_nodes)
        op_nodes = filter(lambda n: n["type"] == "op", node_data)
        node_costs = map(lambda op_node: op_cost(op_node, current_mapping, arch_graph, gate_costs,
                                                 allow_missing_edge=allow_missing_edge),
                         op_nodes)
        try:
            # It is assumed node_costs is not empty.
            sync_cost += max(node_costs)
            depth += 1
        except ValueError:
            continue  # Skip empty sequences
    return Cost(depth=depth, synchronous_cost=sync_cost, asynchronous_cost=-1, cumulative_cost=-1)


def asynchronous_cost(circuit: DAGCircuit,
                      current_mapping: Mapping[Reg, ArchNode],
                      arch_graph: Union[nx.Graph, nx.DiGraph],
                      gate_costs: Dict[str, int] = None,
                      allow_missing_edge: bool = False) -> int:
    """Compute the cost of executing the gates on the architecture graph as soon as possible.

        Asynchronous here means that once the predecessors in the DAG are both done processing,
        the gate will immediately start being performed.

    :raises: KeyError when a 2-qubit operation is between nodes
        that does not exist in the arch_graph (by the permutation).
    :param circuit: The circuit to perform on the hardware.
    :param current_mapping: A mapping from circuit nodes to arch_graph nodes.
    :param arch_graph: The architecture graph representing the underlying hardware.
        Optionally weighted.
    :param gate_costs: A mapping from basis operation name to cost.
    :param allow_missing_edge: Will ignore the existence of edges, and fix edge weight to 1.
    :return: The asynchronous cost.
    """
    distance: Dict[Any, int] = {}  # stores {v : length}
    graph: nx.DiGraph = circuit.multi_graph
    for node in nx.topological_sort(graph):
        v_data = graph.nodes[node]
        node_cost: int
        if v_data["type"] == "op":
            node_cost = op_cost(v_data, current_mapping, arch_graph, gate_costs,
                                allow_missing_edge=allow_missing_edge)
        else:
            node_cost = 0

        # Find the maximum-distance predecessor and add that to the node cost.
        distance[node] = max((distance[u] for u in graph.pred[node]), default=0) + node_cost
    return max(distance.values())


def cumulative_cost(circuit: DAGCircuit,
                    current_mapping: Mapping[Reg, ArchNode],
                    arch_graph: Union[nx.Graph, nx.DiGraph],
                    gate_costs: Dict[str, int] = None,
                    allow_missing_edge: bool = False) -> int:
    """Calculate the sum of all gate costs"""
    op_nodes = (node[1] for node in circuit.multi_graph.nodes(data=True) if node[1]["type"] == "op")
    return sum(op_cost(op_node, current_mapping, arch_graph, gate_costs,
                       allow_missing_edge=allow_missing_edge) for op_node in op_nodes)


def op_cost(op_node: Any,
            current_mapping: Mapping[Reg, ArchNode],
            arch_graph: Union[nx.Graph, nx.DiGraph],
            gate_costs: Dict[str, int] = None,
            allow_missing_edge: bool = False) -> int:
    """Calculate the cost of performing the operation in the architecture.

    :raises KeyError: when a 2-qubit operation is between nodes
        that does not exist in the arch_graph (by the permutation).
    :param op_node: A DAGCircuit operation node to compute the cost for performing it.
    :param current_mapping: A mapping from circuit nodes to arch_graph nodes.
    :param arch_graph: The architecture graph representing the underlying hardware.
        Optionally weighted.
    :param gate_costs: A mapping from basis operation name to cost.
    :param allow_missing_edge: Will ignore the existence of edges, and fix edge weight to 1.
    :return: The asynchronous cost.
    """
    if gate_costs is None:
        gcosts = default_gate_costs()
    else:
        gcosts = gate_costs

    # multi-qubit operations
    if len(op_node['qargs']) == 2 and op_node['name'] != "barrier":
        qarg0, qarg1 = op_node['qargs']
        place0, place1 = current_mapping[qarg0], current_mapping[qarg1]
        weight: int
        if allow_missing_edge:
            # Set all weights to a constant 1.
            weight = 1
        else:
            edge: Dict
            try:
                # Assert the edge exists.
                edge = arch_graph.edges[place0, place1]
            except KeyError:
                if op_node['name'] == "swap":
                    # Swap is symmetric so we try the other way around.
                    edge = arch_graph.edges[place1, place0]
                else:
                    raise
            # Default edge weight 1
            weight = edge.get('weight', 1)
        return gcosts[op_node['name']] * weight

    # Single-qubit gate.
    return gcosts[op_node['name']]

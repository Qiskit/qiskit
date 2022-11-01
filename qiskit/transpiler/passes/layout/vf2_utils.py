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

"""This module contains common utils for vf2 layout passes."""

from collections import defaultdict
import statistics
import random

from retworkx import PyDiGraph, PyGraph

from qiskit.circuit import ControlFlowOp, ForLoopOp
from qiskit.converters import circuit_to_dag


def build_interaction_graph(dag, strict_direction=True):
    """Build an interaction graph from a dag."""
    im_graph = PyDiGraph(multigraph=False) if strict_direction else PyGraph(multigraph=False)
    im_graph_node_map = {}
    reverse_im_graph_node_map = {}

    class MultiQEncountered(Exception):
        """Used to singal an error-status return from the DAG visitor."""

    def _visit(dag, weight, wire_map):
        for node in dag.op_nodes(include_directives=False):
            if isinstance(node.op, ControlFlowOp):
                if isinstance(node.op, ForLoopOp):
                    inner_weight = len(node.op.params[0]) * weight
                else:
                    inner_weight = weight
                for block in node.op.blocks:
                    inner_wire_map = {
                        inner: wire_map[outer] for outer, inner in zip(node.qargs, block.qubits)
                    }
                    _visit(circuit_to_dag(block), inner_weight, inner_wire_map)
                continue
            len_args = len(node.qargs)
            qargs = [wire_map[q] for q in node.qargs]
            if len_args == 1:
                if qargs[0] not in im_graph_node_map:
                    weights = defaultdict(int)
                    weights[node.name] += weight
                    im_graph_node_map[qargs[0]] = im_graph.add_node(weights)
                    reverse_im_graph_node_map[im_graph_node_map[qargs[0]]] = qargs[0]
                else:
                    im_graph[im_graph_node_map[qargs[0]]][node.op.name] += weight
            if len_args == 2:
                if qargs[0] not in im_graph_node_map:
                    im_graph_node_map[qargs[0]] = im_graph.add_node(defaultdict(int))
                    reverse_im_graph_node_map[im_graph_node_map[qargs[0]]] = qargs[0]
                if qargs[1] not in im_graph_node_map:
                    im_graph_node_map[qargs[1]] = im_graph.add_node(defaultdict(int))
                    reverse_im_graph_node_map[im_graph_node_map[qargs[1]]] = qargs[1]
                edge = (im_graph_node_map[qargs[0]], im_graph_node_map[qargs[1]])
                if im_graph.has_edge(*edge):
                    im_graph.get_edge_data(*edge)[node.name] += weight
                else:
                    weights = defaultdict(int)
                    weights[node.name] += weight
                    im_graph.add_edge(*edge, weights)
            if len_args > 2:
                raise MultiQEncountered()

    try:
        _visit(dag, 1, {bit: bit for bit in dag.qubits})
    except MultiQEncountered:
        return None
    return im_graph, im_graph_node_map, reverse_im_graph_node_map


def score_layout(avg_error_map, layout, bit_map, reverse_bit_map, im_graph, strict_direction=False):
    """Score a layout given an average error map."""
    bits = layout.get_virtual_bits()
    fidelity = 1
    for bit, node_index in bit_map.items():
        gate_count = sum(im_graph[node_index].values())
        error_rate = avg_error_map.get((bits[bit],))
        if error_rate is not None:
            fidelity *= (1 - avg_error_map[(bits[bit],)]) ** gate_count
    for edge in im_graph.edge_index_map().values():
        gate_count = sum(edge[2].values())
        qargs = (bits[reverse_bit_map[edge[0]]], bits[reverse_bit_map[edge[1]]])
        if not strict_direction and qargs not in avg_error_map:
            qargs = (qargs[1], qargs[0])
        if qargs in avg_error_map:
            fidelity *= (1 - avg_error_map[qargs]) ** gate_count
    return 1 - fidelity


def build_average_error_map(target, properties, coupling_map):
    """Build an average error map used for scoring layouts pre-basis translation."""
    avg_map = {}
    num_qubits = 0
    if coupling_map is not None:
        num_qubits = coupling_map.size()
    if target is not None:
        for qargs in target.qargs:
            qarg_error = 0.0
            count = 0
            for op in target.operation_names_for_qargs(qargs):
                inst_props = target[op].get(qargs, None)
                if inst_props is not None and inst_props.error is not None:
                    count += 1
                    qarg_error += inst_props.error
            if count > 0:
                avg_map[qargs] = qarg_error / count
    elif properties is not None:
        errors = defaultdict(list)
        for qubit in range(len(properties.qubits)):
            errors[(qubit,)].append(properties.readout_error(qubit))
        for gate in properties.gates:
            qubits = tuple(gate.qubits)
            for param in gate.parameters:
                if param.name == "gate_error":
                    errors[qubits].append(param.value)
        avg_map = {k: statistics.mean(v) for k, v in errors.items()}
    elif coupling_map is not None:
        for qubit in range(num_qubits):
            avg_map[(qubit,)] = (
                coupling_map.graph.out_degree(qubit) + coupling_map.graph.in_degree(qubit)
            ) / num_qubits
        for edge in coupling_map.graph.edge_list():
            avg_map[edge] = (avg_map[(edge[0],)] + avg_map[(edge[1],)]) / 2
    return avg_map


def shuffle_coupling_graph(coupling_map, seed, strict_direction=True):
    """Create a shuffled coupling graph from a coupling map."""
    if strict_direction:
        cm_graph = coupling_map.graph
    else:
        cm_graph = coupling_map.graph.to_undirected(multigraph=False)
    cm_nodes = list(cm_graph.node_indexes())
    if seed != -1:
        random.Random(seed).shuffle(cm_nodes)
        shuffled_cm_graph = type(cm_graph)()
        shuffled_cm_graph.add_nodes_from(cm_nodes)
        new_edges = [(cm_nodes[edge[0]], cm_nodes[edge[1]]) for edge in cm_graph.edge_list()]
        shuffled_cm_graph.add_edges_from_no_data(new_edges)
        cm_nodes = [k for k, v in sorted(enumerate(cm_nodes), key=lambda item: item[1])]
        cm_graph = shuffled_cm_graph
    return cm_graph, cm_nodes

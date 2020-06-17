# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Pass for grouping runs of two qubit gates and returning another graph with larger nodes.
"""

# This will be turned into an analysis pass.
# At present this class is only used by a_star_cx.py

import networkx as nx



def group_gates(compiled_dag):
    """Group consecutive runs of gates on a qubit pair."""

    gates = compiled_dag.topological_nodes()
    nqubits = compiled_dag.width()

    single_qubit_gates = [[] for i in range(nqubits)]
    last_index = [-1] * nqubits

    nnodes = 0
    graph = nx.DiGraph()
    qubit_names = sorted(compiled_dag.get_qubits())
    counter = 0
    for i in gates:
        node_i = compiled_dag.multi_graph.node[i]
        if node_i["type"] != "op":
            continue
        if node_i["name"] == "u3" or node_i["name"] == "u2" or node_i["name"] == "u1":
            qubit = qubit_names.index(node_i["qargs"][0])
            if last_index[qubit] == -1:
                single_qubit_gates[qubit] += [node_i]
            else:
                graph.node[last_index[qubit]]["gates"] += [node_i]
        elif node_i["name"] == "cx":
            q_1 = qubit_names.index(node_i["qargs"][0])
            q_2 = qubit_names.index(node_i["qargs"][1])
            if last_index[q_1] == -1 and last_index[q_2] == -1:
                graph.add_node(
                    nnodes,
                    gates=single_qubit_gates[q_1] + single_qubit_gates[q_2] + [node_i],
                    qubits=(q_1, q_2),
                )
                counter += 1
                single_qubit_gates[q_1] = []
                single_qubit_gates[q_2] = []
                last_index[q_1] = nnodes
                last_index[q_2] = nnodes
                nnodes += 1
            elif last_index[q_1] == -1:
                graph.add_node(
                    nnodes, gates=single_qubit_gates[q_1] + [node_i], qubits=(q_1, q_2)
                )
                counter += 1
                graph.add_edge(last_index[q_2], nnodes)
                single_qubit_gates[q_1] = []
                last_index[q_1] = nnodes
                last_index[q_2] = nnodes
                nnodes += 1
            elif last_index[q_2] == -1:
                graph.add_node(
                    nnodes, gates=single_qubit_gates[q_2] + [node_i], qubits=(q_1, q_2)
                )
                counter += 1
                graph.add_edge(last_index[q_1], nnodes)
                single_qubit_gates[q_2] = []
                last_index[q_1] = nnodes
                last_index[q_2] = nnodes
                nnodes += 1
            else:
                if (last_index[q_2] == last_index[q_1]
                        and len(graph.node[last_index[q_2]]["qubits"]) != nqubits):
                    graph.node[last_index[q_1]]["gates"] += [node_i]
                else:
                    graph.add_node(nnodes, gates=[node_i], qubits=(q_1, q_2))
                    counter += 1
                    graph.add_edge(last_index[q_1], nnodes)
                    graph.add_edge(last_index[q_2], nnodes)
                    last_index[q_1] = nnodes
                    last_index[q_2] = nnodes
                    nnodes += 1
        elif node_i["name"] == "barrier":
            for i_inner in range(len(single_qubit_gates)):
                if single_qubit_gates[i_inner] != []:
                    graph.add_node(
                        nnodes, gates=single_qubit_gates[i_inner], qubits=tuple([i_inner])
                    )
                    counter += 1
                    last_index[i_inner] = nnodes
                    single_qubit_gates[i_inner] = []
                    nnodes += 1
            graph.add_node(nnodes, gates=[node_i], qubits=tuple(range(0, nqubits)))
            counter += 1
            for i_inner in range(nqubits):
                graph.add_edge(last_index[i_inner], nnodes)
                last_index[i_inner] = nnodes
            nnodes += 1
        elif node_i["name"] == "measure":
            qubit = qubit_names.index(node_i["qargs"][0])
            if single_qubit_gates[qubit] != []:
                graph.add_node(
                    nnodes, gates=single_qubit_gates[qubit], qubits=tuple([qubit])
                )
                counter += 1
                last_index[qubit] = nnodes
                single_qubit_gates[qubit] = []
                nnodes += 1
            graph.add_node(nnodes, gates=[node_i], qubits=tuple([qubit]))
            counter += 1
            if last_index[qubit] != -1:
                graph.add_edge(last_index[qubit], nnodes)
            last_index[qubit] = nnodes
            nnodes += 1
        else:
            raise Exception("Unexpected operation in circuit!")

    for qubit in range(0, nqubits):
        if single_qubit_gates[qubit] != []:
            graph.add_node(
                nnodes, gates=single_qubit_gates[qubit], qubits=tuple([qubit])
            )
            counter += 1
            last_index[qubit] = nnodes
            single_qubit_gates[qubit] = []
            nnodes += 1

    return graph

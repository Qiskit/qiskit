# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Pass for grouping runs of two qubit gates and returning another graph with larger nodes.
"""
from collections import defaultdict
from qiskit.transpiler import AnalysisPass
# Our goal is to turn this into an analysis pass.  At present this class is only used by a_star_cx.py
class GroupGates(AnalysisPass):
    """Group consecutive runs of gates on a qubit pair."""

    def __init__(self):
        pass

    def run(self, dag):
        """
        sweep dag in a greedy fashion and gather gates operating on the same pair,
        and terminate when a branch out of that pair is encountered.

        Args:
            dag: directed acyclic graph (Note: DiGraph instead of MultiDiGraph)
        """

# group all gate that are applied to two certain qubits (if possible) in a greedy fashion
# return a graph holding the grouped gates
    def group_gates(compiled_dag):

        import networkx as nx
        gates = nx.topological_sort(compiled_dag.multi_graph)
        nqubits = compiled_dag.width()

        single_qubit_gates = [ [] for i in range(nqubits) ]
        last_index = [-1] * nqubits

        nnodes = 0
        graph = nx.DiGraph()
        qubit_names = sorted(compiled_dag.get_qubits())
        counter = 0
        for i in gates:
            nd = compiled_dag.multi_graph.node[i]
            if nd['type'] != 'op':
                continue
            if nd['name'] == 'u3' or nd['name'] == 'u2' or nd['name'] == 'u1':
                qubit = qubit_names.index(nd['qargs'][0])
                if last_index[qubit] == -1:
                    single_qubit_gates[qubit] += [nd]
                else:
                    graph.node[last_index[qubit]]['gates'] += [nd]
            elif nd['name'] == 'cx':
                q1 = qubit_names.index(nd['qargs'][0])
                q2 = qubit_names.index(nd['qargs'][1])
                if last_index[q1] == -1 and last_index[q2] == -1:
                    graph.add_node(nnodes, gates=single_qubit_gates[q1] + single_qubit_gates[q2] + [nd], qubits=(q1,q2))
                    counter += 1
                    single_qubit_gates[q1] = []
                    single_qubit_gates[q2] = []
                    last_index[q1] = nnodes
                    last_index[q2] = nnodes
                    nnodes += 1
                elif last_index[q1] == -1:
                    graph.add_node(nnodes, gates=single_qubit_gates[q1] + [nd], qubits=(q1,q2))
                    counter += 1                
                    graph.add_edge(last_index[q2], nnodes)
                    single_qubit_gates[q1] = []
                    last_index[q1] = nnodes
                    last_index[q2] = nnodes
                    nnodes += 1
                elif last_index[q2] == -1:
                    graph.add_node(nnodes, gates=single_qubit_gates[q2] + [nd], qubits=(q1,q2))
                    counter += 1                
                    graph.add_edge(last_index[q1], nnodes)
                    single_qubit_gates[q2] = []
                    last_index[q1] = nnodes
                    last_index[q2] = nnodes
                    nnodes += 1
                else: 
                    if last_index[q2] == last_index[q1] and len(graph.node[last_index[q2]]['qubits']) != nqubits:
                        graph.node[last_index[q1]]['gates'] += [nd]
                    else: 
                        graph.add_node(nnodes, gates=[nd], qubits=(q1,q2))
                        counter += 1                    
                        graph.add_edge(last_index[q1], nnodes)                    
                        graph.add_edge(last_index[q2], nnodes)
                        last_index[q1] = nnodes
                        last_index[q2] = nnodes
                        nnodes += 1
            elif nd['name'] == 'barrier':
                for i in range(len(single_qubit_gates)):
                    if single_qubit_gates[i] != []:
                        graph.add_node(nnodes, gates=single_qubit_gates[i], qubits=tuple([i]))
                        counter += 1                    
                        last_index[i] = nnodes
                        single_qubit_gates[i] = []
                        nnodes += 1
                graph.add_node(nnodes, gates=[nd], qubits=tuple(range(0,nqubits)))
                counter += 1            
                for i in range(nqubits):
                    graph.add_edge(last_index[i], nnodes)
                    last_index[i] = nnodes
                nnodes += 1                       
            elif nd['name'] == 'measure':
                qubit = qubit_names.index(nd['qargs'][0])
                if single_qubit_gates[qubit] != []:
                    graph.add_node(nnodes, gates=single_qubit_gates[qubit], qubits=tuple([qubit]))
                    counter += 1                
                    last_index[qubit] = nnodes
                    single_qubit_gates[qubit] = []
                    nnodes += 1
                graph.add_node(nnodes, gates=[nd], qubits=tuple([qubit]))
                counter += 1            
                if last_index[qubit] != -1:
                    graph.add_edge(last_index[qubit], nnodes)
                last_index[qubit] = nnodes
                nnodes += 1
            else:
                raise Exception('Unexpected operation in circuit!')

        for qubit in range(0, nqubits):
            if single_qubit_gates[qubit] != []:
                graph.add_node(nnodes, gates=single_qubit_gates[qubit], qubits=tuple([qubit]))
                counter += 1            
                last_index[qubit] = nnodes
                single_qubit_gates[qubit] = []
                nnodes += 1

        return graph

# -*- coding: utf-8 -*-




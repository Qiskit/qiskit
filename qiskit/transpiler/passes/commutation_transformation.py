# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Pass for constructing commutativity aware DAGCircuit from basic DAGCircuit.
The generated DAGCircuit is more relaxed about operation dependencies,
but is not ready for simple scheduling.
"""

from qiskit.transpiler._basepasses import TransformationPass
from qiskit.transpiler.passes import CommutationAnalysis


class CommutationTransformation(TransformationPass):
    """
    A transformation pass to change DAG edges depending on previously discovered
    commutation relations.
    """

    def __init__(self):
        super().__init__()
        self.requires.append(CommutationAnalysis())
        self.preserves.append(CommutationAnalysis())
        self.qreg_op = {}
        self.node_order = {}

    def run(self, dag):
        """
        Construct a new DAG that is commutativity aware. The new DAG is:
        - not friendly to simple scheduling (conflicts might arise),
        but leave more room for optimization.
        - The depth() method will not be accurate before the final scheduling anymore.
        - Preserves the gate count but not edge count in the MultiDiGraph

        Args:
            dag (DAGCircuit): the directed acyclic graph

        Return:
            DAGCircuit: Transformed DAG.
        """

        for wire in dag.wires:
            wire_name = "{0}[{1}]".format(str(wire[0].name), str(wire[1]))
            wire_commutation_set = self.property_set['commutation_set'][wire_name]
            for c_set_ind, c_set in enumerate(wire_commutation_set):
                if dag.multi_graph.node[c_set[0]]['type'] == 'out':
                    continue
                for node1 in c_set:
                    for node2 in c_set:
                        if node1 != node2:
                            wire_to_save = ''
                            for edge in dag.multi_graph.edges([node1], data=True):
                                if edge[2]['name'] != wire_name and edge[1] == node2:
                                    wire_to_save = edge[2]['name']

                            while dag.multi_graph.has_edge(node1, node2):
                                dag.multi_graph.remove_edge(node1, node2)

                            if wire_to_save != '':
                                dag.multi_graph.add_edge(node1, node2, name=wire_to_save)

                    for next_node in wire_commutation_set[c_set_ind + 1]:

                        edge_on_wire = False
                        for temp_edge in dag.multi_graph.edges([node1], data=True):
                            if temp_edge[1] == next_node and temp_edge[2]['name'] == wire_name:
                                edge_on_wire = True

                        if not edge_on_wire:
                            dag.multi_graph.add_edge(node1, next_node, name=wire_name)

        return dag

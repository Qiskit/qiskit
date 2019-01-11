# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Pass for detecting commutativity in a circuit.

Property_set['commutation_set'] is a dictionary that describes
the commutation relations on a given wire, all the gates on a wire
are grouped into a set of gates that commute.

This pass also provides useful methods to determine if two gates
can commute in the circuit.

TODO: the current pass determines commutativity through matrix multiplication.
A rule-based analysis would be potentially faster, but more limited.
"""

from collections import defaultdict
import numpy as np

from qiskit.transpiler._basepasses import AnalysisPass


class CommutationAnalysis(AnalysisPass):
    """An analysis pass to find commutation relations between DAG nodes."""

    def __init__(self, max_depth=100):
        super().__init__()
        self.max_depth = max_depth
        self.wire_op = {}
        self.node_order = {}
        self.node_commute_group = {}

    def run(self, dag):
        """
        Run the pass on the DAG, and write the discovered commutation relations
        into the property_set.
        """
        tops_node = list(dag.node_nums_in_topological_order())

        # Initiation of the node_order
        for num, node in enumerate(tops_node):
            self.node_order[node] = num

        # Initiate the commutation set
        if self.property_set['commutation_set'] is None:
            self.property_set['commutation_set'] = defaultdict(list)

        # Build a dictionary to keep track of the gates on each qubit
        for wire in dag.wires:
            wire_name = "{0}[{1}]".format(str(wire[0].name), str(wire[1]))
            self.wire_op[wire_name] = []
            self.property_set['commutation_set'][wire_name] = []

        # Add edges to the dictionary for each qubit
        for node in tops_node:
            for edge in dag.multi_graph.edges([node], data=True):

                edge_name = edge[2]['name']

                if edge[0] == node:
                    self.wire_op[edge_name].append(edge[0])

                    self.property_set['commutation_set'][(node, edge_name)] = -1

                if dag.multi_graph.node[edge[1]]['type'] == "out":
                    self.wire_op[edge_name].append(edge[1])

        # With traversing the circuit in topological order,
        # the list of gates on a qubit doesn't have to be sorted
        # for key in self.wire_op:
        #     self.wire_op[key].sort(key=_get_node_order)

        for wire in dag.wires:
            wire_name = "{0}[{1}]".format(str(wire[0].name), str(wire[1]))
            for node in self.wire_op[wire_name]:

                if not self.property_set['commutation_set'][wire_name]:
                    self.property_set['commutation_set'][wire_name].append([node])

                if node not in self.property_set['commutation_set'][wire_name][-1]:
                    test_node = self.property_set['commutation_set'][wire_name][-1][-1]
                    if _commute(dag.multi_graph.node[node], dag.multi_graph.node[test_node]):
                        self.property_set['commutation_set'][wire_name][-1].append(node)

                    else:
                        self.property_set['commutation_set'][wire_name].append([node])
                temp_len = len(self.property_set['commutation_set'][wire_name])
                self.property_set['commutation_set'][(node, wire_name)] = temp_len - 1


def _gate_master_def(name, para=None):
    # pylint: disable=too-many-return-statements
    if name == 'h':
        return 1. / np.sqrt(2) * np.array([[1.0, 1.0],
                                           [1.0, -1.0]], dtype=np.complex)
    if name == 'x':
        return np.array([[0.0, 1.0],
                         [1.0, 0.0]], dtype=np.complex)
    if name == 'y':
        return np.array([[0.0, -1.0j],
                         [1.0j, 0.0]], dtype=np.complex)
    if name == 'cx':
        return np.array([[1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0],
                         [0.0, 0.0, 1.0, 0.0]], dtype=np.complex)
    if name == 'cz':
        return np.array([[1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, -1.0]], dtype=np.complex)
    if name == 'cy':
        return np.array([[1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0j],
                         [0.0, 0.0, -1.0j, 0.0]], dtype=np.complex)
    if name == 'z':
        return np.array([[1.0, 0.0],
                         [0.0, -1.0]], dtype=np.complex)
    if name == 't':
        return np.array([[1.0, 0.0],
                         [0.0, np.exp(1j * np.pi / 4.0)]], dtype=np.complex)
    if name == 's':
        return np.array([[1.0, 0.0],
                         [0.0, np.exp(1j * np.pi / 2.0)]], dtype=np.complex)
    if name == 'sdag':
        return np.array([[1.0, 0.0],
                         [0.0, -np.exp(1j * np.pi / 2.0)]], dtype=np.complex)
    if name == 'tdag':
        return np.array([[1.0, 0.0],
                         [0.0, -np.exp(1j * np.pi / 4.0)]], dtype=np.complex)
    if name in ('rz', 'u1'):
        return np.array([[np.exp(-1j * float(para[0]) / 2), 0],
                         [0, np.exp(1j * float(para[0]) / 2)]], dtype=np.complex)
    if name == 'rx':
        return np.array([[np.cos(float(para[0]) / 2), -1j * np.sin(float(para[0]) / 2)],
                         [-1j * np.sin(float(para[0]) / 2), np.cos(float(para[0]) / 2)]],
                        dtype=np.complex)
    if name == 'ry':
        return np.array([[np.cos(float(para[0]) / 2), - np.sin(float(para[0]) / 2)],
                         [np.sin(float(para[0]) / 2), np.cos(float(para[0]) / 2)]],
                        dtype=np.complex)
    if name == 'u2':
        return 1. / np.sqrt(2) * np.array(
            [[1, -np.exp(1j * float(para[1]))],
             [np.exp(1j * float(para[0])), np.exp(1j * (float(para[0]) + float(para[1])))]],
            dtype=np.complex)
    if name == 'u3':
        return 1./np.sqrt(2) * np.array(
            [[np.cos(float(para[0]) / 2.),
              -np.exp(1j * float(para[2])) * np.sin(float(para[0]) / 2.)],
             [np.exp(1j * float(para[1])) * np.sin(float(para[0]) / 2.),
              np.cos(float(para[0]) / 2.) * np.exp(1j * (float(para[2]) + float(para[1])))]],
            dtype=np.complex)

    if name == 'P0':
        return np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex)

    if name == 'P1':
        return np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex)

    if name == 'Id':
        return np.identity(2)

    return None


def _calc_product(node1, node2):

    wire_num = len(set(node1["qargs"] + node2["qargs"]))
    wires = sorted(list(map(lambda x: "{0}[{1}]".format(str(x[0].name), str(x[1])),
                            list(set(node1["qargs"] + node2["qargs"])))))
    final_unitary = np.identity(2 ** wire_num, dtype=np.complex)

    for node in [node1, node2]:

        qstate_list = [np.identity(2)] * wire_num

        if node['name'] == 'cx' or node['name'] == 'cy' or node['name'] == 'cz':

            qstate_list_ext = [np.identity(2)] * wire_num

            node_ctrl = "{0}[{1}]".format(str(node["qargs"][0][0].name), str(node["qargs"][0][1]))
            node_tgt = "{0}[{1}]".format(str(node["qargs"][1][0].name), str(node["qargs"][1][1]))
            ctrl = wires.index(node_ctrl)
            tgt = wires.index(node_tgt)

            qstate_list[ctrl] = _gate_master_def(name='P0')
            qstate_list[tgt] = _gate_master_def(name='Id')
            qstate_list_ext[ctrl] = _gate_master_def(name='P1')
            if node['name'] == 'cx':
                qstate_list_ext[tgt] = _gate_master_def(name='x')
            if node['name'] == 'cy':
                qstate_list_ext[tgt] = _gate_master_def(name='y')
            if node['name'] == 'cz':
                qstate_list_ext[tgt] = _gate_master_def(name='z')

            rt_list = [qstate_list] + [qstate_list_ext]

        else:

            mat = _gate_master_def(name=node['name'], para=node['op'].params)
            node_num = "{0}[{1}]".format(str(node["qargs"][0][0].name),
                                         str(node["qargs"][0][1]))
            qstate_list[wires.index(node_num)] = mat

            rt_list = [qstate_list]

        crt = np.zeros([2 ** wire_num, 2 ** wire_num])

        for state in rt_list:
            crt = crt + _kron_list(state)

        final_unitary = np.dot(crt, final_unitary)
    return final_unitary


def _kron_list(args):
    ret = args[0]
    for item in args[1:]:
        ret = np.kron(ret, item)
    return ret


def _matrix_commute(node1, node2):
    # Good for composite gates or any future
    # user-defined gate of equal or less than 2 qubits.
    ret = False
    if set(node1["qargs"]) & set(node2["qargs"]) == set():
        ret = True
    if _calc_product(node1, node2) is not None:
        ret = np.array_equal(_calc_product(node1, node2),
                             _calc_product(node2, node1))
    return ret


def _commute(node1, node2):
    if node1["type"] != "op" or node2["type"] != "op":
        return False
    return _matrix_commute(node1, node2)

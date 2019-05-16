# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

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
from qiskit.transpiler.exceptions import TranspilerError

from qiskit.transpiler.basepasses import AnalysisPass

_CUTOFF_PRECISION = 1E-10


class CommutationAnalysis(AnalysisPass):
    """An analysis pass to find commutation relations between DAG nodes."""

    def __init__(self):
        super().__init__()
        self.gates_on_wire = {}

    def run(self, dag):
        """
        Run the pass on the DAG, and write the discovered commutation relations
        into the property_set.
        """
        # Initiate the commutation set
        self.property_set['commutation_set'] = defaultdict(list)

        # Build a dictionary to keep track of the gates on each qubit
        for wire in dag.wires:
            wire_name = "{0}[{1}]".format(str(wire[0].name), str(wire[1]))
            self.property_set['commutation_set'][wire_name] = []

        # Add edges to the dictionary for each qubit
        for node in dag.topological_op_nodes():
            for (_, _, edge_data) in dag.edges(node):

                edge_name = edge_data['name']
                self.property_set['commutation_set'][(node, edge_name)] = -1

        for wire in dag.wires:
            wire_name = "{0}[{1}]".format(str(wire[0].name), str(wire[1]))

            for current_gate in dag.nodes_on_wire(wire):

                current_comm_set = self.property_set['commutation_set'][wire_name]
                if not current_comm_set:
                    current_comm_set.append([current_gate])

                if current_gate not in current_comm_set[-1]:
                    prev_gate = current_comm_set[-1][-1]
                    does_commute = False
                    try:
                        does_commute = _commute(current_gate, prev_gate)
                    except TranspilerError:
                        pass
                    if does_commute:
                        current_comm_set[-1].append(current_gate)

                    else:
                        current_comm_set.append([current_gate])

                temp_len = len(current_comm_set)
                self.property_set['commutation_set'][(current_gate, wire_name)] = temp_len - 1


def _gate_master_def(name, params=None):
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
        return np.array([[np.exp(-1j * float(params[0]) / 2), 0],
                         [0, np.exp(1j * float(params[0]) / 2)]], dtype=np.complex)
    if name == 'rx':
        return np.array([[np.cos(float(params[0]) / 2), -1j * np.sin(float(params[0]) / 2)],
                         [-1j * np.sin(float(params[0]) / 2), np.cos(float(params[0]) / 2)]],
                        dtype=np.complex)
    if name == 'ry':
        return np.array([[np.cos(float(params[0]) / 2), - np.sin(float(params[0]) / 2)],
                         [np.sin(float(params[0]) / 2), np.cos(float(params[0]) / 2)]],
                        dtype=np.complex)
    if name == 'u2':
        return 1. / np.sqrt(2) * np.array(
            [[1, -np.exp(1j * float(params[1]))],
             [np.exp(1j * float(params[0])), np.exp(1j * (float(params[0]) + float(params[1])))]],
            dtype=np.complex)

    if name == 'u3':
        return 1./np.sqrt(2) * np.array(
            [[np.cos(float(params[0]) / 2.),
              -np.exp(1j * float(params[2])) * np.sin(float(params[0]) / 2.)],
             [np.exp(1j * float(params[1])) * np.sin(float(params[0]) / 2.),
              np.cos(float(params[0]) / 2.) * np.exp(1j * (float(params[2]) + float(params[1])))]],
            dtype=np.complex)

    if name == 'P0':
        return np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex)

    if name == 'P1':
        return np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex)

    if name == 'Id':
        return np.identity(2)

    raise TranspilerError("The gate %s isn't supported" % name)


def _calc_product(node1, node2):

    wire_num = len(set(node1.qargs + node2.qargs))
    wires = sorted(list(map(lambda x: "{0}[{1}]".format(str(x[0].name), str(x[1])),
                            list(set(node1.qargs + node2.qargs)))))
    final_unitary = np.identity(2 ** wire_num, dtype=np.complex)

    for node in [node1, node2]:

        qstate_list = [np.identity(2)] * wire_num

        if node.name in ['cx', 'cy', 'cz']:

            qstate_list_ext = [np.identity(2)] * wire_num

            node_ctrl = "{0}[{1}]".format(str(node.qargs[0][0].name), str(node.qargs[0][1]))
            node_tgt = "{0}[{1}]".format(str(node.qargs[1][0].name), str(node.qargs[1][1]))
            ctrl = wires.index(node_ctrl)
            tgt = wires.index(node_tgt)

            qstate_list[ctrl] = _gate_master_def(name='P0')
            qstate_list[tgt] = _gate_master_def(name='Id')
            qstate_list_ext[ctrl] = _gate_master_def(name='P1')
            if node.name == 'cx':
                qstate_list_ext[tgt] = _gate_master_def(name='x')
            if node.name == 'cy':
                qstate_list_ext[tgt] = _gate_master_def(name='y')
            if node.name == 'cz':
                qstate_list_ext[tgt] = _gate_master_def(name='z')

            rt_list = [qstate_list] + [qstate_list_ext]

        else:

            mat = _gate_master_def(name=node.name, params=node.op.params)
            node_num = "{0}[{1}]".format(str(node.qargs[0][0].name),
                                         str(node.qargs[0][1]))
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
    if set(node1.qargs) & set(node2.qargs) == set():
        ret = True
    if _calc_product(node1, node2) is not None:
        ret = np.allclose(_calc_product(node1, node2),
                          _calc_product(node2, node1),
                          atol=_CUTOFF_PRECISION)
    return ret


def _commute(node1, node2):
    if node1.type != "op" or node2.type != "op":
        return False
    return _matrix_commute(node1, node2)

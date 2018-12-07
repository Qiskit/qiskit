# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Helper function for converting a dag to a circuit"""
import collections
import networkx as nx

from qiskit.circuit import QuantumCircuit
from qiskit.circuit import ClassicalRegister
from qiskit.circuit import QuantumRegister


def dag_to_circuit(dag):
    """Build a ``QuantumCircuit`` object from a ``DAGCircuit``.

    Args:
        dag (DAGCircuit): the input dag.

    Return:
        QuantumCircuit: the circuit representing the input dag.
    """
    qregs = collections.OrderedDict()
    for qreg in dag.qregs.values():
        qreg_tmp = QuantumRegister(qreg.size, name=qreg.name)
        qregs[qreg.name] = qreg_tmp
    cregs = collections.OrderedDict()
    for creg in dag.cregs.values():
        creg_tmp = ClassicalRegister(creg.size, name=creg.name)
        cregs[creg.name] = creg_tmp

    name = dag.name or None
    circuit = QuantumCircuit(*qregs.values(), *cregs.values(), name=name)

    graph = dag.multi_graph
    for node in nx.topological_sort(graph):
        n = graph.nodes[node]
        if n['type'] == 'op':
            if n['op'].name == 'U':
                name = 'u_base'
            elif n['op'].name == 'CX':
                name = 'cx_base'
            elif n['op'].name == 'id':
                name = 'iden'
            else:
                name = n['op'].name

            instr_method = getattr(circuit, name)
            qubits = []
            for qubit in n['qargs']:
                qubits.append(qregs[qubit[0].name][qubit[1]])

            clbits = []
            for clbit in n['cargs']:
                clbits.append(cregs[clbit[0].name][clbit[1]])
            params = n['op'].param

            if name in ['snapshot', 'save', 'noise', 'load']:
                result = instr_method(params[0])
            else:
                result = instr_method(*params, *qubits, *clbits)
            if 'condition' in n and n['condition']:
                result.c_if(*n['condition'])
    return circuit

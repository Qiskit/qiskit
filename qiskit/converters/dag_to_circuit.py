# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Helper function for converting a dag to a circuit"""
import collections
import random
import string
import networkx as nx

from qiskit._quantumcircuit import QuantumCircuit
from qiskit._classicalregister import ClassicalRegister
from qiskit._quantumregister import QuantumRegister


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

    random_name = QuantumCircuit.cls_prefix() + \
        str(''.join(random.choice(string.ascii_lowercase) for i in range(8)))
    name = dag.name or random_name
    circuit = QuantumCircuit(*qregs.values(), *cregs.values(), name=name)

    graph = dag.multi_graph
    for node in nx.topological_sort(graph):
        n = graph.nodes[node]
        if n['type'] == 'op':
            if n['op'].name == 'U':
                name = 'u_base'
            else:
                name = n['op'].name

            instr_method = getattr(circuit, name)
            qubits = []
            for qubit in n['op'].qargs:
                qubits.append(qregs[qubit[0].name][qubit[1]])

            clbits = []
            for clbit in n['op'].cargs:
                clbits.append(cregs[clbit[0].name][clbit[1]])
            params = n['op'].param
            result = instr_method(*params, *qubits, *clbits)
            if 'condition' in n and n['condition']:
                result.c_if(*n['condition'])

    return circuit

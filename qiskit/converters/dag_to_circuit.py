# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Helper function for converting a dag to a circuit"""
from copy import deepcopy
import random
import string
import networkx as nx

from qiskit.circuit import QuantumCircuit


def dag_to_circuit(dag):
    """Build a ``QuantumCircuit`` object from a ``DAGCircuit``.

    Args:
        dag (DAGCircuit): the input dag.

    Return:
        QuantumCircuit: the circuit representing the input dag.
    """
    circuit = QuantumCircuit()
    random_name = QuantumCircuit.cls_prefix() + \
        str(''.join(random.choice(string.ascii_lowercase) for i in range(8)))
    circuit.name = dag.name or random_name
    for qreg in dag.qregs.values():
        circuit.add_register(qreg)
    for creg in dag.cregs.values():
        circuit.add_register(creg)
    graph = dag.multi_graph
    for node in nx.topological_sort(graph):
        n = graph.nodes[node]
        if n['type'] == 'op':
            op = deepcopy(n['op'])
            op.qargs = n['qargs']
            op.cargs = n['cargs']
            op.circuit = circuit
            if 'condition' in n and n['condition']:
                op = op.c_if(*n['condition'])
            circuit._attach(op)

    return circuit

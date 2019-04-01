# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
A pass for transforming a circuit with virtual qubits into a circuit with physical qubits.
"""
from qiskit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.mapper import Layout, CouplingMap
from qiskit.transpiler import TransformationPass


class ApplyLayout(TransformationPass):
    """
    Transform a DAGCircuit with virtual qubits into a DAGCircuit with physical qubits.
    """

    def __init__(self, coupling: CouplingMap, initial_layout: Layout = None):
        """
        Transform a DAGCircuit with virtual qubits into a DAGCircuit with physical qubits
        defined in the coupling by applying a given `initial_layout`.
        Args:
            coupling: initial layout to be applied
            initial_layout: initial layout to be applied
        """
        super().__init__()
        self._coupling = coupling
        self._initial_layout = initial_layout

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """
        Runs the ApplyLayout pass on `dag`.
        Args:
            dag: DAG to map.
        Returns:
            A mapped DAG (with physical qubits).
        """
        if not self._initial_layout:
            self._initial_layout = self.property_set["layout"]

        q = QuantumRegister(self._coupling.size(), 'q')

        layout = self._initial_layout
        new_dag = DAGCircuit()
        new_dag.add_qreg(q)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)
        for node in dag.nodes_in_topological_order():
            if node.type == 'op':
                qargs = [q[layout[qarg]] for qarg in node.op.qargs]
                if node.op.name == "swap":
                    layout.swap(*node.op.qargs)     # must do before apply_operation_back
                new_dag.apply_operation_back(node.op, qargs, node.cargs, node.condition)

        return new_dag

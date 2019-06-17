# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
A pass for transforming a circuit with virtual qubits into a circuit with physical qubits.
"""
from qiskit.circuit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass


class ApplyLayout(TransformationPass):
    """
    Transform a DAGCircuit with virtual qubits into a DAGCircuit with physical qubits.
    """

    def __init__(self, coupling, initial_layout=None):
        """
        Transform a DAGCircuit with virtual qubits into a DAGCircuit with physical qubits
        defined in the `coupling` by applying a given `initial_layout`.
        Args:
            coupling (CouplingMap): coupling graph to which the circuit is mapped
            initial_layout (Layout): initial layout to be applied
        """
        super().__init__()
        self._coupling = coupling
        self._initial_layout = initial_layout

    def run(self, dag):
        """
        Runs the ApplyLayout pass on `dag`.
        Args:
            dag (DAGCircuit): DAG to map.
        Returns:
            DAGCircuit: A mapped DAG (with physical qubits).
        """
        if not self._initial_layout:
            self._initial_layout = self.property_set["layout"]

        q = QuantumRegister(self._coupling.size(), 'q')

        layout = self._initial_layout
        new_dag = DAGCircuit()
        new_dag.add_qreg(q)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)
        for node in dag.topological_op_nodes():
            if node.type == 'op':
                qargs = [q[layout[qarg]] for qarg in node.qargs]
                new_dag.apply_operation_back(node.op, qargs, node.cargs, node.condition)

        return new_dag

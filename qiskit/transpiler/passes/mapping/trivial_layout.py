# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""A pass for choosing a Layout of a circuit onto a Coupling graph, using a simple
round-robin order.

This pass associates a physical qubit (int) to each virtual qubit
of the circuit (tuple(QuantumRegister, int)) in increasing order.
"""

from qiskit.mapper import Layout
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError


class TrivialLayout(AnalysisPass):
    """
    Chooses a Layout by assigning n circuit qubits to device qubits 0, .., n-1.

    Does not assume any ancilla.
    """

    def __init__(self, coupling_map):
        """
        Choose a TrivialLayout.

        Args:
            coupling_map (Coupling): directed graph representing a coupling map.

        Raises:
            TranspilerError: if invalid options
        """
        super().__init__()
        self.coupling_map = coupling_map

    def run(self, dag):
        """
        Pick a layout by assigning n circuit qubits to device qubits 0, .., n-1.

        Args:
            dag (DAGCircuit): DAG to find layout for.

        Raises:
            TranspilerError: if dag wider than self.coupling_map
        """
        num_dag_qubits = sum([qreg.size for qreg in dag.qregs.values()])
        if num_dag_qubits > self.coupling_map.size():
            raise TranspilerError('Number of qubits greater than device.')
        self.property_set['layout'] = Layout.generate_trivial_layout(*dag.qregs.values())

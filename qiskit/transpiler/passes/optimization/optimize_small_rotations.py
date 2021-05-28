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

"""Optimize by removing small rotations """
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
import qiskit.converters.circuit_to_dag
from qiskit.circuit.library import CRXGate, CRYGate, CRZGate, PhaseGate, RXGate, RYGate, RZGate
from qiskit.dagcircuit import DAGCircuit

from qiskit.transpiler.basepasses import TransformationPass


class RemoveSmallRotations(TransformationPass):
    """Return a circuit with small rotation gates removed."""

    def __init__(self, epsilon: float = 0):
        """Remove all small rotations from a circuit

        Args:
            epsilon: Threshold for rotation angle to be removed
        """
        super().__init__()

        self.epsilon = epsilon
        self._empty_dag1 = qiskit.converters.circuit_to_dag(QuantumCircuit(1))
        self._empty_dag2 = qiskit.converters.circuit_to_dag(QuantumCircuit(2))

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the pass on `dag`.

        Args:
            dag: input dag.

        Returns:
            Output dag with small rotations removed
        """
        for node in dag.op_nodes():
            if isinstance(node.op, (PhaseGate, RXGate, RYGate, RZGate)):
                phi = node.op.params[0]
                if np.abs(phi) <= self.epsilon:
                    dag.substitute_node_with_dag(node, self._empty_dag1)
            elif isinstance(node.op, (CRXGate, CRYGate, CRZGate)):
                phi = node.op.params[0]
                if np.abs(phi) <= self.epsilon:
                    dag.substitute_node_with_dag(node, self._empty_dag2)
        return dag

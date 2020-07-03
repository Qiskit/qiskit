# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Pass to remove instructions on idle qubits."""

from qiskit.circuit.delay import Delay
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError


class RemoveOpsOnIdleQubits(TransformationPass):
    """Pass to remove instructions on idle qubits."""

    def run(self, dag):
        """Remove delays on idle qubits.

        Args:
            dag (DAGCircuit): DAG to be transformed.

        Returns:
            DAGCircuit: A transformed DAG.

        Raises:
            TranspilerError: if the circuit is not mapped on physical qubits.
        """
        if len(dag.qregs) != 1 or dag.qregs.get('q', None) is None:
            raise TranspilerError('RemoveOpsOnIdleQubits runs on physical circuits only')

        for q in dag.qubits:
            idling = True
            for node in dag.nodes_on_wire(q, only_ops=True):
                if not isinstance(node.op, Delay):
                    idling = False
                    break
            if idling:
                for node in list(dag.nodes_on_wire(q, only_ops=True)):
                    dag.remove_op_node(node)

        return dag

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

"""Transform a circuit with virtual qubits into a circuit with physical qubits."""

from qiskit.circuit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError


class ApplyLayout(TransformationPass):
    """Transform a circuit with virtual qubits into a circuit with physical qubits.

    Transforms a DAGCircuit with virtual qubits into a DAGCircuit with physical qubits
    by applying the Layout given in `property_set`.
    Requires either of passes to set/select Layout, e.g. `SetLayout`, `TrivialLayout`.
    Assumes the Layout has full physical qubits.
    """

    def run(self, dag):
        """Run the ApplyLayout pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to map.

        Returns:
            DAGCircuit: A mapped DAG (with physical qubits).

        Raises:
            TranspilerError: if no layout is found in `property_set` or no full physical qubits.
        """
        layout = self.property_set["layout"]
        if not layout:
            raise TranspilerError(
                "No 'layout' is found in property_set. Please run a Layout pass in advance.")
        if len(layout) != (1 + max(layout.get_physical_bits())):
            raise TranspilerError(
                "The 'layout' must be full (with ancilla).")

        q = QuantumRegister(len(layout), 'q')

        new_dag = DAGCircuit()
        new_dag.add_qreg(q)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)
        for node in dag.topological_op_nodes():
            if node.type == 'op':
                qargs = [q[layout[qarg]] for qarg in node.qargs]
                new_dag.apply_operation_back(node.op, qargs, node.cargs)

        return new_dag

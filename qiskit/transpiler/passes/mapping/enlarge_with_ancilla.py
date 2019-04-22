# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""
Transformation pass that extends the circuit with new virtual qubits (i.e. ancilla).
Which qubits to add are previously allocated in the 'layout' property, by a previous pass.
"""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError


class EnlargeWithAncilla(TransformationPass):
    """Extends the DAG circuit with virtual qubits (ancilla) that are specified in
    the layout, but not present in the circuit."""

    def __init__(self, layout=None):
        """
        Args:
            layout (Layout): layout of qubits to consider
        """
        super().__init__()
        self.layout = layout

    def run(self, dag):
        """
        Extends dag with virtual qubits that are in layout but not in the circuit yet.

        Args:
            dag (DAGCircuit): DAG to extend.

        Returns:
            DAGCircuit: An extended DAG.

        Raises:
            TranspilerError: If there is not layout in the property set or not set at init time.
        """
        self.layout = self.layout or self.property_set['layout']

        if self.layout is None:
            raise TranspilerError("EnlargeWithAncilla requires property_set[\"layout\"] or"
                                  " \"layout\" parameter to run")

        layout_virtual_qubits = self.layout.get_virtual_bits().keys()
        new_qregs = set(virtual_qubit[0] for virtual_qubit in layout_virtual_qubits
                        if virtual_qubit not in dag.wires)

        for qreg in new_qregs:
            dag.add_qreg(qreg)

        return dag

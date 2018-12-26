# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""
This pass extends the DAG with idle physical qubits in the layout
"""

from qiskit.transpiler._basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.circuit import QuantumRegister

class EnlargeWithAncilla(TransformationPass):
    """Extends the DAG with idle physical qubits in the self.property_set["layout"]."""

    def __init__(self, layout=None):
        """
        Args:
            layout (Layout): layout of qubits to consider
        """
        super().__init__()
        self.layout = layout
        self.ancilla_name = 'ancilla'

    def run(self, dag):
        """
        Extends `dag` with idle physical qubits in the self.property_set["layout"]
        (or `layout` kwarg from `__init__`).

        Args:
            dag (DAGCircuit): DAG to extend.

        Returns:
            DAGCircuit: A extended DAG.
        """
        if self.layout is None:
            if self.property_set["layout"]:
                self.layout = self.property_set["layout"]
            else:
                raise TranspilerError(
                    "EnlargeWithAncilla requieres self.property_set[\"layout\"] to run")

        amount_of_idle_qubits = len(self.layout.idle_physical_bits())
        if amount_of_idle_qubits:
            if self.ancilla_name in dag.qregs:
                save_prefix = QuantumRegister.prefix
                QuantumRegister.prefix = self.ancilla_name
                dag.add_qreg(QuantumRegister(amount_of_idle_qubits))
                QuantumRegister.prefix = save_prefix
            else:
                dag.add_qreg(QuantumRegister(amount_of_idle_qubits, name=self.ancilla_name))
        return dag

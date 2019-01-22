# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
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
        (or `layout` kwarg from `__init__`). If an extension is performed, the DAG
        will be extended with an additional quantum register with the name  "ancilla"
        (or "ancillaN" if the name is already taken, where N is an integer).

        Args:
            dag (DAGCircuit): DAG to extend.

        Returns:
            DAGCircuit: A extended DAG.

        Raises:
            TranspilerError: If there is not layout in the property set or not set at init time.
        """
        if self.layout is None:
            if self.property_set["layout"]:
                self.layout = self.property_set["layout"]
            else:
                raise TranspilerError(
                    "EnlargeWithAncilla requires self.property_set[\"layout\"] to run")

        # Idle physical qubits are those physical qubits that no virtual qubit corresponds to.
        # Add extra virtual qubits to make the DAG and CouplingMap the same size.
        num_idle_physical_qubits = len(self.layout.idle_physical_bits())
        if num_idle_physical_qubits:
            if self.ancilla_name in dag.qregs:
                save_prefix = QuantumRegister.prefix
                QuantumRegister.prefix = self.ancilla_name
                dag.add_qreg(QuantumRegister(num_idle_physical_qubits))
                QuantumRegister.prefix = save_prefix
            else:
                dag.add_qreg(QuantumRegister(num_idle_physical_qubits, name=self.ancilla_name))
        return dag

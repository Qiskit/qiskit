# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Extend the dag with virtual qubits that are in layout but not in the circuit yet."""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError


class EnlargeWithAncilla(TransformationPass):
    """Extend the dag with virtual qubits that are in layout but not in the circuit yet.

    Extend the DAG circuit with new virtual qubits (ancilla) that are specified
    in the layout, but not present in the circuit. Which qubits to add are
    previously allocated in the ``layout`` property, by a previous pass.
    """

    def run(self, dag):
        """Run the EnlargeWithAncilla pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to extend.

        Returns:
            DAGCircuit: An extended DAG.

        Raises:
            TranspilerError: If there is not layout in the property set or not set at init time.
        """
        layout = self.property_set['layout']

        if layout is None:
            raise TranspilerError('EnlargeWithAncilla requires property_set["layout"]')

        layout_virtual_qubits = layout.get_virtual_bits().keys()
        new_qregs = {virtual_qubit.register for virtual_qubit in layout_virtual_qubits
                     if virtual_qubit not in dag.wires}

        for qreg in new_qregs:
            dag.add_qreg(qreg)

        return dag

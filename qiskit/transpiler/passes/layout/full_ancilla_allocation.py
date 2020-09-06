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

"""Allocate all idle nodes from the coupling map as ancilla on the layout."""

from qiskit.circuit import QuantumRegister
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError


class FullAncillaAllocation(AnalysisPass):
    """Allocate all idle nodes from the coupling map as ancilla on the layout.

    A pass for allocating all idle physical qubits (those that exist in coupling
    map but not the dag circuit) as ancilla. It will also choose new virtual
    qubits to correspond to those physical ancilla.

    Note:
        This is an analysis pass, and only responsible for choosing physical
        ancilla locations and their corresponding virtual qubits.
        A separate transformation pass must add those virtual qubits to the
        circuit.
    """

    def __init__(self, coupling_map):
        """FullAncillaAllocation initializer.

        Args:
            coupling_map (Coupling): directed graph representing a coupling map.
        """
        super().__init__()
        self.coupling_map = coupling_map
        self.ancilla_name = 'ancilla'

    def run(self, dag):
        """Run the FullAncillaAllocation pass on `dag`.

        Extend the layout with new (physical qubit, virtual qubit) pairs.
        The dag signals which virtual qubits are already in the circuit.
        This pass will allocate new virtual qubits such that no collision occurs
        (i.e. Layout bijectivity is preserved)

        The coupling_map and layout together determine which physical qubits are free.

        Args:
            dag (DAGCircuit): circuit to analyze

        Returns:
            DAGCircuit: returns the same dag circuit, unmodified

        Raises:
            TranspilerError: If there is not layout in the property set or not set at init time.
        """
        layout = self.property_set.get('layout')

        if layout is None:
            raise TranspilerError('FullAncillaAllocation pass requires property_set["layout"].')

        layout_physical_qubits = layout.get_physical_bits().keys()
        coupling_physical_qubits = self.coupling_map.physical_qubits
        idle_physical_qubits = [q for q in coupling_physical_qubits
                                if q not in layout_physical_qubits]

        if idle_physical_qubits:
            if self.ancilla_name in dag.qregs:
                save_prefix = QuantumRegister.prefix
                QuantumRegister.prefix = self.ancilla_name
                qreg = QuantumRegister(len(idle_physical_qubits))
                QuantumRegister.prefix = save_prefix
            else:
                qreg = QuantumRegister(len(idle_physical_qubits), name=self.ancilla_name)

        for idx, idle_q in enumerate(idle_physical_qubits):
            self.property_set['layout'][idle_q] = qreg[idx]

        return dag

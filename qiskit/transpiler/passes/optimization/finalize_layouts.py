# This code is part of Qiskit.
#
# (C) Copyright IBM 2024
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Finalize layout-related attributes."""

import logging

from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.layout import Layout

logger = logging.getLogger(__name__)


class FinalizeLayouts(AnalysisPass):
    """Finalize 'layout' and 'final_layout' attributes, taking 'virtual_permutation_layout'
    into account when exists.
    """

    def run(self, dag):
        """Run the FinalizeLayouts pass on ``dag``.

        Args:
            dag (DAGCircuit): the DAG circuit.
        """

        if (virtual_permutation_layout := self.property_set["virtual_permutation_layout"]) is None:
            return

        self.property_set.pop("virtual_permutation_layout")

        # virtual_permutation_layout is usually created before extending the layout with ancillas,
        # so we extend the permutation to be identity on ancilla qubits
        original_qubit_indices = self.property_set.get("original_qubit_indices", None)
        for oq in original_qubit_indices:
            if oq not in virtual_permutation_layout:
                virtual_permutation_layout[oq] = original_qubit_indices[oq]

        t_qubits = dag.qubits

        if (t_initial_layout := self.property_set.get("layout", None)) is None:
            t_initial_layout = Layout(dict(enumerate(t_qubits)))

        if (t_final_layout := self.property_set.get("final_layout", None)) is None:
            t_final_layout = Layout(dict(enumerate(t_qubits)))

        # Ordered list of original qubits
        original_qubits_reverse = {v: k for k, v in original_qubit_indices.items()}
        original_qubits = []
        for i in range(len(original_qubits_reverse)):
            original_qubits.append(original_qubits_reverse[i])

        virtual_permutation_layout_inv = virtual_permutation_layout.inverse(
            original_qubits, original_qubits
        )

        t_initial_layout_inv = t_initial_layout.inverse(original_qubits, t_qubits)

        # ToDo: this can possibly be made simpler
        new_final_layout = t_initial_layout_inv
        new_final_layout = new_final_layout.compose(virtual_permutation_layout_inv, original_qubits)
        new_final_layout = new_final_layout.compose(t_initial_layout, original_qubits)
        new_final_layout = new_final_layout.compose(t_final_layout, t_qubits)

        self.property_set["layout"] = t_initial_layout
        self.property_set["final_layout"] = new_final_layout

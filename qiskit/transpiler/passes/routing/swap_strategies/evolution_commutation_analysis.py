# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""An analysis class to find evolution gates in which the Paulis commute."""

import warnings

from qiskit.circuit.library import PauliEvolutionGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import AnalysisPass
from qiskit.quantum_info import SparsePauliOp


class CheckCommutingEvolutions(AnalysisPass):
    """Finds PauliEvolutionGates where the operators, that are evolved, all commute."""

    def run(self, dag: DAGCircuit) -> None:
        """Check for ``PauliEvolutionGate``s where the summands all commute.

        Args:
            The DAG circuit in which to look for the commuting evolutions.
        """
        self.property_set["commuting_blocks"] = set()
        self.property_set["1q_blocks"] = set()

        for node in dag.op_nodes():
            if isinstance(node.op, PauliEvolutionGate):
                operator = node.op.operator
                if self.single_qubit_terms_only(operator):
                    self.property_set["1q_blocks"].add(node)
                elif self.summands_commute(node.op.operator):
                    self.property_set["commuting_blocks"].add(node)

    @staticmethod
    def single_qubit_terms_only(operator: SparsePauliOp) -> bool:
        """Determine if the Paulis are made of single qubit terms only.

        Args:
            operator: The operator to check if it consists only of single qubit terms.

        Returns:
            True if the operator consists of only single qubit terms (like ``IIX + IZI``),
            and False otherwise.
        """

        for pauli in operator.paulis:
            num_terms = sum([char != "I" for char in str(pauli)])
            if num_terms > 1:
                return False

        return True

    @staticmethod
    def summands_commute(operator: SparsePauliOp) -> bool:
        """Check if all summands in the operator we evolve commute.

        Args:
            operator: The operator on which we check if all summands commute.

        Returns:
            True if all summands commute, False otherwise.
        """

        if not isinstance(operator, SparsePauliOp):
            warnings.warn(
                "PauliEvolutionGate does not only contain SparsePauliOp -- not checking commutativity."
            )
            return False

        # get a list of summands that commute
        commuting_subparts = operator.paulis.group_qubit_wise_commuting()

        # if all commute we only have one summand!
        return len(commuting_subparts) == 1

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

"""Check if a property reached a fixed point."""

from qiskit.transpiler.basepasses import AnalysisPass


class GatesInBasis(AnalysisPass):
    """Check if a a dag is all in the basis"""

    def __init__(self, basis_gates):
        """FixedPoint initializer.

        Args:
            basis_gates (list): The list of basis gate names to check
        """
        super().__init__()
        self._basis_gates = set(basis_gates)

    def run(self, dag):
        """Run the GatesInBasis pass on `dag`."""
        gates_out_of_basis = False
        gates = set(x.op.name for x in dag.gate_nodes())
        for gate in gates:
            if gate not in self._basis_gates:
                gates_out_of_basis = True
                break

        self.property_set["all_gates_in_basis"] = not gates_out_of_basis

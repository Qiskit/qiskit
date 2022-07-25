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

"""Check if all gates in the DAGCircuit are in the specified basis gates."""

from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError


class GatesInBasis(AnalysisPass):
    """Check if all gates in a DAG are in a given set of gates"""

    def __init__(self, basis_gates=None, target=None):
        """Initialize the GatesInBasis pass.

        Args:
            basis_gates (list): The list of strings representing the set of basis gates.
            target (Target): The target representing the backend. If specified
                this will be used instead of the ``basis_gates`` parameter

        Raises:
            TranspilerError: If neither basis_gates or target is set.
        """
        super().__init__()
        if basis_gates is None and target is None:
            raise TranspilerError(
                "A value for 'basis_gates' or 'target' must be set to use this pass"
            )
        if basis_gates is not None:
            self._basis_gates = set(basis_gates).union(
                {"measure", "reset", "barrier", "snapshot", "delay"}
            )
        self._target = target

    def run(self, dag):
        """Run the GatesInBasis pass on `dag`."""
        gates_out_of_basis = False
        if self._target is not None:
            qubit_map = {qubit: index for index, qubit in enumerate(dag.qubits)}
            for gate in dag.op_nodes():
                # Barrier is universal and supported by all backends
                if gate.name == "barrier":
                    continue
                if not self._target.instruction_supported(
                    gate.name, tuple(qubit_map[bit] for bit in gate.qargs)
                ):
                    gates_out_of_basis = True
                    break
        else:
            for gate in dag._op_names:
                if gate not in self._basis_gates:
                    gates_out_of_basis = True
                    break
        self.property_set["all_gates_in_basis"] = not gates_out_of_basis

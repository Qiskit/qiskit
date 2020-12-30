# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Synthesize a single qubit gate to a discrete basis set."""

from typing import List, Union

from qiskit.circuit import QuantumCircuit, Gate
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit.dagcircuit import DAGCircuit


class SolovayKitaevSynthesis(TransformationPass):
    """Synthesize gates according to their basis gates."""

    def __init__(self, n: int, basis_gates: List[Union[str, Gate]]):
        """SynthesizeUnitaries initializer.

        Args:
            n: The recursion depth.
            basis_gates: List of gate names to target.
        """
        super().__init__()
        self._n = n  # pylint: disable=invalid-name
        self._basis_gates = basis_gates

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the UnitarySynthesis pass on `dag`.

        Args:
            dag: input dag.

        Returns:
            Output dag with UnitaryGates synthesized to target basis.
        """
        for node in dag.nodes():
            if not node.type == 'op':
                pass  # skip all nodes that do not represent operations

            if not node.op.num_qubits == 1:
                pass  # ignore all non-single qubit gates, possible raise error here?

            matrix = node.op.to_matrix()

            # TODO call solovay kitaev on ``matrix`` here` using the basis gates from
            # self._basis_gates
            approximation = QuantumCircuit(1)
            approximation.unitary(matrix, [0])

            substitute = circuit_to_dag(approximation)
            dag.substitute_node_with_dag(node, substitute)

        return dag

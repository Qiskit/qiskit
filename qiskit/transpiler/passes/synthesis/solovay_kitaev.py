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
import numpy as np

from qiskit.circuit import QuantumCircuit, Gate
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit.dagcircuit import DAGCircuit

from .solovay_kitaev_utils import GateSequence


class SolovayKitaev():
    """The Solovay Kitaev discrete decomposition algorithm."""

    def __init__(self, basis_gates: List[Union[str, Gate]]) -> None:
        self._basis_gates = basis_gates
        self._basic_approximations = self.generate_basic_approximations(
            basis_gates)

    def generate_basic_approximations(self, basis_gates: List[Union[str, Gate]]
                                      ) -> List[GateSequence]:
        """Generate the list of basic approximations."""
        return []

    def run(self, gate_matrix: np.ndarray, recursion_degree: int) -> QuantumCircuit:
        r"""Run the algorithm.

        Args:
            gate_matrix: The 2x2 matrix representing the gate. Does not need to be SU(2).
            recursion_degree: The recursion degree, called :math:`n` in the paper.
        """
        # gate_matrix_su2 = make gate_matrix SU(2)
        # gate_matrix_so3 = make SO(3)
        # call _recurse to get the result
        # return the result as QuantumCircuit

        # this is just a placeholder
        approximation = QuantumCircuit(1)
        approximation.unitary(gate_matrix, [0])
        return approximation

    def _recurse(self, matrix: GateSequence, n: int) -> GateSequence:
        """Recursion wrapper for the algorithm.

        Args:
            matrix: The SO(3) gate matrix.
            n: The recursion count.
        """
        pass


class SolovayKitaevDecomposition(TransformationPass):
    """Synthesize gates according to their basis gates."""

    def __init__(self, recursion_degree: int, basis_gates: List[Union[str, Gate]]) -> None:
        """SynthesizeUnitaries initializer.

        Args:
            recursion_degree: The recursion depth.
            basis_gates: List of gate names to target.
        """
        super().__init__()
        self._recursion_degree = recursion_degree
        self._sk = SolovayKitaev(basis_gates)

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the UnitarySynthesis pass on `dag`.

        Args:
            dag: input dag.

        Returns:
            Output dag with UnitaryGates synthesized to target basis.
        """
        for node in dag.nodes():
            if node.type != 'op':
                continue  # skip all nodes that do not represent operations

            if not node.op.num_qubits == 1:
                continue  # ignore all non-single qubit gates, possible raise error here?

            matrix = node.op.to_matrix()

            # call solovay kitaev
            approximation = self._sk.run(matrix, self._recursion_degree)

            # convert to a dag and replace the gate by the approximation
            substitute = circuit_to_dag(approximation)
            dag.substitute_node_with_dag(node, substitute)

        return dag

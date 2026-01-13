# This code is part of Qiskit.
#
# (C) Copyright IBM 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Convert standard gates into Pauli product rotation gates for PBC"""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit
from qiskit._accelerate.pbc_transformation import pbc_transformation


class PBCTransformation(TransformationPass):
    """
    Map gates to a list of equivalent Pauli product rotations and a global phase.
    Each element of the list is of the form
    ((Pauli string, phase rescale factor, [qubit indices]), global phase).
    For gates that didn't have a phase (e.g. X)
    the phase rescale factor is simply the phase of the rotation gate. The convention is
    `original_gate = PauliEvolutionGate(pauli, phase) * e^{i global_phase * phase}`
    """

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the PBC Transformayion optimization pass on ``dag``.

        Args:
            dag: the input DAG.

        Returns:
            The output DAG.
        """
        dag = pbc_transformation(dag)

        return dag

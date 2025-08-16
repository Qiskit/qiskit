# This code is part of Qiskit.
#
# (C) Copyright IBM 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Move clifford gates to the end of the circuit, changing rotation gates to multi-qubit rotations."""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit._accelerate import litinski_transformation as litinski_transformation_rs


class LitinskiTransformation(TransformationPass):
    """
    Applies Litinski transform to a circuit.

    The transform applies to a circuit containing Clifford + RZ-rotation gates (including T and Tdg),
    and moves Clifford gates to the end of the circuit, while changing rotation gates to multi-qubit
    rotations.

    The pass supports all of the Clifford gates in the list returned by :func:`.get_clifford_gate_names`,
    namely
    ``["id", "x", "y", "z", "h", "s", "sdg", "sx", "sxdg", "cx", "cz", "cy", "swap", "iswap", "ecr", "dcx"]``.
    The list of supported RZ-rotations is ``["t", "tdg", "rz"]``.
    """

    def run(self, dag):
        """Run the LitiskiTransformation pass on ``dag``.

        Args:
            dag (DAGCircuit): the input DAG.

        Returns:
            DAGCircuit: the output DAG.
        """
        new_dag = litinski_transformation_rs.run(dag)

        # If the pass did not do anything, the result is None
        if new_dag is None:
            return dag

        return new_dag

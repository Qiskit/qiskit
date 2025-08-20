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
from qiskit.dagcircuit import DAGCircuit
from qiskit._accelerate.litinski_transformation import run_litinski_transformation


class LitinskiTransformation(TransformationPass):
    """
    Applies Litinski transform to a circuit.

    The transform applies to a circuit containing Clifford + RZ-rotation gates (including T and Tdg),
    and moves Clifford gates to the end of the circuit, while changing rotation gates to multi-qubit
    rotations (represented using PauliEvolution gates).

    The pass supports all of the Clifford gates in the list returned by
    :func:`.get_clifford_gate_names`:

    ``["id", "x", "y", "z", "h", "s", "sdg", "sx", "sxdg", "cx", "cz", "cy",
    "swap","iswap", "ecr", "dcx"]``

    The list of supported RZ-rotations is:

    ``["t", "tdg", "rz"]``

    References:

        [1]: Litinski. A Game of Surface Codes.
             `Quantum 3, 128 (2019) <https://quantum-journal.org/papers/q-2019-03-05-128>`_

    """

    def __init__(self, fix_clifford: bool = True):
        """

        Args:
            fix_clifford: if ``False`` (non-default), the returned circuit contains
                only PauliEvolution gates, with the final Clifford gates omitted.
                Note that in this case the operators of the original and synthesized
                circuits will generally not be equivalent.
        """
        super().__init__()
        self.fix_clifford = fix_clifford

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the LitiskiTransformation pass on ``dag``.

        Args:
            dag: the input DAG.

        Returns:
            The output DAG.

        Raises:
            TranspilerError: if the circuit contains gates
                not supported by the pass.
        """
        new_dag = run_litinski_transformation(dag, self.fix_clifford)

        # If the pass did not do anything, the result is None
        if new_dag is None:
            return dag

        return new_dag

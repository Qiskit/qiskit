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

"""Convert rotation gates into {Clifford,T,Tdg} when their angles are integer multiples of 2*pi/8"""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit
from qiskit._accelerate.discretize_rotations import discretize_rotations


class DiscretizeRotations(TransformationPass):
    """Convert rotation gates into {Clifford,T,Tdg} when their angles are integer multiples of 2*pi/8."""

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
        new_dag = discretize_rotations(dag)

        # If the pass did not do anything, the result is None
        if new_dag is None:
            return dag

        return new_dag

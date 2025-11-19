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

"""Convert rotation gates into {Clifford,T,Tdg} when their angles are integer multiples of pi/4"""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit
from qiskit._accelerate.discretize_rotations import discretize_rotations


class DiscretizeRotations(TransformationPass):
    """Convert single-qubit rotation gates :class:`.RZGate`, :class:`.RXGate` and :class:`.RYGate`,
    into {Clifford,T,Tdg} when their angles are integer multiples of pi/4.
    Note that odd multiples of pi/4 require a single :class:`.TGate` and :class:`.TdgGate`,
    as well as some Clifford gates,
    while even multiples of pi/4, or equivalently, integer multiples of pi/2,
    can be written using only Clifford gates.
    """

    def __init__(self, approximation_degree: float = 1.0):
        """
        Args:
            approximation_degree: Used in the tolerance computations.
        """
        super().__init__()
        self.approximation_degree = approximation_degree

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the Discretize Rotations optimization pass on ``dag``.

        Args:
            dag: the input DAG.

        Returns:
            The output DAG.
        """
        new_dag = discretize_rotations(dag, self.approximation_degree)

        # If the pass did not do anything, the result is None
        if new_dag is None:
            return dag

        return new_dag

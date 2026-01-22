# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Combine consecutive T/Tdg gates in a Clifford+T circuit."""

from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit._accelerate.optimize_clifford_t import optimize_clifford_t


class OptimizeCliffordT(TransformationPass):
    """An optimization pass for Clifford+T circuits.

    Currently all the pass does is merging pairs of consecutive T-gates into
    S-gates, and pair of consecutive Tdg-gates into Sdg-gates.
    """

    def run(self, dag: DAGCircuit):
        """
        Run the OptimizeCliffordT pass on `dag`.

        The pass applies to a Clifford + T/Tdg circuit, and outputs an optimized
        Clifford + T/Tdg circuit.

        Args:
            dag: The directed acyclic graph to run on.

        Returns:
            DAGCircuit: Transformed DAG.
        """

        optimize_clifford_t(dag)
        return dag

# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Optimize sequences of single-qubit Clifford+T gates."""

from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit._accelerate.optimize_clifford_t import optimize_clifford_t


class OptimizeCliffordT(TransformationPass):
    """
    Optimize sequences of consecutive Clifford+T gates.

    This pass rewrites maximal chains of consecutive single-qubit
    Clifford+T gates, reducing each chain to an equivalent sequence
    that uses the minimum possible number of T gates.

    For a chain of length :math:`m`, the pass runs in linear time,
    :math:`O(m)`.
    """

    def run(self, dag: DAGCircuit):
        """
        Run the OptimizeCliffordT pass on `dag`.

        Args:
            dag: The directed acyclic graph to run on.

        Returns:
            DAGCircuit: Transformed DAG.
        """

        optimize_clifford_t(dag)
        return dag

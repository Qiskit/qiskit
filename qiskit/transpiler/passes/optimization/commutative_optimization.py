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

"""Commutative Optimization transpiler pass."""


from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit.commutation_library import SessionCommutationChecker as scc

from qiskit._accelerate import commutative_optimization

from qiskit.transpiler.passes.utils.control_flow import trivial_recurse


class CommutativeOptimization(TransformationPass):
    """
    Cancel/merge gates exploiting commutativity relations.

    The pass will:

    * Cancel pairs of inverse gates.
    * Cancels pairs of gates that are inverse up to a global phase (adjusting
      the global phase accordingly).
    * Combines different types of RZ-rotations into an RZ-gate.
    * Combines different types of RX-rotations into an RX-gate.

    This pass generalizes both :class:`.CommutativeCancellation` and
    :class:`.CommutativeInverseCancellation` transpiler passes.
    """

    def __init__(self, approximation_degree: float = 1.0, max_qubits: int = 4):
        """
        Args:
            approximation_degree: Used in the tolerance computations.
            max_qubits: Limits the number of qubits in matrix-based commutativity and
                inverse checks.
        """
        super().__init__()
        self.commutation_checker = scc.cc
        self.approximation_degree = approximation_degree
        self.max_qubits = max_qubits

    @trivial_recurse
    def run(self, dag):
        """Run the CommutativeOptimization pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """
        result = commutative_optimization.commutative_optimization(
            dag, self.commutation_checker, self.approximation_degree, self.max_qubits
        )

        # If the pass did not do anything, the result is None
        if result is None:
            return dag

        return result

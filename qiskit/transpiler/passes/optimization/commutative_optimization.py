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

"""Remove CIC and CC."""


from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit.commutation_library import SessionCommutationChecker as scc

from qiskit._accelerate import commutative_optimization

from qiskit.transpiler.passes.utils.control_flow import trivial_recurse


class CommutativeOptimization(TransformationPass):
    """ToDo"""

    def __init__(self):
        """
        ToDo
        """
        super().__init__()

        # ToDo: not sure about this
        self._commutation_checker = scc.cc

    @trivial_recurse
    def run(self, dag):
        """Run the CommutativeCancellation pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """
        result = commutative_optimization.commutative_optimization(dag, self._commutation_checker)

        # If the pass did not do anything, the result is None
        if result is None:
            return dag

        return result

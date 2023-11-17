# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Remove reset when it is the final instruction on a qubit."""

from qiskit.circuit import Reset
from qiskit.dagcircuit import DAGOutNode
from qiskit.transpiler.basepasses import TransformationPass


class RemoveFinalReset(TransformationPass):
    """Remove reset when it is the final instruction on a qubit wire."""

    def run(self, dag):
        """Run the RemoveFinalReset pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """
        resets = dag.op_nodes(Reset)
        for reset in resets:
            successor = next(dag.successors(reset))
            if isinstance(successor, DAGOutNode):
                dag.remove_op_node(reset)
        return dag

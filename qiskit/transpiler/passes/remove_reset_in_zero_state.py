# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""
Transpiler pass to remove reset gate when the qubit is in zero state
"""

from qiskit.circuit import Reset
from qiskit.transpiler.basepasses import TransformationPass


class RemoveResetInZeroState(TransformationPass):
    """Remove reset gate when the qubit is in zero state"""

    def run(self, dag):
        """Return a new circuit that has been optimized."""
        resets = dag.op_nodes(Reset)
        for reset in resets:
            predecessor = next(dag.predecessors(reset))
            if predecessor.type == 'in':
                dag.remove_op_node(reset)
        return dag

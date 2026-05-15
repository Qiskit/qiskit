# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""
UnrollBoxes transpiler pass.
"""

from typing import Callable, Optional

from qiskit.circuit import BoxOp
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils.control_flow import trivial_recurse


class UnrollBoxes(TransformationPass):
    """Unroll BoxOp operations by replacing them with their internal circuit body."""

    def __init__(
        self,
        known_annotations: Optional[Callable[[dict], bool]] = None,
    ) -> None:
        """Create a new UnrollBoxes pass.

        Args:
            known_annotations: Predicate returning True for annotations that
                are safe to ignore when deciding whether to unroll a BoxOp.
        """

        super().__init__()
        self.known_annotations = known_annotations or (lambda ann: True)

    def _validate_annotations(self, box_op: BoxOp) -> bool:
        for ann_dict in getattr(box_op, "annotations", []):
            if not self.known_annotations(ann_dict):
                return False
        return True

    @trivial_recurse
    def run(self, dag):
        for node in dag.op_nodes(BoxOp):
            box_op = node.op
            if not self._validate_annotations(box_op):
                continue
            inner_dag = circuit_to_dag(box_op.blocks[0])
            dag.substitute_node_with_dag(node, inner_dag)
        return dag

"""
UnrollBoxes transpiler pass.
"""

from typing import Callable, Optional
from qiskit.circuit import BoxOp
from qiskit.transpiler import TransformationPass
from qiskit.transpiler.passes.utils.control_flow import trivial_recurse
from qiskit.converters import circuit_to_dag


class UnrollBoxes(TransformationPass):
    """Unroll BoxOp operations by replacing them with their internal circuit body."""

    def __init__(
        self,
        recursive: bool = True,
        known_annotations: Optional[Callable[[dict], bool]] = None,
        max_depth: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.recursive = recursive
        self.known_annotations = known_annotations or (lambda ann: True)
        self.max_depth = max_depth

    def _validate_annotations(self, box_op: BoxOp, depth: int = 0) -> bool:
        if self.max_depth is not None and depth >= self.max_depth:
            return False

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
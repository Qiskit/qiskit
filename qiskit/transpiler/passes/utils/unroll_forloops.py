# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""UnrollForLoops transpilation pass"""

from qiskit.circuit import ForLoopOp, ContinueLoopOp, BreakLoopOp, IfElseOp
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes.utils import control_flow
from qiskit.converters import circuit_to_dag
from qiskit.circuit.classical.expr import Range


class UnrollForLoops(TransformationPass):
    """``UnrollForLoops`` transpilation pass unrolls for-loops when possible.

    This pass unrolls :class:`~.ForLoopOp` instructions whose ``indexset`` is a
    Python :class:`range`, a tuple of integers, or a **constant**
    :class:`~.expr.Range` (materialized via :meth:`~.expr.Range.values`).
    Non-constant :class:`~.expr.Range` indexsets are skipped by default because
    loop bounds are only known at runtime; set ``strict=True`` to raise instead.
    """

    def __init__(self, max_target_depth=-1, *, strict=False):
        """Things like ``for x in {0, 3, 4} {rx(x) qr[1];}`` will turn into
        ``rx(0) qr[1]; rx(3) qr[1]; rx(4) qr[1];``.

        Constant :class:`~.expr.Range` indexsets are materialized to a Python
        :class:`range` and unrolled the same way. Non-constant
        :class:`~.expr.Range` indexsets are left unchanged unless ``strict`` is
        set.

        .. note::
            The ``UnrollForLoops`` unrolls only one level of block depth. No inner loop will
            be considered by ``max_target_depth``.

        Args:
            max_target_depth (int): Optional. Checks if the unrolled block is over a particular
                subcircuit depth. To disable the check, use ``-1`` (Default).
            strict (bool): If ``True``, raise :class:`~.TranspilerError` when a
                :class:`~.ForLoopOp` uses a non-constant :class:`~.expr.Range`
                indexset. If ``False`` (default), skip such loops.
        """
        super().__init__()
        self.max_target_depth = max_target_depth
        self.strict = strict

    @control_flow.trivial_recurse
    def run(self, dag):
        """Run the UnrollForLoops pass on ``dag``.

        Args:
            dag (DAGCircuit): the directed acyclic graph to run on.

        Returns:
            DAGCircuit: Transformed DAG.
        """
        for forloop_op in dag.op_nodes(ForLoopOp):
            (indexset, loop_param, body) = forloop_op.op.params

            if isinstance(indexset, Range):
                if not indexset.const:
                    if self.strict:
                        raise TranspilerError(
                            "Cannot unroll for_loop: indexset is a non-constant "
                            "expr.Range whose bounds are only known at runtime."
                        )
                    continue
                indexset = indexset.values()

            # skip unrolling if it results in bigger than max_target_depth
            if 0 < self.max_target_depth < len(indexset) * body.depth():
                continue

            # skip unroll when break_loop or continue_loop inside body
            if _body_contains_continue_or_break(body):
                continue

            unrolled_dag = circuit_to_dag(body).copy_empty_like()
            unrolled_dag.global_phase = 0
            for index_value in indexset:
                bound_body = (
                    body.assign_parameters({loop_param: index_value}) if loop_param else body
                )
                unrolled_dag.compose(circuit_to_dag(bound_body), inplace=True)
            dag.substitute_node_with_dag(forloop_op, unrolled_dag)

        return dag


def _body_contains_continue_or_break(circuit):
    """Checks if a circuit contains ``continue``s or ``break``s. Conditional bodies are inspected."""
    for inst in circuit.data:
        operation = inst.operation
        if isinstance(operation, (ContinueLoopOp, BreakLoopOp)):
            return True
        if isinstance(operation, IfElseOp):
            for block in operation.params:
                if _body_contains_continue_or_break(block):
                    return True
    return False

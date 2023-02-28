# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" UnrollForLoops transpilation pass """

from qiskit.circuit import ForLoopOp, ContinueLoopOp, BreakLoopOp, IfElseOp
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow
from qiskit.converters import circuit_to_dag


class UnrollForLoops(TransformationPass):
    """UnrollForLoops transpilation pass unrolls for loops when possible. Things like
    `for x in {0, 3, 4} {rx(x) qr[1];}` will turn into `rx(0) qr[1]; rx(3) qr[1]; rx(4) qr[1];`.
    """

    @control_flow.trivial_recurse
    def run(self, dag):
        """Run the UnrollForLoops pass on `dag`.

        Args:
            dag (DAGCircuit): the directed acyclic graph to run on.

        Returns:
            DAGCircuit: Transformed DAG.
        """
        for forloop_op in dag.op_nodes(ForLoopOp):
            (indexset, loop_parameter, body) = forloop_op.op.params

            # do not unroll when break_loop or continue_loop inside body
            if UnrollForLoops.body_contains_continue_or_break(body):
                continue

            unrolled_dag = circuit_to_dag(body).copy_empty_like()
            for index_value in indexset:
                bound_body_dag = circuit_to_dag(body.bind_parameters({loop_parameter: index_value}))
                unrolled_dag.compose(bound_body_dag, inplace=True)
            dag.substitute_node_with_dag(forloop_op, unrolled_dag)

        return dag

    @classmethod
    def body_contains_continue_or_break(cls, circuit):
        """Checks if a circuit contains ``continue``s or ``break``s. Conditional bodies are inspected."""
        for inst in circuit.data:
            operation = inst.operation
            for type_ in [ContinueLoopOp, BreakLoopOp]:
                if isinstance(operation, type_):
                    return True
            if isinstance(operation, IfElseOp):
                for block in operation.params:
                    if UnrollForLoops.body_contains_continue_or_break(block):
                        return True
        return False

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Remove the swaps followed by measurement (and adapt the measurement)."""

from qiskit.circuit import Measure
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow
from qiskit.dagcircuit import DAGOpNode, DAGOutNode


class OptimizeSwapBeforeMeasure(TransformationPass):
    """Remove the swaps followed by measurement (and adapt the measurement).

    Transpiler pass to remove swaps in front of measurements by re-targeting
    the classical bit of the measure instruction.
    """

    @control_flow.trivial_recurse
    def run(self, dag):
        """Run the OptimizeSwapBeforeMeasure pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """
        tail = self._collect_tail(dag)
        if not any(isinstance(node.op, SwapGate) for node in tail):
            return dag

        for node in tail:
            dag.remove_op_node(node)

        # ``tail`` is topologically ordered, so a single forward pass suffices. ``wire_of[q]``
        # is the qubit whose original contents currently sit at position ``q``, given the swaps
        # seen so far. Measurements are re-created on that qubit, in their original order.
        wire_of = {qubit: qubit for qubit in dag.qubits}
        for node in tail:
            if isinstance(node.op, SwapGate):
                a, b = node.qargs
                wire_of[a], wire_of[b] = wire_of[b], wire_of[a]
            else:
                dag.apply_operation_back(
                    Measure(), (wire_of[node.qargs[0]],), node.cargs, check=False
                )
        return dag

    @staticmethod
    def _collect_tail(dag):
        """Return the topologically ordered "tail" of ``dag``: the unconditional swaps and
        measurements all of whose successors are output nodes or themselves part of the tail."""
        tail = set()
        reverse_order = []
        worklist = [
            pred
            for out_node in dag.output_map.values()
            for pred in dag.predecessors(out_node)
        ]
        while worklist:
            node = worklist.pop()
            if node in tail or not isinstance(node, DAGOpNode):
                continue
            if not (
                isinstance(node.op, Measure)
                or (
                    isinstance(node.op, SwapGate)
                    and getattr(node.op, "_condition", None) is None
                )
            ):
                continue
            if all(
                isinstance(succ, DAGOutNode) or succ in tail for succ in dag.successors(node)
            ):
                tail.add(node)
                reverse_order.append(node)
                worklist.extend(dag.predecessors(node))
        reverse_order.reverse()
        return reverse_order

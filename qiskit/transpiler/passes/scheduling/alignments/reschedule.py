# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Rescheduler pass to adjust node start times."""

from typing import List

from qiskit.circuit.delay import Delay
from qiskit.circuit.gate import Gate
from qiskit.circuit.measure import Measure
from qiskit.dagcircuit import DAGCircuit, DAGOpNode, DAGOutNode
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError


class ConstrainedReschedule(AnalysisPass):
    """Rescheduler pass that updates node start times to conform to the hardware alignments.

    This pass shifts DAG node start times previously scheduled with one of
    the scheduling passes, e.g. :class:`ASAPSchedule` or :class:`ALAPSchedule`,
    so that every instruction start time satisfies alignment constraints.

    Examples:

        We assume executing the following circuit on a backend with 16 dt of acquire alignment.

        .. parsed-literal::

                 ┌───┐┌────────────────┐┌─┐
            q_0: ┤ X ├┤ Delay(100[dt]) ├┤M├
                 └───┘└────────────────┘└╥┘
            c: 1/════════════════════════╩═
                                         0

        Note that delay of 100 dt induces a misalignment of 4 dt at the measurement.
        This pass appends an extra 12 dt time shift to the input circuit.

        .. parsed-literal::

                 ┌───┐┌────────────────┐┌─┐
            q_0: ┤ X ├┤ Delay(112[dt]) ├┤M├
                 └───┘└────────────────┘└╥┘
            c: 1/════════════════════════╩═
                                         0

    Notes:

        Your backend may execute circuits violating these alignment constraints.
        However, you may obtain erroneous measurement result because of the
        untracked phase originating in the instruction misalignment.
    """

    def __init__(
        self,
        acquire_alignment: int = 1,
        pulse_alignment: int = 1,
    ):
        """Create new rescheduler pass.

        The alignment values depend on the control electronics of your quantum processor.

        Args:
            acquire_alignment: Integer number representing the minimum time resolution to
                trigger acquisition instruction in units of ``dt``.
            pulse_alignment: Integer number representing the minimum time resolution to
                trigger gate instruction in units of ``dt``.
        """
        super().__init__()
        self.acquire_align = acquire_alignment
        self.pulse_align = pulse_alignment

    @classmethod
    def _get_next_gate(cls, dag: DAGCircuit, node: DAGOpNode) -> List[DAGOpNode]:
        """Get next non-delay nodes.

        Args:
            dag: DAG circuit to be rescheduled with constraints.
            node: Current node.

        Returns:
            A list of non-delay successors.
        """
        op_nodes = []
        for next_node in dag.quantum_successors(node):
            if isinstance(next_node, DAGOutNode):
                continue
            if isinstance(next_node.op, Delay):
                # Ignore delays. We are only interested in start time of instruction nodes.
                op_nodes.extend(cls._get_next_gate(dag, next_node))
            else:
                op_nodes.append(next_node)

        return op_nodes

    def _push_node_back(self, dag: DAGCircuit, node: DAGOpNode, shift: int):
        """Update start time of current node. Successors are also shifted to avoid overlap.

        Args:
            dag: DAG circuit to be rescheduled with constraints.
            node: Current node.
            shift: Amount of required time shift.
        """
        node_start_time = self.property_set["node_start_time"]
        new_t1 = node_start_time[node] + node.op.duration + shift

        # Check successors for overlap
        overlaps = {n: new_t1 - node_start_time[n] for n in self._get_next_gate(dag, node)}

        # Recursively shift next node until overlap is resolved
        for successor, t_overlap in overlaps.items():
            if t_overlap > 0:
                self._push_node_back(dag, successor, t_overlap)

        # Update start time of this node after all overlaps are resolved
        node_start_time[node] += shift

    def run(self, dag: DAGCircuit):
        """Run rescheduler.

        This pass should perform rescheduling to satisfy:

            - All DAGOpNode are placed at start time satisfying hardware alignment constraints.
            - The end time of current does not overlap with the start time of successor nodes.
            - Compiler directives are not necessary satisfying the constraints.

        Assumptions:

            - Topological order and absolute time order of DAGOpNode are consistent.

        Based on the configurations above, rescheduler pass takes following strategy.

        1. Scan node from the beginning, i.e. from left of the circuit. The rescheduler
            calls ``node_start_time`` from the property set,
            and retrieves the scheduled start time of current node.
        2. If the start time of the node violates the alignment constraints,
            the scheduler increases the start time until it satisfies the constraint.
        3. Check overlap with successor nodes. If any overlap occurs, the rescheduler
            recursively pushs the successor nodes backward towards the end of the wire.
            Note that shifted location doesn't need to satisfy the constraints,
            thus it will be a minimum delay to resolve the overlap with the ancestor node.
        4. Repeat 1-3 until the node at the end of the wire. This will resolve
            all misalignment without creating overlap between the nodes.

        Args:
            dag: DAG circuit to be rescheduled with constraints.

        Raises:
            TranspilerError: If circuit is not scheduled.
        """

        if "node_start_time" not in self.property_set:
            raise TranspilerError(
                f"The input circuit {dag.name} is not scheduled. Call one of scheduling passes "
                f"before running the {self.__class__.__name__} pass."
            )

        node_start_time = self.property_set["node_start_time"]

        for node in dag.topological_op_nodes():
            if node_start_time[node] == 0:
                # Every instruction can start at t=0
                continue

            if isinstance(node.op, Gate):
                alignment = self.pulse_align
            elif isinstance(node.op, Measure):
                alignment = self.acquire_align
            else:
                # Directive or delay. These can start at arbitrary time.
                continue

            try:
                misalignment = node_start_time[node] % alignment
                if misalignment == 0:
                    continue
                shift = max(0, alignment - misalignment)
            except KeyError as ex:
                raise TranspilerError(
                    f"Start time of {repr(node)} is not found. This node is likely added after "
                    "this circuit is scheduled. Run scheduler again."
                ) from ex
            if shift > 0:
                self._push_node_back(dag, node, shift)

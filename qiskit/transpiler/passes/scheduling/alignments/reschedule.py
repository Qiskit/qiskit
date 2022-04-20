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

from qiskit.circuit.gate import Gate
from qiskit.circuit.measure import Measure
from qiskit.dagcircuit import DAGCircuit, DAGOpNode, DAGOutNode
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError


class ConstrainedReschedule(AnalysisPass):
    """Rescheduler pass that updates node start times to conform to the hardware alignments.

    This pass shifts DAG node start times previously scheduled with one of
    the scheduling passes, e.g. :class:`ASAPScheduleAnalysis` or :class:`ALAPScheduleAnalysis`,
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
        for next_node in dag.successors(node):
            if isinstance(next_node, DAGOutNode):
                continue
            op_nodes.append(next_node)

        return op_nodes

    def _push_node_back(self, dag: DAGCircuit, node: DAGOpNode, shift: int):
        """Update start time of current node. Successors are also shifted to avoid overlap.

        .. note::

            This logic assumes the all bits in the qregs and cregs synchronously start and end,
            i.e. occupy the same time slot, but qregs and cregs can take
            different time slot due to classical I/O latencies.

        Args:
            dag: DAG circuit to be rescheduled with constraints.
            node: Current node.
            shift: Amount of required time shift.
        """
        node_start_time = self.property_set["node_start_time"]
        conditional_latency = self.property_set.get("conditional_latency", 0)
        clbit_write_latency = self.property_set.get("clbit_write_latency", 0)

        # Compute shifted t1 of this node separately for qreg and creg
        this_t0 = node_start_time[node]
        new_t1q = this_t0 + node.op.duration + shift
        this_qubits = set(node.qargs)
        if isinstance(node.op, Measure):
            # creg access ends at the end of instruction
            new_t1c = new_t1q
            this_clbits = set(node.cargs)
        else:
            if node.op.condition_bits:
                # conditional access ends at the beginning of node start time
                new_t1c = this_t0 + shift
                this_clbits = set(node.op.condition_bits)
            else:
                new_t1c = None
                this_clbits = set()

        # Check successors for overlap
        for next_node in self._get_next_gate(dag, node):
            # Compute next node start time separately for qreg and creg
            next_t0q = node_start_time[next_node]
            next_qubits = set(next_node.qargs)
            if isinstance(next_node.op, Measure):
                # creg access starts after write latency
                next_t0c = next_t0q + clbit_write_latency
                next_clbits = set(next_node.cargs)
            else:
                if next_node.op.condition_bits:
                    # conditional access starts before node start time
                    next_t0c = next_t0q - conditional_latency
                    next_clbits = set(next_node.op.condition_bits)
                else:
                    next_t0c = None
                    next_clbits = set()
            # Compute overlap if there is qubits overlap
            if any(this_qubits & next_qubits):
                qreg_overlap = new_t1q - next_t0q
            else:
                qreg_overlap = 0
            # Compute overlap if there is clbits overlap
            if any(this_clbits & next_clbits):
                creg_overlap = new_t1c - next_t0c
            else:
                creg_overlap = 0
            # Shift next node if there is finite overlap in either in qubits or clbits
            overlap = max(qreg_overlap, creg_overlap)
            if overlap > 0:
                self._push_node_back(dag, next_node, overlap)

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
            - All bits in either qargs or cargs associated with node synchronously start.
            - Start time of qargs and cargs may different due to I/O latency.

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

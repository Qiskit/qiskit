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

"""Compact Scheduling."""
from qiskit.circuit import Gate
from qiskit.dagcircuit import DAGInNode
from qiskit.transpiler.exceptions import TranspilerError

from qiskit.transpiler.passes.scheduling.scheduling.base_scheduler import BaseScheduler
from qiskit.transpiler.passes.scheduling.scheduling.alap import ALAPScheduleAnalysis


class CompactScheduleAnalysis(BaseScheduler):
    """Compact scheduling pass, which schedules the head non-conditional gates on each qubit
    are scheduled as late as possible while the tail non-conditional gates on each qubit
    are scheduled as soon as possible.
    """

    def run(self, dag):
        """Run the compact scheduling pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to schedule.

        Returns:
            DAGCircuit: A scheduled DAG.

        Raises:
            TranspilerError: if the circuit is not mapped on physical qubits.
            TranspilerError: if conditional bit is added to non-supported instruction.
        """
        if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
            raise TranspilerError("Compact scheduling runs on physical circuits only")

        # run ALAP schedule first
        alap_schedule = ALAPScheduleAnalysis(self.durations)
        alap_schedule.run(dag)
        node_start_time = alap_schedule.property_set["node_start_time"]

        # move floating non-conditional gate towards front as possible
        bit_indices = {bit: index for index, bit in enumerate(dag.qubits)}
        for node in reversed(list(dag.topological_op_nodes())):
            if not isinstance(node.op, Gate) or node.op.condition_bits:
                continue

            start_time = node_start_time[node]
            asap_time = 0
            for prev in dag.predecessors(node):
                if isinstance(prev, DAGInNode):
                    asap_time = start_time
                    break

                prev_duration = self._get_node_duration(prev, bit_indices, dag)
                prev_stop_time = node_start_time[prev] + prev_duration
                asap_time = max(asap_time, prev_stop_time)

            if asap_time < start_time:
                node_start_time[node] = asap_time

        self.property_set["node_start_time"] = node_start_time

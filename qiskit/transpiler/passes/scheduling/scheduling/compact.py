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
    """Compact scheduling pass, in which the head non-conditional gates are scheduled
    as late as possible (ALAP) while the tail non-conditional gates are scheduled
    as soon as possible (ASAP).

    For example, given a below circuit,
                    ┌───┐               ┌───┐ ░ ┌─┐
               q_0: ┤ H ├──■─────────■──┤ H ├─░─┤M├──────
                    └───┘┌─┴─┐     ┌─┴─┐└───┘ ░ └╥┘┌─┐
               q_1: ─────┤ X ├──■──┤ X ├──────░──╫─┤M├───
                    ┌───┐└───┘┌─┴─┐├───┤      ░  ║ └╥┘┌─┐
               q_2: ┤ H ├─────┤ X ├┤ H ├──────░──╫──╫─┤M├
                    └───┘     └───┘└───┘      ░  ║  ║ └╥┘
            meas: 3/═════════════════════════════╩══╩══╩═
                                                 0  1  2
    Compact scheduling algorithm schedules it as below.
                          ┌───┐            ┌────────────────┐           ┌───┐        ░ ┌─┐
               q_0: ──────┤ H ├─────────■──┤ Delay(900[dt]) ├──■────────┤ H ├────────░─┤M├──────
                    ┌─────┴───┴──────┐┌─┴─┐└────────────────┘┌─┴─┐┌─────┴───┴──────┐ ░ └╥┘┌─┐
               q_1: ┤ Delay(200[dt]) ├┤ X ├────────■─────────┤ X ├┤ Delay(200[dt]) ├─░──╫─┤M├───
                    ├────────────────┤├───┤      ┌─┴─┐       ├───┤├────────────────┤ ░  ║ └╥┘┌─┐
               q_2: ┤ Delay(700[dt]) ├┤ H ├──────┤ X ├───────┤ H ├┤ Delay(700[dt]) ├─░──╫──╫─┤M├
                    └────────────────┘└───┘      └───┘       └───┘└────────────────┘ ░  ║  ║ └╥┘
            meas: 3/════════════════════════════════════════════════════════════════════╩══╩══╩═
                                                                                        0  1  2
    Notice that the first Hadamard on qubit 2 ALAP while the second Hadamard on qubit 2 ASAP.

    Note that the specification of Compact scheduling defines only how to schedule the head and
    the tail gates. That means there is no specification for how to schedule gates in the middle
    (e.g. if there is one more Hadamard on qubit 0 between two CNOT gates, its position may vary
    depending on the implementation).

    Also note that conditional gates and non-``Gate`` operations are not taken into accounts in
    Compact scheduling. That means they are scheduled (without unnecessary delays) but may not
    in the compact manner.
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

        # move floating non-conditional gate forward as possible
        bit_indices = {bit: index for index, bit in enumerate(dag.qubits)}
        for node in dag.topological_op_nodes():
            if not isinstance(node.op, Gate) or node.op.condition_bits:
                continue

            asap_time = 0
            start_time = node_start_time[node]
            for prev in dag.predecessors(node):
                if isinstance(prev, DAGInNode):
                    asap_time = start_time  # no update of node_start_time[node]
                    break

                prev_duration = self._get_node_duration(prev, bit_indices, dag)
                prev_stop_time = node_start_time[prev] + prev_duration
                asap_time = max(asap_time, prev_stop_time)

            if asap_time < start_time:
                node_start_time[node] = asap_time

        self.property_set["node_start_time"] = node_start_time

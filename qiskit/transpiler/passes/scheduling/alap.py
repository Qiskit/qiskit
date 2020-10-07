# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""ALAP Scheduling."""
from collections import defaultdict
from typing import List

from qiskit.circuit.delay import Delay
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError


class ALAPSchedule(TransformationPass):
    """ALAP Scheduling."""

    def __init__(self, durations):
        """ALAPSchedule initializer.

        Args:
            durations (InstructionDurations): Durations of instructions to be used in scheduling
        """
        super().__init__()
        self.durations = durations

    def run(self, dag, time_unit=None):  # pylint: disable=arguments-differ
        """Run the ALAPSchedule pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to schedule.
            time_unit (str): Time unit to be used in scheduling: 'dt' or 's'.

        Returns:
            DAGCircuit: A scheduled DAG.

        Raises:
            TranspilerError: if the circuit is not mapped on physical qubits.
        """
        if len(dag.qregs) != 1 or dag.qregs.get('q', None) is None:
            raise TranspilerError('ALAP schedule runs on physical circuits only')

        if not time_unit:
            time_unit = self.property_set['time_unit']

        new_dag = DAGCircuit()
        for qreg in dag.qregs.values():
            new_dag.add_qreg(qreg)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)

        qubit_time_available = defaultdict(int)

        def pad_with_delays(qubits: List[int], until, unit) -> None:
            """Pad idle time-slots in ``qubits`` with delays in ``unit`` until ``until``."""
            for q in qubits:
                if qubit_time_available[q] < until:
                    idle_duration = until - qubit_time_available[q]
                    new_dag.apply_operation_front(Delay(idle_duration, unit), [q], [])

        for node in reversed(list(dag.topological_op_nodes())):
            start_time = max(qubit_time_available[q] for q in node.qargs)
            pad_with_delays(node.qargs, until=start_time, unit=time_unit)

            new_node = new_dag.apply_operation_front(node.op, node.qargs, node.cargs,
                                                     node.condition)
            duration = self.durations.get(node.op, node.qargs, unit=time_unit)
            # set duration for each instruction (tricky but necessary)
            new_node.op.duration = duration
            new_node.op.unit = time_unit

            stop_time = start_time + duration
            # update time table
            for q in node.qargs:
                qubit_time_available[q] = stop_time

        working_qubits = qubit_time_available.keys()
        circuit_duration = max(qubit_time_available[q] for q in working_qubits)
        pad_with_delays(new_dag.qubits, until=circuit_duration, unit=time_unit)

        new_dag.name = dag.name
        new_dag.duration = circuit_duration
        new_dag.unit = time_unit
        return new_dag

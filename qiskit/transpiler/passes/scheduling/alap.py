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
import itertools
from collections import defaultdict
from typing import List

from qiskit.circuit import Delay, Measure
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes.scheduling.time_unit_conversion import TimeUnitConversion


class ALAPSchedule(TransformationPass):
    """ALAP Scheduling pass, which schedules the **stop** time of instructions as late as possible.

    For circuits with instructions writing or reading clbits (e.g. measurements, conditional gates),
    the scheduler assumes clbits I/O operations take no time, ``measure`` locks clbits to be written
    at its end and ``c_if`` locks clbits to be read at its beginning.

    Notes:
        The ALAP scheduler may not schedule a circuit exactly the same as any real backend does
        when the circuit contains control flows (e.g. conditional instructions).
    """

    def __init__(self, durations):
        """ALAPSchedule initializer.

        Args:
            durations (InstructionDurations): Durations of instructions to be used in scheduling
        """
        super().__init__()
        self.durations = durations
        # ensure op node durations are attached and in consistent unit
        self.requires.append(TimeUnitConversion(durations))

    def run(self, dag):
        """Run the ALAPSchedule pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to schedule.

        Returns:
            DAGCircuit: A scheduled DAG.

        Raises:
            TranspilerError: if the circuit is not mapped on physical qubits.
        """
        if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
            raise TranspilerError("ALAP schedule runs on physical circuits only")

        time_unit = self.property_set["time_unit"]
        new_dag = DAGCircuit()
        for qreg in dag.qregs.values():
            new_dag.add_qreg(qreg)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)

        qubit_time_available = defaultdict(int)
        clbit_readable = defaultdict(int)
        clbit_writeable = defaultdict(int)

        def pad_with_delays(qubits: List[int], until, unit) -> None:
            """Pad idle time-slots in ``qubits`` with delays in ``unit`` until ``until``."""
            for q in qubits:
                if qubit_time_available[q] < until:
                    idle_duration = until - qubit_time_available[q]
                    new_dag.apply_operation_front(Delay(idle_duration, unit), [q], [])

        bit_indices = {bit: index for index, bit in enumerate(dag.qubits)}
        for node in reversed(list(dag.topological_op_nodes())):
            # validate node.op.duration
            if node.op.duration is None:
                indices = [bit_indices[qarg] for qarg in node.qargs]
                raise TranspilerError(
                    f"Duration of {node.op.name} on qubits {indices} is not found."
                )
            if isinstance(node.op.duration, ParameterExpression):
                indices = [bit_indices[qarg] for qarg in node.qargs]
                raise TranspilerError(
                    f"Parameterized duration ({node.op.duration}) "
                    f"of {node.op.name} on qubits {indices} is not bounded."
                )
            # choose appropriate clbit available time depending on op
            clbit_time_available = (
                clbit_writeable if isinstance(node.op, Measure) else clbit_readable
            )
            # correction to change clbit start time to qubit start time
            delta = 0 if isinstance(node.op, Measure) else node.op.duration
            # must wait for op.condition_bits as well as node.cargs
            start_time = max(
                itertools.chain(
                    (qubit_time_available[q] for q in node.qargs),
                    (clbit_time_available[c] - delta for c in node.cargs + node.op.condition_bits),
                )
            )

            pad_with_delays(node.qargs, until=start_time, unit=time_unit)

            new_dag.apply_operation_front(node.op, node.qargs, node.cargs)

            stop_time = start_time + node.op.duration
            # update time table
            for q in node.qargs:
                qubit_time_available[q] = stop_time
            for c in node.cargs:  # measure
                clbit_writeable[c] = clbit_readable[c] = start_time
            for c in node.op.condition_bits:  # conditional op
                clbit_writeable[c] = max(stop_time, clbit_writeable[c])

        working_qubits = qubit_time_available.keys()
        circuit_duration = max(qubit_time_available[q] for q in working_qubits)
        pad_with_delays(new_dag.qubits, until=circuit_duration, unit=time_unit)

        new_dag.name = dag.name
        new_dag.metadata = dag.metadata
        # set circuit duration and unit to indicate it is scheduled
        new_dag.duration = circuit_duration
        new_dag.unit = time_unit
        return new_dag

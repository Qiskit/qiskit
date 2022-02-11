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
from qiskit.circuit import Delay, Qubit
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.dagcircuit import DAGCircuit, DAGInNode
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

        idle_before = {q: 0 for q in dag.qubits + dag.clbits}
        bit_indices = {bit: index for index, bit in enumerate(dag.qubits)}
        for node in reversed(list(dag.topological_op_nodes())):
            # validate node.op.duration
            if node.op.duration is None:
                indices = [bit_indices[qarg] for qarg in node.qargs]
                if dag.has_calibration_for(node):
                    node.op.duration = dag.calibrations[node.op.name][
                        (tuple(indices), tuple(float(p) for p in node.op.params))
                    ].duration

                if node.op.duration is None:
                    raise TranspilerError(
                        f"Duration of {node.op.name} on qubits {indices} is not found."
                    )
            if isinstance(node.op.duration, ParameterExpression):
                indices = [bit_indices[qarg] for qarg in node.qargs]
                raise TranspilerError(
                    f"Parameterized duration ({node.op.duration}) "
                    f"of {node.op.name} on qubits {indices} is not bounded."
                )

            this_t0 = max(idle_before[q] for q in node.qargs + node.cargs + node.op.condition_bits)
            this_t1 = this_t0 + node.op.duration

            for prev_node in dag.predecessors(node):
                if isinstance(prev_node, DAGInNode):
                    continue
                # Keep current bit status until operation complete
                for blocked_bit in set(node.op.condition_bits) & set(prev_node.cargs):
                    idle_before[blocked_bit] = this_t1

            for bit in node.qargs + node.cargs:
                before = idle_before[bit]
                delta = this_t0 - before
                if before > 0 and delta > 0 and isinstance(bit, Qubit):
                    new_dag.apply_operation_front(Delay(delta, time_unit), [bit], [])
                idle_before[bit] = this_t1

            new_dag.apply_operation_front(node.op, node.qargs, node.cargs)

        circuit_duration = max(idle_before.values())
        for bit, before in idle_before.items():
            delta = circuit_duration - before
            if not (delta > 0 and isinstance(bit, Qubit)):
                continue
            new_dag.apply_operation_front(Delay(delta, time_unit), [bit], [])

        new_dag.name = dag.name
        new_dag.metadata = dag.metadata
        new_dag.calibrations = dag.calibrations

        # set circuit duration and unit to indicate it is scheduled
        new_dag.duration = circuit_duration
        new_dag.unit = time_unit

        return new_dag

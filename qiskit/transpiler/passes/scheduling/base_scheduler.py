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

"""Base circuit scheduling pass."""
import warnings

from qiskit.circuit import Delay, Gate, Measure, Reset
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.dagcircuit import DAGOpNode, DAGCircuit, DAGOutNode
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.instruction_durations import InstructionDurations
from qiskit.transpiler.passes.scheduling.time_unit_conversion import TimeUnitConversion
from qiskit.transpiler.target import Target


class BaseSchedulerTransform(TransformationPass):
    """Base scheduler pass.

    .. warning::

        This base class is not part of the public interface for this module
        it should not be used to develop new scheduling passes as the passes
        which are using this are pending a future deprecation and subsequent
        removal. If you are developing new scheduling passes look at the
        ``BaseScheduler`` class instead which is used in the new scheduling
        pass workflow.

    Policy of topological node ordering in scheduling

        The DAG representation of ``QuantumCircuit`` respects the node ordering also in the
        classical register wires, though theoretically two conditional instructions
        conditioned on the same register are commute, i.e. read-access to the
        classical register doesn't change its state.

        .. code-block:: text

            qc = QuantumCircuit(2, 1)
            qc.delay(100, 0)
            qc.x(0)
            qc.x(1)

        The scheduler SHOULD comply with above topological ordering policy of the DAG circuit.
        Accordingly, the `asap`-scheduled circuit will become

        .. code-block:: text

                 ┌────────────────┐   ┌───┐
            q_0: ┤ Delay(100[dt]) ├───┤ X ├──────────────
                 ├────────────────┤   └───┘      ┌───┐
            q_1: ┤ Delay(100[dt]) ├──────────────┤ X ├───
                 └────────────────┘              └───┘


        Note that this scheduling might be inefficient in some cases,
        because the second conditional operation can start without waiting the delay of 100 dt.
        However, such optimization should be done by another pass,
        otherwise scheduling may break topological ordering of the original circuit.
    """

    CONDITIONAL_SUPPORTED = (Gate, Delay)

    def __init__(
        self,
        durations: InstructionDurations = None,
        clbit_write_latency: int = 0,
        conditional_latency: int = 0,
        target: Target = None,
    ):
        """Scheduler initializer.

        Args:
            durations: Durations of instructions to be used in scheduling
            clbit_write_latency: A control flow constraints. Because standard superconducting
                quantum processor implement dispersive QND readout, the actual data transfer
                to the clbit happens after the round-trip stimulus signal is buffered
                and discriminated into quantum state.
                The interval ``[t0, t0 + clbit_write_latency]`` is regarded as idle time
                for clbits associated with the measure instruction.
                This defaults to 0 dt which is identical to Qiskit Pulse scheduler.
            conditional_latency: A control flow constraints. This value represents
                a latency of reading a classical register for the conditional operation.
                The gate operation occurs after this latency. This appears as a delay
                in front of the DAGOpNode of the gate.
                This defaults to 0 dt.
            target: The :class:`~.Target` representing the target backend, if both
                ``durations`` and this are specified then this argument will take
                precedence and ``durations`` will be ignored.
        """
        super().__init__()
        self.durations = durations
        # Ensure op node durations are attached and in consistent unit
        if target is not None:
            self.durations = target.durations()
        self.requires.append(TimeUnitConversion(self.durations))

        # Control flow constraints.
        self.clbit_write_latency = clbit_write_latency
        self.conditional_latency = conditional_latency

        self.target = target

    def _get_node_duration(
        self,
        node: DAGOpNode,
        dag: DAGCircuit,
    ) -> int:
        """A helper method to get duration from node or calibration."""
        indices = [dag.find_bit(qarg).index for qarg in node.qargs]

        if dag._has_calibration_for(node):
            # If node has calibration, this value should be the highest priority
            cal_key = tuple(indices), tuple(float(p) for p in node.op.params)
            duration = dag._calibrations_prop[node.op.name][cal_key].duration
        elif node.name == "delay":
            duration = node.op.duration
        else:
            try:
                duration = self.durations.get(node.op, indices)
            except TranspilerError:
                duration = None

        if isinstance(node.op, Reset):
            warnings.warn(
                "Qiskit scheduler assumes Reset works similarly to Measure instruction. "
                "Actual behavior depends on the control system of your quantum backend. "
                "Your backend may provide a plugin scheduler pass."
            )
        elif isinstance(node.op, Measure):
            is_mid_circuit = not any(
                isinstance(x, DAGOutNode) for x in dag.quantum_successors(node)
            )
            if is_mid_circuit:
                warnings.warn(
                    "Qiskit scheduler assumes mid-circuit measurement works as a standard instruction. "
                    "Actual backend may apply custom scheduling. "
                    "Your backend may provide a plugin scheduler pass."
                )

        if isinstance(duration, ParameterExpression):
            raise TranspilerError(
                f"Parameterized duration ({duration}) "
                f"of {node.op.name} on qubits {indices} is not bounded."
            )
        if duration is None:
            raise TranspilerError(f"Duration of {node.op.name} on qubits {indices} is not found.")

        return duration

    def _delay_supported(self, qarg: int) -> bool:
        """Delay operation is supported on the qubit (qarg) or not."""
        if self.target is None or self.target.instruction_supported("delay", qargs=(qarg,)):
            return True
        return False

    def run(self, dag: DAGCircuit):
        raise NotImplementedError

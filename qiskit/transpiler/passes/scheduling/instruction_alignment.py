# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Analysis passes for hardware alignment constraints."""

from typing import List

import warnings

from qiskit.circuit.delay import Delay
from qiskit.circuit.gate import Gate
from qiskit.circuit.measure import Measure
from qiskit.dagcircuit import DAGCircuit, DAGOpNode, DAGOutNode
from qiskit.pulse import Play
from qiskit.transpiler.basepasses import TransformationPass, AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError


class ConstrainedReschedule(AnalysisPass):
    """Rescheduler pass that updates node start times to conform to the hardware alignments.

    This is a control electronics aware analysis pass.

    In many quantum computing architectures gates (instructions) are implemented with
    shaped analog stimulus signals. These signals are digitally stored in the
    waveform memory of the control electronics and converted into analog voltage signals
    by electronic components called digital to analog converters (DAC).

    In a typical hardware implementation of superconducting quantum processors,
    a single qubit instruction is implemented by a
    microwave signal with the duration of around several tens of ns with a per-sample
    time resolution of ~0.1-10ns, as reported by ``backend.configuration().dt``.
    In such systems requiring higher DAC bandwidth, control electronics often
    defines a `pulse granularity`, in other words a data chunk, to allow the DAC to
    perform the signal conversion in parallel to gain the bandwidth.

    A control electronics, i.e. micro-architecture, of the real quantum backend may
    impose some constraints on the start time of microinstructions.
    In Qiskit SDK, the duration of :class:`qiskit.circuit.Delay` can take arbitrary
    value in units of dt, thus circuits involving delays may violate the constraints,
    which may result in failure in the circuit execution on the backend.

    This pass shifts DAG node start times previously scheduled with one of
    the scheduling passes, e.g. :class:`ASAPSchedule` or :class:`ALAPSchedule`,
    so that every instruction start time satisfies alignment constraints described below.

    Pulse alignment constraints

        This value is reported by ``timing_constraints["pulse_alignment"]`` in the backend
        configuration in units of dt. The start time of the all pulse instruction should be
        multiple of this value. Violation of this constraint may result in the
        backend execution failure.

        In most of the senarios, the scheduled start time of ``DAGOpNode`` corresponds to the
        start time of the underlying pulse instruction composing the node operation.
        However, this assumption can be intentionally broken by defining a pulse gate,
        i.e. calibration, with the schedule involving pre-buffer, i.e. some random pulse delay
        followed by a pulse instruction. Because this pass is not aware of such edge case,
        the user must take special care of pulse gates if any.

    Acquire alignment constraints

        This value is reported by ``timing_constraints["acquire_alignment"]`` in the backend
        configuration in units of dt. The start time of the :class:`~qiskit.circuit.Measure`
        instruction should be multiple of this value.

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
            TranspilerError: Alignment is necessary but scheduling is not performed.
        """
        # Rescheduling is not necessary
        if self.acquire_align == 1 and self.pulse_align == 1:
            return

        run_rescheduler = False

        # Check delay durations
        for delay_node in dag.op_nodes(Delay):
            dur = delay_node.op.duration
            if not (dur % self.acquire_align == 0 and dur % self.pulse_align == 0):
                run_rescheduler = True
                break

        # Check custom gate durations
        for inst_defs in dag.calibrations.values():
            for caldef in inst_defs.values():
                dur = caldef.duration
                if not (dur % self.acquire_align == 0 and dur % self.pulse_align == 0):
                    run_rescheduler = True
                    break

        if not run_rescheduler:
            return

        # Need scheduling for alignment. Error is caused when DAG circuit is not scheduled.
        # This check should be placed here for backward compatibility.
        # Now rescheduler pass is set to all preset pass managers,
        # but scheduling is disabled by default.
        # If this pass detect any violation risk, it will ask user to schedule the circuit.
        if "node_start_time" not in self.property_set:
            raise TranspilerError(
                f"Input DAG {dag.name} likely needs alignment but no scheduling is performed. "
                "Set scheduling method or add explicit scheduling pass to the pass manager."
            )
        node_start_time = self.property_set["node_start_time"]

        for node in dag.topological_op_nodes():
            if isinstance(node.op, Gate):
                alignment = self.pulse_align
            elif isinstance(node.op, Measure):
                alignment = self.acquire_align
            else:
                # Directive or delay. These can be start at arbitrary time.
                continue

            try:
                shift = max(0, alignment - node_start_time[node] % alignment)
            except KeyError as ex:
                raise TranspilerError(
                    f"Start time of {repr(node)} is not found. This node is likely added after "
                    "this circuit is scheduled. Run scheduler again."
                ) from ex
            if shift > 0:
                self._push_node_back(dag, node, shift)


class AlignMeasures(TransformationPass):
    """Deprecated. Measurement alignment."""

    def __new__(cls, alignment: int = 1) -> ConstrainedReschedule:
        """Create new pass.

        Args:
            alignment: Integer number representing the minimum time resolution to
                trigger measure instruction in units of ``dt``. This value depends on
                the control electronics of your quantum processor.

        Returns:
            ConstrainedReschedule instance that is a drop-in-replacement of this class.
        """
        warnings.warn(
            f"{cls.__name__} has been deprecated as of Qiskit 20.0. "
            f"Use ConstrainedReschedule pass instead.",
            FutureWarning,
        )
        return ConstrainedReschedule(acquire_alignment=alignment)

    def run(self, dag):
        raise NotImplementedError


class ValidatePulseGates(AnalysisPass):
    """Check custom gate length.

    This is a control electronics aware analysis pass.

    Quantum gates (instructions) are often implemented with shaped analog stimulus signals.
    These signals may be digitally stored in the waveform memory of the control electronics
    and converted into analog voltage signals by electronic components known as
    digital to analog converters (DAC).

    In Qiskit SDK, we can define the pulse-level implementation of custom quantum gate
    instructions, as a `pulse gate
    <https://qiskit.org/documentation/tutorials/circuits_advanced/05_pulse_gates.html>`__,
    thus user gates should satisfy all waveform memory constraints imposed by the backend.

    This pass validates all attached calibration entries and raises ``TranspilerError`` to
    kill the transpilation process if any invalid calibration entry is found.
    This pass saves users from waiting until job execution time to get an invalid pulse error from
    the backend control electronics.
    """

    def __init__(
        self,
        granularity: int = 1,
        min_length: int = 1,
    ):
        """Create new pass.

        Args:
            granularity: Integer number representing the minimum time resolution to
                define the pulse gate length in units of ``dt``. This value depends on
                the control electronics of your quantum processor.
            min_length: Integer number representing the minimum data point length to
                define the pulse gate in units of ``dt``. This value depends on
                the control electronics of your quantum processor.
        """
        super().__init__()
        self.granularity = granularity
        self.min_length = min_length

    def run(self, dag: DAGCircuit):
        """Run the measurement alignment pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to be checked.

        Returns:
            DAGCircuit: DAG with consistent timing and op nodes annotated with duration.

        Raises:
            TranspilerError: When pulse gate violate pulse controller constraints.
        """
        if self.granularity == 1 and self.min_length == 1:
            # we can define arbitrary length pulse with dt resolution
            return

        for gate, insts in dag.calibrations.items():
            for qubit_param_pair, schedule in insts.items():
                for _, inst in schedule.instructions:
                    if isinstance(inst, Play):
                        pulse = inst.pulse
                        if pulse.duration % self.granularity != 0:
                            raise TranspilerError(
                                f"Pulse duration is not multiple of {self.granularity}. "
                                "This pulse cannot be played on the specified backend. "
                                f"Please modify the duration of the custom gate pulse {pulse.name} "
                                f"which is associated with the gate {gate} of "
                                f"qubit {qubit_param_pair[0]}."
                            )
                        if pulse.duration < self.min_length:
                            raise TranspilerError(
                                f"Pulse gate duration is less than {self.min_length}. "
                                "This pulse cannot be played on the specified backend. "
                                f"Please modify the duration of the custom gate pulse {pulse.name} "
                                f"which is associated with the gate {gate} of "
                                "qubit {qubit_param_pair[0]}."
                            )

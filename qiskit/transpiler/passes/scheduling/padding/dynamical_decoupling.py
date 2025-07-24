# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Dynamical Decoupling insertion pass."""
from __future__ import annotations

import logging
import numpy as np

from qiskit.circuit import Gate, ParameterExpression, Qubit
from qiskit.circuit.delay import Delay
from qiskit.circuit.library.standard_gates import IGate, UGate, U3Gate
from qiskit.circuit.reset import Reset
from qiskit.dagcircuit import DAGCircuit, DAGNode, DAGInNode, DAGOpNode, DAGOutNode
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.synthesis.one_qubit import OneQubitEulerDecomposer
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.instruction_durations import InstructionDurations
from qiskit.transpiler.passes.optimization import Optimize1qGates
from qiskit.transpiler.passes.scheduling.padding.base_padding import BasePadding
from qiskit.transpiler.target import Target
from qiskit._accelerate.pad_dynamical_decoupling import pad_dynamical_decoupling


logger = logging.getLogger(__name__)


class PadDynamicalDecoupling(BasePadding):
    """Dynamical decoupling insertion pass.

    This pass works on a scheduled, physical circuit. It scans the circuit for
    idle periods of time (i.e. those containing delay instructions) and inserts
    a DD sequence of gates in those spots. These gates amount to the identity,
    so do not alter the logical action of the circuit, but have the effect of
    mitigating decoherence in those idle periods.

    As a special case, the pass allows a length-1 sequence (e.g. ``[XGate()]``).
    In this case the DD insertion happens only when the gate inverse can be
    absorbed into a neighboring gate in the circuit (so we would still be
    replacing Delay with something that is equivalent to the identity).
    This can be used, for instance, as a Hahn echo.

    This pass ensures that the inserted sequence preserves the circuit exactly
    (including global phase).

    .. plot::
       :alt: Output from the previous code.
       :include-source:

        import numpy as np
        from qiskit.circuit import QuantumCircuit
        from qiskit.circuit.library import XGate
        from qiskit.transpiler import PassManager, InstructionDurations, Target, CouplingMap
        from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDynamicalDecoupling
        from qiskit.visualization import timeline_drawer

        circ = QuantumCircuit(4)
        circ.h(0)
        circ.cx(0, 1)
        circ.cx(1, 2)
        circ.cx(2, 3)
        circ.measure_all()
        durations = InstructionDurations(
            [("h", 0, 50), ("cx", [0, 1], 700), ("reset", None, 10),
             ("cx", [1, 2], 200), ("cx", [2, 3], 300),
             ("x", None, 50), ("measure", None, 1000)],
            dt=1e-7
        )
        target = Target.from_configuration(
            ["h", "cx", "reset", "x", "measure"],
            num_qubits=4,
            coupling_map=CouplingMap.from_line(4, bidirectional=False),
            instruction_durations=durations,
            dt=1e-7,
        )

        # balanced X-X sequence on all qubits
        dd_sequence = [XGate(), XGate()]
        pm = PassManager([ALAPScheduleAnalysis(durations),
                          PadDynamicalDecoupling(durations, dd_sequence)])
        circ_dd = pm.run(circ)
        timeline_drawer(circ_dd, target=target)

        # Uhrig sequence on qubit 0
        n = 8
        dd_sequence = [XGate()] * n
        def uhrig_pulse_location(k):
            return np.sin(np.pi * (k + 1) / (2 * n + 2)) ** 2
        spacing = []
        for k in range(n):
            spacing.append(uhrig_pulse_location(k) - sum(spacing))
        spacing.append(1 - sum(spacing))
        pm = PassManager(
            [
                ALAPScheduleAnalysis(durations),
                PadDynamicalDecoupling(durations, dd_sequence, qubits=[0], spacing=spacing),
            ]
        )
        circ_dd = pm.run(circ)
        timeline_drawer(circ_dd, target=target)

    .. note::

        You may need to call alignment pass before running dynamical decoupling to guarantee
        your circuit satisfies acquisition alignment constraints.
    """

    def __init__(
        self,
        durations: InstructionDurations = None,
        dd_sequence: list[Gate] = None,
        qubits: list[int] | None = None,
        spacing: list[float] | None = None,
        skip_reset_qubits: bool = True,
        pulse_alignment: int = 1,
        extra_slack_distribution: str = "middle",
        target: Target = None,
    ):
        """Dynamical decoupling initializer.

        Args:
            durations: Durations of instructions to be used in scheduling.
            dd_sequence: Sequence of gates to apply in idle spots.
            qubits: Physical qubits on which to apply DD.
                If None, all qubits will undergo DD (when possible).
            spacing: A list of spacings between the DD gates.
                The available slack will be divided according to this.
                The list length must be one more than the length of dd_sequence,
                and the elements must sum to 1. If None, a balanced spacing
                will be used [d/2, d, d, ..., d, d, d/2].
            skip_reset_qubits: If True, does not insert DD on idle periods that
                immediately follow initialized/reset qubits
                (as qubits in the ground state are less susceptible to decoherence).
            pulse_alignment: The hardware constraints for gate timing allocation.
                This is usually provided from ``backend.configuration().timing_constraints``.
                If provided, the delay length, i.e. ``spacing``, is implicitly adjusted to
                satisfy this constraint.
            extra_slack_distribution: The option to control the behavior of DD sequence generation.
                The duration of the DD sequence should be identical to an idle time in the
                scheduled quantum circuit, however, the delay in between gates comprising the sequence
                should be integer number in units of dt, and it might be further truncated
                when ``pulse_alignment`` is specified. This sometimes results in the duration of
                the created sequence being shorter than the idle time
                that you want to fill with the sequence, i.e. `extra slack`.
                This option takes following values.

                    - "middle": Put the extra slack to the interval at the middle of the sequence.
                    - "edges": Divide the extra slack as evenly as possible into
                      intervals at beginning and end of the sequence.
            target: The :class:`~.Target` representing the target backend.
                Target takes precedence over other arguments when they can be inferred from target.
                Therefore specifying target as well as other arguments like ``durations`` or
                ``pulse_alignment`` will cause those other arguments to be ignored.

        Raises:
            TranspilerError: When invalid DD sequence is specified.
            TranspilerError: When pulse gate with the duration which is
                non-multiple of the alignment constraint value is found.
            TypeError: If ``dd_sequence`` is not specified
        """
        super().__init__(target=target, durations=durations)
        self._durations = durations
        if dd_sequence is None:
            raise TypeError("required argument 'dd_sequence' is not specified")
        self._dd_sequence = dd_sequence
        self._qubits = qubits
        self._skip_reset_qubits = skip_reset_qubits
        self._alignment = pulse_alignment
        self._spacing = spacing
        self._extra_slack_distribution = extra_slack_distribution

        self._no_dd_qubits: set[int] = set()
        self._dd_sequence_lengths: dict[Qubit, list[int]] = {}
        self._sequence_phase = 0
        if target is not None:
            # The priority order for instruction durations is: target > standalone.
            self._durations = target.durations()
            self._alignment = target.pulse_alignment
            for gate in dd_sequence:
                if gate.name not in target.operation_names:
                    raise TranspilerError(
                        f"{gate.name} in dd_sequence is not supported in the target"
                    )

    def _pre_runhook(self, dag: DAGCircuit):
        super()._pre_runhook(dag)

        durations = InstructionDurations()
        if self._durations is not None:
            durations.update(self._durations, getattr(self._durations, "dt", None))
        num_pulses = len(self._dd_sequence)

        # Check if physical circuit is given
        if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
            raise TranspilerError("DD runs on physical circuits only.")

        # Set default spacing otherwise validate user input
        if self._spacing is None:
            mid = 1 / num_pulses
            end = mid / 2
            self._spacing = [end] + [mid] * (num_pulses - 1) + [end]
        else:
            if sum(self._spacing) != 1 or any(a < 0 for a in self._spacing):
                raise TranspilerError(
                    "The spacings must be given in terms of fractions "
                    "of the slack period and sum to 1."
                )

        # Check if DD sequence is identity
        if num_pulses != 1:
            if num_pulses % 2 != 0:
                raise TranspilerError("DD sequence must contain an even number of gates (or 1).")
            noop = np.eye(2)
            for gate in self._dd_sequence:
                noop = noop.dot(gate.to_matrix())
            if not matrix_equal(noop, IGate().to_matrix(), ignore_phase=True):
                raise TranspilerError("The DD sequence does not make an identity operation.")
            self._sequence_phase = np.angle(noop[0][0])

        # Compute no DD qubits on which any gate in dd_sequence is not supported in the target
        for qarg, _ in enumerate(dag.qubits):
            for gate in self._dd_sequence:
                if not self.__gate_supported(gate, qarg):
                    self._no_dd_qubits.add(qarg)
                    logger.debug(
                        "No DD on qubit %d as gate %s is not supported on it", qarg, gate.name
                    )
                    break
        # Precompute qubit-wise DD sequence length for performance
        for physical_index, qubit in enumerate(dag.qubits):
            if not self.__is_dd_qubit(physical_index):
                continue

            sequence_lengths = []
            for index, gate in enumerate(self._dd_sequence):
                gate_length = durations.get(gate, physical_index)
                sequence_lengths.append(gate_length)
                # Update gate duration. This is necessary for current timeline drawer, i.e. scheduled.
                gate = gate.to_mutable()
                self._dd_sequence[index] = gate
                gate.duration = gate_length
            self._dd_sequence_lengths[qubit] = sequence_lengths

    def __gate_supported(self, gate: Gate, qarg: int) -> bool:
        """A gate is supported on the qubit (qarg) or not."""
        if self.target is None or self.target.instruction_supported(gate.name, qargs=(qarg,)):
            return True
        return False

    def __is_dd_qubit(self, qubit_index: int) -> bool:
        """DD can be inserted in the qubit or not."""
        if (qubit_index in self._no_dd_qubits) or (
            self._qubits and qubit_index not in self._qubits
        ):
            return False
        return True

    def _pad(
        self,
        dag: DAGCircuit,
        qubit: Qubit,
        t_start: int,
        t_end: int,
        next_node: DAGNode,
        prev_node: DAGNode,
    ):  
        # NOTE: The :meth:`._pad` method is currently being ported as it is the core logic within the ``PadDynamicalDecoupling`` pass
        # and the :meth:`.run` method would need to be ported along with the :class:`.BasePadding` class when the ``PadDelay`` pass is also ported.
        pad_dynamical_decoupling(
            t_end,
            t_start,
            self._alignment,
            prev_node,
            next_node,
            self._no_dd_qubits,
            self._qubits,
            qubit,
            dag,
            self.property_set,
            self._skip_reset_qubits,
            self._dd_sequence_lengths,
            self._sequence_phase,
            self._dd_sequence,
            hasattr(prev_node, "op") and isinstance(prev_node.op, Reset),
            self._spacing,
            self._extra_slack_distribution,
        )

    @staticmethod
    def _resolve_params(gate: Gate) -> tuple:
        """Return gate params with any bound parameters replaced with floats"""
        params = []
        for p in gate.params:
            if isinstance(p, ParameterExpression) and not p.parameters:
                params.append(float(p))
            else:
                params.append(p)
        return tuple(params)


def _format_node(node: DAGNode) -> str:
    """Util to format the DAGNode, DAGInNode, and DAGOutNode."""
    if isinstance(node, (DAGInNode, DAGOutNode)):
        return f"{node.__class__.__name__} on qarg {node.wire}"
    return f"DAGNode {node.name} on qargs {node.qargs}"

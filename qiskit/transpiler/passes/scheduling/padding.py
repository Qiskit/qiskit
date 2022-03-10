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

"""Padding pass to fill empty timeslot."""

from typing import List, Optional
from itertools import zip_longest

import numpy as np

from qiskit.circuit import Qubit, Gate
from qiskit.circuit.delay import Delay
from qiskit.circuit.reset import Reset
from qiskit.circuit.library.standard_gates import IGate, UGate, U3Gate
from qiskit.dagcircuit import DAGCircuit, DAGNode, DAGOutNode, DAGInNode, DAGOpNode
from qiskit.transpiler.instruction_durations import InstructionDurations
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.optimization import Optimize1qGates
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.quantum_info.synthesis import OneQubitEulerDecomposer
from qiskit.transpiler.exceptions import TranspilerError


class BasePadding(TransformationPass):
    """The base class of padding pass.

    This pass requires one of scheduling passes to be executed before itself.
    Since there are multiple scheduling strategies, the selection of scheduling
    pass is left in the hands of the pass manager designer.
    Once a scheduling analysis pass is run, ``node_start_time`` is generated
    in the :attr:`property_set`.  This information is represented by a python dictionary of
    the expected instruction execution times keyed on the node instances.
    Entries in the dictionary are only created for non-delay nodes.
    The padding pass expects all ``DAGOpNode`` in the circuit to be scheduled.

    This base class doesn't define any sequence to interleave, but it manages
    the location where the sequence is inserted, and provides a set of information necessary
    to construct the proper sequence. Thus, a subclass of this pass just needs to implement
    :meth:`_pad` method, in which the subclass constructs a circuit block to insert.
    This mechanism removes lots of boilerplate logic to manage whole DAG circuits.

    Note that padding pass subclasses should define interleaving sequences satisfying:

        - Interleaved sequence does not change start time of other nodes
        - Interleaved sequence should have total duration of the provided ``time_interval``.

    Any manipulation violating these constraints may prevent this base pass from correctly
    tracking the start time of each instruction,
    which may result in violation of hardware alignment constraints.
    """

    def run(self, dag: DAGCircuit):
        """Run the padding pass on ``dag``.

        Args:
            dag: DAG to be checked.

        Returns:
            DAGCircuit: DAG with idle time filled with instructions.

        Raises:
            TranspilerError: When a particular node is not scheduled, likely some transform pass
                is inserted before this node is called.
        """
        self._pre_runhook(dag)

        node_start_time = self.property_set["node_start_time"]

        new_dag = DAGCircuit()
        for qreg in dag.qregs.values():
            new_dag.add_qreg(qreg)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)

        new_dag.name = dag.name
        new_dag.metadata = dag.metadata
        new_dag.unit = self.property_set["time_unit"]
        new_dag.calibrations = dag.calibrations

        idle_after = {bit: 0 for bit in dag.qubits}

        # Compute fresh circuit duration from the node start time dictionary and op duration.
        # Note that pre-scheduled duration may change within the alignment passes, i.e.
        # if some instruction time t0 violating the hardware alignment constraint,
        # the alignment pass may delay t0 and accordingly the circuit duration changes.
        circuit_duration = 0
        for node in dag.topological_op_nodes():
            if node in node_start_time:
                t0 = node_start_time[node]
                t1 = t0 + node.op.duration
                circuit_duration = max(circuit_duration, t1)

                if isinstance(node.op, Delay):
                    # The padding class considers a delay instruction as idle time
                    # rather than instruction. Delay node is removed so that
                    # we can extract non-delay predecessors.
                    dag.remove_op_node(node)
                    continue

                for bit in node.qargs:

                    # Fill idle time with some sequence
                    if t0 - idle_after[bit] > 0:
                        # Find previous node on the wire, i.e. always the latest node on the wire
                        prev_node = next(new_dag.predecessors(new_dag.output_map[bit]))
                        self._pad(
                            dag=new_dag,
                            qubit=bit,
                            t_start=idle_after[bit],
                            t_end=t0,
                            next_node=node,
                            prev_node=prev_node,
                        )

                    idle_after[bit] = t1

                new_dag.apply_operation_back(node.op, node.qargs, node.cargs)
            else:
                raise TranspilerError(
                    f"Operation {repr(node)} is likely added after the circuit is scheduled. "
                    "Schedule the circuit again if you transformed it."
                )

        # Add delays until the end of circuit.
        for bit in new_dag.qubits:
            if circuit_duration - idle_after[bit] > 0:
                node = new_dag.output_map[bit]
                prev_node = next(new_dag.predecessors(node))
                self._pad(
                    dag=new_dag,
                    qubit=bit,
                    t_start=idle_after[bit],
                    t_end=circuit_duration,
                    next_node=node,
                    prev_node=prev_node,
                )

        new_dag.duration = circuit_duration

        # Invalidate old schedule information since delays are filled with sequence.
        del self.property_set["node_start_time"]

        return new_dag

    def _pre_runhook(self, dag: DAGCircuit):
        """Extra routine inserted before running the padding pass.

        Args:
            dag: DAG circuit that sequence is applied.

        Raises:
            TranspilerError: If the whole circuit or instruction is not scheduled.
        """
        if "node_start_time" not in self.property_set:
            raise TranspilerError(
                f"The input circuit {dag.name} is not scheduled. Call one of scheduling passes "
                f"before running the {self.__class__.__name__} pass."
            )

    def _pad(
        self,
        dag: DAGCircuit,
        qubit: Qubit,
        t_start: int,
        t_end: int,
        next_node: DAGNode,
        prev_node: DAGNode,
    ):
        """Interleave instruction sequence in between two nodes.

        Args:
            dag: DAG circuit that sequence is applied.
            qubit: The wire that the sequence is applied on.
            t_start: Absolute start time of this interval.
            t_end: Absolute end time of this interval.
            next_node: Node that follows the sequence.
            prev_node: Node ahead of the sequence.
        """
        raise NotImplementedError


class PadDelay(BasePadding):
    """Padding idle time with Delay instructions.

    Consecutive delays will be merged in the output of this pass.

    .. code-block::python

        durations = InstructionDurations([("x", None, 160), ("cx", None, 800)])

        qc = QuantumCircuit(2)
        qc.delay(100, 0)
        qc.x(1)
        qc.cx(0, 1)

    The ASAP-scheduled circuit output may become

    .. parsed-literal::

             ┌────────────────┐
        q_0: ┤ Delay(160[dt]) ├──■──
             └─────┬───┬──────┘┌─┴─┐
        q_1: ──────┤ X ├───────┤ X ├
                   └───┘       └───┘

    Note that the additional idle time of 60dt on the ``q_0`` wire coming from the duration difference
    between ``Delay`` of 100dt (``q_0``) and ``XGate`` of 160 dt (``q_1``) is absorbed in
    the delay instruction on the ``q_0`` wire, i.e. in total 160 dt.

    See :class:`BasePadding` pass for details.
    """

    def __init__(self, fill_very_end: bool = True):
        """Create new padding delay pass.

        Args:
            fill_very_end: Set ``True`` to fill the end of circuit with delay.
        """
        super().__init__()
        self.fill_very_end = fill_very_end

    def _pad(
        self,
        dag: DAGCircuit,
        qubit: Qubit,
        t_start: int,
        t_end: int,
        next_node: DAGNode,
        prev_node: DAGNode,
    ):
        if not self.fill_very_end and isinstance(next_node, DAGOutNode):
            return

        time_interval = t_end - t_start
        dag.apply_operation_back(Delay(time_interval, dag.unit), [qubit])


class DynamicalDecoupling(BasePadding):
    """Dynamical decoupling insertion pass.

    This pass works on a scheduled, physical circuit. It scans the circuit for
    idle periods of time (i.e. those containing delay instructions) and inserts
    a DD sequence of gates in those spots. These gates amount to the identity,
    so do not alter the logical action of the circuit, but have the effect of
    mitigating decoherence in those idle periods.

    As a special case, the pass allows a length-1 sequence (e.g. [XGate()]).
    In this case the DD insertion happens only when the gate inverse can be
    absorbed into a neighboring gate in the circuit (so we would still be
    replacing Delay with something that is equivalent to the identity).
    This can be used, for instance, as a Hahn echo.

    This pass ensures that the inserted sequence preserves the circuit exactly
    (including global phase).

    .. jupyter-execute::

        import numpy as np
        from qiskit.circuit import QuantumCircuit
        from qiskit.circuit.library import XGate
        from qiskit.transpiler import PassManager, InstructionDurations
        from qiskit.transpiler.passes import ALAPSchedule, DynamicalDecoupling
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
             ("x", None, 50), ("measure", None, 1000)]
        )

    .. jupyter-execute::

        # balanced X-X sequence on all qubits
        dd_sequence = [XGate(), XGate()]
        pm = PassManager([ALAPSchedule(durations),
                          DynamicalDecoupling(durations, dd_sequence)])
        circ_dd = pm.run(circ)
        timeline_drawer(circ_dd)

    .. jupyter-execute::

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
                ALAPSchedule(durations),
                DynamicalDecoupling(durations, dd_sequence, qubits=[0], spacing=spacing),
            ]
        )
        circ_dd = pm.run(circ)
        timeline_drawer(circ_dd)

    .. note::

        You may need to call alignment pass before running dynamical decoupling to guarantee
        your circuit satisfies acquisition alignment constraints.
    """

    def __init__(
        self,
        durations: InstructionDurations,
        dd_sequence: List[Gate],
        qubits: Optional[List[int]] = None,
        spacing: Optional[List[float]] = None,
        skip_reset_qubits: bool = True,
        pulse_alignment: int = 1,
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
                (as qubits in the ground state are less susceptile to decoherence).
            pulse_alignment: The hardware constraints for gate timing allocation.
                This is usually provided from ``backend.configuration().timing_constraints``.
                If provided, the delay length, i.e. ``spacing``, is implicitly adjusted to
                satisfy this constraint.

        Raises:
            TranspilerError: When invalid DD sequence is specified.
            TranspilerError: When pulse gate with the duration which is
                non-multiple of the alignment constraint value is found.
        """
        super().__init__()
        self._durations = durations
        self._dd_sequence = dd_sequence
        self._qubits = qubits
        self._skip_reset_qubits = skip_reset_qubits
        self._alignment = pulse_alignment
        self._spacing = spacing

        self._dd_sequence_lengths = dict()
        self._sequence_phase = 0

    def _pre_runhook(self, dag: DAGCircuit):
        super()._pre_runhook(dag)

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

        # Precompute qubit-wise DD sequence length for performance
        for qubit in dag.qubits:
            physical_index = dag.qubits.index(qubit)
            if self._qubits and physical_index not in self._qubits:
                continue

            gate_length_sum = 0
            for gate in self._dd_sequence:
                try:
                    # Check calibration.
                    gate_length = dag.calibrations[gate.name][(physical_index, gate.params)]
                    if gate_length % self._alignment != 0:
                        # This is necessary to implement lightweight scheduling logic for this pass.
                        # Usually the pulse alignment constraint and pulse data chunk size take
                        # the same value, however, we can intentionally violate this pattern
                        # at the gate level. For example, we can create a schedule consisting of
                        # a pi-pulse of 32 dt followed by a post buffer, i.e. delay, of 4 dt
                        # on the device with 16 dt constraint. Note that the pi-pulse length
                        # is multiple of 16 dt but the gate length of 36 is not multiple of it.
                        # Such pulse gate should be excluded.
                        raise TranspilerError(
                            f"Pulse gate {gate.name} with length non-multiple of {self._alignment} "
                            f"is not acceptable in {self.__class__.__name__} pass."
                        )
                except KeyError:
                    gate_length = self._durations.get(gate, physical_index)
                gate_length_sum += gate_length
            self._dd_sequence_lengths[qubit] = gate_length_sum

    def _pad(
        self,
        dag: DAGCircuit,
        qubit: Qubit,
        t_start: int,
        t_end: int,
        next_node: DAGNode,
        prev_node: DAGNode,
    ):
        # This routine takes care of the pulse alignment constraint for the DD sequence.
        # Note that the alignment constraint acts on the t0 of the DAGOpNode.
        # Now this constarained scheduling problem is simplified to the problem of
        # finding delay amount which is multiple of the constraint value by assuming
        # that the duration of every DAGOpNode is also multiple of the constraint value.
        #
        # For example, given the constraint value of 16 and XY4 with 160 dt gates.
        # Here we assume current interval is 992 dt.
        #
        # relative spacing := [0.125, 0.25, 0.25, 0.25, 0.125]
        # slack = 992 dt - 4 x 160 dt = 352 dt
        #
        # unconstraind sequence: 44dt-X1-88dt-Y2-88dt-X3-88dt-Y4-44dt
        # constraind sequence  : 32dt-X1-80dt-Y2-80dt-X3-80dt-Y4-32dt + extra slack 48 dt
        #
        # Now we evenly split extra slack into start and end of the sequence.
        # The distributed slack should be multiple of 16.
        # Start = +16, End += 32
        #
        # final sequence       : 48dt-X1-80dt-Y2-80dt-X3-80dt-Y4-64dt / in total 992 dt
        #
        # Now we verify t0 of every node starts from multiple of 16 dt.
        #
        # X1:  48 dt (3 x 16 dt)
        # Y2:  48 dt + 160 dt + 80 dt = 288 dt (18 x 16 dt)
        # Y3: 288 dt + 160 dt + 80 dt = 528 dt (33 x 16 dt)
        # Y4: 368 dt + 160 dt + 80 dt = 768 dt (48 x 16 dt)
        #
        # As you can see, constraints on t0 are all satified without explicit scheduling.
        time_interval = t_end - t_start

        if self._qubits and dag.qubits.index(qubit) not in self._qubits:
            # Target physical qubit is not the target of this DD sequence.
            dag.apply_operation_back(Delay(time_interval, dag.unit), [qubit])
            return

        if self._skip_reset_qubits and (
            isinstance(prev_node, DAGInNode) or isinstance(prev_node.op, Reset)
        ):
            # Previous node is the start edge or reset, i.e. qubit is ground state.
            dag.apply_operation_back(Delay(time_interval, dag.unit), [qubit])
            return

        slack = time_interval - self._dd_sequence_lengths[qubit]
        sequence_gphase = self._sequence_phase

        if slack <= 0:
            # Interval too short.
            dag.apply_operation_back(Delay(time_interval, dag.unit), [qubit])
            return

        if len(self._dd_sequence) == 1:
            # Special case of using a single gate for DD
            u_inv = self._dd_sequence[0].inverse().to_matrix()
            theta, phi, lam, phase = OneQubitEulerDecomposer().angles_and_phase(u_inv)
            if isinstance(next_node, DAGOpNode) and isinstance(next_node.op, (UGate, U3Gate)):
                # Absorb the inverse into the successor (from left in circuit)
                theta_r, phi_r, lam_r = next_node.op.params
                next_node.op.params = Optimize1qGates.compose_u3(
                    theta_r, phi_r, lam_r, theta, phi, lam
                )
                sequence_gphase += phase
            elif isinstance(prev_node, DAGOpNode) and isinstance(prev_node.op, (UGate, U3Gate)):
                # Absorb the inverse into the predecessor (from right in circuit)
                theta_l, phi_l, lam_l = prev_node.op.params
                prev_node.op.params = Optimize1qGates.compose_u3(
                    theta, phi, lam, theta_l, phi_l, lam_l
                )
                sequence_gphase += phase
            else:
                # Don't do anything if there's no single-qubit gate to absorb the inverse
                dag.apply_operation_back(Delay(time_interval, dag.unit), [qubit])
                return

        def _constrained_length(values):
            return self._alignment * np.floor(values / self._alignment)

        # (1) Compute DD intervals satisfying the constraint
        taus = _constrained_length(slack * np.asarray(self._spacing))
        extra_slack = slack - np.sum(taus)

        # (2) Distribute extra slack as evenly as possible
        to_begin_edge = _constrained_length(extra_slack / 2)
        taus[0] += to_begin_edge
        taus[-1] += extra_slack - to_begin_edge

        # (3) Construct DD sequence with delays
        for tau, gate in zip_longest(taus, self._dd_sequence):
            if tau > 0:
                dag.apply_operation_back(Delay(tau, dag.unit), [qubit])
            if gate is not None:
                dag.apply_operation_back(gate, [qubit])

        dag.global_phase = self._mod_2pi(dag.global_phase + sequence_gphase)

    @staticmethod
    def _mod_2pi(angle: float, atol: float = 0):
        """Wrap angle into interval [-π,π). If within atol of the endpoint, clamp to -π"""
        wrapped = (angle + np.pi) % (2 * np.pi) - np.pi
        if abs(wrapped - np.pi) < atol:
            wrapped = -np.pi
        return wrapped

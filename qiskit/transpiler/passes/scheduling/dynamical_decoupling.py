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

"""Dynamical Decoupling insertion pass."""

import itertools

import numpy as np
from qiskit.circuit.delay import Delay
from qiskit.circuit.reset import Reset
from qiskit.circuit.library.standard_gates import IGate, UGate
from qiskit.quantum_info import Operator
from qiskit.quantum_info.synthesis import OneQubitEulerDecomposer
from qiskit.transpiler.passes.optimization import Optimize1qGates
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError


class DynamicalDecoupling(TransformationPass):
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

    .. jupyter-execute::

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

        dd_sequence = [XGate(), XGate()]

        pm = PassManager([ALAPSchedule(durations),
                          DynamicalDecoupling(durations, dd_sequence)])
        circ_dd = pm.run(circ)
        timeline_drawer(circ_dd)
    """

    def __init__(self, durations, dd_sequence, qubits=None, spacing=None, skip_reset_qubits=True):
        """Dynamical decoupling initializer.

        Args:
            durations (InstructionDurations): Durations of instructions to be
                used in scheduling.
            dd_sequence (list[Gate]): sequence of gates to apply in idle spots.
            qubits (list[int]): physical qubits on which to apply DD.
                If None, all qubits will undergo DD (when possible).
            spacing (callable): a function that specifies spacing between DD
                gates. It maps natural numbers, i, to the fraction of the total
                slack that must be allocated to the i'th delay window. If None,
                equal spacing will be used.
            skip_reset_qubits (bool): if True, does not insert DD on idle
                periods that immediately follow initialized/reset qubits (as
                qubits in the ground state are less susceptile to decoherence).
        """
        super().__init__()
        self._durations = durations
        self._dd_sequence = dd_sequence
        self._qubits = qubits
        self._spacing = spacing
        self._skip_reset_qubits = skip_reset_qubits

    def run(self, dag):
        """Run the DynamicalDecoupling pass on dag.

        Args:
            dag (DAGCircuit): a scheduled DAG.

        Returns:
            DAGCircuit: equivalent circuit with delays interrupted by DD,
                where possible.

        Raises:
            TranspilerError: if the circuit is not mapped on physical qubits.
        """
        if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
            raise TranspilerError("DD runs on physical circuits only.")

        if dag.duration is None:
            raise TranspilerError("DD runs after circuit is scheduled.")

        # we support only even number of DD pulses (or just 1, as a special case)
        if len(self._dd_sequence) != 1:
            if len(self._dd_sequence) % 2 != 0:
                raise TranspilerError("DD sequence must contain an even number of gates (or 1).")
            noop = np.eye(2)
            for gate in self._dd_sequence:
                noop = noop.dot(gate.to_matrix())
            if not np.allclose(noop, np.eye(2), rtol=1e-5, atol=1e-8):
                raise TranspilerError("The DD sequence does not make an identity operation.")

        if self._qubits is None:
            self._qubits = range(dag.num_qubits())

        if self._spacing:
            raise TranspilerError("only balanced spacing is supported now.")

        new_dag = dag._copy_circuit_metadata()

        for nd in dag.topological_op_nodes():
            if not isinstance(nd.op, Delay):
                new_dag.apply_operation_back(nd.op, nd.qargs, nd.cargs)
                continue

            dag_qubit = nd.qargs[0]
            physical_qubit = new_dag.qubits.index(dag_qubit)
            if physical_qubit not in self._qubits:  # skip unwanted qubits
                new_dag.apply_operation_back(nd.op, nd.qargs, nd.cargs)
                continue

            pred = next(dag.predecessors(nd))
            succ = next(dag.successors(nd))
            if self._skip_reset_qubits:  # discount initial delays
                if pred.type == "in" or isinstance(pred, Reset):
                    new_dag.apply_operation_back(nd.op, nd.qargs, nd.cargs)
                    continue

            dd_sequence_duration = 0
            for gate in self._dd_sequence:
                gate.duration = self._durations.get(gate, physical_qubit)
                dd_sequence_duration += gate.duration

            slack = nd.op.duration - dd_sequence_duration
            if slack <= 0:  # dd doesn't fit
                new_dag.apply_operation_back(nd.op, nd.qargs, nd.cargs)
                continue

            if len(self._dd_sequence) == 1:  # special case of using a single gate for DD
                udg = self._dd_sequence[0].inverse().to_matrix()
                theta, phi, lam, phase = OneQubitEulerDecomposer().angles_and_phase(udg)
                # absorb the inverse into the successor (from left in circuit)
                if succ.type == "op" and isinstance(succ.op, UGate):
                    begin = int(slack / 2)
                    new_dag.apply_operation_back(Delay(begin), [dag_qubit])
                    new_dag.apply_operation_back(self._dd_sequence[0], [dag_qubit])
                    new_dag.apply_operation_back(Delay(slack - begin), [dag_qubit])
                    theta_r, phi_r, lam_r = succ.op.params
                    succ.op.params = Optimize1qGates.compose_u3(
                        theta_r, phi_r, lam_r, theta, phi, lam
                    )
                    new_dag.global_phase = _mod_2pi(new_dag.global_phase + phase)
                # absorb the inverse into the predecessor (from right in circuit)
                elif pred.type == "op" and isinstance(pred.op, UGate):
                    begin = int(slack / 2)
                    new_dag.apply_operation_back(Delay(begin), [dag_qubit])
                    new_dag.apply_operation_back(self._dd_sequence[0], [dag_qubit])
                    new_dag.apply_operation_back(Delay(slack - begin), [dag_qubit])
                    theta_l, phi_l, lam_l = pred.op.params
                    pred.op.params = Optimize1qGates.compose_u3(
                        theta, phi, lam, theta_l, phi_l, lam_l
                    )
                    new_dag.global_phase = _mod_2pi(new_dag.global_phase + phase)
                # don't do anything if there's no single-qubit gate to absorb the inverse
                else:
                    new_dag.apply_operation_back(nd.op, nd.qargs, nd.cargs)
            else:
                # balanced spacing (d/2, d, d, ..., d, d, d/2)
                # careful here that we add up to the original delay duration
                num_pulses = len(self._dd_sequence)
                mid = int(slack / num_pulses)
                end = int(mid / 2)
                unused_slack = slack - 2 * end - (num_pulses - 1) * mid
                delays = (
                    [end]
                    + [mid] * int((num_pulses - 1) / 2)
                    + [mid + unused_slack]
                    + [mid] * int((num_pulses - 1) / 2)
                    + [end]
                )
                for idle, gate in itertools.zip_longest(delays, self._dd_sequence):
                    new_dag.apply_operation_back(Delay(idle), [dag_qubit])
                    if gate is not None:
                        new_dag.apply_operation_back(gate, [dag_qubit])

        return new_dag


def _mod_2pi(angle: float, atol: float = 0):
    """Wrap angle into interval [-π,π). If within atol of the endpoint, clamp to -π"""
    wrapped = (angle + np.pi) % (2 * np.pi) - np.pi
    if abs(wrapped - np.pi) < atol:
        wrapped = -np.pi
    return wrapped

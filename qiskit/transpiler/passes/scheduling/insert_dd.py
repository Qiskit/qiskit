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

from collections import defaultdict
from typing import List
import numpy as np

from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import Delay
from qiskit.circuit.library import UGate
from qiskit.transpiler.passes import Optimize1qGates
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError


class InsertDD(TransformationPass):
    """Dynamical decoupling insertion pass.

    This pass works on a scheduled, physical circuit. It scans the circuit for
    idle periods of time (i.e. those containing delay instructions) and inserts
    a DD sequence of gates in those spots. These gates amount to the identity,
    so do not alter the logical action of the circuit, but have the effect of
    mitigating decoherence in those idle periods.

    The pass does not insert DD on idle periods that immediately follow qubit
    reset, as qubits in the ground state are less susceptile to decoherence.
    """

    def __init__(self, durations, dd_sequence, qubits=None, spacing=None):
        """Dynamical decoupling initializer.

        Args:
            durations (InstructionDurations): Durations of instructions to be
                used in scheduling.
            dd_sequence (list[str]): sequence of gates (by name) to apply
                in idle spots.
            qubits (list[int]): physical qubits on which to apply DD.
                If None, all qubits will undergo DD.
            spacing (callable): a function that specifies spacing between DD
                gates. It maps natural numbers, i, to the fraction of the total
                slack that must be allocated to the i'th delay window. If None,
                equal spacing will be used.
        """
        super().__init__()
        self._durations = durations
        self._dd_sequence = dd_sequence
        self._qubits = qubits
        self._spacing = spacing

        #def udd10_pos(j):
        #    return np.sin(np.pi*j/(2*40 + 2))**2

        #def equispaced_pos(j):
        #    return 1/num

    def run(self, dag):
        """Run the InsertDD pass on dag.

        Args:
            dag (DAGCircuit): a scheduled DAG.

        Returns:
            DAGCircuit: equivalent circuit with delays interrupted by DD,
                where possible.

        Raises:
            TranspilerError: if the circuit is not mapped on physical qubits.
        """
        if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
            raise TranspilerError('DD runs on physical circuits only.')

        if dag.duration is None:
            raise TranspilerError('DD runs after circuit is scheduled.')

        new_dag = dag._copy_circuit_metadata()

        for nd in dag.topological_op_nodes():
            if isinstance(nd.op, Delay):
                qubit = nd.qargs[0]
                pred = next(dag.predecessors(nd))
                succ = next(dag.successors(nd))
                if pred.type == 'in':  # discount initial delays
                    new_dag.apply_operation_back(nd.op, nd.qargs, nd.cargs)
                else:
                    dd_sequence = []
                    dd_sequence_duration = 0
                    for g in self._dd_sequence:
                        g.duration = self._durations.get(g, qubit)
                        dd_sequence.append(g)
                        dd_sequence_duration += g.duration
                    slack = nd.op.duration - dd_sequence_duration
                    if slack <= 0:     # dd doesn't fit
                        new_dag.apply_operation_back(nd.op, nd.qargs, nd.cargs)
                    else:
                        if len(self._dd_sequence) == 2:
                            begin = int(slack/4)
                            new_dag.apply_operation_back(Delay(begin), [qubit])
                            new_dag.apply_operation_back(dd_sequence[0], [qubit])
                            new_dag.apply_operation_back(Delay(slack-2*begin), [qubit])
                            new_dag.apply_operation_back(dd_sequence[1], [qubit])
                            new_dag.apply_operation_back(Delay(begin), [qubit])
                        elif len(self._dd_sequence) == 1: 
                            if isinstance(succ.op, UGate): # only do it if the rest can absorb
                                begin = int(slack/2)
                                new_dag.apply_operation_back(Delay(begin), [qubit])
                                new_dag.apply_operation_back(dd_sequence[0], [qubit])
                                new_dag.apply_operation_back(Delay(slack-begin), [qubit])
                                # absorb the gate into the successor (from left in circuit)
                                theta, phi, lam = succ.op.params
                                succ.op.params = Optimize1qGates.compose_u3(theta, phi, lam, np.pi, 0, np.pi)
                            else:
                                new_dag.apply_operation_back(nd.op, nd.qargs, nd.cargs)
                        else:
                            raise TranspilerError(
                                    f'Provided value for dd_sequence: {self._dd_sequence} is not valid')
            else:
                new_dag.apply_operation_back(nd.op, nd.qargs, nd.cargs)

        new_dag.duration = dag.duration
        new_dag.unit = dag.unit
        return new_dag

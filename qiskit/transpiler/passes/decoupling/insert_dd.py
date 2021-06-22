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

"""Dynamical Decoupling insertion pass."""
from collections import defaultdict
from typing import List
import numpy as np

from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import Delay
from qiskit.circuit.library import U3Gate, XGate
from qiskit.transpiler.passes import Optimize1qGates
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError


class InsertDD(TransformationPass):
    """DD insertion pass."""

    def __init__(self, durations, dd_sequence):
        """ASAPSchedule initializer.
        Args:
            durations (InstructionDurations): Durations of instructions to be used in scheduling
            dd_sequence (list[str]): sequence of gates to apply in idle spots
        """
        super().__init__()
        self._durations = durations
        self._dd_sequence = dd_sequence

    def run(self, dag):
        """Run the InsertDD pass on `dag`.
        Args:
            dag (DAGCircuit): a scheduled DAG.
        Returns:
            DAGCircuit: equivalent circuit with no extra delays but DD where possible.
        Raises:
            TranspilerError: if the circuit is not mapped on physical qubits.
        """
        if len(dag.qregs) != 1 or dag.qregs.get('q', None) is None:
            raise TranspilerError('DD runs on physical circuits only')

        new_dag = DAGCircuit()
        for qreg in dag.qregs.values():
            new_dag.add_qreg(qreg)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)

        for nd in dag.topological_op_nodes():
            if isinstance(nd.op, Delay):
                qubit = nd.qargs[0]
                pred = next(dag.predecessors(nd))
                succ = next(dag.successors(nd))
                if pred.type == 'in':  # discount initial delays
                    new_dag.apply_operation_back(nd.op, nd.qargs, nd.cargs)
                else:
                    dd_sequence = []
                    for g in self._dd_sequence:
                        x = XGate()
                        x.duration = self._durations.get(g, qubit)
                        dd_sequence.append(x)
                    dd_sequence_duration = sum([self._durations.get(g, qubit) for g in self._dd_sequence])
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
                            if isinstance(succ.op, U3Gate): # only do it if the rest can absorb
                                begin = int(slack/2)
                                new_dag.apply_operation_back(Delay(begin), [qubit])
                                new_dag.apply_operation_back(dd_sequence[0], [qubit])
                                new_dag.apply_operation_back(Delay(slack-begin), [qubit])
                                # absorb an X gate into the successor (from left in circuit)
                                theta, phi, lam = succ.op.params
                                succ.op.params = Optimize1qGates.compose_u3(theta, phi, lam, np.pi, 0, np.pi)
                            else:
                                new_dag.apply_operation_back(nd.op, nd.qargs, nd.cargs)
                        else:
                            raise TranspilerError('whats this sequence you tryna do?')
            else:
                new_dag.apply_operation_back(nd.op, nd.qargs, nd.cargs)

        new_dag.qubit_time_available = dag.qubit_time_available
        new_dag.duration = dag.duration
        return new_dag

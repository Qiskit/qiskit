# -*- coding: utf-8 -*-

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

from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes.scheduling.asap import ASAPSchedule


class ALAPSchedule(TransformationPass):
    """ALAP Scheduling."""

    def __init__(self, backend):
        """ALAPSchedule initializer.

        Args:
            backend (Backend): .
        """
        super().__init__()
        self.asap = ASAPSchedule(backend)

    def run(self, dag):
        """Run the ALAPSchedule pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to schedule.

        Returns:
            DAGCircuit: A scheduled DAG.

        Raises:
            TranspilerError: if ...
        """
        if len(dag.qregs) != 1 or dag.qregs.get('q', None) is None:
            raise TranspilerError('ALAP schedule runs on physical circuits only')

        new_dag = dag.reverse_ops()
        new_dag = self.asap.run(new_dag)
        new_dag = new_dag.reverse_ops()

        new_dag.name = dag.name
        return new_dag

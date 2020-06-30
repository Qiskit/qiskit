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

"""Pass to convert time unit of delays into dt."""
import warnings

from qiskit.circuit.delay import Delay
from qiskit.circuit import ParameterExpression
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError


class DelayInDt(TransformationPass):
    """Pass to convert time unit of delays into dt."""

    def __init__(self, dt):
        """DelayInDt initializer.

        Args:
            dt (float): .
        """
        super().__init__()
        self.dt = dt

    def run(self, dag):
        """Update durations in dt for all delays whose units are not dt (None).

        Args:
            dag (DAGCircuit): DAG to be converted.

        Returns:
            DAGCircuit: A converted DAG.

        Raises:
            TranspilerError: if ...
        """
        if len(dag.qregs) != 1 or dag.qregs.get('q', None) is None:
            raise TranspilerError('DelayInDt runs on physical circuits only')

        for node in dag.op_nodes(op=Delay):
            if isinstance(node.op.duration, ParameterExpression):
                try:
                    node.op.duration = float(node.op.duration)
                except TypeError:
                    raise TranspilerError("Durations of delays must be bounded before scheduling."
                                          "Use 'QuantumCircuit.bind_parameters' for that.")
                if node.op.unit == 'dt':
                    node.op.duration = int(node.op.duration)

            if node.op.unit != 'dt':  # convert unit of duration to dt
                if self.dt is None:
                    raise TranspilerError('If using unit in delay, backend must have dt.')
                if node.op.unit == 'ps':
                    scale = 1e-12
                elif node.op.unit == 'ns':
                    scale = 1e-9
                elif node.op.unit == 'us':
                    scale = 1e-6
                elif node.op.unit == 's':
                    scale = 1.0
                else:
                    raise TranspilerError('Invalid unit %s in delay instruction.' % node.op.unit)
                duration_in_sec = scale * node.op.duration
                duration_in_dt = round(duration_in_sec / self.dt)
                rounding_error = abs(duration_in_sec - duration_in_dt * self.dt)
                if rounding_error > 1e-15:
                    warnings.warn("Duration of delay is rounded to %d dt = %e s from %e"
                                  % (duration_in_dt, duration_in_dt * self.dt, duration_in_sec),
                                  UserWarning)
                node.op.duration = duration_in_dt
                node.op.unit = None

        return dag

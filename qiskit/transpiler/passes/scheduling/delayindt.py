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

from qiskit.circuit import ParameterExpression
from qiskit.circuit.delay import Delay
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.util import apply_prefix


class DelayInDt(TransformationPass):
    """Pass to convert time unit of delays into dt."""

    def __init__(self, dt_in_sec):
        """DelayInDt initializer.

        Args:
            dt_in_sec (float): Sampling time [sec] used for the conversion.
        """
        super().__init__()
        self.dt = dt_in_sec

    def run(self, dag):
        """Convert durations of delays in the circuit to be in dt.

        Args:
            dag (DAGCircuit): DAG to be converted.

        Returns:
            DAGCircuit: A converted DAG.

        Raises:
            TranspilerError: if failing to the unit conversion for some reason.
        """
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
                    raise TranspilerError("If using unit in delay, backend must have dt.")
                if node.op.unit == 's':
                    duration_in_sec = node.op.duration
                else:
                    try:
                        duration_in_sec = apply_prefix(node.op.duration, node.op.unit)
                    except Exception:
                        raise TranspilerError("Invalid unit %s in delay." % node.op.unit)
                duration_in_dt = round(duration_in_sec / self.dt)
                rounding_error = abs(duration_in_sec - duration_in_dt * self.dt)
                if rounding_error > 1e-15:
                    warnings.warn("Duration of delay is rounded to %d dt = %e s from %e"
                                  % (duration_in_dt, duration_in_dt * self.dt, duration_in_sec),
                                  UserWarning)
                node.op.duration = duration_in_dt
                node.op.unit = 'dt'

        return dag


def delay_in_dt(circuit, dt_in_sec):
    """Convert durations of delays in the circuit to be in dt.

    Args:
        circuit (QuantumCircuit): Circuit to be converted.
        dt_in_sec (float): Sample time [sec] used for the conversion.

    Returns:
        QuantumCircuit: A converted circuit.
    """
    from qiskit.converters.circuit_to_dag import circuit_to_dag
    from qiskit.converters.dag_to_circuit import dag_to_circuit
    dag = circuit_to_dag(circuit)
    dag = DelayInDt(dt_in_sec).run(dag)
    return dag_to_circuit(dag)

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

"""Utilities for scheduling passes."""
import warnings

from qiskit.circuit.barrier import Barrier
from qiskit.circuit.measure import Measure
from qiskit.dagcircuit import DAGNode
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.instruction_durations import InstructionDurations


class DurationMapper:
    def __init__(self, instruction_durations: InstructionDurations):
        self.instruction_durations = instruction_durations

    def get(self, node: DAGNode):
        duration = node.op.duration
        if duration is None:
            if isinstance(node.op, Barrier):
                duration = 0
            else:  # consult instruction_durations
                qubits = [q.index for q in node.qargs]
                duration = self.instruction_durations.get(node.op.name, qubits)

        return duration


# class DurationMapper:
#     def __init__(self, backend):
#         # TODO: backend.properties() should let us know all about instruction durations
#         if not backend.configuration().open_pulse:
#             raise TranspilerError("DurationMapper needs backend.configuration().dt")
#         self.backend_prop = backend.properties()
#         self.dt = backend.configuration().dt
#         # To know duration of measures, to be removed
#         self.all_qubits = tuple([i for i in range(backend.configuration().num_qubits)])
#         self.inst_map = backend.defaults().instruction_schedule_map
#
#     def get(self, node: DAGNode):
#         duration = node.op.duration
#         if duration is None:
#             if isinstance(node.op, Barrier):
#                 duration = 0
#             else:  # consult backend properties
#                 qubits = [q.index for q in node.qargs]
#                 if isinstance(node.op, Measure):
#                     duration = self.inst_map.get(node.op.name, self.all_qubits).duration
#                 else:
#                     duration = self.backend_prop.gate_length(node.op.name, qubits)
#
#         # convert seconds (float) to dts (int)
#         if isinstance(duration, float):
#             org = duration
#             duration = round(duration / self.dt)
#             rounding_error = abs(org - duration * self.dt)
#             if rounding_error > 1e-15:
#                 warnings.warn("Duration of %s is rounded to %d dt = %e s from %e"
#                               % (node.op.name, duration, duration * self.dt, org),
#                               UserWarning)
#
#         return duration

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

"""Durations of instructions, one of transpiler configurations."""
import warnings
from typing import Optional, List, Tuple

from qiskit.transpiler.exceptions import TranspilerError


class InstructionDurations:
    """Helper class to provide integer durations for safe scheduling."""

    def __init__(self,
                 instruction_durations: Optional[List[Tuple[str, Optional[List[int]], int]]] = None,
                 schedule_dt=None):
        self.duration_by_name = {}
        self.duration_by_name_qubits = {}
        self.schedule_dt = schedule_dt
        if instruction_durations:
            self.update(instruction_durations, schedule_dt)

    @classmethod
    def from_backend(cls, backend):
        """Construct the instruction durations from the backend."""
        if backend is None:
            return InstructionDurations()
        # TODO: backend.properties() should let us know all about instruction durations
        if not backend.configuration().open_pulse:
            raise TranspilerError("DurationMapper needs backend.configuration().dt")

        dt = backend.configuration().dt  # pylint: disable=invalid-name
        instruction_durations = []
        # backend.properties._gates -> instruction_durations
        for gate, insts in backend.properties()._gates.items():
            for qubits, props in insts.items():
                if 'gate_length' in props:
                    gate_length = props['gate_length'][0]  # Throw away datetime at index 1
                    duration = round(gate_length / dt)
                    rounding_error = abs(gate_length - duration * dt)
                    if rounding_error > 1e-15:
                        warnings.warn("Duration of %s is rounded to %d dt = %e s from %e"
                                      % (gate, duration, duration * dt, gate_length),
                                      UserWarning)
                    instruction_durations.append((gate, qubits, duration))
        # To know duration of measures, to be removed
        inst_map = backend.defaults().instruction_schedule_map
        all_qubits = tuple(range(backend.configuration().num_qubits))
        meas_duration = inst_map.get('measure', all_qubits).duration
        for q in all_qubits:
            instruction_durations.append(('measure', [q], meas_duration))
        return InstructionDurations(instruction_durations, dt)

    def update(self,
               instruction_durations: Optional[List[Tuple[str, Optional[List[int]], int]]] = None,
               dt=None):
        """Merge/extend self with instruction_durations."""
        if self.schedule_dt and dt and self.schedule_dt != dt:
            raise TranspilerError("dt must be the same to update")

        self.schedule_dt = dt or self.schedule_dt

        if instruction_durations:
            for name, qubits, duration in instruction_durations:
                if not isinstance(duration, int):
                    raise TranspilerError("duration value must be integer.")

                if isinstance(qubits, int):
                    qubits = [qubits]

                if qubits is None:
                    self.duration_by_name[name] = duration
                else:
                    self.duration_by_name_qubits[(name, tuple(qubits))] = duration

        return self

    def get(self, name, qubits):
        """Get the duration of the instruction with the name and the qubits."""
        if name in {'barrier', 'timestep'}:
            return 0

        if isinstance(qubits, int):
            qubits = [qubits]

        key = (name, tuple(qubits))
        if key in self.duration_by_name_qubits:
            return self.duration_by_name_qubits[key]

        if name in self.duration_by_name:
            return self.duration_by_name[name]

        raise TranspilerError("No value is found for key={}".format(key))

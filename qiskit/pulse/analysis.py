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

"""Analysis passes and functions for pulse programs."""
import collections

import numpy as np

from qiskit import pulse
from qiskit.pulse import instructions, commands
from qiskit.pulse.basepasses import AnalysisPass
from qiskit.pulse import InstructionScheduleMap


class MeasurementAlignmentAnalysis(AnalysisPass):
    def __init__(
        self,
        inst_map: InstructionScheduleMap,
        cal_gate: str = 'u3',
        max_calibration_duration: int = None,
    ):
        """Calculate the latest time to align all measures in a program."""
        super().__init__()
        self.inst_map = inst_map
        self.cal_gate = cal_gate
        self.max_calibration_duration = max_calibration_duration

    def run(
        self,
        program: pulse.Program,
    ) -> pulse.Program:
        """Set the the max between the duration of the calibration time and
        the absolute time of the latest scheduled acquire."""
        if self.max_calibration_duration is None:
            align_time = self.get_max_calibration_duration()
        else:
            align_time = self.max_calibration_duration
        for schedule in program.schedules:
            last_acquire = 0
            acquire_times = [time for time, inst in schedule.instructions
                             if isinstance(inst, (instructions.Acquire,
                                                  commands.AcquireInstruction))]
            if acquire_times:
                last_acquire = max(acquire_times)
            align_time = max(align_time, last_acquire)
        self.analysis.meas_align_time = align_time
        return program

    def get_max_calibration_duration(self):
        """Return the time needed to allow for readout discrimination calibration pulses."""
        max_calibration_duration = 0
        for qubits in self.inst_map.qubits_with_instruction(self.cal_gate):
            cmd = self.inst_map.get(self.cal_gate, qubits, np.pi, 0, np.pi)
            max_calibration_duration = max(cmd.duration, max_calibration_duration)
        return max_calibration_duration


class AmalgamatedAcquires(AnalysisPass):
    """Produce a dictionary of acquire instructions that ocurr at the same time.
    across channels."""
    def run(
        self,
        program: pulse.Program,
    ) -> pulse.Program:
        """Set a dictionary of acquire instructions across qubits at the same time."""
        acquire_instruction_maps = []
        acquire_instruction_map = collections.defaultdict(list)

        for schedule in program.schedules:
            acquire_instruction_map = collections.defaultdict(list)
            for time, instruction in schedule.instructions:
                if isinstance(instruction, (commands.AcquireInstruction, instructions.Acquire)):
                    # Acquires have a single AcquireChannel per inst, but we have to bundle them
                    # together into the Qobj as one instruction with many channels
                    acquire_instruction_map[(time, instruction.command)].append(instruction)
            acquire_instruction_maps.append(acquire_instruction_map)

        self.analysis.acquire_instruction_maps = acquire_instruction_maps
        return program


class MaxMemorySlotUsed(AnalysisPass):
    """Produce the maximum memory slot used in each program."""
    def run(
        self,
        program: pulse.Program,
    ) -> pulse.Program:
        """Set a dictionary of acquire instructions across qubits at the same time."""
        max_memory_slots = []
        for schedule in program.schedules:
            max_memory_slot = 0
            for time, instruction in schedule.instructions:
                if isinstance(instruction, (commands.AcquireInstruction, instructions.Acquire)):
                    if instruction.mem_slot:
                        max_memory_slot = max(max_memory_slot, instruction.mem_slot.index)

            max_memory_slots.append(max_memory_slot)
        self.analysis.max_memory_slot_used = max_memory_slots
        return program

# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Measurement command.
"""
# import logging
# from typing import List, Union, Set
#
# from qiskit.pulse.channels import (PulseChannel, MemorySlot, RegisterSlot)
# from qiskit.pulse.channels import Qubit
# from qiskit.pulse.common.interfaces import Pulse
# from qiskit.pulse.common.timeslots import Interval, Timeslot, TimeslotOccupancy
# from qiskit.pulse.exceptions import PulseError
# from .sample_pulse import SamplePulse
#
# logger = logging.getLogger(__name__)
#
#
# class MeasurementPulse(Pulse):
#     """Pulse to drive a measurement pulse shape and acquire the results to a `Qubit`. """
#
#     def __init__(self,
#                  shapes: Union[SamplePulse, List[SamplePulse]],
#                  qubits: Union[Qubit, List[Qubit]],
#                  mem_slots: Union[MemorySlot, List[MemorySlot]],
#                  discriminator: Discriminator = None,
#                  kernel: Kernel = None,
#                  reg_slots: Union[RegisterSlot, List[RegisterSlot]] = None):
#         if isinstance(shapes, SamplePulse):
#             shapes = [shapes]
#         if len(shapes) != len(qubits):
#             raise PulseError("#pulses must be equals to #qubits")
#         if reg_slots:
#             if len(qubits) != len(reg_slots):
#                 raise PulseError("#reg_slots must be equals to #qubits")
#         self._drive_pulses = []
#         for shape, q in zip(shapes, qubits):
#             self._drive_pulses.append(DrivePulse(shape, q.measure))
#         # TODO: check if all of the meas_pulse duration is the same
#         self._acquire_pulse = AcquirePulse(shapes[0].duration,
#                                            qubits,
#                                            discriminator,
#                                            kernel,
#                                            reg_slots)
#
#         durations = [shape.duration for shape in shapes]
#         # TODO: more precise time-slots
#         slots = [Timeslot(Interval(0, dur), q.measure) for dur, q in zip(durations, qubits)]
#         slots.extend([Timeslot(Interval(0, dur), q.acquire) for dur, q in zip(durations, qubits)])
#         slots.extend([Timeslot(Interval(0, dur), mem) for dur, mem in zip(durations, mem_slots)])
#         super().__init__(TimeslotOccupancy(slots))
#
#     def duration(self):
#         return self._acquire_pulse.duration
#
#     @property
#     def channelset(self) -> Set[PulseChannel]:
#         channels = []
#         for pulse in self._drive_pulses:
#             channels.extend(pulse.channelset)
#         channels.extend(self._acquire_pulse.channelset)
#         return {channels}

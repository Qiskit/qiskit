# -*- coding: utf-8 -*

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

# pylint: disable=missing-docstring,invalid-name,no-member
# pylint: disable=attribute-defined-outside-init

import numpy as np

from qiskit.pulse import Schedule, Gaussian, DriveChannel, Play, Waveform


def build_sample_pulse_schedule(number_of_unique_pulses, number_of_channels):
    rng = np.random.RandomState(42)
    sched = Schedule()
    for _ in range(number_of_unique_pulses):
        for channel in range(number_of_channels):
            sched.append(
                Play(Waveform(rng.random(50)), DriveChannel(channel)),
                inplace=True,
            )
    return sched


def build_parametric_pulse_schedule(number_of_unique_pulses,
                                    number_of_channels):
    sched = Schedule()
    for _ in range(number_of_unique_pulses):
        for channel in range(number_of_channels):
            sched.append(
                Play(
                    Gaussian(duration=25, sigma=4, amp=0.5j),
                    DriveChannel(channel),
                ),
                inplace=True,
            )
    return sched


class ScheduleConstructionBench:
    params = ([1, 2, 5], [8, 128, 2048])
    param_names = ['number_of_unique_pulses', 'number_of_channels']
    timeout = 600

    def setup(self, unique_pulses, channels):
        self.sample_sched = build_sample_pulse_schedule(unique_pulses,
                                                        channels)
        self.parametric_sched = build_parametric_pulse_schedule(unique_pulses,
                                                                channels)

    def time_sample_pulse_schedule_construction(self,
                                                unique_pulses,
                                                channels):
        build_sample_pulse_schedule(unique_pulses, channels)

    def time_parametric_pulse_schedule_construction(self,
                                                    unique_pulses,
                                                    channels):
        build_parametric_pulse_schedule(unique_pulses, channels)

    def time_append_instruction(self, _, __):
        self.sample_sched.append(self.parametric_sched, inplace=True)

    def time_insert_instruction_left_to_right(self, _, __):
        sched = self.sample_sched.shift(self.parametric_sched.stop_time)
        sched.insert(
            self.parametric_sched.start_time,
            self.parametric_sched,
            inplace=True,
        )

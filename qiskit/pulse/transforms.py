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

"""
Basic rescheduling functions which take schedules or instructions
(and possibly some arguments) and return new schedules.
"""
from collections import defaultdict

from . import Play
from .schedule import Schedule


def compress_pulses(schedule: Schedule,
                    by_channel: bool = False) -> Schedule:
    """Optimization pass to replace identical pulses.

    Args:
        schedule (Schedule): Schedule to compress.
        by_channel (bool): Compress pulses by channel.

    Returns:
        Compressed schedule.
    """

    new_schedule = Schedule(name=schedule.name)
    existing_pulses = defaultdict(list)

    for time, inst in schedule.instructions:
        if isinstance(inst, Play):
            channel = inst.channel.name if by_channel else '__all__'
            if inst.pulse in existing_pulses[channel]:
                idx = existing_pulses[channel].index(inst.pulse)
                identical_pulse = existing_pulses[channel][idx]
                new_schedule |= Play(identical_pulse, inst.channel, inst.name) << time
            else:
                existing_pulses[channel].append(inst.pulse)
                new_schedule |= inst << time
        else:
            new_schedule |= inst << time

    return new_schedule

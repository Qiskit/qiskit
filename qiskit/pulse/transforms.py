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
from typing import List

from qiskit.pulse import Play, Schedule


def compress_pulses(schedules: List[Schedule],
                    by_schedule: bool = False,
                    by_channel: bool = False) -> List[Schedule]:
    """Optimization pass to replace identical pulses.

    Args:
        schedules (list): Schedules to compress.
        by_schedule (bool): Compress pulses by schedule.
        by_channel (bool): Compress pulses by channel.

    Returns:
        Compressed schedules.
    """

    def _identifier(sched: str, chan: str) -> str:
        if not by_channel and not by_schedule:
            return '__all__'

        pairs = [(chan, by_channel), (sched, by_schedule)]
        return '_'.join([p[0] for p in pairs if p[1]])

    existing_pulses = defaultdict(list)
    new_schedules = []

    for schedule in schedules:
        new_schedule = Schedule(name=schedule.name)

        for time, inst in schedule.instructions:
            if isinstance(inst, Play):
                identifier = _identifier(schedule.name, inst.channel.name)
                if inst.pulse in existing_pulses[identifier]:
                    idx = existing_pulses[identifier].index(inst.pulse)
                    identical_pulse = existing_pulses[identifier][idx]
                    new_schedule |= Play(identical_pulse, inst.channel, inst.name) << time
                else:
                    existing_pulses[identifier].append(inst.pulse)
                    new_schedule |= inst << time
            else:
                new_schedule |= inst << time

        new_schedules.append(new_schedule)

    return new_schedules

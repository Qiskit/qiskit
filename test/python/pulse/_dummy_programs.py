# This code is part of Qiskit.
#
# (C) Copyright IBM 2024
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A pulse schedule library for unit tests."""

from qiskit.pulse import builder
from qiskit.pulse.library import symbolic_pulses
from qiskit.pulse import channels
from qiskit.pulse.schedule import ScheduleBlock


def play_gaussian() -> ScheduleBlock:
    """Create simple schedule of playing a single Gaussian pulse."""

    with builder.build() as out:
        builder.play(
            pulse=symbolic_pulses.Gaussian(
                duration=100,
                amp=0.1,
                sigma=25,
                angle=0.0,
            ),
            channel=channels.DriveChannel(0),
        )
    return out


def play_and_inner_block_sequential() -> ScheduleBlock:
    """Create sequential schedule with nested program."""

    with builder.build() as subroutine:
        builder.shift_phase(1.0, channel=channels.DriveChannel(0))
        builder.play(
            pulse=symbolic_pulses.Gaussian(
                duration=100,
                amp=0.1,
                sigma=25,
                angle=0.0,
            ),
            channel=channels.DriveChannel(0),
        )

    with builder.build() as out:
        with builder.align_sequential():
            builder.append_schedule(subroutine)
            builder.play(
                pulse=symbolic_pulses.Gaussian(
                    duration=40,
                    amp=0.1,
                    sigma=10,
                    angle=0.0,
                ),
                channel=channels.DriveChannel(1),
            )

    return out

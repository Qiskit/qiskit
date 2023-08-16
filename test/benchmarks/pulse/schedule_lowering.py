# This code is part of Qiskit.
#
# (C) Copyright IBM 2023
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

from qiskit.pulse import builder, library, channels
from qiskit.pulse.transforms import target_qobj_transform


def build_complicated_schedule():

    with builder.build() as schedule:
        with builder.align_sequential():
            with builder.align_right():
                with builder.phase_offset(np.pi, channels.ControlChannel(2)):
                    with builder.align_sequential():
                        for _ in range(5):
                            builder.play(
                                library.GaussianSquare(640, 0.1, 64, 384),
                                channels.ControlChannel(2),
                            )
                builder.play(
                    library.Constant(1920, 0.1),
                    channels.DriveChannel(1),
                )
                builder.barrier(
                    channels.DriveChannel(0),
                    channels.DriveChannel(1),
                    channels.DriveChannel(2),
                )
                builder.delay(800, channels.DriveChannel(1))
                with builder.align_left():
                    builder.play(
                        library.Drag(160, 0.3, 40, 1.5),
                        channels.DriveChannel(0),
                    )
                    builder.play(
                        library.Drag(320, 0.2, 80, 1.5),
                        channels.DriveChannel(1),
                    )
                    builder.play(
                        library.Drag(480, 0.1, 120, 1.5),
                        channels.DriveChannel(2),
                    )
            builder.reference("sub")
            with builder.align_left():
                for i in range(3):
                    builder.play(
                        library.GaussianSquare(1600, 0.1, 64, 1344),
                        channels.MeasureChannel(i),
                    )
                    builder.acquire(
                        1600,
                        channels.AcquireChannel(i),
                        channels.MemorySlot(i),
                    )

    with builder.build() as subroutine:
        for i in range(3):
            samples = np.random.random(160)
            builder.play(samples, channels.DriveChannel(i))
    schedule.assign_references({("sub",): subroutine}, inplace=True)

    return schedule


class ScheduleLoweringBench:
    def setup(self):
        self.schedule_block = build_complicated_schedule()

    def time_lowering(self):
        # Lower schedule block to generate job payload
        target_qobj_transform(self.schedule_block)

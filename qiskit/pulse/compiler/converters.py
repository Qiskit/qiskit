# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A base pass for Qiskit PulseIR compilation."""


from qiskit.pulse.schedule import ScheduleBlock

PulseIR = object


def schedule_to_ir(schedule: ScheduleBlock) -> PulseIR:
    """Convert ScheduleBlock into PulseIR.

    Args:
        schedule: Schedule to convert.

    Returns:
        PulseIR used internally in the pulse compiler.
    """
    raise NotImplementedError


def ir_to_schedule(pulse_ir: PulseIR) -> ScheduleBlock:
    """Convert PulseIR to ScheduleBlock.

    Args:
        pulse_ir: PulseIR to convert.

    Returns:
        ScheduleBlock that end-user may interact with.
    """
    raise NotImplementedError

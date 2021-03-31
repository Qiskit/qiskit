# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""A collection of set of transforms."""

# TODO: replace this with proper pulse transformation passes

from typing import Union, Iterable, Tuple

from qiskit.pulse import transforms
from qiskit.pulse.schedule import ScheduleBlock, Schedule
from qiskit.pulse.instructions import Instruction
from qiskit.pulse.exceptions import PulseError


InstructionSched = Union[Tuple[int, Instruction], Instruction]


def base_qobj_transform(sched: Union[ScheduleBlock,
                                     Schedule,
                                     InstructionSched,
                                     Iterable[InstructionSched]],
                        remove_directives: bool = True) -> Schedule:
    """A basic pulse program transformation for OpenPulse API execution.

    Args:
        sched: Input program to transform.
        remove_directives: Set `True` to remove compiler directives.

    Returns:
        Transformed program for execution.

    Raises:
        PulseError: When input program cannot be converted into ``Schedule``.
    """
    if not isinstance(sched, Schedule):
        # convert into schedule representation
        if isinstance(sched, ScheduleBlock):
            sched = transforms.block_to_schedule(sched)
        else:
            try:
                sched = Schedule(*_format_schedule_component(sched))
            except PulseError:
                raise PulseError('Input schedule is not valid data format.')

    # remove subroutines, i.e. Call instructions
    sched = transforms.inline_subroutines(sched)

    # inline nested schedules
    sched = transforms.flatten(sched)

    # remove directives, e.g. barriers
    if remove_directives:
        sched = transforms.remove_directives(sched)

    return sched


def _format_schedule_component(sched: Union[InstructionSched, Iterable[InstructionSched]]):
    """A helper function to convert instructions into list of instructions."""
    # TODO remove schedule initialization with *args, Qiskit-terra/#5093

    if hasattr(sched, '__iter__'):
        # (t0, inst), or list of it
        if isinstance(sched[0], int):
            # (t0, inst) tuple
            return [sched]
        else:
            return sched
    else:
        return [sched]

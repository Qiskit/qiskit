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
from qiskit.pulse.ir import IrBlock, IrInstruction


def schedule_to_ir(schedule: ScheduleBlock) -> IrBlock:
    """Convert ScheduleBlock into PulseIR.

    Args:
        schedule: Schedule to convert.

    Returns:
        PulseIR used internally in the pulse compiler.
    """
    out = IrBlock(alignment=schedule.alignment_context)

    def _wrap_recursive(_elm):
        if isinstance(_elm, ScheduleBlock):
            return schedule_to_ir(_elm)
        return IrInstruction(instruction=_elm)

    for element in schedule.blocks:
        wrapped_element = _wrap_recursive(element)
        out.add_element(wrapped_element)
    return out


def ir_to_schedule(pulse_ir: IrBlock) -> ScheduleBlock:
    """Convert PulseIR to ScheduleBlock.

    Args:
        pulse_ir: PulseIR to convert.

    Returns:
        ScheduleBlock that end-user may interact with.
    """
    out = ScheduleBlock(alignment_context=pulse_ir.alignment)

    def _unwrap_recursive(_elm):
        if isinstance(_elm, IrBlock):
            return ir_to_schedule(_elm)
        return _elm.instruction

    for element in pulse_ir.elements:
        unwrapped_element = _unwrap_recursive(element)
        out.append(unwrapped_element, inplace=True)

    return out

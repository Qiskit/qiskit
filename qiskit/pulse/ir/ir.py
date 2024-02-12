# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=cyclic-import

"""
=========
Pulse IR
=========

"""

from __future__ import annotations
from typing import List
from abc import ABC, abstractmethod

import numpy as np

from qiskit.pulse.exceptions import PulseError

from qiskit.pulse.transforms import AlignmentKind
from qiskit.pulse.instructions import Instruction


class IrElement(ABC):
    """Base class for Pulse IR elements"""

    @property
    @abstractmethod
    def initial_time(self) -> int | None:
        """Return the initial time of the element"""
        pass

    @property
    @abstractmethod
    def duration(self) -> int | None:
        """Return the duration of the element"""
        pass

    @abstractmethod
    def shift_initial_time(self, value: int):
        """Shift ``initial_time``

        Shifts ``initial_time`` to ``initial_time+value``.

        Args:
            value: The integer value by which ``initial_time`` is to be shifted.
        """
        pass

    @property
    def final_time(self) -> int | None:
        """Return the final time of the element"""
        try:
            return self.initial_time + self.duration
        except TypeError:
            return None


class IrInstruction(IrElement):
    """Pulse IR Instruction

    A Pulse IR instruction represents a ``ScheduleBlock`` instruction, with the addition of
    an ``initial_time`` property.
    """

    def __init__(self, instruction: Instruction, initial_time: int | None = None):
        """Pulse IR instructions

        Args:
            instruction: the Pulse `Instruction` represented by this IR instruction.
            initial_time (Optional): Starting time of the instruction. Defaults to ``None``
        """
        self._instruction = instruction
        if initial_time is None:
            self._initial_time = None
        else:
            self.initial_time = initial_time

    @property
    def instruction(self) -> Instruction:
        """Return the instruction associated with the IrInstruction"""
        return self._instruction

    @property
    def initial_time(self) -> int | None:
        """A clock time (in terms of system ``dt``) when this instruction is issued.

        .. note::
            initial_time value defaults to ``None`` and can only be set to non-negative integer.
        """
        return self._initial_time

    @property
    def duration(self) -> int:
        """The duration of the instruction (in terms of system ``dt``)."""
        return self._instruction.duration

    @initial_time.setter
    def initial_time(self, value: int):
        """Set ``initial_time``

        Args:
            value: The integer value of ``initial_time``.

        Raises:
            PulseError: if ``value`` is not ``None`` and not non-negative integer.
        """
        if not isinstance(value, (int, np.integer)) or value < 0:
            raise PulseError("initial_time must be a non-negative integer")
        self._initial_time = value

    def shift_initial_time(self, value: int) -> None:
        """Shift ``initial_time``

        Shifts ``initial_time`` to ``initial_time+value``.

        Args:
            value: The integer value by which ``initial_time`` is to be shifted.

        Raises:
            PulseError: If the instruction is not scheduled.
        """
        if self.initial_time is None:
            raise PulseError("Can not shift initial_time of an untimed element")

        # validation of new initial_time is done in initial_time setter.
        self.initial_time = self.initial_time + value

    def __eq__(self, other: "IrInstruction") -> bool:
        """Return True iff self and other are equal

        Args:
            other: The IR instruction to compare to this one.

        Returns:
            True iff equal.
        """
        return (
            type(self) is type(other)
            and self._instruction == other.instruction
            and self._initial_time == other.initial_time
        )

    def __repr__(self) -> str:
        """IrInstruction representation"""
        return f"IrInstruction({self.instruction}, t0={self.initial_time})"


class IrBlock(IrElement):
    """IR representation of instruction sequences

    ``IrBlock`` is the backbone of the intermediate representation used in the Qiskit Pulse compiler.
    A pulse program is represented as a single ``IrBlock`` object, with elements
    which include ``IrInstruction`` objects and other nested ``IrBlock`` objects.
    """

    def __init__(self, alignment: AlignmentKind):
        """Create ``IrBlock`` object

        Args:
            alignment: The ``AlignmentKind`` to be used for scheduling.
        """
        self._elements = []
        self._alignment = alignment

    @property
    def initial_time(self) -> int | None:
        """Return the initial time ``initial_time`` of the object"""
        elements_initial_times = [element.initial_time for element in self._elements]
        if None in elements_initial_times:
            return None
        else:
            return min(elements_initial_times, default=None)

    @property
    def final_time(self) -> int | None:
        """Return the final time of the ``IrBlock``object"""
        elements_final_times = [element.final_time for element in self._elements]
        if None in elements_final_times:
            return None
        else:
            return max(elements_final_times, default=None)

    @property
    def duration(self) -> int | None:
        """Return the duration of the ir block"""
        try:
            return self.final_time - self.initial_time
        except TypeError:
            return None

    @property
    def elements(self) -> List[IrElement]:
        """Return the elements of the ``IrBlock`` object"""
        return self._elements

    @property
    def alignment(self) -> AlignmentKind:
        """Return the alignment of the ``IrBlock`` object"""
        return self._alignment

    def has_child_ir(self) -> bool:
        """Check if IrBlock has child IrBlock object

        Returns:
            ``True`` if object has ``IrBlock`` object in its elements, and ``False`` otherwise.

        """
        for element in self._elements:
            if isinstance(element, IrBlock):
                return True
        return False

    def add_element(
        self,
        element: IrElement | List[IrElement],
    ):
        """Adds IR element or list thereof to the ``IrBlock`` object.

        Args:
            element: `IrElement` object, or list thereof to add.
        """
        if not isinstance(element, list):
            element = [element]

        self._elements.extend(element)

    def shift_initial_time(self, value: int):
        """Shifts ``initial_time`` of the ``IrBlock`` elements

        The ``initial_time`` of all elements in the ``IrBlock`` are shifted by ``value``,
        including recursively if any element is ``IrBlock``.

        Args:
            value: The integer value by which ``initial_time`` is to be shifted.

        Raises:
            PulseError: if the object is not scheduled (initial_time is None)
        """
        if self.initial_time is None:
            raise PulseError("Can not shift initial_time of IrBlock with unscheduled elements")

        for element in self.elements:
            element.shift_initial_time(value)

    def __eq__(self, other: "IrBlock") -> bool:
        """Return True iff self and other are equal
        Specifically, iff all of their properties are identical.

        Args:
            other: The IrBlock to compare to this one.

        Returns:
            True iff equal.
        """
        if (
            type(self) is not type(self)
            or self._alignment != other._alignment
            or len(self._elements) != len(other._elements)
        ):
            return False
        for element_self, element_other in zip(self._elements, other._elements):
            if element_other != element_self:
                return False

        return True

    def __len__(self) -> int:
        """Return the length of the IR, defined as the number of elements in it"""
        return len(self._elements)

    def __repr__(self) -> str:
        """IrBlock representation"""
        inst_count = len(
            [element for element in self.elements if isinstance(element, IrInstruction)]
        )
        block_count = len(self) - inst_count
        reprstr = f"IrBlock({self.alignment}"
        if inst_count > 0:
            reprstr += f", {inst_count} IrInstructions"
        if block_count > 0:
            reprstr += f", {block_count} IrBlocks"
        if self.initial_time is not None:
            reprstr += f", initial_time={self.initial_time}"
        if self.duration is not None:
            reprstr += f", duration={self.duration}"
        reprstr += ")"
        return reprstr

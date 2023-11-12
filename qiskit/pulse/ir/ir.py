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

from typing import Union, Optional, List
from abc import ABC, abstractmethod

import numpy as np

from qiskit.pulse.exceptions import PulseError

from qiskit.pulse.transforms import AlignmentKind, AlignSequential
from qiskit.pulse.instructions import Instruction


class IrElement(ABC):
    """Base class for Pulse IR elements"""

    @property
    @abstractmethod
    def initial_time(self) -> Union[int, None]:
        """Return the initial time of the element"""
        pass

    @property
    @abstractmethod
    def duration(self) -> Union[int, None]:
        """Return the duration of the element"""
        pass

    @abstractmethod
    def shift_initial_time(self, value: int):
        """Shift ``initial_time``

        Shifts ``initial_time`` to ``initial_time+value``.

        Args:
            value: The integer value by which ``initial_time`` is to be shifted.

        Raises:
            PulseError: if ``initial_time`` is ``None``.
        """
        if self.initial_time is None:
            raise PulseError("Can not shift initial_time of an untimed element")

    @property
    def final_time(self) -> Union[int, None]:
        """Return the final time of the element"""
        return self.initial_time + self.duration


class IrInstruction(IrElement):
    """Pulse IR Instruction

    A Pulse IR instruction represents a ``ScheduleBlock`` instruction, with the addition of
    an ``initial_time`` property.
    """

    def __init__(self, instruction: Instruction, initial_time: Optional[int] = None):
        """Pulse IR instructions

        Args:
            instruction: the Pulse `Instruction` represented by this IR instruction.
            initial_time (Optional): Starting time of the instruction. Defaults to ``None``

        Raises:
            PulseError: if ``initial_time`` is not ``None`` and not non-negative integer.
        """
        if initial_time is not None and (
            not isinstance(initial_time, (int, np.integer)) or initial_time < 0
        ):
            raise PulseError("initial_time must be a non-negative integer.")

        self._instruction = instruction
        self._initial_time = initial_time

    @property
    def instruction(self) -> Instruction:
        """Return the instruction associated with the IrInstruction"""
        return self._instruction

    @property
    def initial_time(self) -> Union[int, None]:
        return self._initial_time

    @property
    def duration(self) -> int:
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

    def shift_initial_time(self, value: int):
        """Shift ``initial_time``

        Shifts ``initial_time`` to ``initial_time+value``.

        Args:
            value: The integer value by which ``initial_time`` is to be shifted.
        """
        super().shift_initial_time(value)

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


class IrBlock(IrElement):
    """
    ``IrBlock`` is the backbone of the  intermediate representation used in the Qiskit Pulse compiler.
    A pulse program is represented as a single ``IrBlock`` object, with elements
    which include IR instructions and other
    nested ``IrBlock`` objects. This structure mimics that of :class:`.qiskit.pulse.ScheduleBlock`
    which is the main pulse program format used in Qiskit.
    """

    def __init__(self, alignment: Optional[AlignmentKind] = AlignSequential):
        """Create ``IrBlock`` object

        Args:
            alignment(Optional): The ``AlignmentKind`` to be used for scheduling. Defaults
                to ``AlignSequential``.
        """
        self._elements = []
        self._alignment = alignment

    @property
    def initial_time(self) -> Union[int, None]:
        """Return the initial time ``initial_time`` of the object"""
        elements_initial_times = [element.initial_time for element in self._elements]
        if None in elements_initial_times:
            return None
        else:
            return min(elements_initial_times, default=None)

    @property
    def final_time(self) -> Union[int, None]:
        """Return the final time of the ``IrBlock``object

        The final time is defined as ``initial_time+duration``"""
        elements_final_times = [element.final_time for element in self._elements]
        if None in elements_final_times:
            return None
        else:
            return max(elements_final_times, default=None)

    @property
    def duration(self) -> Union[int, None]:
        """Return the duration of the ir block"""
        return self.final_time - self.initial_time

    @property
    def elements(self) -> List[IrElement]:
        """Return the elements of the ``IrBlock`` object"""
        return self._elements

    @property
    def alignment(self) -> AlignmentKind:
        """Return the alignment of the ``IrBlock`` object"""
        return self._alignment

    def has_child_ir(self) -> bool:
        """Check if PulseIR has child IrBlock object

        Returns:
            ``True`` if object has ``IrBlock`` object in its elements, and ``False`` otherwise.

        """
        for element in self._elements:
            if isinstance(element, IrBlock):
                return True
        return False

    def add_element(
        self,
        element: Union[IrElement, List[IrElement]],
    ):
        """Adds IR element or list thereof to the ``IrBlock`` object.

        Args:
            element: `IrElement` object, or list thereof to add.
        """
        if not isinstance(element, list):
            element = [element]

        self._elements.extend(element)

    def shift_initial_time(self, value: int, start_ind: int = 0):
        """Shifts ``initial_time`` of the ``IrBlock`` elements

        According to the value of ``start_ind``, all or some of the ``initial_time`` of elements in
        the ``PulseIR`` are shifted, including recursively if any element is ``PulseIR``.

        Args:
            value: The integer value by which ``initial_time`` is to be shifted.
            start_ind: The index of the first element to be shifted. All elements with
                higher indices will be shifted. Default value - 0 (all elements are shifted).

        Raises:
            PulseError: if any element of the object is not scheduled (initial_time==None)
        """
        for element in self.elements[start_ind:]:
            if element.initial_time is None:
                raise PulseError("Can not shift initial_time of IrBlock with unscheduled elements")
            element.shift_initial_time(value)

    def flatten(self):
        """Flatten the ``IrBlock`` into a single block (in place)

        Recursively flattens all child ``IrBlock`` until only instructions remain."""
        new_elements = self._get_flatten_elements()
        self._elements = new_elements

    def _get_flatten_elements(self) -> List[IrElement]:
        """Return a list of flatten instructions

        Recursively lists all instructions in the ``IrBlock``.

        Raises:
            PulseError: If any instruction in the ``IrBlock`` is not scheduled
                (has ``initial_time==None``)
        """
        flatten_elements = []
        for element in self._elements:
            if isinstance(element, IrBlock):
                flatten_elements.extend(element._get_flatten_elements())
            else:
                if element.initial_time is None:
                    raise PulseError("Can not flatten an unscheduled PulseIR")
                flatten_elements.append(element)
        return flatten_elements

    def sort_by_initial_time(self):
        """Sort elements by ``initial_time`` in place (not recursively)

        Raises:
            PulseError: If any element is not scheduled (has ``initial_time==None``).
        """
        initial_times = [element.initial_time for element in self._elements]
        if None in initial_times:
            raise PulseError("Can not sort an unscheduled PulseIR")
        self._elements = [self._elements[i] for i in np.argsort(initial_times)]

    def remove_instruction(self, instruction: IrInstruction):
        """Remove instruction from ``IrBlock`` (in place)

        If the instruction does not exist in the object, no change is made and no error is raised.
        To avoid ambiguity, unscheduled instructions can not be removed.

        Args:
            instruction: The instruction to be removed.

        Raises:
            PulseError: If the instruction is not scheduled (``instruction.initial_time`` is ``None``).
        """
        if instruction.initial_time is None:
            raise PulseError(
                "Removing un-scheduled instructions could be ambiguous and is not allowed"
            )
        if instruction in self._elements:
            self._elements.remove(instruction)

    def __eq__(self, other: "IrBlock") -> bool:
        """Return True iff self and other are equal
        Specifically, iff all of their properties are identical.

        Args:
            other: The PulseIR to compare to this one.

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

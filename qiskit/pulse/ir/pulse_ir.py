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

from typing import Union, Optional, List, Set, Dict

import numpy as np

from qiskit.pulse.exceptions import PulseError

from qiskit.pulse import (
    LogicalElement,
    Frame,
    MixedFrame,
    Qubit,
)

from qiskit.pulse.transforms import AlignmentKind, AlignSequential

from .ir_instructions import (
    BaseIRInstruction,
    GenericInstruction,
    AcquireInstruction,
)


class PulseIR:
    """
    ``PulseIR`` is the backbone of the  intermediate representation used in the Qiskit Pulse compiler.
    A pulse program is represented as a single ``PulseIR`` object, with elements
    which include IR instructions (base class :class:`~.BaseIRInstruction) and other
    nested ``PulseIR`` objects. This structure mimics that of :class:`.qiskit.pulse.ScheduleBlock`
    which is the main pulse program format used in Qiskit.

    The IR is ment to support compilation needs, including partial instructions broadcasting,
    scheduling, validation and more.
    """

    def __init__(self, alignment: Optional[AlignmentKind] = AlignSequential):
        """Create ``PulseIR`` object

        Args:
            alignment(Optional): The ``AlignmentKind`` to be used for scheduling. Defaults
                to ``AlignSequential``.
        """
        self._elements = []
        self._alignment = alignment

    @property
    def initial_time(self) -> int:
        """Return the initial time ``initial_time`` of the object"""
        elements_initial_times = [element.initial_time for element in self._elements]
        if None in elements_initial_times:
            return None
        else:
            return min(elements_initial_times, default=None)

    @property
    def final_time(self) -> int:
        """Return the final time of the ``PulseIR``object

        The final time is defined as ``initial_time+duration``"""
        elements_final_times = [element.final_time for element in self._elements]
        if None in elements_final_times:
            return None
        else:
            return max(elements_final_times, default=None)

    @property
    def elements(self) -> List[Union[GenericInstruction, AcquireInstruction, "PulseIR"]]:
        """Return the elements of the ``PulseIR`` object"""
        return self._elements

    @property
    def alignment(self) -> AlignmentKind:
        """Return the alignment of the ``PulseIR`` object"""
        return self._alignment

    def has_child_ir(self) -> bool:
        """Check if PulseIR has child PulseIR object

        Returns:
            ``True`` if object has ``PulseIR`` object in its elements, and ``False`` otherwise.

        """
        for element in self._elements:
            if isinstance(element, PulseIR):
                return True
        return False

    def add_element(
        self,
        element: Union[
            GenericInstruction,
            AcquireInstruction,
            "PulseIR",
            List[Union[GenericInstruction, AcquireInstruction, "PulseIR"]],
        ],
    ):
        """Adds IR element (instruction or ``PulseIR``) or list thereof to the ``PulseIR`` object.

        Args:
            element: Instruction, ``PulseIR`` object, or list thereof to add.
        """
        if not isinstance(element, list):
            element = [element]

        self._elements.extend(element)

    def shift_initial_time(self, value: int, start_ind: int = 0):
        """Shifts ``initial_time`` of the ``PulseIR``'s elements

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
                raise PulseError("Can not shift initial_time of PulseIR with unscheduled elements")
            element.shift_initial_time(value)

    def frames(self) -> Set[Frame]:
        """Return the ``Frame``\\ s associated with the ``PulseIR`` object

        Recursively lists all :class:`~.Frame`\\ s associated with the ``PulseIR``,
        and returns them as a set.
        """
        frames = set()
        for element in self._elements:
            if isinstance(element, GenericInstruction) and element.frame is not None:
                frames.add(element.frame)
            elif isinstance(element, PulseIR):
                frames |= element.frames()
        return frames

    def logical_elements(self) -> Set[LogicalElement]:
        """Return the ``LogicalElement``\\ s associated with the ``PulseIR`` object

        Recursively lists all :class:`~.LogicalElement`\\ s associated with the ``PulseIR``,
        and returns them as a set.
        """
        logical_elements = set()
        for element in self._elements:
            if isinstance(element, GenericInstruction) and element.logical_element is not None:
                logical_elements.add(element.logical_element)
            if isinstance(element, AcquireInstruction):
                logical_elements.add(element.qubit)
            elif isinstance(element, PulseIR):
                logical_elements |= element.logical_elements()
        return logical_elements

    def mixed_frames(
        self,
        broadcasting_info: Optional[Dict[Union[LogicalElement, Frame], Set[MixedFrame]]] = None,
    ) -> Set[MixedFrame]:
        """Return the ``MixedFrame``s associated with the ``PulseIR`` object

        Recursively lists all :class:`~.MixedFrame`s associated with the ``PulseIR``,
        and returns them as a set. If partial instructions exist (:class:`~.GenericInstrucion`
        with only one of :class:`~.Frame` or :class:`~.LogicalElement` defined),
        they are either ignored or broadcasted to all relevant ``MixedFrames`` according
        to ``broadcasting_info``.

        Args:
            broadcasting_info: An optional dictionary mapping ``LogicalElement``s and ``Frame``s
                of partial instruction to a set of ``MixedFrames`` for which they are to be broadcasted.
                Default value - ``None`` (no broadcasting). if provided, the dictionary has to include
                all ``LogicalElement``s and ``Frame``s of partial instructions in the ``PulseIR``.
        """
        mixed_frames = set()
        for element in self._elements:
            if isinstance(element, GenericInstruction):
                if element.logical_element is not None and element.frame is not None:
                    mixed_frames.add(MixedFrame(element.logical_element, element.frame))
                elif broadcasting_info is not None:
                    mixed_frames |= broadcasting_info[element.logical_element or element.frame]
            elif isinstance(element, PulseIR):
                mixed_frames |= element.mixed_frames(broadcasting_info=broadcasting_info)
        return mixed_frames

    def _get_partial_instruction_info(self) -> Set[Union[LogicalElement, Frame]]:
        """Return the ``LogicalElement``s and ``Frame``s of all partial instructions

        partial instructions are :class:`~.GenericInstrucion` with only one of
        :class:`~.Frame` or :class:`~.LogicalElement` defined.
        """
        unbroadcasted = set()
        for element in self._elements:
            if isinstance(element, GenericInstruction):
                if element.logical_element is None or element.frame is None:
                    unbroadcasted.add(element.logical_element or element.frame)
            elif isinstance(element, PulseIR):
                unbroadcasted |= element._get_partial_instruction_info()
        return unbroadcasted

    def get_broadcasting_info(self) -> Dict[Union[LogicalElement, Frame], Set[MixedFrame]]:
        """Return the broadcasting info of the ``PulseIR`` object

        Partial instructions (:class:`~.GenericInstrucion` with only one of :class:`~.Frame`
         or :class:`~.LogicalElement` defined) need to be broadcasted to every :class:`~.MixedFrame`
         associated with them in the ``PulseIR``.

        Returns:
            A dictionary mapping every :class:`~.LogicalElement` and `~.Frame` of a partial
            instruction to a set of all associated ``MixedFrame``s.
        """
        mixed_frames = self.mixed_frames()
        partial_instructions_info = self._get_partial_instruction_info()
        broadcasting_info = {}

        for element_frame in partial_instructions_info:
            associated_mixed_frames = set()
            for mixed_frame in mixed_frames:
                if element_frame in (mixed_frame.logical_element, mixed_frame.frame):
                    associated_mixed_frames.add(mixed_frame)
            broadcasting_info[element_frame] = associated_mixed_frames

        return broadcasting_info

    def flatten(self):
        """Flatten the ``PulseIR`` into a single block (in place)

        Recursively flattens all child ``PulseIR`` until only instructions remain."""
        new_elements = self._get_flatten_elements()
        self._elements = new_elements

    def _get_flatten_elements(self) -> List[Union[BaseIRInstruction, "PulseIR"]]:
        """Return a list of flatten instructions

        Recursively lists all instructions in the ``PulseIR``.

        Raises:
            PulseError: If any instruction in the ``PulseIR`` is not scheduled
                (has ``initial_time==None``)
        """
        flatten_elements = []
        for element in self._elements:
            if isinstance(element, PulseIR):
                flatten_elements.extend(element._get_flatten_elements())
            else:
                if element.initial_time is None:
                    raise PulseError("Can not flatten an unscheduled PulseIR")
                flatten_elements.append(element)
        return flatten_elements

    def get_instructions_by_mixed_frame(
        self,
        mixed_frame: MixedFrame,
        recursive: Optional[bool] = True,
    ) -> List[GenericInstruction]:
        """Return all instructions associated with a given ``MixedFrame``

        Args:
            mixed_frame: The ``MixedFrame`` whose instructions are to be returned.
            recursive (Optional): If ``True`` recursively looks for instructions,
                else ignores instructions of child ``PulseIR``. Default - ``True``.
        """

        instructions = []
        for element in self._elements:
            if isinstance(element, GenericInstruction):
                if mixed_frame == MixedFrame(element.logical_element, element.frame):
                    instructions.append(element)
            elif recursive and isinstance(element, PulseIR):
                instructions.extend(
                    element.get_instructions_by_mixed_frame(mixed_frame, recursive=recursive)
                )
        return instructions

    def get_acquire_instructions(
        self, qubit: Optional[Qubit] = None, recursive: Optional[bool] = True
    ) -> List[GenericInstruction]:
        """Return ``AcquireInstruction``\\ s

        Args:
            qubit: Optionally return only instructions associated with this qubit.
                Default value - None (return all instructions).
            recursive: If ``True`` recursively looks for instructions,
                else ignores instructions of child ``PulseIR``. Default - ``True``.
        """
        instructions = []
        for element in self._elements:
            if isinstance(element, AcquireInstruction) and (
                qubit is None or element.qubit == qubit
            ):
                instructions.append(element)
            elif recursive and isinstance(element, PulseIR):
                instructions.extend(element.get_acquire_instructions(qubit, recursive=recursive))
        return instructions

    def sort_by_initial_time(self):
        """Sort elements by ``initial_time`` in place (not recursively)

        Raises:
            PulseError: If any element is not scheduled (has ``initial_time==None``).
        """
        initial_times = [element.initial_time for element in self._elements]
        if None in initial_times:
            raise PulseError("Can not sort an unscheduled PulseIR")
        self._elements = [self._elements[i] for i in np.argsort(initial_times)]

    def remove_instruction(self, instruction: BaseIRInstruction):
        """Remove instruction from ``PulseIR`` (in place)

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

    def remove_partial_instructions(self, recursive: bool = False):
        """Remove partial instructions (in place)

        Partial instructions are :class:`~.GenericInstrucion` with only one of
        :class:`~.Frame` or :class:`~.LogicalElement` defined. Typically, partial
        instructions need to be removed after broadcasting is completed.

        Args:
            recursive: If ``True`` recursively removes partial instructions,
                else ignores instructions of child ``PulseIR``. Default - ``False``.
        """
        self._elements = list(
            filter(
                lambda element: not (
                    isinstance(element, GenericInstruction)
                    and (element.logical_element is None or element.frame is None)
                ),
                self._elements,
            )
        )
        if recursive:
            for element in self._elements:
                if isinstance(element, PulseIR):
                    element.remove_partial_instructions(recursive=recursive)

    def __repr__(self):
        repr_str = f"PulseIR[alignment={self._alignment.__name__},["
        if len(self._elements) > 0:
            for element in self._elements:
                repr_str += str(element) + ", "
            repr_str = repr_str[:-2]
        repr_str += "]]"

        return repr_str

    def __eq__(self, other: "PulseIR") -> bool:
        """Return True iff self and other are equal
        Specifically, iff all of their properties are identical.

        Args:
            other: The PulseIR to compare to this one.

        Returns:
            True iff equal.
        """
        if self._alignment != other._alignment or len(self._elements) != len(other._elements):
            return False
        for element_self, element_other in zip(self._elements, other._elements):
            if element_other != element_self:
                return False

        return True

    def __len__(self) -> int:
        """Return the length of the IR, defined as the number of elements in it"""
        return len(self._elements)

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
IR Instructions
=========

"""

from typing import Optional
from abc import ABC, abstractmethod

import numpy as np

from qiskit.pulse.exceptions import PulseError

from qiskit.pulse import LogicalElement, Frame, SymbolicPulse, Waveform, Qubit, MemorySlot


class BaseIRInstruction(ABC):
    """Base class for PulseIR instruction

    ``BaseIRInstruction`` serves as the base class for all instructions in PulseIR.
    It sets common properties like ``duration``, ``initial_time``, and ``operand``.
    """

    @abstractmethod
    def __init__(self, duration: int, operand, initial_time: Optional[int] = None):
        """Base class for PulseIR instruction.

        Args:
            duration: Duration of the instruction (in terms of system ``dt``).
            operand: The operation of the instruction.
            initial_time (Optional): Starting time of the instruction. Defaults to ``None``

        Raises:
            PulseError: if ``duration`` is not non-negative integer.
            PulseError: if ``initial_time`` is not ``None`` and not non-negative integer.
        """

        if not isinstance(duration, (int, np.integer)) or duration < 0:
            raise PulseError("duration must be a non-negative integer.")
        if initial_time is not None and (
            not isinstance(initial_time, (int, np.integer)) or initial_time < 0
        ):
            raise PulseError("initial_time must be a non-negative integer.")

        self._duration = duration
        self._operand = operand
        self._initial_time = initial_time

    @property
    def operand(self):
        """Return the operand"""
        return self._operand

    @property
    def duration(self) -> int:
        """Return the duration"""
        return self._duration

    @property
    def initial_time(self) -> int:
        """Return the initial time ``initial_time``"""
        return self._initial_time

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

        Raises:
            PulseError: if ``initial_time`` is ``None``.
        """
        if self._initial_time is None:
            raise PulseError("Can not shift initial_time of an untimed instruction")

        # validation of new initial_time is done in initial_time setter.
        self.initial_time = self.initial_time + value

    @property
    def final_time(self) -> int:
        """Return the final time

        The final time is defined as ``initial_time+duration``.
        """
        return self._initial_time + self._duration


class GenericInstruction(BaseIRInstruction):
    """PulseIR Generic Instruction

    ``GenericInstruction`` encompasses the most common instruction types in PulseIR.
    The allowed instruction types are listed ``allowed_types`` and include:

    * Play
    * Delay
    * SetFrequency
    * ShiftFrequency
    * SetPhase
    * ShiftPhase

    Instructions are defined on both ``LogicalElement`` and ``Frame`` or in some cases only
    one of the two. If only one of ``LogicalElement`` and ``Frame`` are provided, the instruction
    is called partial instruction, and will require broadcasting during compilation.
    """

    allowed_types = ["Play", "Delay", "SetFrequency", "ShiftFrequency", "SetPhase", "ShiftPhase"]

    def __init__(
        self,
        instruction_type: str,
        operand,
        logical_element: Optional[LogicalElement] = None,
        frame: Optional[Frame] = None,
        initial_time: Optional[int] = None,
    ):
        """Create PulseIR Generic Instruction

        Args:
            instruction_type: The type of instruction.
            operand: The operand describing the operation of the instruction.
            logical_element (Optional): The logical element associated with the instruction.
                Defaults to ``None``
            frame (Optional): The frame associated with the instruction. Defaults to ``None``
            initial_time (Optional): Starting time of the instruction. Defaults to ``None``

        Raises:
            PulseError: if ``initial_time`` is not ``None`` and not non-negative integer.
        """
        duration = self._validate_instruction(instruction_type, operand, logical_element, frame)
        self._instruction_type = instruction_type
        self._logical_element = logical_element
        self._frame = frame
        super().__init__(duration, operand, initial_time)

    def _validate_instruction(
        self, instruction_type: str, operand, logical_element: LogicalElement, frame: Frame
    ) -> int:
        """Validate instruction parameters and set duration

        Args:
            instruction_type: string of instruction type.
            operand: The operand associated with the instruction.
            logical_element: The logical element associated with the instruction.
            frame: The frame associated with the instruction.

        Returns:
            The duration of the instruction.

        Raises:
            PulseError: If ``instruction_type`` not in allowed types.
            PulseError: If ``operand`` does not match the requirements of the given instruction type.
            PulseError: If an insufficient combination of ``logical_element`` and ``frame`` is provided.
        """
        if instruction_type not in self.__class__.allowed_types:
            raise PulseError(f"{instruction_type} is not a recognized instruction")

        if instruction_type == "Delay":
            if not isinstance(operand, (int, np.integer)) or operand < 0:
                raise PulseError(
                    "The operand of a Delay instruction must be a non-negative integer."
                )
            if logical_element is None:
                raise PulseError("Delay instruction must have an associated logical element.")
            duration = operand

        elif instruction_type == "Play":
            if not isinstance(operand, (SymbolicPulse, Waveform)):
                raise PulseError(
                    f"Play instruction is incompatible with operand of type {type(operand)}."
                )
            if logical_element is None or frame is None:
                raise PulseError(
                    "Play instruction must have an associated logical element and frame."
                )
            duration = operand.duration

        elif instruction_type in ["SetFrequency", "ShiftFrequency", "SetPhase", "ShiftPhase"]:
            if isinstance(operand, (int, np.integer)):
                operand = float(operand)
            if not isinstance(operand, float):
                raise PulseError(
                    "The operand of a Set/Shift Frequency/Phase instruction must be a float."
                )
            if frame is None:
                raise PulseError(
                    "Set/Shift Frequency/Phase instruction must have an associated frame."
                )
            duration = 0

        return duration

    @property
    def instruction_type(self) -> str:
        """Return instruction type"""
        return self._instruction_type

    @property
    def logical_element(self) -> LogicalElement:
        """Return the ``LogicalElement`` associated with the instruction"""
        return self._logical_element

    @property
    def frame(self) -> Frame:
        """Return the ``Frame`` associated with the instruction"""
        return self._frame

    def __repr__(self):
        repr_str = f"{self._instruction_type}(operand={self._operand},"
        if self._logical_element is not None:
            repr_str += f"logical_element={self._logical_element},"
        if self._frame is not None:
            repr_str += f"frame={self._frame},"
        repr_str += f"duration={self._duration}"
        if self._initial_time is not None:
            repr_str += f",initial_time={self._initial_time}"
        repr_str += ")"
        return repr_str

    def __eq__(self, other: "GenericInstruction") -> bool:
        """Return True iff self and other are equal

        Specifically, iff all of their properties are identical.

        Args:
            other: The logical element to compare to this one.

        Returns:
            True iff equal.
        """
        return (
            self.logical_element == other.logical_element
            and self.frame == other.frame
            and self.operand == other.operand
            and self.instruction_type == other.instruction_type
            and self.initial_time == other.initial_time
        )


class AcquireInstruction(BaseIRInstruction):
    """PulseIR Acquire Instruction

    Unlike :class:`~.GenericInstruction` which is associated with ``LogicalElement`` and ``Frame``,
    the acquire operation is associated with a ``Qubit`` (which is being acquired) and a ``MemorySlot``
    (where the measurement result is to be stored).
    """

    def __init__(
        self,
        qubit: Qubit,
        memory_slot: MemorySlot,
        duration: int,
        initial_time: Optional[int] = None,
    ):
        """Create ``AquireInstruction``

        Args:
            qubit: The qubit being acquired.
            memory_slot: The memory slot where the measurement result is to be stored.
            duration: Acquire duration.
            initial_time (Optional): Starting time of the instruction. Defaults to ``None``
        """
        self._qubit = qubit
        self._memory_slot = memory_slot
        operand = duration
        super().__init__(duration, operand, initial_time)

    @property
    def qubit(self) -> Qubit:
        """Return the ``Qubit`` associated with the instruction"""
        return self._qubit

    @property
    def memory_slot(self) -> MemorySlot:
        """Return the ``MemorySlot`` associated with the instruction"""
        return self._memory_slot

    def __repr__(self):
        return (
            "Acquire"
            "("
            f"qubit={self._qubit},"
            f"memory_slot={self._memory_slot},"
            f"duration={self._duration},"
            f"initial_time={self._initial_time}"
            ")"
        )

    def __eq__(self, other: "AcquireInstruction") -> bool:
        """Return True iff self and other are equal

        Specifically, iff all of their properties are identical.

        Args:
            other: The logical element to compare to this one.

        Returns:
            True iff equal.
        """
        return (
            self.qubit == other.qubit
            and self.memory_slot == other.memory_slot
            and self.duration == other.duration
            and self.initial_time == other.initial_time
        )

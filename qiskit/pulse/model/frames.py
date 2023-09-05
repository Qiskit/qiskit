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

"""
Frames
"""

from abc import ABCMeta
from typing import Optional

import numpy as np

from qiskit.pulse.exceptions import PulseError


class Frame(metaclass=ABCMeta):
    """Base class for pulse module frame.

    Because pulses used in Quantum HW are typically AC pulses, the carrier frequency and phase
    must be defined. The :class:`Frame` is the object which sets the frequency and phase for the carrier,
    and each pulse, and most instructions are associated with a frame. The different types of frames
    dictate how the frequency and phase duo are defined.

    Instructions on :class:`Frame`s like set/shift frequency/phase are broadcasted
    to every :class:`MixedFrame` which involves the same :class:`Frame`. The default
    initial phase for every frame is 0.
    """

    def __init__(self, identifier):
        """Create ``Frame``.

        Args:
            identifier: A unique identifier used to identify the frame.
        """
        self._identifier = identifier
        self._hash = hash((type(self), identifier))

    @property
    def identifier(self):
        """Return the identifier of the frame"""
        return self._identifier

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._identifier})"

    def __eq__(self, other: "Frame") -> bool:
        """Return True iff self and other are equal, specifically, iff they have the same type and name.

        Args:
            other: The frame to compare to this one.

        Returns:
            True iff equal.
        """
        return type(self) is type(other) and self._identifier == other._identifier

    def __hash__(self) -> int:
        return self._hash


class GenericFrame(Frame):
    """Pulse module GenericFrame.

    The :class:`GenericFrame` is used for custom user defined frames, which are not associated with any
    backend defaults. It is especially useful when the frame doesn't correspond to any frame of
    the typical qubit model, like qudit control for example.

    :class:`GenericFrame`s are identified by their unique name.
    """

    def __init__(self, identifier: str, frequency: float, phase: Optional[float] = 0.0):
        """Create ``GenericFrame``.

        Args:
            identifier: A unique identifier used to identify the frame.
            frequency: The initial frequency set for the frame.
            phase: The initial phase set for the frame. Default value 0.

        """
        self._frequency = frequency
        self._phase = phase
        super().__init__(identifier)

    @property
    def name(self) -> str:
        """Return the name of the frame."""
        return self._identifier

    @property
    def frequency(self) -> float:
        """Return the initial frequency of the generic frame."""
        return self._frequency

    @property
    def phase(self) -> float:
        """Return the initial phase of the generic frame."""
        return self._phase

    def __eq__(self, other: "GenericFrame") -> bool:
        """Return True iff self and other are equal, specifically, iff they have the same type
        name, frequency and phase.

        Args:
            other: The generic frame to compare to this one.

        Returns:
            True iff equal.
        """
        return (
            type(self) is type(other)
            and self._identifier == other._identifier
            and self._frequency == other._frequency
            and self._phase == other._phase
        )

    def __hash__(self) -> int:
        return self._hash


class QubitFrame(Frame):
    """A frame associated with the driving of a qubit.

    :class:`QubitFrame` is a frame associated with the driving of a specific qubit.
    The initial frequency of
    the frame will be taken as the default driving frequency provided by the backend
    during compilation.
    """

    def __init__(self, qubit_index: int):
        """Create ``QubitFrame``.

        Args:
            qubit_index: The index of the qubit represented by the frame.
        """
        self._validate_index(qubit_index)
        super().__init__(qubit_index)

    @property
    def qubit_index(self) -> int:
        """Return the qubit index of the qubit frame."""
        return self._identifier

    def _validate_index(self, index) -> None:
        """Raise a ``PulseError`` if the qubit index is invalid. Namely, check if the index is a
        non-negative integer.

        Raises:
            PulseError: If ``identifier`` (index) is a negative integer.
        """
        pass
        if not isinstance(index, (int, np.integer)) or index < 0:
            raise PulseError("Qubit index must be a non-negative integer")


class MeasurementFrame(Frame):
    """A frame associated with the dispersive readout of a qubit.

    ``MeasurementFrame`` is a frame associated with the dispersive readout of a specific qubit,
    which requires a stimulus tone driven at frequency off resonant to qubit drive.

    If not set otherwise, the initial frequency of the frame will be taken as the default
    measurement frequency provided by the backend during compilation.
    """

    def __init__(self, qubit_index: int):
        """Create ``MeasurementFrame``.

        Args:
            qubit_index: The index of the qubit represented by the frame.
        """
        self._validate_index(qubit_index)
        super().__init__(qubit_index)

    @property
    def qubit_index(self) -> int:
        """Return the qubit index of the measurement frame."""
        return self._identifier

    def _validate_index(self, index) -> None:
        """Raise a ``PulseError`` if the qubit index is invalid. Namely, check if the index is a
        non-negative integer.

        Raises:
            PulseError: If ``index`` is a negative integer.
        """
        pass
        if not isinstance(index, (int, np.integer)) or index < 0:
            raise PulseError("Qubit index must be a non-negative integer")

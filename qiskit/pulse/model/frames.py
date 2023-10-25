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

from abc import ABC

import numpy as np

from qiskit.pulse.exceptions import PulseError


class Frame(ABC):
    """Base class for pulse module frame.

    Because pulses used in Quantum hardware are typically AC pulses, the carrier frequency and phase
    must be defined. The :class:`Frame` is the object which identifies the frequency and phase for
    the carrier.
    and each pulse and most other instructions are associated with a frame. The different types of frames
    dictate how the frequency and phase duo are defined.

    The default initial phase for every frame is 0.
    """

    def __init__(self, identifier):
        """Create ``Frame``.

        Args:
            identifier: A unique identifier used to hash the Frame.
        """
        self._hash = hash((type(self), identifier))

    def __eq__(self, other: "Frame") -> bool:
        """Return True iff self and other are equal, specifically, iff they have the same type and hash.

        Args:
            other: The frame to compare to this one.

        Returns:
            True iff equal.
        """
        return type(self) is type(other) and self._hash == other._hash

    def __hash__(self) -> int:
        return self._hash


class GenericFrame(Frame):
    """Pulse module GenericFrame.

    The :class:`GenericFrame` is used for custom user defined frames, which are not associated with any
    backend defaults. It is especially useful when the frame doesn't correspond to any frame of
    the typical qubit model, like qudit control for example. Because no backend defaults exist for
    these frames, during compilation an initial frequency and phase will need to be provided.

    :class:`GenericFrame` objects are identified by their unique name.
    """

    def __init__(self, name: str):
        """Create ``GenericFrame``.

        Args:
            name: A unique identifier used to identify the frame.
        """
        self._name = name
        super().__init__(name)

    @property
    def name(self) -> str:
        """Return the name of the frame."""
        return self._name

    def __repr__(self) -> str:
        return f"GenericFrame({self._name})"


class QubitFrame(Frame):
    """A frame associated with the driving of a qubit.

    :class:`QubitFrame` is a frame associated with the driving of a specific qubit.
    The initial frequency of
    the frame will be taken as the default driving frequency provided by the backend
    during compilation.
    """

    def __init__(self, index: int):
        """Create ``QubitFrame``.

        Args:
            index: The index of the qubit represented by the frame.
        """
        self._validate_index(index)
        self._index = index
        super().__init__("QubitFrame" + str(index))

    @property
    def index(self) -> int:
        """Return the qubit index of the qubit frame."""
        return self._index

    def _validate_index(self, index) -> None:
        """Raise a ``PulseError`` if the qubit index is invalid. Namely, check if the index is a
        non-negative integer.

        Raises:
            PulseError: If ``identifier`` (index) is a negative integer.
        """
        if not isinstance(index, (int, np.integer)) or index < 0:
            raise PulseError("Qubit index must be a non-negative integer")

    def __repr__(self) -> str:
        return f"QubitFrame({self._index})"


class MeasurementFrame(Frame):
    """A frame associated with the measurement of a qubit.

    ``MeasurementFrame`` is a frame associated with the readout of a specific qubit,
    which requires a stimulus tone driven at frequency off resonant to qubit drive.

    If not set otherwise, the initial frequency of the frame will be taken as the default
    measurement frequency provided by the backend during compilation.
    """

    def __init__(self, index: int):
        """Create ``MeasurementFrame``.

        Args:
            index: The index of the qubit represented by the frame.
        """
        self._validate_index(index)
        self._index = index
        super().__init__("MeasurementFrame" + str(index))

    @property
    def index(self) -> int:
        """Return the qubit index of the measurement frame."""
        return self._index

    def _validate_index(self, index) -> None:
        """Raise a ``PulseError`` if the qubit index is invalid. Namely, check if the index is a
        non-negative integer.

        Raises:
            PulseError: If ``index`` is a negative integer.
        """
        if not isinstance(index, (int, np.integer)) or index < 0:
            raise PulseError("Qubit index must be a non-negative integer")

    def __repr__(self) -> str:
        return f"MeasurementFrame({self._index})"

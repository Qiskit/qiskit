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
.. _pulse-logical-elements-frames:

=======================================
Logical Elements & Frames (:mod:`qiskit.pulse.logical_elements_frames`)
=======================================

Pulse is meant to be agnostic to the underlying hardware implementation, while still allowing
low-level control. Qiskit Pulse's logical element and frames are meant to create a flexible framework
to define where pulses are applied, and what would be their carrier frequency and phase
(because typically AC pulses are used). Each :class:`LogicalElement` represents a separate component
in the quantum system on which instructions could be applied. On the other hand, each ``Frame`` represents
a frequency and phase duo for the carrier of the pulse.

This logical and virtual representation allows the user to write template pulse
programs without worrying about the exact details of the HW implementation (are the pulses to be played
via the same port? Which NCO is used?), while still allowing for effective utilization of the quantum
HW. The burden of mapping the different combinations of ``LogicalElement``s and ``Frame``s to HW aware
objects is left to the Pulse Compiler.

LogicalElement
=============
``LogicalElement``s are identified by their index and name. Currently, the most prominent example
is the ``Qubit``.

.. autosummary::
   :toctree: ../stubs/

   Qubit
   Coupler

Frame
=============
``Frame``s are identified by their name. A ``GenericFrame`` is used to specify custom frequency
and phase duos, while ``QubitFrame`` and ``MeasurementFrame`` are used to indicate that backend
defaults are to be used (for the qubit's driving frequency and measurement frequency respectively).

.. autosummary::
   :toctree: ../stubs/

   GenericFrame
   QubitFrame
   MeasurementFrame


MixedFrame
=============
The combination of a ``LogicalElement`` and ``Frame`` is dubbed a ``MixedFrame`` and is similar
to the legacy ``Channel`` objects. It should be noted that the legacy ``AcquireChannel``
has no counterpart in the new model, because they have a one-to-one correspondence with the
qubits, and so acquire instructions are simply associated with a ``Qubit``.

.. autosummary::
   :toctree: ../stubs/

   MixedFrame
   CRMixedFrame








"""
from abc import ABCMeta, abstractmethod
from typing import Optional

import numpy as np

from qiskit.pulse.exceptions import PulseError


class LogicalElement(metaclass=ABCMeta):
    """Base class of logical elements.

    ``LogicalElement``s are abstraction of the quantum HW component which can be controlled by the user
    ("apply pulses on").
    Every played pulse is associated with a ``LogicalElement`` on which it is being played.
    Logical elements identified by their index, and a unique name for each class such that the
    objects name is given by ``<class name><index>``.

    To implement a new logical element inherit from :class:`LogicalElement` the ``name`` method needs to
    be overridden with a proper name for the class.
    """

    def __init__(self, index):
        """Create ``LogicalElement``.

        Args:
            index: The index of the logical element.
        """
        self._validate_index(index)
        self._index = index
        self._hash = hash(self.name)

    @property
    def index(self):
        """Return the ``index`` of this logical element."""
        return self._index

    @abstractmethod
    def _validate_index(self, index) -> None:
        """Raise a PulseError if the logical element ``index`` is invalid.

        Raises:
            PulseError: If ``index`` is not valid.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this logical element."""
        pass

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other: "LogicalElement") -> bool:
        """Return True iff self and other are equal, specifically, iff they have the same type
        and the same ``identifier``.

        Args:
            other: The logical element to compare to this one.

        Returns:
            True iff equal.
        """
        return type(self) is type(other) and self._index == other._index

    def __hash__(self) -> int:
        return self._hash


class Qubit(LogicalElement):
    """Qubit logical element.

    ``Qubit`` represents the different qubits in the system, as identified by
    their (positive integer) index.
    """

    # pylint: disable=useless-parent-delegation
    def __init__(self, index: int):
        """Qubit logical element.

        Args:
            index: Qubit index.
        """
        super().__init__(index)

    def _validate_index(self, index) -> None:
        """Raise a ``PulseError`` if the qubit index is invalid. Namely, check if the index is a
        non-negative integer.

        Raises:
            PulseError: If ``index`` is a negative integer.
        """
        if not isinstance(index, (int, np.integer)) or index < 0:
            raise PulseError("Qubit index must be a non-negative integer")

    @property
    def name(self) -> str:
        """Return the name of this qubit"""
        return f"Q{self.index}"


class Coupler(LogicalElement):
    """Coupler logical element.

    ``Coupler`` represents an element which couples two qubits, and can be controlled on its own.
    It is identified by the tuple of indices of the coupled qubits.
    """

    def __init__(self, qubit_index_1: int, qubit_index_2: int):
        """Coupler logical element.

        The coupler ``index`` is defined as the ``tuple`` (``qubit_index_1``,``qubit_index_2``).

        Args:
            qubit_index_1: Index of the first qubit coupled by the coupler.
            qubit_index_2: Index of the second qubit coupled by the coupler.
        """
        super().__init__((qubit_index_1, qubit_index_2))

    def _validate_index(self, index) -> None:
        """Raise a ``PulseError`` if the coupler ``index`` is invalid. Namely,
        check if coupled qubit indices are non-negative integers.

        Raises:
            PulseError: If ``index`` is invalid.
        """
        for qubit_index in index:
            if not isinstance(qubit_index, (int, np.integer)) or qubit_index < 0:
                raise PulseError("Both indices of coupled qubits must be non-negative integers")

    @property
    def name(self) -> str:
        """Return the name of this coupler"""
        return f"Coupler{self.index}"


class Frame:
    """Base class for pulse module frame.

    Because pulses used in Quantum HW are typically AC pulses, the carrier frequency and phase
    must be defined. The ``Frame`` is the object which sets the frequency and phase for the carrier,
    and each pulse is associated with a frame.

    The different types of frames dictate how the frequency and phase duo are defined:

    - ``GenericFrame`` is used to custom frames, where the frequency is defined by the user.
    - ``QubitFrame`` is associated with the default driving frequency of a qubit.
    - ``MeasurementFrame`` is associated with the default measurement frequency of a qubit.

    Instructions on ``Frame``s like set/shift frequency/phase are broadcasted to every ``MixedFrame``
    which involves the same ``Frame``. The default initial phase for every frame is 0.
    """

    def __init__(self, name: str):
        """Create ``Frame``.

        Args:
            name: A unique identifier used to identify the frame.
        """
        self._name = name
        self._hash = hash(name)

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the frame."""
        pass

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other: "Frame") -> bool:
        """Return True iff self and other are equal, specifically, iff they have the same type and name.

        Args:
            other: The frame to compare to this one.

        Returns:
            True iff equal.
        """
        return type(self) is type(other) and self._name == other._name

    def __hash__(self) -> int:
        return self._hash


class GenericFrame(Frame):
    """Pulse module GenericFrame.

    The ``GenericFrame`` is used for custom user defined frames, which are not associated with any
    backend defaults. It is especially useful when the frame doesn't correspond to any frame of
    the typical qubit model, like qudit control for example.

    ``GenericFrame``s are identified by their unique name.

    """

    def __init__(self, name: str, frequency: float, phase: Optional[float] = 0.0):
        """Create ``Frame``.

        Args:
            name: A unique identifier used to identify the frame.
            frequency: The initial frequency set for the frame.
            phase: The initial phase set for the frame. Default value 0.

        """
        self._frequency = frequency
        self._phase = phase
        super().__init__(name)

    @property
    def name(self) -> str:
        """Return the name of the frame."""
        return f"GenericFrame({self._name})"

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
            and self._name == other._name
            and self._frequency == other._frequency
            and self._phase == other._phase
        )

    def __hash__(self) -> int:
        return self._hash


class QubitFrame(Frame):
    """A frame associated with the driving of a qubit.

    ``QubitFrame`` is a frame associated with the driving of a specific qubit. The initial frequency of
    the frame will be taken as the default driving frequency provided by the backend
    during compilation.
    """

    def __init__(self, qubit_index: int):
        """Create ``QubitFrame``.

        Args:
            qubit_index: The index of the qubit represented by the frame.
        """
        self._validate_index(qubit_index)
        self._index = qubit_index
        super().__init__(self.name)

    @property
    def name(self) -> str:
        """Return the name of the qubit frame."""
        return f"QubitFrame{self.qubit_index}"

    @property
    def qubit_index(self) -> int:
        """Return the qubit index of the qubit frame."""
        return self._index

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
    """A frame associated with the measurement of a qubit.

    ``MeasurementFrame`` is a frame associated with the measurement of a specific qubit.
    If not set otherwise, the initial frequency of the frame will be taken as the default
    measurement frequency provided by the backend during compilation.
    """

    def __init__(self, qubit_index: int):
        """Create ``MeasurementFrame``.

        Args:
            qubit_index: The index of the qubit represented by the frame.
        """
        self._validate_index(qubit_index)
        self._index = qubit_index
        super().__init__(self.name)

    @property
    def qubit_index(self) -> int:
        """Return the qubit index of the measurement frame."""
        return self._index

    @property
    def name(self) -> str:
        """Return the name of the frame."""
        return f"MeasurementFrame{self.qubit_index}"

    def _validate_index(self, index) -> None:
        """Raise a ``PulseError`` if the qubit index is invalid. Namely, check if the index is a
        non-negative integer.

        Raises:
            PulseError: If ``index`` is a negative integer.
        """
        pass
        if not isinstance(index, (int, np.integer)) or index < 0:
            raise PulseError("Qubit index must be a non-negative integer")


class MixedFrame:
    """Representation of a ``LogicalElement`` and ``Frame`` combination.

    Every pulse needs to be associated with both a ``LogicalElement`` and a ``Frame``. The combination
    of the two is called a mixed frame, and is represented by a ``MixedFrame`` object. The ``MixedFrame``
    is closely related to the legacy ``Channel``, but provides a more flexible framework.

    In most cases the ``MixedFrame`` is used more by the compiler, and a pulse program can be written
    without ``MixedFrame``s, by setting ``LogicalElement`` and ``Frame`` independently. However,
    in some cases using ``MixedFrame``s can better convey the meaning of the code, and change the
    compilation process. One example are shift/set frequency/phase instructions which are not
    broadcasted to other ``MixedFrame``s if applied on a specific ``MixedFrame`` (unlike the behavior
    of ``Frame``). A second example is the ``CRMixedFrame`` which indicates to the compiler what is the
    purpose of the pulses played on the mixed frame, and allows for extra validation - namely, validating
    that qubits are coupled.
    """

    def __init__(self, logical_element: LogicalElement, frame: Frame):
        """Create ``MixedFrame``.

        Args:
            logical_element: The logical element associated with the mixed frame.
            frame: The frame associated with the mixed frame.
        """
        self._logical_element = logical_element
        self._frame = frame
        self._hash = hash((self._logical_element, self._frame))

    @property
    def logical_element(self) -> LogicalElement:
        """Return the ``LogicalElement`` of this mixed frame."""
        return self._logical_element

    @property
    def frame(self) -> Frame:
        """Return the ``Frame`` of this mixed frame."""
        return self._frame

    @property
    def name(self) -> str:
        """Return the name of the mixed frame."""
        return f"MixedFrame({self.logical_element},{self.frame})"

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other: "MixedFrame") -> bool:
        """Return True iff self and other are equal, specifically, iff they have the same logical
        element and frame.

        Args:
            other: The mixed frame to compare to this one.

        Returns:
            True iff equal.
        """
        return self._logical_element == other._logical_element and self._frame == other._frame

    def __hash__(self) -> int:
        return self._hash


class CRMixedFrame(MixedFrame):
    """A mixed frame for Cross Resonance control.

    ``CRMixedFrame`` is identical to ``MixedFrame`` but is devoted to the common case of cross
    resonance control. In this case the ``LogicalElement`` and ``Frame`` associated with the
    ``MixedFrame`` are of types ``Qubit`` and ``QubitFrame`` respectively.

    ``CRMixedFrame`` and ``MixedFrame`` of the same elements will not only function in the same way,
    but will also be equal to one another. ``CRMixedFrame`` is used for improved readability
    and type hinting, as well as compilation validation - the compiler will verify that the qubits
    are in fact coupled (this need not be the case with a ``MixedFrame`` associated with ``Qubit``
    and ``QubitFrame``).
    """

    def __init__(self, qubit: Qubit, qubit_frame: QubitFrame):
        """Create ``CRMixedFrame``.

        Args:
            qubit: The ``Qubit`` object associated with the mixed frame.
            qubit_frame: The ``QubitFrame`` associated with the mixed frame.
        """
        super().__init__(logical_element=qubit, frame=qubit_frame)

    @property
    def qubit(self) -> Qubit:
        """Return the ``Qubit`` object of this mixed frame."""
        return self.logical_element

    @property
    def qubit_frame(self) -> QubitFrame:
        """Return the ``QubitFrame`` of this mixed frame."""
        return self.frame

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

"""Device component classes."""

from abc import ABC, abstractmethod


class DeviceComponent(ABC):
    """Class representing a device component."""

    @abstractmethod
    def __str__(self):
        pass

    def __repr__(self):
        return f"<{self.__class__.__name__}({str(self)})>"


class Qubit(DeviceComponent):
    """Class representing a qubit device component."""

    def __init__(self, index: int):
        self._index = index

    def __str__(self):
        return f"Q{self._index}"


class Resonator(DeviceComponent):
    """Class representing a resonator device component."""

    def __init__(self, index: int):
        self._index = index

    def __str__(self):
        return f"R{self._index}"


class UnknownComponent(DeviceComponent):
    """Class representing unknown device component."""

    def __init__(self, component: str):
        self._component = component

    def __str__(self):
        return self._component


def to_component(string: str) -> DeviceComponent:
    """Convert the input string to a ``DeviceComponent`` instance.

    Args:
        string: String to be converted.

    Returns:
        A ``DeviceComponent`` instance.

    Raises:
        ValueError: If input string is not a valid device component.
    """
    if string.startswith("Q"):
        return Qubit(int(string[1:]))
    elif string.startswith("R"):
        return Resonator(int(string[1:]))
    else:
        return UnknownComponent(string)

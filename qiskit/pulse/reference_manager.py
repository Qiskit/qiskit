# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Management of scehdule block reference."""

from collections.abc import MutableMapping, ItemsView, KeysView, ValuesView
from typing import TYPE_CHECKING, Sequence, Any

from qiskit.pulse.exceptions import PulseError

if TYPE_CHECKING:
    from .schedule import ScheduleBlock
    from .channels import Channel


# A variable to represent an empty (unassigned) subrotuine in the reference table.
_NOT_ASSIGNED = None


class ReferenceManager(MutableMapping):
    """Helper class to manage reference to subroutine.

    This dict-like object manages mapping to program and channels both
    keyed on the subroutine name. New entry should be created with
    :meth:`define_reference` method otherwise it raises an error when
    it is directly created through ``__setitem__`` as in a standard python dictionary.

    This class is intended to be manipulated by :class:`ScheduleBlock` instance,
    but user can still access this object through :attr:`ScheduleBlock.references`.

    This class implements pretty printing of reference information so that
    user can easily understand the structure of the program.

    Dictionary methods :meth:`item` and :meth:`values` are based on one for the program mapping
    and thus channel information is dropped. Note that channels information is
    available through the assigned schedule block, i.e. :attr:`ScheduleBlock.channels`.
    Channels stored in the reference are used to validate a program assigned with ``__setitem__``.

    This object is intended to be QPY-serializable.
    """

    def __init__(self, scope: str):
        """Create new reference manager."""
        self._scope = scope
        self._keys = []
        self._programs = []
        self._channels = []

    @property
    def scope(self) -> str:
        """Returns the target scope of this manager."""
        return self._scope

    def define_reference(self, key: str, channels: Sequence["Channel"]):
        """Create new reference entry to this mapping.

        Args:
            key: Key string of the reference.
            channels: List of pulse channels associated with the subroutine.
        """
        if key not in self._keys:
            self._keys.append(key)
            self._channels.append(channels)
            self._programs.append(_NOT_ASSIGNED)

    # pylint: disable=arguments-differ
    def update(self, other: "ReferenceManager"):
        if not isinstance(other, ReferenceManager):
            raise TypeError(
                f"Invalid object type for updating the ReferenceManager: '{other.__class__.__name__}'."
            )

        duplicated = set(self._keys) & set(other._keys)

        if duplicated:
            raise PulseError(
                f"Reference key {duplicated} in the reference '{other._scope}' conflicts with "
                f"the current reference '{self._scope}'. "
                f"Please call '{other._scope}' as a subroutine."
            )

        self._keys += other._keys
        self._channels += other._channels
        self._programs += other._programs

    def get(self, key: str, default: Any = None) -> "ScheduleBlock":
        try:
            return self._programs[self._keys.index(key)]
        except ValueError:
            return default

    def items(self) -> ItemsView:
        return ItemsView(dict(zip(self._keys, self._programs)))

    def keys(self) -> KeysView:
        return KeysView(dict(zip(self._keys, self._programs)))

    def values(self) -> ValuesView:
        return ValuesView(dict(zip(self._keys, self._programs)))

    def clear(self):
        self._keys.clear()
        self._programs.clear()
        self._channels.clear()

    def __setitem__(self, key: str, value: "ScheduleBlock"):
        if key not in self._keys:
            raise PulseError(f"Reference key {key} is not defined in the scope '{self._scope}'.")
        index = self._keys.index(key)

        # Check channel equality
        # Use channel-name based equality since channel index can be parametrized.
        # UUID based equality causes problem when channel indices are separately created.
        sub_channels = set(c.name for c in value.channels)
        ref_channels = set(c.name for c in self._channels[index])
        if set(sub_channels) != set(ref_channels):
            raise PulseError(
                f"Channels of assigned program {sub_channels} are not identical to the "
                f"channels associated with the reference instruction {ref_channels}."
            )
        if self._programs[index] is not None and self._programs[index] != value:
            raise PulseError(
                f"Subroutine '{key}' has been already assigned. "
                "Newly assigned schedule has conflict with the previous assignment."
            )
        self._programs[index] = value

    def __getitem__(self, key: str):
        if key not in self._keys:
            raise PulseError(f"Reference key '{key}' does not exist in the scope {self._scope}.")
        index = self._keys.index(key)

        return self._programs[index]

    def __delitem__(self, key: str):
        if key in self:
            index = self._keys.index(key)
            del self._keys[index]
            del self._programs[index]
            del self._channels[index]

    def __contains__(self, key: str):
        if key in self._keys:
            return True
        return False

    def __iter__(self):
        return self._keys.__iter__()

    def __len__(self):
        return len(self._keys)

    def __repr__(self):
        keys = ", ".join(self._keys)
        return f"{self.__class__.__name__}(keys=[{keys}])"

    def __str__(self):
        out = f"{self.__class__.__name__}"
        out += f"\n  - current scope: {repr(self._scope)}"
        out += f"\n  - # of references: {len(self)}"
        out += "\n  - mappings:"
        for index, key in enumerate(self.keys()):
            prog_repr = repr(self._programs[index])
            if len(prog_repr) > 50:
                prog_repr = prog_repr[:50] + "..."
            chan_repr = ", ".join(map(lambda c: c.name, self._channels[index]))
            out += f"\n    * {repr(key)}: {prog_repr}, channels=[{chan_repr}]"
        return out

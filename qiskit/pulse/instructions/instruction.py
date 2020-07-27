# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
``Instruction`` s are single operations within a :py:class:`~qiskit.pulse.Schedule`, and can be
used the same way as :py:class:`~qiskit.pulse.Schedule` s.

For example::

    duration = 10
    channel = DriveChannel(0)
    sched = Schedule()
    sched += Delay(duration, channel)  # Delay is a specific subclass of Instruction
"""
import warnings

from abc import ABC

from typing import Callable, Dict, Iterable, List, Optional, Tuple
import numpy as np

from ..channels import Channel
from ..exceptions import PulseError
from ..interfaces import ScheduleComponent
from ..schedule import Schedule

# pylint: disable=missing-return-doc


class Instruction(ScheduleComponent, ABC):
    """The smallest schedulable unit: a single instruction. It has a fixed duration and specified
    channels.
    """

    def __init__(self,
                 operands: Tuple,
                 duration: int,
                 channels: Tuple[Channel],
                 name: Optional[str] = None):
        """Instruction initializer.

        Args:
            operands: The argument list.
            duration: Length of time taken by the instruction in terms of dt.
            channels: Tuple of pulse channels that this instruction operates on.
            name: Optional display name for this instruction.

        Raises:
            PulseError: If duration is negative.
            PulseError: If the input ``channels`` are not all of
                type :class:`Channel`.
        """
        if not isinstance(duration, (int, np.integer)):
            raise PulseError("Instruction duration must be an integer, "
                             "got {} instead.".format(duration))
        if duration < 0:
            raise PulseError("{} duration of {} is invalid: must be nonnegative."
                             "".format(self.__class__.__name__, duration))
        self._duration = duration

        for channel in channels:
            if not isinstance(channel, Channel):
                raise PulseError("Expected a channel, got {} instead.".format(channel))

        self._channels = channels
        self._timeslots = {channel: [(0, self.duration)] for channel in channels}
        self._operands = operands
        self._name = name
        self._hash = None

    @property
    def name(self) -> str:
        """Name of this instruction."""
        return self._name

    @property
    def command(self) -> None:
        """The associated command. Commands are deprecated, so this method will be deprecated
        shortly.

        Returns:
            Command: The deprecated command if available.
        """
        warnings.warn("The `command` method is deprecated. Commands have been removed and this "
                      "method returns None.", DeprecationWarning)

    @property
    def id(self) -> int:  # pylint: disable=invalid-name
        """Unique identifier for this instruction."""
        return id(self)

    @property
    def operands(self) -> Tuple:
        """Return instruction operands."""
        return self._operands

    @property
    def channels(self) -> Tuple[Channel]:
        """Returns channels that this schedule uses."""
        return self._channels

    @property
    def timeslots(self) -> Dict[Channel, List[Tuple[int, int]]]:
        """Occupied time slots by this instruction."""
        warnings.warn("Access to Instruction timeslots is deprecated.")
        return self._timeslots

    @property
    def start_time(self) -> int:
        """Relative begin time of this instruction."""
        return 0

    @property
    def stop_time(self) -> int:
        """Relative end time of this instruction."""
        return self.duration

    @property
    def duration(self) -> int:
        """Duration of this instruction."""
        return self._duration

    @property
    def _children(self) -> Tuple[ScheduleComponent]:
        """Instruction has no child nodes."""
        return ()

    @property
    def instructions(self) -> Tuple[Tuple[int, 'Instruction']]:
        """Iterable for getting instructions from Schedule tree."""
        return tuple(self._instructions())

    def ch_duration(self, *channels: List[Channel]) -> int:
        """Return duration of the supplied channels in this Instruction.

        Args:
            *channels: Supplied channels
        """
        return self.ch_stop_time(*channels)

    def ch_start_time(self, *channels: List[Channel]) -> int:
        """Return minimum start time for supplied channels.

        Args:
            *channels: Supplied channels
        """
        return 0

    def ch_stop_time(self, *channels: List[Channel]) -> int:
        """Return maximum start time for supplied channels.

        Args:
            *channels: Supplied channels
        """
        if any(chan in self.channels for chan in channels):
            return self.duration
        return 0

    def _instructions(self, time: int = 0) -> Iterable[Tuple[int, 'Instruction']]:
        """Iterable for flattening Schedule tree.

        Args:
            time: Shifted time of this node due to parent

        Yields:
            Tuple[int, ScheduleComponent]: Tuple containing time `ScheduleComponent` starts
                at and the flattened `ScheduleComponent`
        """
        yield (time, self)

    def flatten(self) -> 'Instruction':
        """Return itself as already single instruction."""
        return self

    def shift(self: ScheduleComponent, time: int, name: Optional[str] = None) -> Schedule:
        """Return a new schedule shifted forward by `time`.

        Args:
            time: Time to shift by
            name: Name of the new schedule. Defaults to name of self
        """
        if name is None:
            name = self.name
        return Schedule((time, self), name=name)

    def insert(self, start_time: int, schedule: ScheduleComponent,
               name: Optional[str] = None) -> Schedule:
        """Return a new :class:`~qiskit.pulse.Schedule` with ``schedule`` inserted within
        ``self`` at ``start_time``.

        Args:
            start_time: Time to insert the schedule schedule
            schedule: Schedule to insert
            name: Name of the new schedule. Defaults to name of self
        """
        if name is None:
            name = self.name
        return Schedule(self, (start_time, schedule), name=name)

    def append(self, schedule: ScheduleComponent,
               name: Optional[str] = None) -> Schedule:
        """Return a new :class:`~qiskit.pulse.Schedule` with ``schedule`` inserted at the
        maximum time over all channels shared between ``self`` and ``schedule``.

        Args:
            schedule: schedule to be appended
            name: Name of the new schedule. Defaults to name of self
        """
        common_channels = set(self.channels) & set(schedule.channels)
        time = self.ch_stop_time(*common_channels)
        return self.insert(time, schedule, name=name)

    def draw(self, dt: float = 1, style=None,
             filename: Optional[str] = None, interp_method: Optional[Callable] = None,
             scale: float = 1, plot_all: bool = False,
             plot_range: Optional[Tuple[float]] = None,
             interactive: bool = False, table: bool = True,
             label: bool = False, framechange: bool = True,
             channels: Optional[List[Channel]] = None):
        """Plot the instruction.

        Args:
            dt: Time interval of samples
            style (Optional[SchedStyle]): A style sheet to configure plot appearance
            filename: Name required to save pulse image
            interp_method: A function for interpolation
            scale: Relative visual scaling of waveform amplitudes
            plot_all: Plot empty channels
            plot_range: A tuple of time range to plot
            interactive: When set true show the circuit in a new window
                (this depends on the matplotlib backend being used supporting this)
            table: Draw event table for supported instructions
            label: Label individual instructions
            framechange: Add framechange indicators
            channels: A list of channel names to plot

        Returns:
            matplotlib.figure: A matplotlib figure object of the pulse schedule
        """
        # pylint: disable=invalid-name, cyclic-import
        from qiskit import visualization

        return visualization.pulse_drawer(self, dt=dt, style=style,
                                          filename=filename, interp_method=interp_method,
                                          scale=scale,
                                          plot_all=plot_all, plot_range=plot_range,
                                          interactive=interactive, table=table,
                                          label=label, framechange=framechange,
                                          channels=channels)

    def __eq__(self, other: 'Instruction') -> bool:
        """Check if this Instruction is equal to the `other` instruction.

        Equality is determined by the instruction sharing the same operands and channels.
        """
        return isinstance(other, type(self)) and self.operands == other.operands

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash((type(self), self.operands, self.name))
        return self._hash

    def __add__(self, other: ScheduleComponent) -> Schedule:
        """Return a new schedule with `other` inserted within `self` at `start_time`."""
        return self.append(other)

    def __or__(self, other: ScheduleComponent) -> Schedule:
        """Return a new schedule which is the union of `self` and `other`."""
        return self.insert(0, other)

    def __lshift__(self, time: int) -> Schedule:
        """Return a new schedule which is shifted forward by `time`."""
        return self.shift(time)

    def __repr__(self) -> str:
        operands = ', '.join(str(op) for op in self.operands)
        return "{}({}{})".format(self.__class__.__name__, operands,
                                 ", name='{}'".format(self.name) if self.name else "")

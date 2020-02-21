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

from typing import Tuple, List, Iterable, Callable, Optional, Union

from qiskit.pulse.channels import Channel
from qiskit.pulse.interfaces import ScheduleComponent
from qiskit.pulse.schedule import Schedule
from qiskit.pulse.timeslots import Interval, Timeslot, TimeslotCollection

# pylint: disable=missing-return-doc

# TODO: After migrating instruction implementations, add property+abstractmethod operands


class Instruction(ScheduleComponent, ABC):
    """The smallest schedulable unit: a single instruction. It has a fixed duration and specified
    channels.
    """

    def __init__(self, duration: Union['Command', int],
                 *channels: Channel,
                 name: Optional[str] = None):
        """Instruction initializer.

        Args:
            duration: Length of time taken by the instruction in terms of dt.
            *channels: List of pulse channels that this instruction operates on.
            name: Display name for this instruction.
        """
        self._name = name

        self._command = None
        if not isinstance(duration, int):
            # TODO: Add deprecation warning once all instructions are migrated
            self._command = duration
            if name is None:
                self._name = duration.name
            duration = duration.duration

        self._timeslots = TimeslotCollection(*(Timeslot(Interval(0, duration), channel)
                                               for channel in channels if channel is not None))

    @property
    def name(self) -> str:
        """Name of this instruction."""
        return self._name

    @property
    def command(self) -> 'Command':
        """The associated command."""
        return self._command

    @property
    def channels(self) -> Tuple[Channel]:
        """Returns channels that this schedule uses."""
        return self.timeslots.channels

    @property
    def timeslots(self) -> TimeslotCollection:
        """Occupied time slots by this instruction."""
        return self._timeslots

    @property
    def start_time(self) -> int:
        """Relative begin time of this instruction."""
        return self.timeslots.start_time

    @property
    def stop_time(self) -> int:
        """Relative end time of this instruction."""
        return self.timeslots.stop_time

    @property
    def duration(self) -> int:
        """Duration of this instruction."""
        return self.timeslots.duration

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
        return self.timeslots.ch_duration(*channels)

    def ch_start_time(self, *channels: List[Channel]) -> int:
        """Return minimum start time for supplied channels.

        Args:
            *channels: Supplied channels
        """
        return self.timeslots.ch_start_time(*channels)

    def ch_stop_time(self, *channels: List[Channel]) -> int:
        """Return maximum start time for supplied channels.

        Args:
            *channels: Supplied channels
        """
        return self.timeslots.ch_stop_time(*channels)

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

    def union(self, *schedules: List[ScheduleComponent], name: Optional[str] = None) -> 'Schedule':
        """Return a new schedule which is the union of `self` and `schedule`.

        Args:
            *schedules: Schedules to be take the union with this Instruction.
            name: Name of the new schedule. Defaults to name of self
        """
        if name is None:
            name = self.name
        return Schedule(self, *schedules, name=name)

    def shift(self: ScheduleComponent, time: int, name: Optional[str] = None) -> 'Schedule':
        """Return a new schedule shifted forward by `time`.

        Args:
            time: Time to shift by
            name: Name of the new schedule. Defaults to name of self
        """
        if name is None:
            name = self.name
        return Schedule((time, self), name=name)

    def insert(self, start_time: int, schedule: ScheduleComponent,
               name: Optional[str] = None) -> 'Schedule':
        """Return a new :class:`~qiskit.pulse.Schedule` with ``schedule`` inserted within
        ``self`` at ``start_time``.

        Args:
            start_time: Time to insert the schedule schedule
            schedule: Schedule to insert
            name: Name of the new schedule. Defaults to name of self
        """
        return self.union((start_time, schedule), name=name)

    def append(self, schedule: ScheduleComponent,
               name: Optional[str] = None) -> 'Schedule':
        """Return a new :class:`~qiskit.pulse.Schedule` with ``schedule`` inserted at the
        maximum time over all channels shared between ``self`` and ``schedule``.

        Args:
            schedule: schedule to be appended
            name: Name of the new schedule. Defaults to name of self
        """
        common_channels = set(self.channels) & set(schedule.channels)
        time = self.ch_stop_time(*common_channels)
        return self.insert(time, schedule, name=name)

    def draw(self, dt: float = 1, style: Optional['SchedStyle'] = None,
             filename: Optional[str] = None, interp_method: Optional[Callable] = None,
             scale: float = 1, channels_to_plot: Optional[List[Channel]] = None,
             plot_all: bool = False, plot_range: Optional[Tuple[float]] = None,
             interactive: bool = False, table: bool = True,
             label: bool = False, framechange: bool = True,
             scaling: float = None,
             channels: Optional[List[Channel]] = None):
        """Plot the instruction.

        Args:
            dt: Time interval of samples
            style: A style sheet to configure plot appearance
            filename: Name required to save pulse image
            interp_method: A function for interpolation
            scale: Relative visual scaling of waveform amplitudes
            channels_to_plot: Deprecated, see `channels`
            plot_all: Plot empty channels
            plot_range: A tuple of time range to plot
            interactive: When set true show the circuit in a new window
                (this depends on the matplotlib backend being used supporting this)
            table: Draw event table for supported instructions
            label: Label individual instructions
            framechange: Add framechange indicators
            scaling: Deprecated, see `scale`
            channels: A list of channel names to plot

        Returns:
            matplotlib.figure: A matplotlib figure object of the pulse schedule
        """
        # pylint: disable=invalid-name, cyclic-import
        if scaling is not None:
            warnings.warn('The parameter "scaling" is being replaced by "scale"',
                          DeprecationWarning, 3)
            scale = scaling

        from qiskit import visualization

        if channels_to_plot:
            warnings.warn('The parameter "channels_to_plot" is being replaced by "channels"',
                          DeprecationWarning, 3)
            channels = channels_to_plot

        return visualization.pulse_drawer(self, dt=dt, style=style,
                                          filename=filename, interp_method=interp_method,
                                          scale=scale,
                                          plot_all=plot_all, plot_range=plot_range,
                                          interactive=interactive, table=table,
                                          label=label, framechange=framechange,
                                          channels=channels)

    def __eq__(self, other: 'Instruction'):
        """Check if this Instruction is equal to the `other` instruction.

        Equality is determined by the instruction sharing the same operands and channels.
        """
        if self.command:
            # Backwards compatibility for Instructions with Commands
            return (self.command == other.command) and (set(self.channels) == set(other.channels))
        return ((self.duration == other.duration) and
                (set(self.channels) == set(other.channels)) and
                (isinstance(other, type(self))))

    def __hash__(self):
        if self.command:
            # Backwards compatibility for Instructions with Commands
            return hash((hash(tuple(self.command)), self.channels.__hash__()))
        return hash((hash(tuple(self.duration)), self.channels.__hash__()))

    def __add__(self, other: ScheduleComponent) -> 'Schedule':
        """Return a new schedule with `other` inserted within `self` at `start_time`."""
        return self.append(other)

    def __or__(self, other: ScheduleComponent) -> 'Schedule':
        """Return a new schedule which is the union of `self` and `other`."""
        return self.union(other)

    def __lshift__(self, time: int) -> 'Schedule':
        """Return a new schedule which is shifted forward by `time`."""
        return self.shift(time)

    def __repr__(self):
        return "%s(%s, %s)" % (self.__class__.__name__,
                               self.duration,
                               ', '.join(str(ch) for ch in self.channels))

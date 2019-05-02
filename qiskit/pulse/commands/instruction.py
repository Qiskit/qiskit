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
Instruction = Leaf node of schedule.
"""
import logging
from typing import Tuple, List, Iterable, Callable

from qiskit.pulse import ops
from qiskit.pulse.channels import Channel
from qiskit.pulse.interfaces import ScheduleComponent
from qiskit.pulse.timeslots import Interval, Timeslot, TimeslotCollection
from qiskit.pulse.exceptions import PulseError

logger = logging.getLogger(__name__)

# pylint: disable=missing-return-doc,missing-type-doc


class Instruction(ScheduleComponent):
    """An abstract class for leaf nodes of schedule."""

    def __init__(self, command, *channels: List[Channel],
                 timeslots: TimeslotCollection = None, name=None):
        """
        command (Command): Pulse command to schedule
        *channels: List of pulse channels to schedule with command
        timeslots: Optional list of timeslots. If channels are supplied timeslots
            cannot also be given
        name: Name of Instruction
        """
        self._command = command
        self._name = name if name else self._command.name

        if timeslots and channels:
            raise PulseError('Channels and timeslots may not both be supplied.')

        if not timeslots:
            duration = command.duration
            self._timeslots = TimeslotCollection(*(Timeslot(Interval(0, duration), channel)
                                                   for channel in channels))
        else:
            self._timeslots = timeslots

        channels = self.channels

        self._buffer = max(chan.buffer for chan in channels) if channels else 0

    @property
    def name(self) -> str:
        """Name of this instruction."""
        return self._name

    @property
    def command(self):
        """Acquire command.

        Returns: Command
        """
        return self._command

    @property
    def channels(self) -> Tuple[Channel]:
        """Returns channels that this schedule uses."""
        return self.timeslots.channels

    @property
    def timeslots(self) -> TimeslotCollection:
        """Occupied time slots by this instruction. """
        return self._timeslots

    @property
    def start_time(self) -> int:
        """Relative begin time of this instruction. """
        return self.timeslots.start_time

    @property
    def stop_time(self) -> int:
        """Relative end time of this instruction. """
        return self.timeslots.stop_time

    @property
    def duration(self) -> int:
        """Duration of this instruction."""
        return self.timeslots.duration

    @property
    def buffer(self) -> int:
        """Buffer for schedule. To be used when appending"""
        return self._buffer

    @property
    def _children(self) -> Tuple[ScheduleComponent]:
        """Instruction has no child nodes. """
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
                at and the flattened `ScheduleComponent`.
        """
        yield (time, self)

    def flatten(self) -> 'Instruction':
        """Return itself as already single instruction."""
        return self

    def union(self, *schedules: List[ScheduleComponent], name: str = None) -> 'ScheduleComponent':
        """Return a new schedule which is the union of `self` and `schedule`.

        Args:
            *schedules: Schedules to be take the union with the parent `Schedule`.
            name: Name of the new schedule. Defaults to name of parent
        """
        return ops.union(self, *schedules, name=name)

    def shift(self: ScheduleComponent, time: int, name: str = None) -> 'ScheduleComponent':
        """Return a new schedule shifted forward by `time`.

        Args:
            time: Time to shift by
            name: Name of the new schedule. Defaults to name of parent
        """
        return ops.shift(self, time, name=name)

    def insert(self, start_time: int, schedule: ScheduleComponent, buffer: bool = False,
               name: str = None) -> 'ScheduleComponent':
        """Return a new schedule with `schedule` inserted within `self` at `start_time`.

        Args:
            start_time: time to be inserted
            schedule: schedule to be inserted
            buffer: Obey buffer when inserting
            name: Name of the new schedule. Defaults to name of parent
        """
        return ops.insert(self, start_time, schedule, buffer=buffer, name=name)

    def append(self, schedule: ScheduleComponent, buffer: bool = True,
               name: str = None) -> 'ScheduleComponent':
        """Return a new schedule with `schedule` inserted at the maximum time over
        all channels shared between `self` and `schedule`.

        Args:
            schedule: schedule to be appended
            buffer: Obey buffer when appending
            name: Name of the new schedule. Defaults to name of parent
        """
        return ops.append(self, schedule, buffer=buffer, name=name)

    def draw(self, dt: float = 1, style=None,
             filename: str = None, interp_method: Callable = None, scaling: float = 1,
             channels_to_plot: List[Channel] = None, plot_all: bool = False,
             plot_range: Tuple[float] = None, interactive: bool = False,
             table: bool = True, label: bool = False,
             framechange: bool = True):
        """Plot the instruction.

        Args:
            dt: Time interval of samples
            style (OPStyleSched): A style sheet to configure plot appearance
            filename: Name required to save pulse image
            interp_method: A function for interpolation
            scaling (float): Relative visual scaling of waveform amplitudes
            channels_to_plot: A list of channel names to plot
            plot_all: Plot empty channels
            plot_range: A tuple of time range to plot
            interactive: When set true show the circuit in a new window
                (this depends on the matplotlib backend being used supporting this)
            table: Draw event table for supported commands
            label: Label individual instructions
            framechange: Add framechange indicators


        Returns:
            matplotlib.figure: A matplotlib figure object of the pulse schedule.
        """
        # pylint: disable=invalid-name, cyclic-import

        from qiskit.tools import visualization

        return visualization.pulse_drawer(self, dt=dt, style=style,
                                          filename=filename, interp_method=interp_method,
                                          scaling=scaling, channels_to_plot=channels_to_plot,
                                          plot_all=plot_all, plot_range=plot_range,
                                          interactive=interactive, table=table,
                                          label=label, framechange=framechange)

    def __add__(self, schedule: ScheduleComponent) -> 'ScheduleComponent':
        """Return a new schedule with `schedule` inserted within `self` at `start_time`."""
        return self.append(schedule)

    def __or__(self, schedule: ScheduleComponent) -> 'ScheduleComponent':
        """Return a new schedule which is the union of `self` and `schedule`."""
        return self.union(schedule)

    def __lshift__(self, time: int) -> 'ScheduleComponent':
        """Return a new schedule which is shifted forward by `time`."""
        return self.shift(time)

    def __repr__(self):
        return "%s" % (self._command)

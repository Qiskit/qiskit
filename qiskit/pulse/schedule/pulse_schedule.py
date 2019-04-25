# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Schedule.
"""
import logging
from copy import copy
from operator import attrgetter
from typing import List, Tuple, Callable

from qiskit.pulse.commands import Instruction
from qiskit.pulse.common.interfaces import ScheduleComponent
from qiskit.pulse.common.timeslots import TimeslotCollection
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.channels import DeviceSpecification, Channel

logger = logging.getLogger(__name__)


class Schedule(ScheduleComponent):
    """Schedule of instructions. The composite node of a schedule tree."""

    def __init__(self, name: str = None, start_time: int = 0):
        """Create empty schedule.

        Args:
            name (str, optional): Name of this schedule. Defaults to None.
            start_time (int, optional): Begin time of this schedule. Defaults to 0.
        """
        self._name = name
        self._start_time = start_time
        self._occupancy = TimeslotCollection(timeslots=[])
        self._children = ()

    @property
    def name(self) -> str:
        """Name of this schedule."""
        return self._name

    def insert(self, start_time: int, schedule: ScheduleComponent) -> 'Schedule':
        """Return a new schedule with inserting a `schedule` at `start_time`.

        Args:
            start_time (int): time to be inserted
            schedule (ScheduleComponent): schedule to be inserted

        Returns:
            Schedule: a new schedule inserted a `schedule` at `start_time`

        Raises:
            PulseError: when an invalid schedule is specified or failed to insert
        """
        if not isinstance(schedule, ScheduleComponent):
            raise PulseError("Invalid to be inserted: %s" % schedule.__class__.__name__)
        news = copy(self)
        try:
            news._insert(start_time, schedule)
        except PulseError as err:
            raise PulseError(err.message)
        return news

    def _insert(self, start_time: int, schedule: ScheduleComponent):
        """Insert a new `schedule` at `start_time`.
        Args:
            start_time (int): start time of the schedule
            schedule (ScheduleComponent): schedule to be inserted
        Raises:
            PulseError: when an invalid schedule is specified or failed to insert
        """
        if schedule == self:
            raise PulseError("Cannot insert self to avoid infinite recursion")
        shifted = schedule.occupancy.shifted(start_time)
        if self._occupancy.is_mergeable_with(shifted):
            self._occupancy = self._occupancy.merged(shifted)
            self._children += (schedule.shifted(start_time),)
        else:
            logger.warning("Fail to insert %s at %s due to timing overlap", schedule, start_time)
            raise PulseError("Fail to insert %s at %s due to overlap" % (str(schedule), start_time))

    def append(self, schedule: ScheduleComponent) -> 'Schedule':
        """Return a new schedule with appending a `schedule` at the timing
        just after the last instruction finishes.

        Args:
            schedule (ScheduleComponent): schedule to be appended

        Returns:
            Schedule: a new schedule appended a `schedule`

        Raises:
            PulseError: when an invalid schedule is specified or failed to append
        """
        if not isinstance(schedule, ScheduleComponent):
            raise PulseError("Invalid to be appended: %s" % schedule.__class__.__name__)
        news = copy(self)
        try:
            news._insert(self.stop_time, schedule)
        except PulseError:
            logger.warning("Fail to append %s due to timing overlap", schedule)
            raise PulseError("Fail to append %s due to overlap" % str(schedule))
        return news

    @property
    def duration(self) -> int:
        return self.stop_time - self.start_time

    @property
    def occupancy(self) -> TimeslotCollection:
        return self._occupancy

    def shifted(self, shift: int) -> ScheduleComponent:
        news = copy(self)
        news._start_time += shift
        news._occupancy = self._occupancy.shifted(shift)
        return news

    @property
    def start_time(self) -> int:
        return self._occupancy.start_time(default=self._start_time)

    @property
    def stop_time(self) -> int:
        return self._occupancy.stop_time(default=self._start_time)

    @property
    def children(self) -> Tuple[ScheduleComponent, ...]:
        return self._children

    def __add__(self, schedule: ScheduleComponent):
        return self.append(schedule)

    def __or__(self, schedule: ScheduleComponent):
        return self.insert(0, schedule)

    def __str__(self):
        res = "Schedule(%s):\n" % (self._name or "")
        instructions = sorted(self.flat_instruction_sequence(), key=attrgetter("start_time"))
        res += '\n'.join([str(i) for i in instructions])
        return res

    def flat_instruction_sequence(self) -> List[Instruction]:
        """Return instruction sequence of this schedule.
        Each instruction has absolute start time.
        """
        return [_ for _ in Schedule._flatten_generator(self)]

    @staticmethod
    def _flatten_generator(node: ScheduleComponent, time: int = 0):
        if isinstance(node, Schedule):
            for child in node.children:
                yield from Schedule._flatten_generator(child, time + node._start_time)
        elif isinstance(node, Instruction):
            yield node.shifted(time)
        else:
            raise PulseError("Unknown ScheduleComponent type: %s" % node.__class__.__name__)

    def draw(self, device: DeviceSpecification, dt: float = 1, style=None,
             filename: str = None, interp_method: Callable = None, scaling: float = None,
             channels_to_plot: List[Channel] = None, plot_all: bool = False,
             plot_range: Tuple[float] = None, interactive: bool = False):
        """Plot the interpolated envelope of pulse.

        Args:
            device: Device information to organize channels.
            dt: Time interval of samples.
            style (OPStyleSched): A style sheet to configure plot appearance.
            filename: Name required to save pulse image.
            interp_method: A function for interpolation.
            scaling: scaling of waveform amplitude.
            channels_to_plot: A list of channel names to plot.
            plot_all: Plot empty channels.
            plot_range: A tuple of time range to plot.
            interactive: When set true show the circuit in a new window
                (this depends on the matplotlib backend being used supporting this).

        Returns:
            matplotlib.figure: A matplotlib figure object of the pulse schedule.
        """
        # pylint: disable=invalid-name, cyclic-import

        from qiskit.tools import visualization

        return visualization.pulse_drawer(self, device=device, dt=dt, style=style,
                                          filename=filename, interp_method=interp_method,
                                          scaling=scaling, channels_to_plot=channels_to_plot,
                                          plot_all=plot_all, plot_range=plot_range,
                                          interactive=interactive)

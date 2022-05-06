# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from typing import Tuple, Dict, Union, Iterator, List, Sequence, Optional, TYPE_CHECKING

from collections import defaultdict
import numpy as np

from qiskit.pulse.exceptions import PulseError

if TYPE_CHECKING:
    from qiskit.pulse.instructions import Instruction
    from qiskit.pulse.schedule import Schedule
    from qiskit.pulse.channels import Channel


class TimeslotManager:

    def __init__(self):
        self._timeslots = defaultdict(list)
        self._idle_after = None

    def __len__(self):
        return sum(len(slots) for slots in self._timeslots.values())

    @property
    def timeslots(self) -> Dict["Channel", List[Tuple[int, int]]]:
        return self._timeslots

    @property
    def channels(self) -> Tuple["Channel"]:
        return tuple(self._timeslots.keys())

    @property
    def duration(self) -> int:
        if self._idle_after is None:
            self._idle_after = self.find_idle_after()

        return self._idle_after

    def regenerate(self, t0_inst_tups: Iterator[Tuple[int, Union["Instruction", "Schedule"]]]):
        self.clear()
        self._idle_after = None

        for t0, inst in t0_inst_tups:
            if not np.issubdtype(type(t0), np.integer):
                raise PulseError("Schedule start time must be an integer.")
            if t0 < 0:
                PulseError(f"An instruction on {inst.channels} has a negative starting time.")

            if hasattr(inst, "timeslots"):
                # TODO Cleaner logic is needed here
                from qiskit.pulse.transforms import flatten
                flat_sched = flatten(inst)
                reservation = {
                    chan: (t0 + ts, t0 + tf) for chan, (ts, tf) in flat_sched.timeslots.items()
                }
            else:
                reservation = {chan: (t0, t0 + inst.duration) for chan in inst.channels}

            overlaps = set(reservation.keys() & self._timeslots.keys())
            while overlaps:
                chan = overlaps.pop()
                interval = reservation.pop(chan)
                self._check_overlap(inst_name=inst.name, chan=chan, interval=interval, t0=t0)
                self._timeslots[chan].append(interval)

            self._timeslots.update(reservation)

    def add(self, inst_name: str, channels: Sequence["Channel"], interval: Tuple[int, int]):
        for chan in channels:
            self._check_overlap(inst_name=inst_name, chan=chan, interval=interval)
            self._timeslots[chan].append(interval)

    def _check_overlap(
        self,
        inst_name: str,
        chan: "Channel",
        interval: Tuple[int, int],
        t0: Optional[int] = None,
    ):
        def _check_overlap(occupied: Tuple[int, int]):
            return max(interval[0], occupied[0]) < min(interval[1], occupied[1])
        if any(_check_overlap(occupied) for occupied in self._timeslots[chan]):
            t0 = t0 if t0 is not None else interval[0]
            raise PulseError(
                f"Schedule(name={inst_name}) cannot be inserted at time {t0} "
                f"because its instruction on channel {chan} scheduled from time "
                f"{interval[0]} to {interval[1]} overlaps with an existing instruction."
            )

    def find_idle_after(self, channels: Optional[Sequence["Channel"]] = None) -> int:
        if channels is not None:
            channels = list(self._timeslots.keys())

        if all(chan not in self._timeslots for chan in channels):
            return 0

        idle_after = 0
        for chan in channels:
            if chan not in self._timeslots:
                continue
            chan_idle_after = max(interval[1] for interval in self._timeslots[chan])
            idle_after = max(idle_after, chan_idle_after)
        return idle_after

    def find_start_at(self, channels: Optional[Sequence["Channel"]] = None) -> int:
        if channels is not None:
            channels = list(self._timeslots.keys())

        if all(chan not in self._timeslots for chan in channels):
            return 0

        start_at = self.duration
        for chan in channels:
            if chan not in self._timeslots:
                continue
            chan_start_at = min(interval[0] for interval in self._timeslots[chan])
            start_at = min(start_at, chan_start_at)
        return start_at

    def clear(self):
        self._timeslots.clear()

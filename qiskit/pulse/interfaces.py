# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
ScheduleComponent has been deprecated.
It is a common interface for components of schedule (Instruction and Schedule).
"""
import warnings
from abc import ABCMeta, abstractmethod
from typing import Tuple, List, Union, Optional

from qiskit.pulse.channels import Channel


class ScheduleComponent(metaclass=ABCMeta):
    """ScheduleComponent has been deprecated.
    It has been replaced by `qiskit.pulse.values.Value` and `qiskit.pulse.values.NamedValue``.
    Anywhere that currently accepts a ``ScheduleComponent`` should instead accept a
    ``Union[Schedule, Instruction]``.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of ScheduleComponent."""
        warnings.warn(
            "ScheduleComponent is deprecated and will be removed in a future release. "
            "Anywhere that currently accepts a ``ScheduleComponent`` should instead "
            "accept a ``Union[Schedule, Instruction]`` ",
            DeprecationWarning,
        )
        pass

    @property
    @abstractmethod
    def channels(self) -> List[Channel]:
        """Return channels used by schedule."""
        warnings.warn(
            "ScheduleComponent is deprecated and will be removed in a future release. "
            "Anywhere that currently accepts a ``ScheduleComponent`` should instead "
            "accept a ``Union[Schedule, Instruction]`` ",
            DeprecationWarning,
        )

        pass

    @property
    @abstractmethod
    def duration(self) -> int:
        """Duration of this schedule component."""
        warnings.warn(
            "ScheduleComponent is deprecated and will be removed in a future release. "
            "Anywhere that currently accepts a ``ScheduleComponent`` should instead "
            "accept a ``Union[Schedule, Instruction]`` ",
            DeprecationWarning,
        )

        pass

    @property
    @abstractmethod
    def start_time(self) -> int:
        """Starting time of this schedule component."""
        warnings.warn(
            "ScheduleComponent is deprecated and will be removed in a future release. "
            "Anywhere that currently accepts a ``ScheduleComponent`` should instead "
            "accept a ``Union[Schedule, Instruction]`` ",
            DeprecationWarning,
        )

        pass

    @property
    @abstractmethod
    def stop_time(self) -> int:
        """Stopping time of this schedule component."""
        warnings.warn(
            "ScheduleComponent is deprecated and will be removed in a future release. "
            "Anywhere that currently accepts a ``ScheduleComponent`` should instead "
            "accept a ``Union[Schedule, Instruction]`` ",
            DeprecationWarning,
        )
        pass

    @abstractmethod
    def ch_duration(self, *channels: List[Channel]) -> int:
        """Duration of the `channels` in schedule component."""
        warnings.warn(
            "ScheduleComponent is deprecated and will be removed in a future release. "
            "Anywhere that currently accepts a ``ScheduleComponent`` should instead "
            "accept a ``Union[Schedule, Instruction]`` ",
            DeprecationWarning,
        )
        pass

    @abstractmethod
    def ch_start_time(self, *channels: List[Channel]) -> int:
        """Starting time of the `channels` in schedule component."""
        warnings.warn(
            "ScheduleComponent is deprecated and will be removed in a future release. "
            "Anywhere that currently accepts a ``ScheduleComponent`` should instead "
            "accept a ``Union[Schedule, Instruction]`` ",
            DeprecationWarning,
        )
        pass

    @abstractmethod
    def ch_stop_time(self, *channels: List[Channel]) -> int:
        """Stopping of the `channels` in schedule component."""
        warnings.warn(
            "ScheduleComponent is deprecated and will be removed in a future release. "
            "Anywhere that currently accepts a ``ScheduleComponent`` should instead "
            "accept a ``Union[Schedule, Instruction]`` ",
            DeprecationWarning,
        )
        pass

    @property
    @abstractmethod
    def _children(self) -> Tuple[Union[int, "ScheduleComponent"]]:
        """Child nodes of this schedule component."""
        warnings.warn(
            "ScheduleComponent is deprecated and will be removed in a future release. "
            "Anywhere that currently accepts a ``ScheduleComponent`` should instead "
            "accept a ``Union[Schedule, Instruction]`` ",
            DeprecationWarning,
        )
        pass

    @property
    @abstractmethod
    def instructions(self) -> Tuple[Tuple[int, "Instructions"]]:
        """Return iterable for all `Instruction`s in `Schedule` tree."""
        warnings.warn(
            "ScheduleComponent is deprecated and will be removed in a future release. "
            "Anywhere that currently accepts a ``ScheduleComponent`` should instead "
            "accept a ``Union[Schedule, Instruction]`` ",
            DeprecationWarning,
        )
        pass

    @abstractmethod
    def flatten(self) -> "ScheduleComponent":
        """Return a new schedule which is the flattened schedule contained all `instructions`."""
        warnings.warn(
            "ScheduleComponent is deprecated and will be removed in a future release. "
            "Anywhere that currently accepts a ``ScheduleComponent`` should instead "
            "accept a ``Union[Schedule, Instruction]`` ",
            DeprecationWarning,
        )
        pass

    @abstractmethod
    def shift(
        self: "ScheduleComponent", time: int, name: Optional[str] = None
    ) -> "ScheduleComponent":
        """Return a new schedule shifted forward by `time`.

        Args:
            time: Time to shift by
            name: Name of the new schedule. Defaults to name of parent
        """
        warnings.warn(
            "ScheduleComponent is deprecated and will be removed in a future release. "
            "Anywhere that currently accepts a ``ScheduleComponent`` should instead "
            "accept a ``Union[Schedule, Instruction]`` ",
            DeprecationWarning,
        )
        pass

    @abstractmethod
    def insert(
        self, start_time: int, schedule: "ScheduleComponent", name: Optional[str] = None
    ) -> "ScheduleComponent":
        """Return a new schedule with `schedule` inserted at `start_time` of `self`.

        Args:
            start_time: time to be inserted
            schedule: schedule to be inserted
            name: Name of the new schedule. Defaults to name of parent
        """
        warnings.warn(
            "ScheduleComponent is deprecated and will be removed in a future release. "
            "Anywhere that currently accepts a ``ScheduleComponent`` should instead "
            "accept a ``Union[Schedule, Instruction]`` ",
            DeprecationWarning,
        )
        pass

    @abstractmethod
    def append(
        self, schedule: "ScheduleComponent", name: Optional[str] = None
    ) -> "ScheduleComponent":
        """Return a new schedule with `schedule` inserted at the maximum time over
        all channels shared between `self` and `schedule`.

        Args:
            schedule: schedule to be appended
            name: Name of the new schedule. Defaults to name of parent
        """
        warnings.warn(
            "ScheduleComponent is deprecated and will be removed in a future release. "
            "Anywhere that currently accepts a ``ScheduleComponent`` should instead "
            "accept a ``Union[Schedule, Instruction]`` ",
            DeprecationWarning,
        )
        pass

    @abstractmethod
    def __add__(self, schedule: "ScheduleComponent") -> "ScheduleComponent":
        """Return a new schedule with `schedule` inserted within `self` at `start_time`."""
        warnings.warn(
            "ScheduleComponent is deprecated and will be removed in a future release. "
            "Anywhere that currently accepts a ``ScheduleComponent`` should instead "
            "accept a ``Union[Schedule, Instruction]`` ",
            DeprecationWarning,
        )
        pass

    @abstractmethod
    def __or__(self, schedule: "ScheduleComponent") -> "ScheduleComponent":
        """Return a new schedule which is the union of `self` and `schedule`."""
        warnings.warn(
            "ScheduleComponent is deprecated and will be removed in a future release. "
            "Anywhere that currently accepts a ``ScheduleComponent`` should instead "
            "accept a ``Union[Schedule, Instruction]`` ",
            DeprecationWarning,
        )
        pass

    @abstractmethod
    def __lshift__(self, time: int) -> "ScheduleComponent":
        """Return a new schedule which is shifted forward by `time`."""
        warnings.warn(
            "ScheduleComponent is deprecated and will be removed in a future release. "
            "Anywhere that currently accepts a ``ScheduleComponent`` should instead "
            "accept a ``Union[Schedule, Instruction]`` ",
            DeprecationWarning,
        )
        pass

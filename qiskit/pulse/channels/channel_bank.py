# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Memory of channel registers.
"""
from qiskit.providers import BaseBackend
from qiskit.pulse import ChannelsError
from .channel_register import ChannelRegister, AcquireChannelRegister
from .output_channel import DriveChannel, ControlChannel, MeasureChannel
from .output_channel_register import DriveChannelRegister
from .pulse_channel import AcquireChannel, SnapshotChannel


class ChannelBank:  # TODO: better name?
    """Implement a channel memory."""

    def __init__(self, backend: BaseBackend = None):
        """
        Create channel registers with default values in backend.
        Args:
            backend:
        """
        self._drive = None
        self._control = None
        self._measure = None
        self._acquire = None
        self._snapshot = None
        if backend:
            # TODO: create channel regs from backend
            pass

    def register(self, reg: ChannelRegister):
        """
        Overwrite the register of the same channel as `reg`.
        Args:
            reg:

        Returns:

        """
        cls = reg.channel_cls
        if cls == DriveChannel:
            self._drive = reg
        elif cls == ControlChannel:
            self._control = reg
        elif cls == MeasureChannel:
            self._measure = reg
        elif cls == AcquireChannel:
            self._acquire = reg
        elif cls == SnapshotChannel:
            self._snapshot = reg
        else:
            raise ChannelsError("Unknown channel: %s", cls.__name__)

    @property
    def drive(self) -> DriveChannelRegister:
        return self._drive

    @drive.setter
    def drive(self, reg):
        raise ChannelsError("No direct set is allowed, use 'register()' for safety set.")

    @property
    def acquire(self) -> AcquireChannelRegister:
        return self._acquire

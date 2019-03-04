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
from .backend_channels import AcquireChannel, SnapshotChannel
from .channel_register import ChannelRegister, OutputChannelRegister
from .output_channels import DriveChannel, ControlChannel, MeasureChannel


class ChannelBank:    # TODO: better name?
    """Implement a channel memory."""

    def __init__(self, backend: BaseBackend = None):
        self._drive = None
        self._control = None
        self._measure = None
        self._acquire = None
        self._snapshot = None
        if backend:
            # TODO create default channel registers
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
    def drive(self) -> OutputChannelRegister:
        return self._drive

    @drive.setter
    def drive(self, reg):
        raise ChannelsError("Cannot set directly, use 'register()' to change")



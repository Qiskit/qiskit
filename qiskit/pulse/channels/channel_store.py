# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Store of channel registers.
"""
import logging
from typing import List

from qiskit.pulse import ChannelsError
from .channel_register import ChannelRegister, AcquireChannelRegister, SnapshotChannelRegister
from .output_channel import DriveChannel, ControlChannel, MeasureChannel
from .output_channel_register import (DriveChannelRegister,
                                      ControlChannelRegister,
                                      MeasureChannelRegister)
from .pulse_channel import AcquireChannel, SnapshotChannel

logger = logging.getLogger(__name__)


class ChannelStore:  # TODO: better name?
    """Implement a channel store, which is a collection of channel registers.
    It must be defined before constructing schedule."""

    @classmethod
    def create_from(cls, backend):
        """
        Create channel store with default values in backend configuration.
        Args:
            backend(Backend):
        """
        config = backend.configuration()

        # system size
        n_qubit = config.n_qubits

        # frequency information
        qubit_lo_freqs = config.defaults['qubit_freq_est']
        qubit_lo_range = config.qubit_lo_range
        meas_lo_freqs = config.defaults['meas_freq_est']
        meas_lo_range = config.meas_lo_range

        # generate channel registers
        channels = [
            DriveChannelRegister(size=n_qubit, lo_freqs=qubit_lo_freqs),
            ControlChannelRegister(size=n_qubit),
            MeasureChannelRegister(size=n_qubit, lo_freqs=meas_lo_freqs),
            AcquireChannelRegister(size=n_qubit),
            SnapshotChannelRegister(size=n_qubit)
        ]

        return ChannelStore(channels)

    def __init__(self, registers: List[ChannelRegister]):
        """
        Create channel registers with default values in backend.
        Args:
            registers(list):
        """
        self._drive = None
        self._control = None
        self._measure = None
        self._acquire = None
        self._snapshot = None
        for reg in registers:
            self._register(reg)

    def _register(self, reg: ChannelRegister):
        """
        Store the register `reg` if the same type register has not yet stored.
        Args:
            reg:
        """
        cls = reg.channel_cls
        if cls == DriveChannel:
            if self._drive is None:
                self._drive = reg
            else:
                raise ChannelsError("Duplicated drive registers are not allowed.")
        elif cls == ControlChannel:
            if self._control is None:
                self._control = reg
            else:
                raise ChannelsError("Duplicated control registers are not allowed.")
        elif cls == MeasureChannel:
            if self._measure is None:
                self._measure = reg
            else:
                raise ChannelsError("Duplicated measure registers are not allowed.")
        elif cls == AcquireChannel:
            if self._acquire is None:
                self._acquire = reg
            else:
                raise ChannelsError("Duplicated acquire registers are not allowed.")
        elif cls == SnapshotChannel:
            if self._snapshot is None:
                self._snapshot = reg
            else:
                raise ChannelsError("Duplicated snapshot registers are not allowed.")
        else:
            raise ChannelsError("Unknown channel: %s", cls.__name__)

    @property
    def drive(self) -> DriveChannelRegister:
        return self._drive

    @property
    def control(self) -> ControlChannelRegister:
        return self._control

    @property
    def measure(self) -> MeasureChannelRegister:
        return self._measure

    @property
    def acquire(self) -> AcquireChannelRegister:
        return self._acquire

    @property
    def snapshot(self) -> SnapshotChannelRegister:
        return self._snapshot

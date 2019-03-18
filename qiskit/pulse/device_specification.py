# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Specification of the device.
"""
import logging
from typing import List

from qiskit.pulse.qubit import Qubit
from .channels.output_channel import DriveChannel, ControlChannel, MeasureChannel
from .channels.pulse_channel import AcquireChannel, SnapshotChannel

logger = logging.getLogger(__name__)


class DeviceSpecification:
    """Implement a device specification, which is usually contructed from backend info."""

    @classmethod
    def create_from(cls, backend):
        """
        Create device specification with values in backend configuration.
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

        # generate channels with assuming their numberings are aligned with qubits
        drives = [DriveChannel(i, qubit_lo_freqs[i]) for i in range(n_qubit)]   # TODO: lo_ranges
        controls = [ControlChannel(i) for i in range(n_qubit)]
        measures = [MeasureChannel(i, meas_lo_freqs[i]) for i in range(n_qubit)]   # TODO: lo_ranges
        acquires = [AcquireChannel(i) for i in range(n_qubit)]
        snapshots = [SnapshotChannel(i) for i in range(n_qubit)]

        qubits = []
        for i in range(n_qubit):
            qubit = Qubit(i,
                          drive_channels=[drives[i]],
                          control_channels=[controls[i]],
                          measure_channels=[measures[i]],
                          acquire_channels=[acquires[i]],
                          snapshot_channels=[snapshots[i]])
            qubits.append(qubit)

        return DeviceSpecification(qubits)

    def __init__(self, qubits: List[Qubit]):
        """
        Create device specification with specified `qubits`.
        Args:
            qubits(list):
        """
        self._qubits = qubits

    @property
    def q(self) -> List[Qubit]:
        return self._qubits

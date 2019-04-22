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

from qiskit.pulse.exceptions import PulseError
from qiskit.validation.exceptions import ModelValidationError
from .output_channel import DriveChannel, ControlChannel, MeasureChannel
from .pulse_channel import AcquireChannel, MemorySlot, RegisterSlot
from .qubit import Qubit

logger = logging.getLogger(__name__)


class DeviceSpecification:
    """Implement a device specification, which is usually constructed from backend info."""

    def __init__(self,
                 qubits: List[Qubit],
                 registers: List[RegisterSlot],
                 mem_slots: List[MemorySlot]):
        """
        Create device specification with specified `qubits`.
        Args:
            qubits:
        """
        self._qubits = qubits
        self._reg_slots = registers
        self._mem_slots = mem_slots

    @classmethod
    def create_from(cls, backend):
        """
        Create device specification with values in backend configuration.
        Args:
            backend(Backend): backend configuration
        Returns:
            DeviceSpecification: created device specification
        Raises:
            PulseError: when an invalid backend is specified
        """
        backend_config = backend.configuration()

        # TODO : Remove usage of config.defaults when backend.defaults() is updated.
        try:
            backend_default = backend.defaults()
        except ModelValidationError:
            from collections import namedtuple
            BackendDefault = namedtuple('BackendDefault', ('qubit_freq_est', 'meas_freq_est'))

            backend_default = BackendDefault(
                qubit_freq_est=backend_config.defaults['qubit_freq_est'],
                meas_freq_est=backend_config.defaults['meas_freq_est']
            )

        # system size
        n_qubits = backend_config.n_qubits
        n_registers = backend_config.n_registers
        n_uchannels = backend_config.n_uchannels

        if n_uchannels > 0 and n_uchannels != n_qubits:
            raise PulseError("This version assumes no U-channels or #U-cannels==#qubits.")

        # frequency information
        qubit_lo_freqs = backend_default.qubit_freq_est
        qubit_lo_ranges = backend_config.qubit_lo_range
        meas_lo_freqs = backend_default.meas_freq_est
        meas_lo_ranges = backend_config.meas_lo_range

        # generate channels with assuming their numberings are aligned with qubits
        drives = [
            DriveChannel(i, qubit_lo_freqs[i], tuple(qubit_lo_ranges[i]))
            for i in range(n_qubits)
        ]
        measures = [
            MeasureChannel(i, meas_lo_freqs[i], tuple(meas_lo_ranges[i]))
            for i in range(n_qubits)
        ]
        acquires = [AcquireChannel(i) for i in range(n_qubits)]
        controls = [ControlChannel(i) for i in range(n_uchannels)]

        qubits = []
        for i in range(n_qubits):
            # TODO: get qubits <-> channels relationship from backend
            qubit = Qubit(i,
                          drive_channels=[drives[i]],
                          control_channels=None if n_uchannels == 0 else controls[i],
                          measure_channels=[measures[i]],
                          acquire_channels=[acquires[i]])
            qubits.append(qubit)

        registers = [RegisterSlot(i) for i in range(n_registers)]
        # TODO: get #mem_slots from backend
        mem_slots = [MemorySlot(i) for i in range(len(qubits))]

        return DeviceSpecification(qubits, registers, mem_slots)

    def __eq__(self, other):
        """Two device specs are the same if they have the same qubits.

        Args:
            other (DeviceSpecification): other DeviceSpecification

        Returns:
            bool: are self and other equal.
        """
        if type(self) is type(other) and \
                self._qubits == other._qubits:
            return True
        return False

    @property
    def q(self) -> List[Qubit]:
        """Return qubits in this device."""
        return self._qubits

    # pylint: disable=invalid-name
    @property
    def c(self) -> List[RegisterSlot]:
        """Return register slots in this device."""
        return self._reg_slots

    @property
    def mem(self) -> List[MemorySlot]:
        """Return memory slots in this device."""
        return self._mem_slots

# -*- coding: utf-8 -*-

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
Specification of the device.
"""
import logging
from typing import List

from qiskit.validation.exceptions import ModelValidationError

from .pulse_channels import DriveChannel, ControlChannel, MeasureChannel
from .channels import AcquireChannel, MemorySlot, RegisterSlot
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
            buffer = backend_default.buffer
        except ModelValidationError:
            try:
                buffer = backend_config.defaults.get('buffer', 0)
            except AttributeError:
                buffer = 0

        # system size
        n_qubits = backend_config.n_qubits
        n_registers = backend_config.n_registers
        n_uchannels = backend_config.n_uchannels

        # generate channels with assuming their numberings are aligned with qubits
        drives = [DriveChannel(i, buffer=buffer) for i in range(n_qubits)]

        measures = [MeasureChannel(i, buffer=buffer) for i in range(n_qubits)]

        controls = [ControlChannel(i, buffer=buffer) for i in range(n_uchannels)]

        acquires = [AcquireChannel(i, buffer=buffer) for i in range(n_qubits)]

        qubits = []
        for i in range(n_qubits):
            # TODO: get qubits <-> channels relationship from backend
            qubit = Qubit(i, drives[i], measures[i], acquires[i],
                          control_channels=[] if not controls else controls)
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

    @property
    def c(self) -> List[RegisterSlot]:
        """Return register slots in this device."""
        return self._reg_slots

    @property
    def mem(self) -> List[MemorySlot]:
        """Return memory slots in this device."""
        return self._mem_slots

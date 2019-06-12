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

# pylint: disable=invalid-name

"""
Specification of the device.
"""
from typing import List

from .channels import AcquireChannel, MemorySlot, RegisterSlot
from .pulse_channels import DriveChannel, ControlChannel, MeasureChannel
from .qubit import Qubit


class PulseSpecification:
    """Implement a device specification."""

    def __init__(self,
                 n_drive: int,
                 n_control: int,
                 n_measure: int,
                 n_registers: int,
                 buffer: int = 0,
                 hamiltonian=None):
        """
        Create pulse specification with number of channels.

        Args:
            n_drive: Number of drive channels.
            n_control: Number of control channels.
            n_measure: Number of acquire channels. Acquire channel is bound to a single memory slot.
            n_registers: Number of classical registers.
            buffer: Buffer that should be placed between instructions on channel.
            hamiltonian (Hamiltonian): Hamiltonian object to extract device topology.
        """
        self._drives = [DriveChannel(idx, buffer) for idx in range(n_drive)]
        self._controls = [ControlChannel(idx, buffer) for idx in range(n_control)]
        self._measures = [MeasureChannel(idx, buffer) for idx in range(n_measure)]
        self._acquires = [AcquireChannel(idx, buffer) for idx in range(n_measure)]
        self._mem_slots = [MemorySlot(idx) for idx in range(n_measure)]
        self._reg_slots = [RegisterSlot(idx) for idx in range(n_registers)]
        self._qubits = []

        if hamiltonian:
            # TODO: create qubit from hamiltonian
            pass

    @classmethod
    def from_device(cls, backend):
        """
        Create pulse specification with values from backend.

        Args:
            backend (BaseBackend): Backend configuration.

        Returns:
            PulseSpecification: New PulseSpecification configured by backend.
        """
        configuration = backend.configuration()
        defaults = backend.defaults()

        # TODO: define n_drive and n_acquire explicitly in backend
        n_drive = len(configuration.qubit_lo_range)
        n_control = configuration.n_uchannels
        n_measure = len(configuration.meas_lo_range)
        n_registers = configuration.n_registers
        buffer = defaults.buffer
        hamiltonian = configuration.hamiltonian

        return PulseSpecification(n_drive=n_drive, n_control=n_control,
                                  n_measure=n_measure, n_registers=n_registers,
                                  buffer=buffer, hamiltonian=hamiltonian)

    def __eq__(self, other):
        """Two device specs are the same if they have the same channels.

        Args:
            other (PulseSpecification): Other PulseSpecification.

        Returns:
            bool: Are self and other equal.
        """
        # pylint: disable=too-many-boolean-expressions
        if type(self) is type(other) and \
                self._drives == other._drives and \
                self._controls == other._controls and \
                self._measures == other._measures and \
                self._acquires == other._acquires and \
                self._mem_slots == other._mem_slots and \
                self._reg_slots == other._reg_slots:
            return True
        return False

    @property
    def d(self) -> List[DriveChannel]:
        """Return drive channel in this device."""
        return self._drives

    @property
    def u(self):
        """Return control channel in this device."""
        return self._controls

    @property
    def m(self):
        """Return measure channel in this device."""
        return self._measures

    @property
    def acq(self):
        """Return acquire channel in this device."""
        return self._acquires

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

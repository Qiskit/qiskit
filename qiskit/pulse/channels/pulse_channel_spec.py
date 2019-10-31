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
Pulse channel wrapper object for the system.
"""
import warnings

from typing import List

from qiskit.pulse.exceptions import PulseError
from .channels import (AcquireChannel, MemorySlot, RegisterSlot, DriveChannel, ControlChannel,
                       MeasureChannel)
from .qubit import Qubit


class PulseChannelSpec:
    """A helper class to support assembling channel objects and their mapping to qubits.
    This class can be initialized with two methods shown as below.

    1. With the `BaseBackend` object of the target pulse backend:
        ```python
        system = PulseChannelSpec.from_backend(backend)
        ```

    2. By specifying the number of pulse elements constituting the target quantum computing system:
        ```python
        system = PulseChannelSpec(n_qubits=5, n_control=6, n_registers=1, buffer=10)
        ```

    Within Qiskit a quantum computing system at the level of pulses is abstracted as
    a combination of multiple types of channels on which instructions are scheduled.
    These most common channel types are the:
      - `PulseChannel`: For performing stimulus of the system.
      - `AcquireChannel`: For scheduling acquisition of qubit data.
      - `MemorySlot`: For persistent storage of measurement results.
      - `RegisterSlot`: For temporary storage of and conditional feedback on measurement results.

    There are also several special types of pulse channels which are the:
      - `DriveChannel`: Used to control a single qubit.
      - `MeasureChannel`: Used to perform measurement stimulus of a single qubit.
      - `ControlChannel`: Used to control an arbitrary Hamiltonian term on the system.
      Typically a two-qubit interaction.

    A collection of above channels is automatically assembled within `PulseChannelSpec`.

    For example, the zeroth drive channel may be accessed by
        ```python
        system.drives[0]
        ```
    or if the channel is connected to the first qubit,
        ```python
        system.qubits[0].drive
        ```
    In the above example, both commands refer to the same object.
    """
    def __init__(self,
                 n_qubits: int,
                 n_control: int,
                 n_registers: int,
                 buffer: int = 0):
        """
        Create pulse channel specification with number of channels.

        Args:
            n_qubits: Number of qubits.
            n_control: Number of control channels.
            n_registers: Number of classical registers.
            buffer: Buffer that should be placed between instructions on channel.
        """
        warnings.warn("The PulseChannelSpec is deprecated. Use backend.configuration() instead. "
                      "The supported methods require some migrations; check out the release "
                      "notes for the complete details.",
                      DeprecationWarning)
        if buffer:
            warnings.warn("Buffers are no longer supported. Please use an explicit Delay.")
        self._drives = [DriveChannel(idx) for idx in range(n_qubits)]
        self._controls = [ControlChannel(idx) for idx in range(n_control)]
        self._measures = [MeasureChannel(idx) for idx in range(n_qubits)]
        self._acquires = [AcquireChannel(idx) for idx in range(n_qubits)]
        self._mem_slots = [MemorySlot(idx) for idx in range(n_qubits)]
        self._reg_slots = [RegisterSlot(idx) for idx in range(n_registers)]

        # create mapping information from channels
        warnings.simplefilter("ignore")  # Suppress Qubit deprecation warnings
        self._qubits = [
            Qubit(ii, DriveChannel(ii), MeasureChannel(ii),
                  AcquireChannel(ii), [ControlChannel(ii)])
            for ii in range(n_qubits)]
        warnings.resetwarnings()

    @classmethod
    def from_backend(cls, backend):
        """
        Create pulse channel specification with values from backend.

        Args:
            backend (BaseBackend): Backend configuration.

        Returns:
            PulseChannelSpec: New PulseSpecification configured by backend.

        Raises:
            PulseError: When OpenPulse is not supported.
        """
        configuration = backend.configuration()

        if not configuration.open_pulse:
            raise PulseError(configuration.backend_name + ' does not support OpenPulse.')

        # TODO: allow for drives/measures which are not identical to number of qubit
        n_qubits = configuration.n_qubits
        n_controls = configuration.n_uchannels
        n_registers = configuration.n_registers

        return PulseChannelSpec(n_qubits=n_qubits, n_control=n_controls,
                                n_registers=n_registers)

    @property
    def drives(self) -> List[DriveChannel]:
        """Return system's drive channels."""
        return self._drives

    @property
    def controls(self):
        """Return system's control channels."""
        return self._controls

    @property
    def measures(self):
        """Return system's measure channels."""
        return self._measures

    @property
    def acquires(self):
        """Return system's acquire channels."""
        return self._acquires

    @property
    def qubits(self) -> List[Qubit]:
        """Return list of qubit in this system."""
        return self._qubits

    @property
    def registers(self) -> List[RegisterSlot]:
        """Return system's register slots."""
        return self._reg_slots

    @property
    def memoryslots(self) -> List[MemorySlot]:
        """Return system's memory slots."""
        return self._mem_slots

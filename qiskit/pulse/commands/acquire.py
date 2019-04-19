# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Acquire.
"""
from typing import Union, List

from qiskit.pulse.channels import Qubit, MemorySlot, RegisterSlot
from qiskit.pulse.common.timeslots import Interval, Timeslot, TimeslotCollection
from qiskit.pulse.exceptions import PulseError
from .instruction import Instruction
from .meas_opts import Discriminator, Kernel
from .pulse_command import PulseCommand


class Acquire(PulseCommand):
    """Acquire."""

    def __init__(self, duration, discriminator=None, kernel=None):
        """Create new acquire command.

        Args:
            duration (int): Duration of acquisition.
            discriminator (Discriminator): Discriminators to be used
                (from the list of available discriminator) if the measurement level is 2.
            kernel (Kernel): The data structures defining the measurement kernels
                to be used (from the list of available kernels) and set of parameters
                (if applicable) if the measurement level is 1 or 2.

        Raises:
            PulseError: when invalid discriminator or kernel object is input.
        """
        super().__init__(duration=duration)

        if discriminator:
            if isinstance(discriminator, Discriminator):
                self.discriminator = discriminator
            else:
                raise PulseError('Invalid discriminator object is specified.')
        else:
            self.discriminator = None

        if kernel:
            if isinstance(kernel, Kernel):
                self.kernel = kernel
            else:
                raise PulseError('Invalid kernel object is specified.')
        else:
            self.kernel = None

    def __eq__(self, other):
        """Two Acquires are the same if they are of the same type
        and have the same kernel and discriminator.

        Args:
            other (Acquire): Other Acquire

        Returns:
            bool: are self and other equal.
        """
        if type(self) is type(other) and \
                self.kernel == other.kernel and \
                self.discriminator == other.discriminator:
            return True
        return False

    def __repr__(self):
        return '%s(%s, duration=%d, kernel=%s, discriminator=%s)' % \
               (self.__class__.__name__, self.name, self.duration,
                self.kernel, self.discriminator)

    def __call__(self,
                 qubits: Union[Qubit, List[Qubit]],
                 mem_slots: Union[MemorySlot, List[MemorySlot]],
                 reg_slots: Union[RegisterSlot, List[RegisterSlot]] = None) -> 'AcquireInstruction':
        return AcquireInstruction(self, qubits, mem_slots, reg_slots)


class AcquireInstruction(Instruction):
    """Pulse to acquire measurement result. """

    def __init__(self,
                 command: Acquire,
                 qubits: Union[Qubit, List[Qubit]],
                 mem_slots: Union[MemorySlot, List[MemorySlot]],
                 reg_slots: Union[RegisterSlot, List[RegisterSlot]] = None,
                 start_time: int = 0):
        if isinstance(qubits, Qubit):
            qubits = [qubits]
        if mem_slots:
            if isinstance(mem_slots, MemorySlot):
                mem_slots = [mem_slots]
            elif len(qubits) != len(mem_slots):
                raise PulseError("#mem_slots must be equals to #qubits")
        if reg_slots:
            if isinstance(reg_slots, RegisterSlot):
                reg_slots = [reg_slots]
            if len(qubits) != len(reg_slots):
                raise PulseError("#reg_slots must be equals to #qubits")
        else:
            reg_slots = []

        # TODO: more precise time-slots when we have `acquisition_latency`
        stop_time = start_time+command.duration
        slots = [Timeslot(Interval(start_time, stop_time), q.acquire) for q in qubits]
        slots.extend([Timeslot(Interval(start_time, stop_time), mem) for mem in mem_slots])

        super().__init__(command, start_time, TimeslotCollection(slots))

        self._qubits = qubits
        self._mem_slots = mem_slots
        self._reg_slots = reg_slots

    @property
    def command(self) -> Acquire:
        """Acquire command. """
        return self._command

    @property
    def qubits(self):
        """Qubits to be acquired. """
        return self._qubits

    @property
    def mem_slots(self):
        """MemorySlots. """
        return self._mem_slots

    @property
    def reg_slots(self):
        """RegisterSlots. """
        return self._reg_slots

    def __repr__(self):
        return '%4d: %s -> q%s' % (self._start_time, self._command, [q.index for q in self._qubits])

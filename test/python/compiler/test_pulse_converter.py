# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=redefined-builtin

"""Converter Test."""

import unittest
from qiskit.test import QiskitTestCase
from qiskit.qobj import PulseQobjInstruction
from qiskit.compiler.pulse_to_qobj import PulseQobjConverter
from qiskit.pulse.commands import SamplePulse, DriveInstruction
from qiskit.pulse.channels import (DeviceSpecification, Qubit, AcquireChannel, DriveChannel,
                                   RegisterSlot, MemorySlot)


class TestPulseConverter(QiskitTestCase):
    """Pulse converter tests."""

    def setUp(self):
        self.device = DeviceSpecification(
            qubits=[
                Qubit(0,
                      drive_channels=[DriveChannel(0, 1.2)],
                      acquire_channels=[AcquireChannel(0)])
            ],
            registers=[
                RegisterSlot(0)
            ])

    def test_pulse_converter(self):

        converter = PulseQobjConverter(PulseQobjInstruction, meas_level=2)

        valid_qobj = PulseQobjInstruction(
            name='gauss',
            ch='d0',
            t0=0
        )

        # place holder for test code

        # TODO: add test after schedule-PR is merged or rebased to this branch.

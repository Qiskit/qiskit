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

"""Test for the CmdDef object."""

from qiskit.test import QiskitTestCase

from qiskit.pulse.channels import SystemTopology
from qiskit.pulse import DriveChannel, AcquireChannel, ControlChannel, MeasureChannel


class TestSystemTopology(QiskitTestCase):
    """Tests for system topology."""

    def setUp(self):
        self.drives = [DriveChannel(ii) for ii in range(2)]
        self.acquires = [AcquireChannel(ii) for ii in range(2)]
        self.controls = [ControlChannel(ii) for ii in range(1)]
        self.measures = [MeasureChannel(ii) for ii in range(2)]

    def test_construction_from_channels(self):
        """Test construction of device topology from channels."""
        topology = SystemTopology(drives=self.drives,
                                  controls=self.controls,
                                  measures=self.measures,
                                  acquires=self.acquires)

        self.assertEqual(topology.qubits[0].drive, DriveChannel(0))
        self.assertEqual(topology.qubits[0].measure, MeasureChannel(0))
        self.assertEqual(topology.qubits[0].acquire, AcquireChannel(0))
        self.assertEqual(topology.qubits[0].controls[0], ControlChannel(0))

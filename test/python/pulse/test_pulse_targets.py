# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test pulse logical elements and frames"""

from qiskit.pulse import (
    PulseError,
    Qubit,
    Coupler,
    Port,
)
from qiskit.test import QiskitTestCase


class TestLogicalElements(QiskitTestCase):
    """Test logical elements."""

    def test_qubit_initialization(self):
        """Test that Qubit type logical elements are created and validated correctly"""
        qubit = Qubit(0)
        self.assertEqual(qubit.index, (0,))
        self.assertEqual(qubit.qubit_index, 0)
        self.assertEqual(str(qubit), "Qubit(0)")

        with self.assertRaises(PulseError):
            Qubit(0.5)
        with self.assertRaises(PulseError):
            Qubit(-0.5)
        with self.assertRaises(PulseError):
            Qubit(-1)

    def test_coupler_initialization(self):
        """Test that Coupler type logical elements are created and validated correctly"""
        coupler = Coupler(0, 3)
        self.assertEqual(coupler.index, (0, 3))
        self.assertEqual(str(coupler), "Coupler(0, 3)")

        coupler = Coupler(0, 3, 2)
        self.assertEqual(coupler.index, (0, 3, 2))

        with self.assertRaises(PulseError):
            Coupler(-1, 0)
        with self.assertRaises(PulseError):
            Coupler(2, -0.5)
        with self.assertRaises(PulseError):
            Coupler(3, -1)
        with self.assertRaises(PulseError):
            Coupler(0, 0, 1)
        with self.assertRaises(PulseError):
            Coupler(0)

    def test_logical_elements_comparison(self):
        """Test the comparison of various logical elements"""
        self.assertEqual(Qubit(0), Qubit(0))
        self.assertNotEqual(Qubit(0), Qubit(1))

        self.assertEqual(Coupler(0, 1), Coupler(0, 1))
        self.assertNotEqual(Coupler(0, 1), Coupler(0, 2))


class TestPorts(QiskitTestCase):
    """Test ports."""

    def test_ports_initialization(self):
        """Test that Ports are created correctly"""
        port = Port("d0")
        self.assertEqual(port.name, "d0")

    def test_ports_comparison(self):
        """Test that Ports are compared correctly"""
        port1 = Port("d0")
        port2 = Port("d0")
        port3 = Port("d1")
        self.assertEqual(port1, port2)
        self.assertNotEqual(port1, port3)

    def test_ports_representation(self):
        """Test Ports repr"""
        port1 = Port("d0")
        self.assertEqual(str(port1), "Port(d0)")

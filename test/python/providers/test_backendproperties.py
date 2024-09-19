# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""TestCase for testing backend properties."""

import copy

from qiskit.providers.fake_provider import Fake5QV1
from qiskit.providers.exceptions import BackendPropertyError
from test import QiskitTestCase  # pylint: disable=wrong-import-order


class BackendpropertiesTestCase(QiskitTestCase):
    """Test usability methods of backend.properties()."""

    # TODO the full file can be removed once BackendV1 is removed, since it is the
    #  only one with backend.properties()

    def setUp(self):
        super().setUp()
        with self.assertWarns(DeprecationWarning):
            self.backend = Fake5QV1()
        self.properties = self.backend.properties()
        self.ref_gate = next(
            g for g in self.backend.configuration().basis_gates if g not in ["id", "rz"]
        )

    def test_gate_property(self):
        """Test for getting the gate properties."""
        self.assertEqual(
            self.properties.gate_property("cx", (0, 1), "gate_error"),
            self.properties._gates["cx"][(0, 1)]["gate_error"],
        )
        self.assertEqual(self.properties.gate_property("cx"), self.properties._gates["cx"])

        with self.assertRaises(BackendPropertyError):
            self.properties.gate_property(self.ref_gate, None, "gate_error")

    def test_gate_error(self):
        """Test for getting the gate errors."""
        self.assertEqual(
            self.properties.gate_error(self.ref_gate, 1),
            self.properties._gates[self.ref_gate][(1,)]["gate_error"][0],
        )
        self.assertEqual(
            self.properties.gate_error(
                self.ref_gate,
                [
                    2,
                ],
            ),
            self.properties._gates[self.ref_gate][(2,)]["gate_error"][0],
        )
        self.assertEqual(
            self.properties.gate_error("cx", [0, 1]),
            self.properties._gates["cx"][(0, 1)]["gate_error"][0],
        )

        with self.assertRaises(BackendPropertyError):
            self.properties.gate_error("cx", 0)

    def test_gate_length(self):
        """Test for getting the gate duration."""
        self.assertEqual(
            self.properties.gate_length(self.ref_gate, 1),
            self.properties._gates[self.ref_gate][(1,)]["gate_length"][0],
        )
        self.assertEqual(
            self.properties.gate_length("cx", [4, 3]),
            self.properties._gates["cx"][(4, 3)]["gate_length"][0],
        )

    def test_qubit_property(self):
        """Test for getting the qubit properties."""
        self.assertEqual(self.properties.qubit_property(0, "T1"), self.properties._qubits[0]["T1"])
        self.assertEqual(
            self.properties.qubit_property(0, "frequency"), self.properties._qubits[0]["frequency"]
        )
        self.assertEqual(self.properties.qubit_property(0), self.properties._qubits[0])

        with self.assertRaises(BackendPropertyError):
            self.properties.qubit_property("T1")

    def test_t1(self):
        """Test for getting the t1 of given qubit."""
        self.assertEqual(self.properties.t1(0), self.properties._qubits[0]["T1"][0])

    def test_t2(self):
        """Test for getting the t2 of a given qubit"""
        self.assertEqual(self.properties.t2(0), self.properties._qubits[0]["T2"][0])

    def test_frequency(self):
        """Test for getting the frequency of given qubit."""
        self.assertEqual(self.properties.frequency(0), self.properties._qubits[0]["frequency"][0])

    def test_readout_error(self):
        """Test for getting the readout error of given qubit."""
        self.assertEqual(
            self.properties.readout_error(0), self.properties._qubits[0]["readout_error"][0]
        )

    def test_readout_length(self):
        """Test for getting the readout length of given qubit."""
        self.assertEqual(
            self.properties.readout_length(0), self.properties._qubits[0]["readout_length"][0]
        )

    def test_apply_prefix(self):
        """Testing unit conversions."""
        self.assertEqual(
            self.properties._apply_prefix(71.9500421005539, "µs"), 7.19500421005539e-05
        )
        self.assertEqual(self.properties._apply_prefix(71.9500421005539, "ms"), 0.0719500421005539)

        with self.assertRaises(BackendPropertyError):
            self.properties._apply_prefix(71.9500421005539, "ws")

    def test_operational(self):
        """Test operation status of a given qubit."""
        self.assertTrue(self.properties.is_qubit_operational(0))

    def test_deepcopy(self):
        """Test that deepcopy creates an identical object."""
        copy_prop = copy.deepcopy(self.properties)
        self.assertEqual(copy_prop, self.properties)

# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test of fake backend generation."""

from qiskit.test import QiskitTestCase
from qiskit.test.mock.utils.fake_backend_builder import FakeBackendBuilder


class FakeBackendBuilderTest(QiskitTestCase):
    """Fake backend builder test."""

    def test_default_parameters(self):
        """Test default parameters."""
        fake_backend = FakeBackendBuilder("FakeTashkent", n_qubits=10).build()

        properties = fake_backend.properties()
        self.assertEqual(len(properties.qubits), 10)
        self.assertEqual(properties.backend_version, "0.0.0")
        self.assertEqual(properties.backend_name, "FakeTashkent")

        configuration = fake_backend.configuration()
        self.assertEqual(configuration.backend_version, "0.0.0")
        self.assertEqual(configuration.backend_name, "FakeTashkent")
        self.assertEqual(configuration.n_qubits, 10)
        self.assertEqual(configuration.basis_gates, ['id', 'u1', 'u2', 'u3', 'cx'])
        self.assertTrue(configuration.local)
        self.assertTrue(configuration.open_pulse)

    def test_set_parameters(self):
        """Test parameters setting."""
        for n_qubits in range(10, 100, 30):
            with self.subTest(n_qubits=n_qubits):
                fake_backend = FakeBackendBuilder("Tashkent",
                                                  n_qubits=n_qubits,
                                                  version="0.0.1",
                                                  basis_gates=['u1'],
                                                  qubit_t1=99.,
                                                  qubit_t2=146.,
                                                  qubit_frequency=5.,
                                                  qubit_readout_error=0.01,
                                                  single_qubit_gates=['u1']).build()

                properties = fake_backend.properties()
                self.assertEqual(properties.backend_version, "0.0.1")
                self.assertEqual(properties.backend_name, "Tashkent")
                self.assertEqual(len(properties.qubits), n_qubits)
                self.assertEqual(len(properties.gates), n_qubits)

                configuration = fake_backend.configuration()
                self.assertEqual(configuration.backend_version, "0.0.1")
                self.assertEqual(configuration.backend_name, "Tashkent")
                self.assertEqual(configuration.n_qubits, n_qubits)
                self.assertEqual(configuration.basis_gates, ['u1'])

    def test_gates(self):
        """Test generated gates."""
        fake_backend = FakeBackendBuilder("Tashkent", n_qubits=4).build()
        properties = fake_backend.properties()

        self.assertEqual(len(properties.gates), 22)

        fake_backend = FakeBackendBuilder("Tashkent", n_qubits=4,
                                          basis_gates=['u1', 'u2', 'cx']).build()
        properties = fake_backend.properties()

        self.assertEqual(len(properties.gates), 14)
        self.assertEqual(len([g for g in properties.gates if g.gate == 'cx']), 6)

    def test_coupling_map_generation(self):
        """Test generation of default coupling map."""
        fake_backend = FakeBackendBuilder("Tashkent", n_qubits=10).build()
        cmap = fake_backend.configuration().coupling_map
        target = [
            [0, 1], [1, 2], [2, 3], [0, 4], [2, 6],
            [4, 5], [5, 6], [6, 7], [5, 9], [8, 9]
        ]
        for couple in cmap:
            with self.subTest(coupling=couple):
                self.assertTrue(couple in target)

        self.assertEqual(len(target), len(cmap))

    def test_configuration(self):
        """Test backend configuration."""
        # TODO: implement

    def test_defaults(self):
        """Test backend defaults."""
        # TODO: implement

    def test_with_coupling_map(self):
        """Test backend generation with coupling map."""
        # TODO: implement

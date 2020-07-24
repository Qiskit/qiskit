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

"""Test of configurable backend generation."""
from qiskit.test import QiskitTestCase
from qiskit.test.mock.utils import ConfigurableFakeBackend


class TestConfigurableFakeBackend(QiskitTestCase):
    """Configurable backend test."""

    def test_default_parameters(self):
        """Test default parameters."""
        fake_backend = ConfigurableFakeBackend("Tashkent", n_qubits=10)

        properties = fake_backend.properties()
        self.assertEqual(len(properties.qubits), 10)
        self.assertEqual(properties.backend_version, "0.0.0")
        self.assertEqual(properties.backend_name, "Tashkent")

        configuration = fake_backend.configuration()
        self.assertEqual(configuration.backend_version, "0.0.0")
        self.assertEqual(configuration.backend_name, "Tashkent")
        self.assertEqual(configuration.n_qubits, 10)
        self.assertEqual(configuration.basis_gates, ['id', 'u1', 'u2', 'u3', 'cx'])
        self.assertTrue(configuration.local)
        self.assertTrue(configuration.open_pulse)

    def test_set_parameters(self):
        """Test parameters setting."""
        for n_qubits in range(10, 100, 30):
            with self.subTest(n_qubits=n_qubits):
                fake_backend = ConfigurableFakeBackend("Tashkent",
                                                       n_qubits=n_qubits,
                                                       version="0.0.1",
                                                       basis_gates=['u1'],
                                                       qubit_t1=99.,
                                                       qubit_t2=146.,
                                                       qubit_frequency=5.,
                                                       qubit_readout_error=0.01,
                                                       single_qubit_gates=['u1'])

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
        fake_backend = ConfigurableFakeBackend("Tashkent", n_qubits=4)
        properties = fake_backend.properties()

        self.assertEqual(len(properties.gates), 22)

        fake_backend = ConfigurableFakeBackend("Tashkent", n_qubits=4,
                                               basis_gates=['u1', 'u2', 'cx'])
        properties = fake_backend.properties()

        self.assertEqual(len(properties.gates), 14)
        self.assertEqual(len([g for g in properties.gates if g.gate == 'cx']), 6)

    def test_coupling_map_generation(self):
        """Test generation of default coupling map."""
        fake_backend = ConfigurableFakeBackend("Tashkent", n_qubits=10)
        cmap = fake_backend.configuration().coupling_map
        target = [
            [0, 1], [0, 4], [1, 2], [1, 5],
            [2, 3], [2, 6], [3, 7], [4, 5],
            [4, 8], [5, 6], [5, 9], [6, 7],
            [8, 9]
        ]
        for couple in cmap:
            with self.subTest(coupling=couple):
                self.assertTrue(couple in target)

        self.assertEqual(len(target), len(cmap))

    def test_configuration(self):
        """Test backend configuration."""
        fake_backend = ConfigurableFakeBackend("Tashkent", n_qubits=10)
        configuration = fake_backend.configuration()

        self.assertEqual(configuration.n_qubits, 10)
        self.assertEqual(configuration.meas_map, [list(range(10))])
        self.assertEqual(len(configuration.hamiltonian['qub']), 10)
        self.assertEqual(len(configuration.hamiltonian['vars']), 33)
        self.assertEqual(len(configuration.u_channel_lo), 13)
        self.assertEqual(len(configuration.meas_lo_range), 10)
        self.assertEqual(len(configuration.qubit_lo_range), 10)

    def test_defaults(self):
        """Test backend defaults."""
        fake_backend = ConfigurableFakeBackend("Tashkent", n_qubits=10)
        defaults = fake_backend.defaults()

        self.assertEqual(len(defaults.cmd_def), 54)
        self.assertEqual(len(defaults.meas_freq_est), 10)
        self.assertEqual(len(defaults.qubit_freq_est), 10)

    def test_with_coupling_map(self):
        """Test backend generation with coupling map."""
        target_coupling_map = [[0, 1], [1, 2], [2, 3]]
        fake_backend = ConfigurableFakeBackend("Tashkent", n_qubits=4,
                                               coupling_map=target_coupling_map)
        cmd_def = fake_backend.defaults().cmd_def
        configured_cmap = fake_backend.configuration().coupling_map
        controlled_not_qubits = [cmd.qubits for cmd in cmd_def if cmd.name == 'cx']

        self.assertEqual(controlled_not_qubits, target_coupling_map)
        self.assertEqual(configured_cmap, target_coupling_map)

# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""Qobj tests."""
import os
import json

from qiskit.test import QiskitTestCase, Path
from qiskit.providers.models import (BackendConfiguration, QASMBackendConfiguration,
                                     PulseBackendConfiguration)


class TestBackendConfiguration(QiskitTestCase):
    """Tests for TestQASMBackendConfiguration."""

    def setUp(self):
        self.examples_base_path = self._get_resource_path('examples',
                                                          Path.SCHEMAS)

    def test_qasm_config(self):
        """Test QASM backend configuration."""
        with open(os.path.join(self.examples_base_path,
                               'backend_configuration_openqasm_example.json'),
                  'r') as example_file:
            qasm_config = json.load(example_file)
        qasm_config_model = BackendConfiguration.from_dict(qasm_config)

        self.assertIsInstance(qasm_config_model, QASMBackendConfiguration)

    def test_pulse_config(self):
        """Test PULSE backend configuration."""
        with open(os.path.join(self.examples_base_path,
                               'backend_configuration_openpulse_example.json'),
                  'r') as example_file:
            pulse_config = json.load(example_file)
        pulse_config_model = BackendConfiguration.from_dict(pulse_config)

        self.assertIsInstance(pulse_config_model, PulseBackendConfiguration)

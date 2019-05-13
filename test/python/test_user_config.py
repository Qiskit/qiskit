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

# pylint: disable=missing-docstring

import os
import tempfile
import unittest

from qiskit import exceptions
from qiskit.test import QiskitTestCase
from qiskit import user_config


class TestUserConfig(QiskitTestCase):

    def test_empty_file_read(self):
        file_path = tempfile.NamedTemporaryFile()
        self.addCleanup(file_path.close)
        config = user_config.UserConfig(file_path.name)
        config.read_config_file()
        self.assertEqual({}, config.settings)

    @unittest.skipIf(os.name == 'nt', 'tempfile fails on appveyor')
    def test_invalid_circuit_drawer(self):
        test_config = """
        [default]
        circuit_drawer = MSPaint
        circuit_mpl_style = default
        """
        file_path = tempfile.NamedTemporaryFile(mode='w')
        self.addCleanup(file_path.close)
        file_path.write(test_config)
        file_path.flush()
        config = user_config.UserConfig(file_path.name)
        self.assertRaises(exceptions.QiskitUserConfigError,
                          config.read_config_file)

    @unittest.skipIf(os.name == 'nt', 'tempfile fails on appveyor')
    def test_circuit_drawer_valid(self):
        test_config = """
        [default]
        circuit_drawer = latex
        circuit_mpl_style = default
        """
        file_path = tempfile.NamedTemporaryFile(mode='w')
        self.addCleanup(file_path.close)
        file_path.write(test_config)
        file_path.flush()
        config = user_config.UserConfig(file_path.name)
        config.read_config_file()
        self.assertEqual({'circuit_drawer': 'latex',
                          'circuit_mpl_style': 'default'}, config.settings)

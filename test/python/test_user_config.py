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

from qiskit import exceptions
from qiskit.test import QiskitTestCase
from qiskit import user_config


class TestUserConfig(QiskitTestCase):

    @classmethod
    def setUpClass(cls):
        cls.file_path = 'temp.txt'

    def test_empty_file_read(self):
        config = user_config.UserConfig(self.file_path)
        config.read_config_file()
        self.assertEqual({},
                         config.settings)

    def test_invalid_optimization_level(self):
        test_config = """
        [default]
        transpile_optimization_level = 76
        """
        self.addCleanup(os.remove, self.file_path)
        with open(self.file_path, 'w') as file:
            file.write(test_config)
            file.flush()
            config = user_config.UserConfig(self.file_path)
            self.assertRaises(exceptions.QiskitUserConfigError,
                              config.read_config_file)

    def test_invalid_circuit_drawer(self):
        test_config = """
        [default]
        circuit_drawer = MSPaint
        """
        self.addCleanup(os.remove, self.file_path)
        with open(self.file_path, 'w') as file:
            file.write(test_config)
            file.flush()
            config = user_config.UserConfig(self.file_path)
            self.assertRaises(exceptions.QiskitUserConfigError,
                              config.read_config_file)

    def test_circuit_drawer_valid(self):
        test_config = """
        [default]
        circuit_drawer = latex
        """
        self.addCleanup(os.remove, self.file_path)
        with open(self.file_path, 'w') as file:
            file.write(test_config)
            file.flush()
            config = user_config.UserConfig(self.file_path)
            config.read_config_file()
            self.assertEqual({'circuit_drawer': 'latex'},
                             config.settings)

    def test_optimization_level_valid(self):
        test_config = """
        [default]
        transpile_optimization_level = 1
        """
        self.addCleanup(os.remove, self.file_path)
        with open(self.file_path, 'w') as file:
            file.write(test_config)
            file.flush()
            config = user_config.UserConfig(self.file_path)
            config.read_config_file()
            self.assertEqual(
                {'transpile_optimization_level': 1},
                config.settings)

    def test_invalid_num_processes(self):
        test_config = """
        [default]
        num_processes = -256
        """
        self.addCleanup(os.remove, self.file_path)
        with open(self.file_path, 'w') as file:
            file.write(test_config)
            file.flush()
            config = user_config.UserConfig(self.file_path)
            self.assertRaises(exceptions.QiskitUserConfigError,
                              config.read_config_file)

    def test_valid_num_processes(self):
        test_config = """
        [default]
        num_processes = 31
        """
        self.addCleanup(os.remove, self.file_path)
        with open(self.file_path, 'w') as file:
            file.write(test_config)
            file.flush()
            config = user_config.UserConfig(self.file_path)
            config.read_config_file()
            self.assertEqual(
                {'num_processes': 31},
                config.settings)

    def test_valid_parallel(self):
        test_config = """
        [default]
        parallel = False
        """
        self.addCleanup(os.remove, self.file_path)
        with open(self.file_path, 'w') as file:
            file.write(test_config)
            file.flush()
            config = user_config.UserConfig(self.file_path)
            config.read_config_file()
            self.assertEqual(
                {'parallel_enabled': False},
                config.settings)

    def test_all_options_valid(self):
        test_config = """
        [default]
        circuit_drawer = latex
        circuit_mpl_style = default
        circuit_mpl_style_path = ~:~/.qiskit
        transpile_optimization_level = 3
        suppress_packaging_warnings = true
        parallel = false
        num_processes = 15
        """
        self.addCleanup(os.remove, self.file_path)
        with open(self.file_path, 'w') as file:
            file.write(test_config)
            file.flush()
            config = user_config.UserConfig(self.file_path)
            config.read_config_file()
            self.assertEqual({'circuit_drawer': 'latex',
                              'circuit_mpl_style': 'default',
                              'circuit_mpl_style_path': ['~', '~/.qiskit'],
                              'transpile_optimization_level': 3,
                              'num_processes': 15,
                              'parallel_enabled': False},
                             config.settings)

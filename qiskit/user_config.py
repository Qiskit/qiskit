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

"""Utils for reading a user preference config files."""

import configparser
import os

from qiskit import exceptions

DEFAULT_FILENAME = os.path.join(os.path.expanduser("~"),
                                '.qiskit', 'settings.conf')


class UserConfig:
    """Class representing a user config file

    The config file format should look like:

    [default]
    circuit_drawer = mpl
    circuit_mpl_style = default

    """
    def __init__(self, filename=None):
        """Create a UserConfig

        Args:
            filename (str): The path to the user config file. If one isn't
                specified ~/.qiskit/settings.conf is used.
        """
        if filename is None:
            self.filename = DEFAULT_FILENAME
        else:
            self.filename = filename
        self.settings = {}
        self.config_parser = configparser.ConfigParser()

    def read_config_file(self):
        """Read config file and parse the contents into the settings attr."""
        if not os.path.isfile(self.filename):
            return
        self.config_parser.read(self.filename)
        if 'default' in self.config_parser.sections():
            circuit_drawer = self.config_parser.get('default',
                                                    'circuit_drawer')
            if circuit_drawer:
                if circuit_drawer not in ['text', 'mpl', 'latex',
                                          'latex_source']:
                    raise exceptions.QiskitUserConfigError(
                        "%s is not a valid circuit drawer backend. Must be "
                        "either 'text', 'mpl', 'latex', or 'latex_source'"
                        % circuit_drawer)
                self.settings['circuit_drawer'] = circuit_drawer

            circuit_mpl_style = self.config_parser.get('default',
                                                       'circuit_mpl_style')
            if circuit_mpl_style:
                if circuit_mpl_style not in ['default', 'bw']:
                    raise exceptions.QiskitUserConfigError(
                        "%s is not a valid mpl circuit style. Must be "
                        "either 'default' or 'bw'"
                        % circuit_mpl_style)
                self.settings['circuit_mpl_style'] = circuit_mpl_style


def get_config():
    """Read the config file from the default location or env var

    It will read a config file at either the default location
    ~/.qiskit/settings.conf or if set the value of the QISKIT_SETTINGS env var.

    It will return the parsed settings dict from the parsed config file.
    Returns:
        dict: The settings dict from the parsed config file.
    """
    filename = os.getenv('QISKIT_SETTINGS', DEFAULT_FILENAME)
    if not os.path.isfile(filename):
        return {}
    user_config = UserConfig(filename)
    user_config.read_config_file()
    return user_config.settings

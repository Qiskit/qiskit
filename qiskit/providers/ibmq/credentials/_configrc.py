# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Utilities for reading and writing credentials from and to configuration files.
"""

import warnings
import os
from ast import literal_eval
from collections import OrderedDict
from configparser import ConfigParser, ParsingError

from qiskit import QiskitError
from .credentials import Credentials

DEFAULT_QISKITRC_FILE = os.path.join(os.path.expanduser("~"),
                                     '.qiskit', 'qiskitrc')


def read_credentials_from_qiskitrc(filename=None):
    """
    Read a configuration file and return a dict with its sections.

    Args:
        filename (str): full path to the qiskitrc file. If `None`, the default
            location is used (`HOME/.qiskit/qiskitrc`).

    Returns:
        dict: dictionary with the contents of the configuration file, with
            the form::

            {credential_unique_id: Credentials}

    Raises:
        QiskitError: if the file was not parseable. Please note that this
            exception is not raised if the file does not exist (instead, an
            empty dict is returned).
    """
    filename = filename or DEFAULT_QISKITRC_FILE
    config_parser = ConfigParser()
    try:
        config_parser.read(filename)
    except ParsingError as ex:
        raise QiskitError(str(ex))

    # Build the credentials dictionary.
    credentials_dict = OrderedDict()
    for name in config_parser.sections():
        single_credentials = dict(config_parser.items(name))
        # Individually convert keys to their right types.
        # TODO: consider generalizing, moving to json configuration or a more
        # robust alternative.
        if 'proxies' in single_credentials.keys():
            single_credentials['proxies'] = literal_eval(
                single_credentials['proxies'])
        if 'verify' in single_credentials.keys():
            single_credentials['verify'] = bool(single_credentials['verify'])
        new_credentials = Credentials(**single_credentials)
        credentials_dict[new_credentials.unique_id()] = new_credentials

    return credentials_dict


def write_qiskit_rc(credentials, filename=None):
    """
    Write credentials to the configuration file.

    Args:
        credentials (dict): dictionary with the credentials, with the form::

            {credentials_unique_id: Credentials}

        filename (str): full path to the qiskitrc file. If `None`, the default
            location is used (`HOME/.qiskit/qiskitrc`).
    """
    def _credentials_object_to_dict(obj):
        return {key: getattr(obj, key)
                for key in ['token', 'url', 'proxies', 'verify']
                if getattr(obj, key)}

    def _section_name(credentials_):
        """Return a string suitable for use as a unique section name."""
        base_name = 'ibmq'
        if credentials_.is_ibmq():
            base_name = '{}_{}_{}_{}'.format(base_name, *credentials_.unique_id())
        return base_name

    filename = filename or DEFAULT_QISKITRC_FILE
    # Create the directories and the file if not found.
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    unrolled_credentials = {
        _section_name(credentials_object): _credentials_object_to_dict(credentials_object)
        for _, credentials_object in credentials.items()
    }

    # Write the configuration file.
    with open(filename, 'w') as config_file:
        config_parser = ConfigParser()
        config_parser.read_dict(unrolled_credentials)
        config_parser.write(config_file)


def store_credentials(credentials, overwrite=False, filename=None):
    """
    Store the credentials for a single account in the configuration file.

    Args:
        credentials (Credentials): credentials instance.
        overwrite (bool): overwrite existing credentials.
        filename (str): full path to the qiskitrc file. If `None`, the default
            location is used (`HOME/.qiskit/qiskitrc`).

    Raises:
        QiskitError: if the account_name could not be assigned.
    """
    # Read the current providers stored in the configuration file.
    filename = filename or DEFAULT_QISKITRC_FILE
    stored_credentials = read_credentials_from_qiskitrc(filename)

    # Check if duplicated credentials are already stored. By convention,
    # we assume (hub, group, project) is always unique.
    if credentials.unique_id() in stored_credentials and not overwrite:
        warnings.warn('Credentials already present. Set overwrite=True to overwrite.')
        return

    # Append and write the credentials to file.
    stored_credentials[credentials.unique_id()] = credentials
    write_qiskit_rc(stored_credentials, filename)


def remove_credentials(credentials, filename=None):
    """Remove credentials from qiskitrc.

    Args:
        credentials (Credentials): credentials.
        filename (str): full path to the qiskitrc file. If `None`, the default
            location is used (`HOME/.qiskit/qiskitrc`).

    Raises:
        QiskitError: If there is no account with that name on the configuration
            file.
    """
    # Set the name of the Provider from the class.
    stored_credentials = read_credentials_from_qiskitrc(filename)

    try:
        del stored_credentials[credentials.unique_id()]
    except KeyError:
        raise QiskitError('The account "%s" does not exist in the '
                          'configuration file')
    write_qiskit_rc(stored_credentials, filename)

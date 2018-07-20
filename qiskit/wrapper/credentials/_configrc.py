# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Utilities for reading and writing credentials from and to configuration files.
"""

import os
from ast import literal_eval
from configparser import ConfigParser, ParsingError

from qiskit import QISKitError
from qiskit.backends.ibmq import IBMQProvider
from ._utils import get_account_name


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

            {'provider_class_name': {'token': 'TOKEN', 'url': 'URL', ... }}

    Raises:
        QISKitError: if the file was not parseable. Please note that this
            exception is not raised if the file does not exist (instead, an
            empty dict is returned).
    """
    filename = filename or DEFAULT_QISKITRC_FILE
    config_parser = ConfigParser()
    try:
        config_parser.read(filename)
    except ParsingError as ex:
        raise QISKitError(str(ex))

    # Build the credentials dictionary.
    credentials_dict = {}
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
        credentials_dict[name] = single_credentials

    return credentials_dict


def write_qiskit_rc(credentials, filename=None):
    """
    Write credentials to the configuration file.

    Args:
        credentials (dict): dictionary with the credentials, with the form::
            {'provider_class_name': {'token': 'TOKEN', 'url': 'URL', ... }}
        filename (str): full path to the qiskitrc file. If `None`, the default
            location is used (`HOME/.qiskit/qiskitrc`).
    """
    filename = filename or DEFAULT_QISKITRC_FILE
    # Create the directories and the file if not found.
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Write the configuration file.
    with open(filename, 'w') as config_file:
        config_parser = ConfigParser()
        config_parser.read_dict(credentials)
        config_parser.write(config_file)


def store_credentials(provider_class=IBMQProvider, overwrite=False,
                      filename=None, **kwargs):
    """
    Store the credentials for a single provider in the configuration file.

    Args:
        provider_class (class): class of the Provider for the credentials.
        overwrite (bool): overwrite existing credentials.
        filename (str): full path to the qiskitrc file. If `None`, the default
            location is used (`HOME/.qiskit/qiskitrc`).
        kwargs (dict): keyword arguments passed to provider class
            initialization.

    Raises:
        QISKitError: If provider already exists and overwrite=False; or if
            the account_name could not be assigned.
    """
    # Set the name of the Provider from the class.
    account_name = get_account_name(provider_class)
    # Read the current providers stored in the configuration file.
    filename = filename or DEFAULT_QISKITRC_FILE
    credentials = read_credentials_from_qiskitrc(filename)
    if account_name in credentials.keys() and not overwrite:
        raise QISKitError('%s is already present and overwrite=False'
                          % account_name)

    # Append the provider, trim the empty options and store it in the file.
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    credentials[account_name] = {**kwargs}
    write_qiskit_rc(credentials, filename)


def remove_credentials(provider_class=IBMQProvider, filename=None):
    """Remove provider credentials from qiskitrc.

    Args:
        provider_class (class): class of the Provider for the credentials.
        filename (str): full path to the qiskitrc file. If `None`, the default
            location is used (`HOME/.qiskit/qiskitrc`).

    Raises:
        QISKitError: If there is no account with that name on the configuration
            file.
    """
    # Set the name of the Provider from the class.
    account_name = get_account_name(provider_class)
    credentials = read_credentials_from_qiskitrc(filename)

    try:
        credentials.pop(account_name)
    except KeyError:
        raise QISKitError('The account "%s" does not exist in the '
                          'configuration file')
    write_qiskit_rc(credentials, filename)

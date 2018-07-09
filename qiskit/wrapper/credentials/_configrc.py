# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Utilities for reading and writing credentials from and to configuration files.
"""

import os
from configparser import ConfigParser, ParsingError
from qiskit import QISKitError


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

            {'provider_name': {'token': 'TOKEN', 'url': 'URL', ... }}

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

    return {name: dict(config_parser.items(name)) for
            name in config_parser.sections()}


def write_qiskit_rc(credentials, filename=None):
    """
    Write credentials to the configuration file.

    Args:
        credentials (dict): dictionary with the credentials, with the form::
            {'provider_name': {'token': 'TOKEN', 'url': 'URL', ... }}
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


def store_credentials(token=None,
                      url='https://quantumexperience.ng.bluemix.net/api',
                      hub=None, group=None, project=None, proxies=None,
                      verify=True, account_name=None, overwrite=False,
                      filename=None):
    """
    Store the credentials for a single provider in the configuration file.

    Args:
        token (str): the token used to register on the online backend such
            as the quantum experience.
        url (str): the url used for online backend such as the quantum
            experience.
        hub (str): the hub used for online backend.
        group (str): the group used for online backend.
        project (str): the project used for online backend.
        proxies (dict): proxy configuration for the API, as a dict with
            'urls' and credential keys.
        verify (bool): if False, ignores SSL certificates errors.
        account_name (str): name for the account in the configuration file
            section.
        overwrite (bool): overwrite existing credentials.
        filename (str): full path to the qiskitrc file. If `None`, the default
            location is used (`HOME/.qiskit/qiskitrc`).

    Raises:
        QISKitError: If provider already exists and overwrite=False; or if
            the account_name could not be assigned.
    """
    # Assign a default name for the credentials section.
    if not account_name:
        if 'quantumexperience' in url:
            account_name = 'ibmq'
        elif 'q-console' in url:
            account_name = 'qnet'
        else:
            raise QISKitError('Cannot parse provider name from credentials.')

    # Read the current providers stored in the configuration file.
    filename = filename or DEFAULT_QISKITRC_FILE
    credentials = read_credentials_from_qiskitrc(filename)
    if account_name in credentials.keys() and not overwrite:
        raise QISKitError('%s is already present and overwrite=False'
                          % account_name)

    # Append the provider, and store it in the file.
    credentials[account_name] = {
        'token': token, 'url': url, 'hub': hub, 'group': group,
        'project': project, 'proxies': proxies, 'verify': verify}
    write_qiskit_rc(credentials, filename)


def remove_credentials(account_name, filename=None):
    """Remove provider credentials from qiskitrc.

     Args:
        account_name (str): Name of the account to be removed.
        filename (str): full path to the qiskitrc file. If `None`, the default
            location is used (`HOME/.qiskit/qiskitrc`).

    Raises:
        QISKitError: If there is no account with that name on the configuration
            file.
    """
    credentials = read_credentials_from_qiskitrc(filename)

    try:
        credentials.pop(account_name)
    except KeyError:
        raise QISKitError('The account "%s" does not exist in the '
                          'configuration file. Available accounts: %s' %
                          (account_name, credentials.keys()))
    write_qiskit_rc(credentials, filename)

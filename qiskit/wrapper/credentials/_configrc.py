# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Utilities for reading and writing credentials from and to configuration files.
"""

import os
import ast
import configparser
from qiskit import QISKitError
from qiskit._util import _parse_ibmq_credentials, _provider_name_from_url
from qiskit.wrapper._register import _register


DEFAULT_QISKITRC_FILE = os.path.join(os.path.expanduser("~"),
                                     '.qiskit', 'qiskitrc')


def has_qiskit_configrc():
    """
    Checks to see if the qikitrc file exists in the default
    location, i.e. HOME/.qiskit/qiskitrc

    Returns:
        bool: True if file exists, False otherwise.
    """
    return os.path.isfile(DEFAULT_QISKITRC_FILE)


def generate_qiskitrc(overwrite=False):
    """
    Generate a blank qiskitrc file.

    Args:
        overwrite (bool): Overwrite existing file. Default is False.

    Returns:
        bool: True if file was written, False otherwise.

    Raises:
        QISKitError:
            Could not write to user home directory.
    """
    # Check for write access to home dir
    if not os.access(os.path.expanduser("~"), os.W_OK):
        raise QISKitError('No write access to home directory.')
    qiskit_conf_dir = os.path.join(os.path.expanduser("~"), '.qiskit')
    if not os.path.exists(qiskit_conf_dir):
        try:
            os.mkdir(qiskit_conf_dir)
        except Exception:
            raise QISKitError(
                'Unable to write QISKit config file to home directory.')
    if os.path.isfile(DEFAULT_QISKITRC_FILE):
        if overwrite:
            os.remove(DEFAULT_QISKITRC_FILE)
        else:
            return False
    # Write a basic file with REGISTRATION section
    cfgfile = open(DEFAULT_QISKITRC_FILE, 'w')
    config = configparser.ConfigParser()
    config.add_section('REGISTRATION')
    config.write(cfgfile)
    cfgfile.close()
    return True


def has_qiskitrc_key(section, key):
    """
    Checks if a given key is already in a given
    section of the configrc.

    Args:
        section (str): Section in which to look.
        key (str): Key of interest.

    Returns:
        bool: True if key exits, False otherwise.

    Raises:
        QISKitError:
            Config file not found.
    """
    out = False
    if DEFAULT_QISKITRC_FILE is None:
        raise QISKitError('QISKit config file not found.')
    config = configparser.ConfigParser()
    config.read(DEFAULT_QISKITRC_FILE)
    if config.has_section(section):
        if key in config.options(section):
            out = True
    else:
        raise QISKitError('Section %s not found in qiskitrc.' % section)
    return out


def write_qiskitrc_key(section, key, value, overwrite=False):
    """
    Writes a single key value to the qiskitrc file.

    Args:
        section (str): Section in which to write.
        key (str): Key to be written.
        value (str): Value to be written
        overwrite (bool): Overwrite key if it exists. Default is False.

    Raises:
        QISKitError:
            Config file not found.
    """
    if not has_qiskit_configrc():
        raise QISKitError('QISKit config file not found.')
    config = configparser.ConfigParser()
    config.read(DEFAULT_QISKITRC_FILE)
    if not config.has_section(section):
        config.add_section(section)
    if has_qiskitrc_key(section, key) and (not overwrite):
        raise QISKitError('%s is already present and overwrite=False' % key)
    elif has_qiskitrc_key(section, key):
        config.remove_option(section, key)
    config.set(section, key, str(value))
    cfgfile = open(DEFAULT_QISKITRC_FILE, 'w')
    config.write(cfgfile)
    cfgfile.close()


def read_qiskitrc_key(section, key):
    """
    Reads a single key value from the qiskitrc file.

    Parameters:
        section (str): The section to search.
        key (str): The key whose value we want.

    Returns:
        value: Value contained in section:key.

    Raises:
        QISKitError:
            Config file not found.
    """
    if not has_qiskit_configrc():
        raise QISKitError('QISKit config file not found.')
    config = configparser.ConfigParser()
    config.read(DEFAULT_QISKITRC_FILE)
    if not config.has_section(section):
        raise QISKitError('Section %s is missing from qiskitrc.' % section)
    if key not in config.options(section):
        raise QISKitError(
            'Key %s is missing in section %s in qiskitrc.' % (key, section))
    # Need to do an eval here in the case value is a dict.
    out = ast.literal_eval(config.get(section, key))
    return out


def store_credentials(token=None,
                      url='https://quantumexperience.ng.bluemix.net/api',
                      hub=None, group=None, project=None, proxies=None,
                      verify=True, overwrite=False):
    """Store provider credentials in local qiskitrc.

    Args:
            token (str): The token used to register on the online backend such
                as the quantum experience.
            url (str): The url used for online backend such as the quantum
                experience.
            hub (str): The hub used for online backend.
            group (str): The group used for online backend.
            project (str): The project used for online backend.
            proxies (dict): Proxy configuration for the API, as a dict with
                'urls' and credential keys.
            verify (bool): If False, ignores SSL certificates errors.
            overwrite (bool): Overwrite existing credentials, default is False.
    Raises:
        QISKitError: If provider already exists and overwrite=False.
    """
    url = _parse_ibmq_credentials(url, hub, group, project)
    provider_name = _provider_name_from_url(url)
    if provider_name is None:
        raise QISKitError('Currently only IBMQ credentials can be stored.')
    pro_dict = {'token': token, 'url': url,
                'proxies': proxies, 'verify': verify}
    if not has_qiskit_configrc():
        generate_qiskitrc()
    if has_qiskitrc_key('REGISTRATION', provider_name) and (not overwrite):
        raise QISKitError('%s is already present and overwrite=False'
                          % provider_name)
    write_qiskitrc_key('REGISTRATION', provider_name,
                       pro_dict, overwrite=True)


def stored_credentials():
    """Returns a list of credentials stored in the
    local qiskitrc file.

    Returns:
        list: List of stored credentials
    Raises:
        QISKitError: qiskitrc file not found.
    """
    if has_qiskit_configrc():
        config = configparser.ConfigParser()
        config.read(DEFAULT_QISKITRC_FILE)
        section = 'REGISTRATION'
        if not config.has_section(section):
            raise QISKitError(
                'Section %s is missing from qiskitrc.' % section)
        else:
            return config.options(section)
    else:
        raise QISKitError('qiskitrc file not found.')


def get_credentials(provider_name=None):
    """Get the provider credentials
    stored in qiskitrc.

    If no provider_name given, returns list
    of provider names in qiskitrc.

    Args:
        provider_name (str): Name of provider.

    Returns:
        list: List of providers in qiskitrc if
            provider_name=None.

    Raises:
        QISKitError: If missing section or qiskitrc file.
    """
    return read_qiskitrc_key('REGISTRATION', provider_name)


def remove_credentials(provider_name):
    """Remove provider credentials
    from qiskitrc.

     Args:
        provider_name (str): Name of provider.

    Raises:
        QISKitError: If missing section and/or provider_name.
    """
    if has_qiskit_configrc():
        config = configparser.ConfigParser()
        config.read(DEFAULT_QISKITRC_FILE)
        section = 'REGISTRATION'
        if not config.has_section(section):
            raise QISKitError(
                'Section %s is missing from qiskitrc.' % section)
        else:
            if provider_name not in config.options(section):
                raise QISKitError(
                    'Key %s is missing in section %s in qiskitrc.'
                    % (provider_name, section))
            else:
                config.remove_option(section, provider_name)
                cfgfile = open(DEFAULT_QISKITRC_FILE, 'w')
                config.write(cfgfile)
                cfgfile.close()
    else:
        raise QISKitError('qiskitrc file not found.')


def qiskitrc_register_providers(specific_provider=None):
    """ Registers providers in qiskitrc

    Args:
        specific_provider (str): Register only this provider, if any.

    Raises:
        QISKitError: if the qiskitrc file cannot be read.
    """
    if not has_qiskit_configrc():
        raise QISKitError('QISKit config file not found.')
    config = configparser.ConfigParser()
    config.read(DEFAULT_QISKITRC_FILE)
    if not config.has_section('REGISTRATION'):
        raise QISKitError("Missing 'REGISTRATION' section in qiskitrc")
    did_register = 0
    for pro in config.options('REGISTRATION'):
        pro_dict = ast.literal_eval(config.get('REGISTRATION', pro))
        if specific_provider is None:
            _register(**pro_dict)
            did_register = 1
        elif pro == specific_provider:
            _register(**pro_dict)
            did_register = 1
            break
    if not did_register and specific_provider is not None:
        raise QISKitError('Provider %s credentials not found.' %
                          specific_provider)


def get_qiskitrc_credentials(provider_name=None):
    """
    Looks for registration information in qiskitrc file.

    Args:
        provider_name (str): Name of provider

    Returns:
        bool: Did registration occur.
    """
    # Look at qiksitrc for saved data
    did_register = 0
    if has_qiskit_configrc():
        qiskitrc_register_providers(specific_provider=provider_name)
        did_register = 1
    return did_register

# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Utilities for working with credentials for the wrapper.
"""
import logging

from qiskit import QISKitError
from ._configrc import read_credentials_from_qiskitrc, store_credentials
from ._environ import read_credentials_from_environ
from ._qconfig import read_credentials_from_qconfig


logger = logging.getLogger(__name__)


def discover_credentials():
    """
    Automatically discover credentials for online providers.

    This method looks for credentials in the following locations, in order,
    and returning as soon as credentials are found::

        1. in the `Qconfig.py` file in the current working directory.
        2. in the environment variables.
        3. in the `qiskitrc` configuration file.

    Returns:
        dict: dictionary with the contents of the configuration file, with
            the form::

            {'provider_name': {'token': 'TOKEN', 'url': 'URL', ... }}
    """
    # 1. Attempt to read them from the `Qconfig.py` file.
    try:
        qconfig_credentials = read_credentials_from_qconfig()
        if qconfig_credentials:
            logger.info('Using credentials from qconfig')
            return qconfig_credentials
    except QISKitError as ex:
        logger.warning(
            'Automatic discovery of qconfig credentials failed: %s', str(ex))

    # 2. Attempt to read them from the environment variables.
    try:
        environ_credentials = read_credentials_from_environ()
        if environ_credentials:
            logger.info('Using credentials from environment variables')
            return environ_credentials
    except QISKitError as ex:
        logger.warning(
            'Automatic discovery of environment credentials failed: %s',
            str(ex))

    # 3. Attempt to read them from the qiskitrc file.
    try:
        provider_credentials = read_credentials_from_qiskitrc()
        if provider_credentials:
            logger.info('Using credentials from qiskitrc')
            return provider_credentials
    except QISKitError as ex:
        logger.warning(
            'Automatic discovery of qiskitrc credentials failed: %s', str(ex))

    return {}

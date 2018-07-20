# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Utilities for reading credentials from environment variables.
"""

import os

from qiskit.backends.ibmq import IBMQProvider
from ._utils import get_account_name

# Dictionary that maps `ENV_VARIABLE_NAME` to credential parameter.
VARIABLES_MAP = {
    'QE_TOKEN': 'token',
    'QE_URL': 'url',
    'QE_HUB': 'hub',
    'QE_GROUP': 'group',
    'QE_PROJECT': 'project'
}


def read_credentials_from_environ():
    """
    Read the environment variables and return its credentials.

    Returns:
        dict: dictionary with the credentials, in the form::

            {'IBMQProvider': {'token': 'TOKEN', 'url': 'URL', ... }}

    """
    # The token is the only required parameter.
    if not os.getenv('QE_TOKEN'):
        return {}

    # Build the credentials based on environment variables.
    credentials = {}
    for envar_name, credential_key in VARIABLES_MAP.items():
        if os.getenv(envar_name):
            credentials[credential_key] = os.getenv(envar_name)

    return {get_account_name(IBMQProvider): credentials}

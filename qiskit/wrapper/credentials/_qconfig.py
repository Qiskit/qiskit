# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Utilities for reading credentials from the deprecated `Qconfig.py` file.
"""

import os
from importlib.util import module_from_spec, spec_from_file_location

from qiskit import QISKitError
from qiskit.backends.ibmq import IBMQProvider
from ._utils import get_account_name


DEFAULT_QCONFIG_FILE = 'Qconfig.py'


def read_credentials_from_qconfig():
    """
    Read a `QConfig.py` file and return its credentials.

    Returns:
        dict: dictionary with the credentials, in the form::

            {'token': 'TOKEN', 'url': 'URL', ... }

    Raises:
        QISKitError: if the Qconfig.py was not parseable. Please note that this
            exception is not raised if the file does not exist (instead, an
            empty dict is returned).
    """
    if not os.path.isfile(DEFAULT_QCONFIG_FILE):
        return {}
    else:
        # Note this is nested inside the else to prevent some tools marking
        # the whole method as deprecated.
        pass
        # TODO: reintroduce when we decide on deprecatin
        # warnings.warn(
        #     "Using 'Qconfig.py' for storing the credentials will be deprecated in"
        #     "upcoming versions (>0.6.0). Using .qiskitrc is recommended",
        #     DeprecationWarning)

    try:
        spec = spec_from_file_location('Qconfig', DEFAULT_QCONFIG_FILE)
        q_config = module_from_spec(spec)
        spec.loader.exec_module(q_config)

        if hasattr(q_config, 'config'):
            credentials = q_config.config.copy()
        else:
            credentials = {}
        credentials['token'] = q_config.APItoken
    except Exception as ex:
        # pylint: disable=broad-except
        raise QISKitError('Error loading Qconfig.py: %s' % str(ex))

    return {get_account_name(IBMQProvider): credentials}

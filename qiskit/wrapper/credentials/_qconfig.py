# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Utilities for reading credentials from the deprecated `Qconfig.py` file.
"""

from importlib.util import spec_from_file_location, module_from_spec
import os

from qiskit import QISKitError


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
    if not os.path.isfile('Qconfig.py'):
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
        spec = spec_from_file_location('Qconfig', 'Qconfig.py')
        q_config = module_from_spec(spec)
        spec.loader.exec_module(q_config)

        credentials = q_config.config.copy()
        credentials['token'] = q_config.APItoken
    except Exception as ex:
        # pylint: disable=broad-except
        raise QISKitError('Error loading Qconfig.py: %s' % str(ex))

    return credentials

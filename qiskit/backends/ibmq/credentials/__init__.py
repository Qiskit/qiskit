# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Utilities for working with credentials for the wrapper.
"""
from collections import OrderedDict
import logging

from qiskit import QiskitError
from .credentials import Credentials
from ._configrc import read_credentials_from_qiskitrc, store_credentials
from ._environ import read_credentials_from_environ
from ._qconfig import read_credentials_from_qconfig

logger = logging.getLogger(__name__)


def discover_credentials(qiskitrc_filename=None):
    """Automatically discover credentials for IBM Q.

    This method looks for credentials in the following locations, in order,
    and returning as soon as credentials are found::

        1. in the `Qconfig.py` file in the current working directory.
        2. in the environment variables.
        3. in the `qiskitrc` configuration file

    Args:
        qiskitrc_filename (str): location for the `qiskitrc` configuration
            file. If `None`, defaults to `{HOME}/.qiskitrc/qiskitrc`.

    Returns:
        dict: dictionary with the contents of the configuration file, with
            the form::

            {credentials_unique_id: Credentials}
    """
    credentials = OrderedDict()

    # dict[str:function] that defines the different locations for looking for
    # credentials, and their precedence order.
    readers = OrderedDict([
        ('qconfig', (read_credentials_from_qconfig, {})),
        ('environment variables', (read_credentials_from_environ, {})),
        ('qiskitrc', (read_credentials_from_qiskitrc,
                      {'filename': qiskitrc_filename}))
    ])

    # Attempt to read the credentials from the different sources.
    for display_name, (reader_function, kwargs) in readers.items():
        try:
            credentials = reader_function(**kwargs)
            logger.info('Using credentials from %s', display_name)
            if credentials:
                break
        except QiskitError as ex:
            logger.warning(
                'Automatic discovery of %s credentials failed: %s',
                display_name, str(ex))

    return credentials

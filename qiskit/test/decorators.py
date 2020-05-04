# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

#  pylint: disable=E1101

"""Decorator for using with Qiskit unit tests."""

import functools
import os
import sys
import unittest
from warnings import warn

from qiskit.util import _has_connection
from .testing_options import get_test_options

HAS_NET_CONNECTION = None


def is_aer_provider_available():
    """Check if the C++ simulator can be instantiated.

    Returns:
        bool: True if simulator executable is available
    """
    # TODO: HACK FROM THE DEPTHS OF DESPAIR AS AER DOES NOT WORK ON MAC
    if sys.platform == 'darwin':
        return False
    try:
        import qiskit.providers.aer  # pylint: disable=unused-import
    except ImportError:
        return False
    return True


def requires_aer_provider(test_item):
    """Decorator that skips test if qiskit aer provider is not available

    Args:
        test_item (callable): function or class to be decorated.

    Returns:
        callable: the decorated function.
    """
    reason = 'Aer provider not found, skipping test'
    return unittest.skipIf(not is_aer_provider_available(), reason)(test_item)


def slow_test(func):
    """Decorator that signals that the test takes minutes to run.

    Args:
        func (callable): test function to be decorated.

    Returns:
        callable: the decorated function.
    """

    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        skip_slow = not TEST_OPTIONS['run_slow']
        if skip_slow:
            raise unittest.SkipTest('Skipping slow tests')

        return func(*args, **kwargs)

    return _wrapper


def _get_credentials():
    """Finds the credentials for a specific test and options.

    Returns:
        Credentials: set of credentials

    Raises:
        SkipTest: when credentials can't be found
    """
    try:
        from qiskit.providers.ibmq.credentials import (Credentials,
                                                       discover_credentials)
    except ImportError:
        raise unittest.SkipTest('qiskit-ibmq-provider could not be found, '
                                'and is required for executing online tests. '
                                'To install, run "pip install qiskit-ibmq-provider" '
                                'or check your installation.')

    if os.getenv('IBMQ_TOKEN') and os.getenv('IBMQ_URL'):
        return Credentials(os.getenv('IBMQ_TOKEN'), os.getenv('IBMQ_URL'))
    elif os.getenv('QISKIT_TESTS_USE_CREDENTIALS_FILE'):
        # Attempt to read the standard credentials.
        discovered_credentials = discover_credentials()

        if discovered_credentials:
            # Decide which credentials to use for testing.
            if len(discovered_credentials) > 1:
                raise unittest.SkipTest(
                    "More than 1 credential set found, use: "
                    "IBMQ_TOKEN and IBMQ_URL env variables to "
                    "set credentials explicitly")

            # Use the first available credentials.
            return list(discovered_credentials.values())[0]
    raise unittest.SkipTest(
        'No IBMQ credentials found for running the test. This is required for '
        'running online tests.')


def requires_qe_access(func):
    """Deprecated in favor of `online_test`"""
    warn("`requires_qe_access` is going to be replaced in favor of `online_test`",
         DeprecationWarning)

    @functools.wraps(func)
    def _wrapper(self, *args, **kwargs):
        if TEST_OPTIONS['skip_online']:
            raise unittest.SkipTest('Skipping online tests')

        credentials = _get_credentials()
        self.using_ibmq_credentials = credentials.is_ibmq()
        kwargs.update({'qe_token': credentials.token,
                       'qe_url': credentials.url})

        return func(self, *args, **kwargs)

    return _wrapper


def online_test(func):
    """Decorator that signals that the test uses the network (and the online API):

    It involves:
        * determines if the test should be skipped by checking environment
            variables.
        * if the `USE_ALTERNATE_ENV_CREDENTIALS` environment variable is
          set, it reads the credentials from an alternative set of environment
          variables.
        * if the test is not skipped, it reads `qe_token` and `qe_url` from
            `Qconfig.py`, environment variables or qiskitrc.
        * if the test is not skipped, it appends `qe_token` and `qe_url` as
            arguments to the test function.

    Args:
        func (callable): test function to be decorated.

    Returns:
        callable: the decorated function.
    """

    @functools.wraps(func)
    def _wrapper(self, *args, **kwargs):
        # To avoid checking the connection in each test
        global HAS_NET_CONNECTION  # pylint: disable=global-statement

        if TEST_OPTIONS['skip_online']:
            raise unittest.SkipTest('Skipping online tests')

        if HAS_NET_CONNECTION is None:
            HAS_NET_CONNECTION = _has_connection('qiskit.org', 443)

        if not HAS_NET_CONNECTION:
            raise unittest.SkipTest("Test requires internet connection.")

        credentials = _get_credentials()
        self.using_ibmq_credentials = credentials.is_ibmq()
        kwargs.update({'qe_token': credentials.token,
                       'qe_url': credentials.url})

        return func(self, *args, **kwargs)

    return _wrapper


TEST_OPTIONS = get_test_options()

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

"""Decorator for using with Qiskit unit tests."""

import functools
import os
import sys
import unittest

from .utils import Path
from .http_recorder import http_recorder
from .testing_options import get_test_options


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


def _get_credentials(test_object, test_options):
    """Finds the credentials for a specific test and options.

    Args:
        test_object (QiskitTestCase): The test object asking for credentials
        test_options (dict): Options after QISKIT_TESTS was parsed by
            get_test_options.

    Returns:
        Credentials: set of credentials

    Raises:
        ImportError: if the
        Exception: when the credential could not be set and they are needed
            for that set of options
    """
    try:
        from qiskit.providers.ibmq.credentials import (Credentials,
                                                       discover_credentials)
    except ImportError:
        raise ImportError('qiskit-ibmq-provider could not be found, and is '
                          'required for mocking or executing online tests.')

    dummy_credentials = Credentials(
        'dummyapiusersloginWithTokenid01',
        'https://quantumexperience.ng.bluemix.net/api')

    if test_options['mock_online']:
        return dummy_credentials

    if os.getenv('USE_ALTERNATE_ENV_CREDENTIALS', ''):
        # Special case: instead of using the standard credentials mechanism,
        # load them from different environment variables. This assumes they
        # will always be in place, as is used by the Travis setup.
        return Credentials(os.getenv('IBMQ_TOKEN'), os.getenv('IBMQ_URL'))
    else:
        # Attempt to read the standard credentials.
        discovered_credentials = discover_credentials()

        if discovered_credentials:
            # Decide which credentials to use for testing.
            if len(discovered_credentials) > 1:
                try:
                    # Attempt to use QE credentials.
                    return discovered_credentials[dummy_credentials.unique_id()]
                except KeyError:
                    pass

            # Use the first available credentials.
            return list(discovered_credentials.values())[0]

    # No user credentials were found.
    if test_options['rec']:
        raise Exception('Could not locate valid credentials. You need them for '
                        'recording tests against the remote API.')

    test_object.log.warning('No user credentials were detected. '
                            'Running with mocked data.')
    test_options['mock_online'] = True
    return dummy_credentials


def requires_qe_access(func):
    """Decorator that signals that the test uses the online API:

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
        if TEST_OPTIONS['skip_online']:
            raise unittest.SkipTest('Skipping online tests')

        credentials = _get_credentials(self, TEST_OPTIONS)
        self.using_ibmq_credentials = credentials.is_ibmq()
        kwargs.update({'qe_token': credentials.token,
                       'qe_url': credentials.url})

        decorated_func = func
        if TEST_OPTIONS['rec'] or TEST_OPTIONS['mock_online']:
            # For recording or for replaying existing cassettes, the test
            # should be decorated with @use_cassette.
            vcr_mode = 'new_episodes' if TEST_OPTIONS['rec'] else 'none'
            decorated_func = http_recorder(
                vcr_mode, Path.CASSETTES.value).use_cassette()(decorated_func)

        return decorated_func(self, *args, **kwargs)

    return _wrapper


TEST_OPTIONS = get_test_options()

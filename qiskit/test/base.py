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

"""Base TestCases for the unit tests.

Implementors of unit tests for Terra are encouraged to subclass
``QiskitTestCase`` in order to take advantage of utility functions (for example,
the environment variables for customizing different options), and the
decorators in the ``decorators`` package.
"""

import inspect
import logging
import os
import unittest
from unittest.util import safe_repr

from .utils import Path, _AssertNoLogsContext, setup_test_logging


__unittest = True  # Allows shorter stack trace for .assertDictAlmostEqual


class QiskitTestCase(unittest.TestCase):
    """Helper class that contains common functionality."""

    @classmethod
    def setUpClass(cls):
        # Determines if the TestCase is using IBMQ credentials.
        cls.using_ibmq_credentials = False

        # Set logging to file and stdout if the LOG_LEVEL envar is set.
        cls.log = logging.getLogger(cls.__name__)
        if os.getenv('LOG_LEVEL'):
            filename = '%s.log' % os.path.splitext(inspect.getfile(cls))[0]
            setup_test_logging(cls.log, os.getenv('LOG_LEVEL'), filename)

    def tearDown(self):
        # Reset the default providers, as in practice they acts as a singleton
        # due to importing the instances from the top-level qiskit namespace.
        from qiskit.providers.basicaer import BasicAer

        BasicAer._backends = BasicAer._verify_backends()

    @staticmethod
    def _get_resource_path(filename, path=Path.TEST):
        """Get the absolute path to a resource.

        Args:
            filename (string): filename or relative path to the resource.
            path (Path): path used as relative to the filename.
        Returns:
            str: the absolute path to the resource.
        """
        return os.path.normpath(os.path.join(path.value, filename))

    def assertNoLogs(self, logger=None, level=None):
        """Assert that no message is sent to the specified logger and level.

        Context manager to test that no message is sent to the specified
        logger and level (the opposite of TestCase.assertLogs()).
        """
        return _AssertNoLogsContext(self, logger, level)

    def assertDictAlmostEqual(self, dict1, dict2, delta=None, msg=None,
                              places=None, default_value=0):
        """Assert two dictionaries with numeric values are almost equal.

        Fail if the two dictionaries are unequal as determined by
        comparing that the difference between values with the same key are
        not greater than delta (default 1e-8), or that difference rounded
        to the given number of decimal places is not zero. If a key in one
        dictionary is not in the other the default_value keyword argument
        will be used for the missing value (default 0). If the two objects
        compare equal then they will automatically compare almost equal.

        Args:
            dict1 (dict): a dictionary.
            dict2 (dict): a dictionary.
            delta (number): threshold for comparison (defaults to 1e-8).
            msg (str): return a custom message on failure.
            places (int): number of decimal places for comparison.
            default_value (number): default value for missing keys.

        Raises:
            TypeError: if the arguments are not valid (both `delta` and
                `places` are specified).
            AssertionError: if the dictionaries are not almost equal.
        """

        error_msg = dicts_almost_equal(dict1, dict2, delta, places, default_value)

        if error_msg:
            msg = self._formatMessage(msg, error_msg)
            raise self.failureException(msg)


def dicts_almost_equal(dict1, dict2, delta=None, places=None, default_value=0):
    """Test if two dictionaries with numeric values are almost equal.

    Fail if the two dictionaries are unequal as determined by
    comparing that the difference between values with the same key are
    not greater than delta (default 1e-8), or that difference rounded
    to the given number of decimal places is not zero. If a key in one
    dictionary is not in the other the default_value keyword argument
    will be used for the missing value (default 0). If the two objects
    compare equal then they will automatically compare almost equal.

    Args:
        dict1 (dict): a dictionary.
        dict2 (dict): a dictionary.
        delta (number): threshold for comparison (defaults to 1e-8).
        places (int): number of decimal places for comparison.
        default_value (number): default value for missing keys.

    Raises:
        TypeError: if the arguments are not valid (both `delta` and
            `places` are specified).

    Returns:
        String: Empty string if dictionaries are almost equal. A description
            of their difference if they are deemed not almost equal.
    """

    def valid_comparison(value):
        """compare value to delta, within places accuracy"""
        if places is not None:
            return round(value, places) == 0
        else:
            return value < delta

    # Check arguments.
    if dict1 == dict2:
        return ''
    if places is not None:
        if delta is not None:
            raise TypeError("specify delta or places not both")
        msg_suffix = ' within %s places' % places
    else:
        delta = delta or 1e-8
        msg_suffix = ' within %s delta' % delta

    # Compare all keys in both dicts, populating error_msg.
    error_msg = ''
    for key in set(dict1.keys()) | set(dict2.keys()):
        val1 = dict1.get(key, default_value)
        val2 = dict2.get(key, default_value)
        if not valid_comparison(abs(val1 - val2)):
            error_msg += '(%s: %s != %s), ' % (safe_repr(key),
                                               safe_repr(val1),
                                               safe_repr(val2))

    if error_msg:
        return error_msg[:-2] + msg_suffix
    else:
        return ''

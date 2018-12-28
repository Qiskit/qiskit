# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

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
        # due to importing the wrapper from qiskit.
        from qiskit.providers.ibmq import IBMQ
        from qiskit.providers.builtinsimulators import BasicAer

        IBMQ._accounts.clear()
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
            TypeError: raises TestCase failureException if the test fails.
        """
        def valid_comparison(value):
            """compare value to delta, within places accuracy"""
            if places is not None:
                return round(value, places) == 0
            else:
                return value < delta

        # Check arguments.
        if dict1 == dict2:
            return
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
            msg = self._formatMessage(msg, error_msg[:-2] + msg_suffix)
            raise self.failureException(msg)

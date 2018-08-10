# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Shared functionality and helpers for the unit tests."""

from enum import Enum
import functools
import inspect
import logging
import os
import time
import unittest
from unittest.util import safe_repr
from qiskit import __path__ as qiskit_path
from qiskit.backends import JobStatus
from qiskit.backends.ibmq import IBMQProvider
from qiskit.backends.local import QasmSimulatorCpp
from qiskit.wrapper.credentials import discover_credentials, get_account_name
from qiskit.wrapper.defaultqiskitprovider import DefaultQISKitProvider


class Path(Enum):
    """Helper with paths commonly used during the tests."""
    # Main SDK path:    qiskit/
    SDK = qiskit_path[0]
    # test.python path: qiskit/test/python/
    TEST = os.path.dirname(__file__)
    # Examples path:    examples/
    EXAMPLES = os.path.join(SDK, '../examples')
    # Schemas path:     qiskit/schemas
    SCHEMAS = os.path.join(SDK, 'schemas')


class QiskitTestCase(unittest.TestCase):
    """Helper class that contains common functionality."""

    @classmethod
    def setUpClass(cls):
        cls.moduleName = os.path.splitext(inspect.getfile(cls))[0]
        cls.log = logging.getLogger(cls.__name__)
        # Determines if the TestCase is using IBMQ credentials.
        cls.using_ibmq_credentials = False

        # Set logging to file and stdout if the LOG_LEVEL environment variable
        # is set.
        if os.getenv('LOG_LEVEL'):
            # Set up formatter.
            log_fmt = ('{}.%(funcName)s:%(levelname)s:%(asctime)s:'
                       ' %(message)s'.format(cls.__name__))
            formatter = logging.Formatter(log_fmt)

            # Set up the file handler.
            log_file_name = '%s.log' % cls.moduleName
            file_handler = logging.FileHandler(log_file_name)
            file_handler.setFormatter(formatter)
            cls.log.addHandler(file_handler)

            # Set the logging level from the environment variable, defaulting
            # to INFO if it is not a valid level.
            level = logging._nameToLevel.get(os.getenv('LOG_LEVEL'),
                                             logging.INFO)
            cls.log.setLevel(level)

    def tearDown(self):
        # Reset the default provider, as in practice it acts as a singleton
        # due to importing the wrapper from qiskit.
        from qiskit.wrapper import _wrapper
        _wrapper._DEFAULT_PROVIDER = DefaultQISKitProvider()

    @staticmethod
    def _get_resource_path(filename, path=Path.TEST):
        """ Get the absolute path to a resource.

        Args:
            filename (string): filename or relative path to the resource.
            path (Path): path used as relative to the filename.
        Returns:
            str: the absolute path to the resource.
        """
        return os.path.normpath(os.path.join(path.value, filename))

    def assertNoLogs(self, logger=None, level=None):
        """
        Context manager to test that no message is sent to the specified
        logger and level (the opposite of TestCase.assertLogs()).
        """
        return _AssertNoLogsContext(self, logger, level)

    def assertDictAlmostEqual(self, dict1, dict2, delta=None, msg=None,
                              places=None, default_value=0):
        """
        Assert two dictionaries with numeric values are almost equal.

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
        if dict1 == dict2:
            # Shortcut
            return
        if delta is not None and places is not None:
            raise TypeError("specify delta or places not both")

        if places is not None:
            success = True
            standard_msg = ''
            # check value for keys in target
            keys1 = set(dict1.keys())
            for key in keys1:
                val1 = dict1.get(key, default_value)
                val2 = dict2.get(key, default_value)
                if round(abs(val1 - val2), places) != 0:
                    success = False
                    standard_msg += '(%s: %s != %s), ' % (safe_repr(key),
                                                          safe_repr(val1),
                                                          safe_repr(val2))
            # check values for keys in counts, not in target
            keys2 = set(dict2.keys()) - keys1
            for key in keys2:
                val1 = dict1.get(key, default_value)
                val2 = dict2.get(key, default_value)
                if round(abs(val1 - val2), places) != 0:
                    success = False
                    standard_msg += '(%s: %s != %s), ' % (safe_repr(key),
                                                          safe_repr(val1),
                                                          safe_repr(val2))
            if success is True:
                return
            standard_msg = standard_msg[:-2] + ' within %s places' % places

        else:
            if delta is None:
                delta = 1e-8  # default delta value
            success = True
            standard_msg = ''
            # check value for keys in target
            keys1 = set(dict1.keys())
            for key in keys1:
                val1 = dict1.get(key, default_value)
                val2 = dict2.get(key, default_value)
                if abs(val1 - val2) > delta:
                    success = False
                    standard_msg += '(%s: %s != %s), ' % (safe_repr(key),
                                                          safe_repr(val1),
                                                          safe_repr(val2))
            # check values for keys in counts, not in target
            keys2 = set(dict2.keys()) - keys1
            for key in keys2:
                val1 = dict1.get(key, default_value)
                val2 = dict2.get(key, default_value)
                if abs(val1 - val2) > delta:
                    success = False
                    standard_msg += '(%s: %s != %s), ' % (safe_repr(key),
                                                          safe_repr(val1),
                                                          safe_repr(val2))
            if success is True:
                return
            standard_msg = standard_msg[:-2] + ' within %s delta' % delta

        msg = self._formatMessage(msg, standard_msg)
        raise self.failureException(msg)


class JobTestCase(QiskitTestCase):
    """Include common functionality when testing jobs."""

    def wait_for_initialization(self, job, timeout=1):
        """Waits until the job progress from `INITIALIZING` to a different
        status."""
        waited = 0
        wait = 0.1
        while job.status['status'] == JobStatus.INITIALIZING:
            time.sleep(wait)
            waited += wait
            if waited > timeout:
                self.fail(
                    msg="The JOB is still initializing after timeout ({}s)"
                    .format(timeout)
                )

    def assertStatus(self, job, status):
        """Assert the intenal job status is the expected one and also tests
        if the shorthand method for that status returns `True`."""
        self.assertEqual(job.status['status'], status)
        if status == JobStatus.CANCELLED:
            self.assertTrue(job.cancelled)
        elif status == JobStatus.DONE:
            self.assertTrue(job.done)
        elif status == JobStatus.VALIDATING:
            self.assertTrue(job.validating)
        elif status == JobStatus.RUNNING:
            self.assertTrue(job.running)
        elif status == JobStatus.QUEUED:
            self.assertTrue(job.queued)


class _AssertNoLogsContext(unittest.case._AssertLogsContext):
    """A context manager used to implement TestCase.assertNoLogs()."""

    # pylint: disable=inconsistent-return-statements
    def __exit__(self, exc_type, exc_value, tb):
        """
        This is a modified version of TestCase._AssertLogsContext.__exit__(...)
        """
        self.logger.handlers = self.old_handlers
        self.logger.propagate = self.old_propagate
        self.logger.setLevel(self.old_level)
        if exc_type is not None:
            # let unexpected exceptions pass through
            return False

        if self.watcher.records:
            msg = 'logs of level {} or higher triggered on {}:\n'.format(
                logging.getLevelName(self.level), self.logger.name)
            for record in self.watcher.records:
                msg += 'logger %s %s:%i: %s\n' % (record.name, record.pathname,
                                                  record.lineno,
                                                  record.getMessage())

            self._raiseFailure(msg)


def slow_test(func):
    """
    Decorator that signals that the test takes minutes to run.

    Args:
        func (callable): test function to be decorated.

    Returns:
        callable: the decorated function.
    """

    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        if SKIP_SLOW_TESTS:
            raise unittest.SkipTest('Skipping slow tests')
        return func(*args, **kwargs)

    return _wrapper


def is_cpp_simulator_available():
    """
    Check if executable for C++ simulator is available in the expected
    location.

    Returns:
        bool: True if simulator executable is available
    """
    try:
        QasmSimulatorCpp()
    except FileNotFoundError:
        return False
    return True


def requires_cpp_simulator(test_item):
    """
    Decorator that skips test if C++ simulator is not available

    Args:
        test_item (callable): function or class to be decorated.

    Returns:
        callable: the decorated function.
    """
    reason = 'C++ simulator not found, skipping test'
    return unittest.skipIf(not is_cpp_simulator_available(), reason)(test_item)


def requires_qe_access(func):
    """
    Decorator that signals that the test uses the online API:
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
    def _wrapper(*args, **kwargs):
        if SKIP_ONLINE_TESTS:
            raise unittest.SkipTest('Skipping online tests')

        # Cleanup the credentials, as this file is shared by the tests.
        from qiskit.wrapper import _wrapper
        _wrapper._DEFAULT_PROVIDER = DefaultQISKitProvider()

        if os.getenv('USE_ALTERNATE_ENV_CREDENTIALS', False):
            # Special case: instead of using the standard credentials mechanism,
            # load them from different environment variables. This assumes they
            # will always be in place, as is used by the Travis setup.
            kwargs.update({
                'qe_token': os.getenv('IBMQ_TOKEN'),
                'qe_url': os.getenv('IBMQ_URL')
            })
            args[0].using_ibmq_credentials = True
        else:
            # Attempt to read the standard credentials.
            account_name = get_account_name(IBMQProvider)
            discovered_credentials = discover_credentials()
            if account_name in discovered_credentials.keys():
                credentials = discovered_credentials[account_name]

                kwargs.update({
                    'qe_token': credentials.get('token'),
                    'qe_url': credentials.get('url'),
                })

                if all(item in credentials.get('url') for
                       item in ['Hubs', 'Groups', 'Projects']):
                    args[0].using_ibmq_credentials = True
            else:
                raise Exception('Could not locate valid credentials')

        return func(*args, **kwargs)

    return _wrapper


def _is_ci_fork_pull_request():
    """
    Check if the tests are being run in a CI environment and if it is a pull
    request.

    Returns:
        bool: True if the tests are executed inside a CI tool, and the changes
            are not against the "master" branch.
    """
    if os.getenv('TRAVIS'):
        # Using Travis CI.
        if os.getenv('TRAVIS_PULL_REQUEST_BRANCH'):
            return True
    elif os.getenv('APPVEYOR'):
        # Using AppVeyor CI.
        if os.getenv('APPVEYOR_PULL_REQUEST_NUMBER'):
            return True
    return False


SKIP_ONLINE_TESTS = os.getenv('SKIP_ONLINE_TESTS', _is_ci_fork_pull_request())
SKIP_SLOW_TESTS = os.getenv('SKIP_SLOW_TESTS', True) not in ['false', 'False', '-1']

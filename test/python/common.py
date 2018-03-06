# -*- coding: utf-8 -*-

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Shared functionality and helpers for the unit tests."""

from enum import Enum
import functools
import inspect
import logging
import os
import unittest

from qiskit import __path__ as qiskit_path


class Path(Enum):
    """Helper with paths commonly used during the tests."""
    # Main SDK path:    qiskit/
    SDK = qiskit_path[0]
    # test.python path: qiskit/test/python/
    TEST = os.path.dirname(__file__)
    # Examples path:    examples/
    EXAMPLES = os.path.join(SDK, '../examples')


class QiskitTestCase(unittest.TestCase):
    """Helper class that contains common functionality."""
    @classmethod
    def setUpClass(cls):
        cls.moduleName = os.path.splitext(inspect.getfile(cls))[0]
        cls.log = logging.getLogger(cls.__name__)

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
        # pylint: disable=invalid-name
        return _AssertNoLogsContext(self, logger, level)


class _AssertNoLogsContext(unittest.case._AssertLogsContext):
    """A context manager used to implement TestCase.assertNoLogs()."""

    LOGGING_FORMAT = "%(levelname)s:%(name)s:%(message)s"

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
        for record in self.watcher.records:
            self._raiseFailure(
                "Something was logged in the logger %s by %s:%i" %
                (record.name, record.pathname, record.lineno))


def requires_qe_access(func):
    """
    Decorator that signals that the test uses the online API:
        * determines if the test should be skipped by checking environment
            variables.
        * if the test is not skipped, it reads `QE_TOKEN` and `QE_URL` from
            `Qconfig.py` or from environment variables.
        * if the test is not skipped, it appends `QE_TOKEN` and `QE_URL` as
            arguments to the test function.
    Args:
        func (callable): test function to be decorated.

    Returns:
        callable: the decorated function.
    """
    @functools.wraps(func)
    def _(*args, **kwargs):
        # pylint: disable=invalid-name
        if SKIP_ONLINE_TESTS:
            raise unittest.SkipTest('Skipping online tests')

        # Try to read the variables from Qconfig.
        try:
            import Qconfig
            QE_TOKEN = Qconfig.APItoken
            QE_URL = Qconfig.config["url"]
        except ImportError:
            # Try to read them from environment variables (ie. Travis).
            QE_TOKEN = os.getenv('QE_TOKEN')
            QE_URL = os.getenv('QE_URL')

        if not QE_TOKEN or not QE_URL:
            raise Exception(
                'Could not locate a valid "Qconfig.py" file nor read the QE '
                'values from the environment')

        kwargs['QE_TOKEN'] = QE_TOKEN
        kwargs['QE_URL'] = QE_URL
        return func(*args, **kwargs)

    return _


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
        if (os.getenv('TRAVIS_REPO_SLUG') !=
                os.getenv('TRAVIS_PULL_REQUEST_SLUG')):
            return True
    elif os.getenv('APPVEYOR'):
        # Using AppVeyor CI.
        if os.getenv('APPVEYOR_PULL_REQUEST_NUMBER'):
            return True
    return False


SKIP_ONLINE_TESTS = os.getenv('SKIP_ONLINE_TESTS', _is_ci_fork_pull_request())

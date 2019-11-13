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

"""Utils for using with Qiskit unit tests."""

import logging
import os
import unittest
from enum import Enum
from itertools import product

from qiskit import __path__ as qiskit_path


class Path(Enum):
    """Helper with paths commonly used during the tests."""

    # Main SDK path:    qiskit/
    SDK = qiskit_path[0]
    # test.python path: qiskit/test/python/
    TEST = os.path.normpath(os.path.join(SDK, '..', 'test', 'python'))
    # Examples path:    examples/
    EXAMPLES = os.path.normpath(os.path.join(SDK, '..', 'examples'))
    # Schemas path:     qiskit/schemas
    SCHEMAS = os.path.normpath(os.path.join(SDK, 'schemas'))
    # Sample QASMs path: qiskit/test/python/qasm
    QASMS = os.path.normpath(os.path.join(TEST, 'qasm'))


def setup_test_logging(logger, log_level, filename):
    """Set logging to file and stdout for a logger.

    Args:
        logger (Logger): logger object to be updated.
        log_level (str): logging level.
        filename (str): name of the output file.
    """
    # Set up formatter.
    log_fmt = ('{}.%(funcName)s:%(levelname)s:%(asctime)s:'
               ' %(message)s'.format(logger.name))
    formatter = logging.Formatter(log_fmt)

    # Set up the file handler.
    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if os.getenv('STREAM_LOG'):
        # Set up the stream handler.
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # Set the logging level from the environment variable, defaulting
    # to INFO if it is not a valid level.
    level = logging._nameToLevel.get(log_level, logging.INFO)
    logger.setLevel(level)


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


class Case(dict):
    """ A test case, see https://ddt.readthedocs.io/en/latest/example.html MyList."""
    pass


def generate_cases(dsc=None, name=None, **kwargs):
    """Combines kwargs in cartesian product and creates Case with them"""
    ret = []
    keys = kwargs.keys()
    vals = kwargs.values()
    for values in product(*vals):
        case = Case(zip(keys, values))
        if dsc is not None:
            setattr(case, "__doc__", dsc.format(**case))
        if name is not None:
            setattr(case, "__name__", name.format(**case))
        ret.append(case)
    return ret

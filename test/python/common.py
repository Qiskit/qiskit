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
import inspect
import logging
import os
import unittest

from qiskit import __path__ as qiskit_path


TRAVIS_FORK_PULL_REQUEST = False
if os.getenv('TRAVIS_PULL_REQUEST_SLUG'):
    if os.getenv('TRAVIS_REPO_SLUG') != os.getenv('TRAVIS_PULL_REQUEST_SLUG'):
        TRAVIS_FORK_PULL_REQUEST = True

LOG_LEVEL = logging.CRITICAL
if os.getenv('LOG_LEVEL'):
    toset = os.getenv('LOG_LEVEL')
    goodlevels = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']
    if not toset in goodlevels:
        raise Exception("The env variable LOG_LEVEL is %s instead of something in %s" % (toset,goodlevels   ))
    LOG_LEVEL = getattr(logging,toset)

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
        cls.log.setLevel(LOG_LEVEL)
        log_fmt = ('{}.%(funcName)s:%(levelname)s:%(asctime)s:'
                   ' %(message)s'.format(cls.__name__))
        formatter = logging.Formatter(log_fmt)

        # logger for the stdout
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        cls.log.addHandler(stream_handler)

    @staticmethod
    def _get_resource_path(filename, path=Path.TEST):
        """ Get the absolute path to a resource.

        Args:
            filename (string): filename or relative path to the resource.
            path (Path): path used as relative to the filename.
        """
        return os.path.normpath(os.path.join(path.value, filename))

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

import inspect
import logging
import os
import unittest


class QiskitTestCase(unittest.TestCase):
    """Helper class that contains common functionality."""
    @classmethod
    def setUpClass(cls):
        # Setup logging to a file 'test_xxx.log'
        cls.moduleName = os.path.splitext(inspect.getfile(cls))[0]
        cls.log = logging.getLogger(cls.__name__)
        cls.log.setLevel(logging.INFO)
        logFileName = cls.moduleName + '.log'
        handler = logging.FileHandler(logFileName)
        handler.setLevel(logging.INFO)
        log_fmt = ('{}.%(funcName)s:%(levelname)s:%(asctime)s:'
                   ' %(message)s'.format(cls.__name__))
        formatter = logging.Formatter(log_fmt)
        handler.setFormatter(formatter)
        cls.log.addHandler(handler)

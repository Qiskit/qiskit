# -*- coding: utf-8 -*-
# pylint: disable=invalid-name,missing-docstring

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

import os
import pprint
import unittest

from qiskit.transpiler import (StageBase, Pipeline)
from .common import QiskitTestCase, TRAVIS_FORK_PULL_REQUEST


class TestTranspiler(QiskitTestCase):
    """
    Test job_pocessor module.
    """

    @classmethod
    def setUpClass(cls):
        super(TestTranspiler, cls).setUpClass()

        try:
            import Qconfig
            cls.QE_TOKEN = Qconfig.APItoken
            cls.QE_URL = Qconfig.config['url']
        except ImportError:
            if 'QE_TOKEN' in os.environ:
                cls.QE_TOKEN = os.environ['QE_TOKEN']
            if 'QE_URL' in os.environ:
                cls.QE_URL = os.environ['QE_URL']

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass


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
"""QISKit unit tests."""

import os


def load_tests(loader, standard_tests, pattern):
    """
    test suite for unittest discovery
    """
    this_dir = os.path.dirname(__file__)
    if pattern in ['test*.py', '*_test.py']:
        package_tests = loader.discover(start_dir=this_dir, pattern=pattern)
        standard_tests.addTests(package_tests)
    elif pattern in ['profile*.py', '*_profile.py']:
        loader.testMethodPrefix = 'profile'
        package_tests = loader.discover(start_dir=this_dir, pattern='test*.py')
        standard_tests.addTests(package_tests)
    return standard_tests

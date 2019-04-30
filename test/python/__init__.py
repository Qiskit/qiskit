# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Qiskit unit tests."""

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
